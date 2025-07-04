import json
import pandas as pd

from itertools import chain

from graph.text_to_kg_state import TextToKGState
from graph.base import BaseNode
from graph.utils.logging_llm_wrapper import LoggingLLMWrapper
from graph.prompts import (
    ADD_LABELS_HUMAN_PROMPT,
    ADD_LABELS_SYSTEM_PROMPT,
)
from graph.utils.models import AddLabelsStructuredOutput

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.base import BaseLanguageModel


class AddLabelsNode(BaseNode):
    name: str = "AddLabelsNode"
    _topics_path: str = "src/data/Hoofdklantsignalen - Subklantsignalen.xlsx"

    def __init__(self, llm: BaseLanguageModel):
        """Adds labels to the JSON-LD knowledge graph."""
        self._topics_dict = self._get_topics_dict()

        super().__init__(llm)

    def _get_topics_dict(self):
        """Returns a dictionary of topics from Excel file."""
        try:
            # Read Excel file
            df = pd.read_excel(self._topics_path)
            
            # Group subtopics by main topic
            topics_dict = df.groupby('Hoofd_klantsignaal')['Sub_klantsignaal'].apply(list).to_dict()
            
            return topics_dict
        except FileNotFoundError:
            print(f"\033[91mTopics file not found at {self._topics_path}\033[0m")
            return {}
        except Exception as e:
            print(f"\033[91mError reading topics file: {e}\033[0m")
            return {}

    def get_node(self):
        """..."""

        def find_topic_set_name(topic_label: str, topic_dict: dict) -> str:
            for topic_set_name, topic_labels in topic_dict.items():
                if topic_label in topic_labels:
                    return topic_set_name
            return None

        def _make_labels_jsonld_str(
            state: TextToKGState, topics_dict: dict, labels: list
        ) -> str:
            """
            Adds labels to the about section of the JSON-LD in the state.
            Args:
                state (TextToKGState): The current state of the pipeline.
                topics_dict (dict): Dictionary containing topic sets and their labels.
                labels (list): List of labels to be added.
            Returns:
                str: Updated JSON-LD string with labels added to the "about" key.
            """
            json_ld = json.loads(state.json_ld_contents[-1])

            # Chech if "about" key exists, if not, initialize it
            if "about" not in json_ld:
                json_ld["about"] = []

            # Add the labels to the "about" key
            for label in labels:
                topic_set_name = find_topic_set_name(label, topics_dict)
                
                # Special handling for "No subtopic found"
                if label == "No subtopic found" and topic_set_name is None:
                    topic_set_name = "No topic found"
                
                print(f"\033[92m    Found topic set name: \033[96m{topic_set_name}\033[0m \033[92mfor label: \033[96m{label}\033[0m")
                if topic_set_name:

                    # Check if "about" value is list
                    if not isinstance(json_ld["about"], list):
                        json_ld["about"] = [json_ld["about"]]

                    json_ld["about"].append(
                        {
                            "@type": "DefinedTerm",
                            "name": label,
                            "inDefinedTermSet": {
                                "@type": "DefinedTermSet",
                                "name": topic_set_name,
                            },
                        }
                    )
            return json.dumps(json_ld)

        def _node(state: TextToKGState) -> TextToKGState:
            """
            ...

            Args:
                state (TextToKGState): The current state of the pipeline.

            Returns:
                TextToKGState: The updated state.
            """
            print("\033[93m    → Node 5: AddLabelsNode ... Adding labels to the json-ld\033[0m")

            # Check if entity extraction failed
            if state.entity_extraction_failed:
                print("    → Entity extraction failed - adding 'No subtopic found' label")
                # Add the hardcoded label to the minimal JSON-LD
                state.json_ld_contents[-1] = _make_labels_jsonld_str(
                    state, self._topics_dict, ["No subtopic found"]
                )
                return state

            # First wrap the base LLM if logging is available
            llm_to_use = self._llm
            if state.db_path and state.wiki_id and state.worker_id:
                llm_to_use = LoggingLLMWrapper(
                    base_llm=self._llm,
                    db_path=state.db_path,
                    source_id=state.wiki_id,
                    call_name="n5_add_labels",
                    worker_id=state.worker_id,
                )

            # Then bind structured model to the (possibly wrapped) llm
            llm = llm_to_use.with_structured_output(AddLabelsStructuredOutput)

            # Define onderwerp and beleving categories
            ONDERWERP_SIGNALS = [
                "Bouwen en verbouwen",
                "Burgerzaken",
                "Dagelijks leven en sociale gelegenheden",
                "Financiële ondersteuning",
                "Maatschappelijke ondersteuning",
                "No topic found",
                "Opruimen, afval en onderhoud",
                "Parkeren",
                "Veiligheid en omgeving",
                "Vervoer",
                "Werk",
                "Wonenen en ondernemen",
                "Zorg",
            ]
            
            BELEVING_SIGNALS = [
                "Informatievoorziening",
                "Houding & Gedrag medewerker",
                "Fysieke dienstverlening",
                "Digitale mogelijkheden",
                "Contact leggen met medewerker",
                "Algemene ervaring",
                "Afhandeling",
                "Processen",
                "Prijs & Kwaliteit",
                "No topic found",
                "Kennis & Vaardigheden medewerker",
            ]

            # Separate topics by category
            onderwerp_labels = []
            beleving_labels = []
            
            for hoofd_signal, sub_signals in self._topics_dict.items():
                if hoofd_signal in ONDERWERP_SIGNALS:
                    onderwerp_labels.extend(sub_signals)
                elif hoofd_signal in BELEVING_SIGNALS:
                    beleving_labels.extend(sub_signals)
            
            # Add "No subtopic found" if not already present
            if "No subtopic found" not in onderwerp_labels:
                onderwerp_labels.append("No subtopic found")
            if "No subtopic found" not in beleving_labels:
                beleving_labels.append("No subtopic found")

            # Format the human prompt
            human_prompt = ADD_LABELS_HUMAN_PROMPT.format(
                ONDERWERP_LIST="\n".join(onderwerp_labels),
                BELEVING_LIST="\n".join(beleving_labels),
                INPUT_TEXT=state.text,
            )

            # Invoke the llm
            response = llm.invoke(
                [
                    SystemMessage(content=ADD_LABELS_SYSTEM_PROMPT),
                    HumanMessage(content=human_prompt),
                ]
            )

            # Extract labels from both categories
            onderwerp_response = response.onderwerp_labels
            beleving_response = response.beleving_labels

            # Create combined topics list for validation
            all_valid_onderwerp = onderwerp_labels
            all_valid_beleving = beleving_labels
            
            # Validate onderwerp labels
            validated_onderwerp = [
                label
                for label in onderwerp_response
                if label in all_valid_onderwerp
                or print(
                    f"    Found onderwerp label '{label}' not in topics list. Skipping this label."
                )
            ]
            
            # Validate beleving labels
            validated_beleving = [
                label
                for label in beleving_response
                if label in all_valid_beleving
                or print(
                    f"    Found beleving label '{label}' not in topics list. Skipping this label."
                )
            ]
            
            # Print selected labels for clarity
            if validated_onderwerp:
                print(f"    → Selected onderwerp labels: {', '.join(validated_onderwerp)}")
            else:
                print(f"    → No onderwerp labels selected")
                
            if validated_beleving:
                print(f"    → Selected beleving labels: {', '.join(validated_beleving)}")
            else:
                print(f"    → No beleving labels selected")
            
            # Combine validated labels for JSON-LD generation
            labels = validated_onderwerp + validated_beleving

            # Add labels and their sets to the last json-ld in the state
            state.json_ld_contents[-1] = _make_labels_jsonld_str(
                state, self._topics_dict, labels
            )

            return state

        return _node
