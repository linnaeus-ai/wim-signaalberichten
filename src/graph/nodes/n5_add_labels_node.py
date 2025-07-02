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
            print(f"Topics file not found at {self._topics_path}")
            return {}
        except Exception as e:
            print(f"Error reading topics file: {e}")
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
                print(f"    Found topic set name: {topic_set_name} for label: {label}")
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
            print("    → Node 5: AddLabelsNode ... Adding labels to the json-ld")

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

            # Formate the human prompt
            topics_list = list(chain.from_iterable(self._topics_dict.values()))
            human_prompt = ADD_LABELS_HUMAN_PROMPT.format(
                TOPICS_LIST="\n".join(topics_list),
                INPUT_TEXT=state.text,
            )

            # Invoke the llm
            response = llm.invoke(
                [
                    SystemMessage(content=ADD_LABELS_SYSTEM_PROMPT),
                    HumanMessage(content=human_prompt),
                ]
            )

            labels = response.labels

            # Check if labels are actually from the topics list
            labels = [
                label
                for label in labels
                if label in topics_list
                or print(
                    f"    Found label '{label}' not in topics list. Skipping this label."
                )
            ]

            # Add labels and their sets to the last json-ld in the state
            state.json_ld_contents[-1] = _make_labels_jsonld_str(
                state, self._topics_dict, labels
            )

            return state

        return _node
