import os
import re
import yaml
import uuid
import json

from graph.text_to_kg_state import TextToKGState
from graph.base import BaseNode
from graph.utils.logging_llm_wrapper import LoggingLLMWrapper
from graph.utils.schema_tools import ultra_shorten_schema_yaml
from graph.prompts import (
    TRANSFORM_TO_KG_SYSTEM_PROMPT,
    TRANSFORM_TO_KG_HUMAN_PROMPT,
)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.base import BaseLanguageModel


class TransformToKGNode(BaseNode):
    name: str = "TransformToKGNode"

    def __init__(self, llm: BaseLanguageModel):
        """Transforms the input text and schema to a JSON-LD knowledge graph."""
        super().__init__(llm)

    def get_node(self):
        """..."""

        def _node(state: TextToKGState) -> TextToKGState:
            """
            ...

            Args:
                state (TextToKGState): The current state of the pipeline.

            Returns:
                TextToKGState: The updated state.
            """

            print(
                f"\033[93m    → Node 3: TransformToKGNode (validation runs: {state.validation_runs})\033[0m"
            )

            # Check if entity extraction failed
            if state.entity_extraction_failed:
                print("\033[93m    → Entity extraction failed - creating minimal JSON-LD\033[0m")
                minimal_jsonld = json.dumps({
                    "@context": "https://schema.org",
                    "@type": "Thing"
                }, indent=2)
                state.json_ld_contents = [minimal_jsonld]
                return state

            # Build mapping of schema classes to entities
            entities = state.entity_extraction_output["entities"]
            schema_to_entities = {}
            for entity_name, class_name, description in entities:
                if class_name in state.schema_definitions:
                    if class_name not in schema_to_entities:
                        schema_to_entities[class_name] = []
                    schema_to_entities[class_name].append(entity_name)

            # Extract relations from n1 output
            relations = state.entity_extraction_output.get("relations", [])
            formatted_relations = []
            for relation in relations:
                if len(relation) >= 3:  # Ensure we have subject, predicate, object
                    formatted_relations.append(
                        f"- {relation[0]} → {relation[1]} → {relation[2]}"
                    )
            relations_text = (
                "\n".join(formatted_relations)
                if formatted_relations
                else "Geen expliciete relaties gevonden"
            )

            # Format the human prompt
            formatted_schemas_list = []
            for key, value in state.schema_definitions.items():
                try:
                    # Ultra-shorten the YAML for more efficient prompting
                    ultra_shortened_yaml = ultra_shorten_schema_yaml(value)

                    # Parse to get the label
                    parsed_yaml = yaml.safe_load(value)

                    # Add entity associations if they exist
                    if key in schema_to_entities and schema_to_entities[key]:
                        entity_list = "\n".join(
                            f"- {entity}" for entity in schema_to_entities[key]
                        )
                        schema_header = f"**Schema: {parsed_yaml.get('label', key)}**\nEntiteiten die dit schema gebruiken:\n{entity_list}\n\n```yaml\n{ultra_shortened_yaml}\n```"
                    else:
                        schema_header = f"**Schema: {parsed_yaml.get('label', key)}**\n\n```yaml\n{ultra_shortened_yaml}\n```"

                    formatted_schemas_list.append(schema_header)
                except yaml.YAMLError as e:
                    print(
                        f"\033[93mYAML parsing error for schema '{key}': {e}. Using raw content.\033[0m"
                    )
                    # Try to ultra-shorten even if YAML parsing failed
                    try:
                        ultra_shortened_yaml = ultra_shorten_schema_yaml(value)
                    except:
                        ultra_shortened_yaml = value  # Fallback to original

                    # Use the content with entity associations
                    if key in schema_to_entities and schema_to_entities[key]:
                        entity_list = "\n".join(
                            f"- {entity}" for entity in schema_to_entities[key]
                        )
                        schema_header = f"**Schema: {key}**\nEntiteiten die dit schema gebruiken:\n{entity_list}\n\n```\n{ultra_shortened_yaml}\n```"
                    else:
                        schema_header = (
                            f"**Schema: {key}**\n\n```\n{ultra_shortened_yaml}\n```"
                        )
                    formatted_schemas_list.append(schema_header)
                except Exception as e:
                    print(
                        f"\033[91mUnexpected error formatting schema '{key}': {e}. Skipping this schema.\033[0m"
                    )
                    continue
            formatted_schemas = "\n\n".join(formatted_schemas_list)

            # Check if this is a rerun or not
            if state.validation_runs > 0:
                # Read the previously generated JSON-LD from state (not file system)
                previous_json_ld = ""
                if state.json_ld_contents:
                    previous_json_ld = state.json_ld_contents[-1]

                rerun_message = f"""**VALIDATIE MISLUKT - Corrigeer de volgende fouten:**

**Vorige JSON-LD die validatie fouten bevat:**
```json
{previous_json_ld}
```

**Validatie errors:**
{state.validation_output[-1] if state.validation_output else "Geen validatie output beschikbaar"}

Analyseer de fouten en genereer een GECORRIGEERDE versie van de JSON-LD."""

                human_prompt = TRANSFORM_TO_KG_HUMAN_PROMPT.format(
                    INPUT_TEXT=state.text,
                    SCHEMA_DEFINITIONS=formatted_schemas,
                    EXTRACTED_RELATIONS=relations_text,
                    RERUN=rerun_message,
                ).strip()
            else:
                human_prompt = TRANSFORM_TO_KG_HUMAN_PROMPT.format(
                    INPUT_TEXT=state.text,
                    SCHEMA_DEFINITIONS=formatted_schemas,
                    EXTRACTED_RELATIONS=relations_text,
                    RERUN="",
                ).strip()
            # import pydevd_pycharm
            # pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

            # Wrap LLM with logging if db path available
            llm_to_use = self._llm
            if state.db_path and state.wiki_id and state.worker_id:
                llm_to_use = LoggingLLMWrapper(
                    base_llm=self._llm,
                    db_path=state.db_path,
                    source_id=state.wiki_id,
                    call_name=f"n3_transform_to_kg_run_{state.validation_runs}",
                    worker_id=state.worker_id
                )

            # Invoke the llm
            response = llm_to_use.invoke(
                [
                    SystemMessage(content=TRANSFORM_TO_KG_SYSTEM_PROMPT),
                    HumanMessage(content=human_prompt),
                ]
            )

            # Ensure the response is only a dict
            def extract_dict_from_string(json_string):
                # Use regular expression to find the JSON part
                pattern = r"(\{.*\})"
                match = re.search(pattern, json_string, re.DOTALL)
                if match:
                    json_part = match.group(0)
                    return json_part
                else:
                    return None

            # Extract the JSON-LD from the response
            json_ld = extract_dict_from_string(response.content)

            if json_ld is None:
                raise Exception("The response does not contain a valid JSON-LD object.")

            # Write the json-ld to a temp file with a uuid
            file_path = f"src/data/tmp/{uuid.uuid1()}.json"

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(json_ld)

            # Update the state with both the file path AND content
            state.json_ld_paths.append(file_path)
            state.json_ld_contents.append(json_ld)
            print(f"\033[92m    → Saved JSON-LD to {file_path}\033[0m")
            return state

        return _node
