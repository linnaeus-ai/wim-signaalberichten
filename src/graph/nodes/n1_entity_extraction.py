import re

from graph.base import BaseNode
from graph.utils.logging_llm_wrapper import LoggingLLMWrapper
from graph.text_to_kg_state import TextToKGState
from graph.prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_HUMAN_PROMPT,
)

from typing import Callable
from langchain_core.messages import SystemMessage, HumanMessage


class EntityExtraction(BaseNode):
    name: str = "EntityExtractionNode"

    def _structure_output(self, response: str) -> dict:
        """
        Parse the LLM response and structure it into summary, entities, and relations.

        Args:
            response (str): Raw response from the LLM

        Returns:
            dict: Structured output with summary, entities, and relations
        """
        # Extract summary
        summary_match = re.search(r"<summary>\n(.*?)\n</summary>", response, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        # Extract entities from CSV section
        entities_pattern = r"<entiteiten>\n(.*?)\n</entiteiten>"
        entities_match = re.search(entities_pattern, response, re.DOTALL)
        entities = []

        if entities_match:
            entities_csv = entities_match.group(1).strip()
            for line in entities_csv.split("\n"):
                if line.strip():
                    parts = [part.strip() for part in line.split(" | ")]
                    if len(parts) >= 3:
                        entities.append([parts[0], parts[1], parts[2]])

        # Extract relations from CSV section
        relations_pattern = r"<relaties>\n(.*?)\n</relaties>"
        relations_match = re.search(relations_pattern, response, re.DOTALL)
        relations = []

        if relations_match:
            relations_csv = relations_match.group(1).strip()
            for line in relations_csv.split("\n"):
                if line.strip():
                    parts = [part.strip() for part in line.split(" | ")]
                    if len(parts) >= 3:
                        relations.append([parts[0], parts[1], parts[2]])

        if summary == "" or entities == [] or relations == []:
            raise ValueError("Incomplete response structure from LLM")

        return {
            "summary": summary,
            "entities": tuple(
                entities
            ),  # (EntiteitNaam, SpecifiekeClass, Engelse beschrijving van de class)
            "relations": tuple(relations),  # (EntiteitA, relatieType, EntiteitB)
        }

    def get_node(self) -> Callable[[TextToKGState], TextToKGState]:
        """
        Get the node function for entity extraction.
        Returns:
            Callable[[TextToKGState], TextToKGState]: A callable that processes the state and returns extracted entities.
        """

        def _node(state: TextToKGState) -> TextToKGState:
            """..."""
            print("    → Running Node 1: EntityExtractionNode...")

            # Wrap LLM with logging if db path available
            llm_to_use = self._llm
            if state.db_path and state.wiki_id and state.worker_id:
                llm_to_use = LoggingLLMWrapper(
                    base_llm=self._llm,
                    db_path=state.db_path,
                    source_id=state.wiki_id,
                    call_name="n1_entity_extraction",
                    worker_id=state.worker_id
                )
            # NOTE: We don't use a structured output because it drastically decreases the specificity of the results
            response = llm_to_use.invoke(
                [
                    SystemMessage(content=ENTITY_EXTRACTION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=ENTITY_EXTRACTION_HUMAN_PROMPT.format(
                            INPUT_TEXT=state.text,
                        )
                    ),
                ]
            )
            if response.content:
                state.entity_extraction_output = self._structure_output(
                    response.content
                )
            else:
                raise Exception("No entities extracted from the input text.")
            return state

        return _node
