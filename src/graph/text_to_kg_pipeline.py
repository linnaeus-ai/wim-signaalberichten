import os
import logging

from typing import Optional, Union, Dict
from langgraph.graph import END, START, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel

from graph.text_to_kg_state import TextToKGState
from graph.nodes import (
    EntityExtraction,
    RetrieveSchemaOrgNode,
    TransformToKGNode,
    ValidatorNode,
    AddLabelsNode,
)
from graph.routers.validator_router import ValidatorRouter

from dotenv import load_dotenv

load_dotenv()


class TextToKGPipeline:
    """Pipeline for converting text to a knowledge graph using various processing nodes."""

    _llms: Dict[str, BaseChatModel]
    _add_labels: bool

    def __init__(self, llm: Union[BaseChatModel, Dict[str, BaseChatModel]], add_labels: bool = False):
        """
        Initialize the pipeline with either a single LLM or node-specific LLMs.
        
        Args:
            llm: Either a single BaseChatModel to use for all nodes, or a dict mapping
                node names to specific models. Valid node names: 'n1', 'n2', 'n3', 'n5'
            add_labels: Whether to include the label addition node
        """
        if not llm:
            raise ValueError("Language model is required")

        # Convert single LLM to dict format for consistency
        if isinstance(llm, BaseChatModel):
            self._llms = {
                'n1': llm,  # EntityExtraction
                'n2': llm,  # RetrieveSchemaOrgNode
                'n3': llm,  # TransformToKGNode
                'n5': llm   # AddLabelsNode
            }
        else:
            # Validate that all required nodes have LLMs
            required_nodes = ['n1', 'n2', 'n3']
            if add_labels:
                required_nodes.append('n5')
            
            missing_nodes = set(required_nodes) - set(llm.keys())
            if missing_nodes:
                raise ValueError(f"Missing LLM configuration for nodes: {missing_nodes}")
            
            self._llms = llm
        
        self._add_labels = add_labels

    @property
    def name(self) -> str:
        """Returns the name of the class.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__

    @property
    def logger(self):
        """
        Get a logger instance specific to this node.

        Returns:
            logging.Logger: A logger instance prefixed with NODE::{node_name}
        """
        return logging.getLogger(f"PIPELINE:{self.name}")

    def to_png(
        self,
        name: Optional[str] = None,
        save_path: str = "./tmp",
    ) -> None:
        """
        Export the pipeline as a PNG image.

        Args:
            name (Optional[str]): The name to use for the PNG file (including extension). If not provided, uses the pipeline name.
            path (str): The path to save the PNG image to.
        """
        name = name or self.name

        png_graph = self.compile().get_graph().draw_mermaid_png()
        file_path = os.path.join(save_path, f"{name}")
        with open(file_path, "wb") as f:
            f.write(png_graph)

        self.logger.info("Pipeline graph saved to %s", file_path)

    def compile(self):
        """
        Compile the text-to-KG pipeline into a state graph.
        Returns:
            The compiled state graph representing the text-to-KG pipeline.
        """
        workflow = StateGraph(TextToKGState)

        # Define all processing nodes with their specific LLMs
        entity_extraction_node = EntityExtraction(llm=self._llms['n1'])
        workflow.add_node(
            entity_extraction_node.name, entity_extraction_node.get_node()
        )

        retrieve_schema_org_node = RetrieveSchemaOrgNode(llm=self._llms['n2'])
        workflow.add_node(
            retrieve_schema_org_node.name, retrieve_schema_org_node.get_node()
        )

        transform_to_kg_node = TransformToKGNode(llm=self._llms['n3'])
        workflow.add_node(transform_to_kg_node.name, transform_to_kg_node.get_node())

        # Validator uses same LLM as n3 (not configurable separately)
        validator_node = ValidatorNode(llm=self._llms['n3'])
        workflow.add_node(validator_node.name, validator_node.get_node())

        # Define workflow graph edges
        workflow.add_edge(START, entity_extraction_node.name)
        workflow.add_edge(entity_extraction_node.name, retrieve_schema_org_node.name)
        workflow.add_edge(retrieve_schema_org_node.name, transform_to_kg_node.name)
        workflow.add_edge(transform_to_kg_node.name, validator_node.name)

        if self._add_labels:
            # Add labels node only if requested
            add_labels_node = AddLabelsNode(llm=self._llms['n5'])
            workflow.add_node(add_labels_node.name, add_labels_node.get_node())
            workflow.add_edge(add_labels_node.name, END)

            # Add a router to determine the next step based on validation
            workflow.add_conditional_edges(
                validator_node.name,
                ValidatorRouter().get_router(),
                {
                    "AddLabelsNode": add_labels_node.name,  # If validation is successful, add labels
                    "TransformToKGNode": transform_to_kg_node.name,  # If validation fails, retry transformation
                },
            )
        else:
            # Without labels, go directly from validator to end or retry
            workflow.add_conditional_edges(
                validator_node.name,
                ValidatorRouter().get_router(),
                {
                    "AddLabelsNode": END,  # If validation is successful, finish
                    "TransformToKGNode": transform_to_kg_node.name,  # If validation fails, retry transformation
                },
            )

        # Compile and return the final graph
        return workflow.compile()
