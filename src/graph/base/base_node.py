import logging
from abc import ABC, abstractmethod
from typing import Any
from langchain_openai import AzureChatOpenAI


class BaseNode(ABC):
    name: str
    _llm: AzureChatOpenAI

    def __init__(self, llm: AzureChatOpenAI):
        """
        Initialize an LLM node with a name and language model.

        Args:
            name (str): The name identifier for this node
            llm (AzureChatOpenAI): The language model instance to use

        Raises:
            ValueError: If llm is None
        """
        if not llm:
            raise ValueError("Language model is required")

        self._llm = llm

    @property
    def logger(self):
        """
        Get a logger instance specific to this node.

        Returns:
            logging.Logger: A logger instance prefixed with NODE::{node_name}
        """
        return logging.getLogger(f"NODE::{self.name}")

    @abstractmethod
    def get_node(self) -> Any:
        """
        Abstract method to get the node instance.

        Returns:
            Any: The node instance.
        """
        pass
