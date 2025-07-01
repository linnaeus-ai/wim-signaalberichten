import logging
from typing import Optional, Callable, Any
from abc import ABC, abstractmethod


class BaseRouter(ABC):
    """
    Abstract base class for all routers in the graph.

    This class provides the basic structure and functionality that all routers must implement.
    Routers are responsible for directing flow between nodes in the graph based on specific conditions
    or logic. Each router must have a unique name and implement the get_router method.

    Attributes:
        _name (str): The name identifier for the router. Can be set at class or instance level.
    """

    name: str

    def __init__(self, name: Optional[str] = None):
        """
        Initialize a new router with a name.

        Args:
            name (Optional[str]): The name identifier for the router. If not provided, uses class-level _name.

        Raises:
            ValueError: If neither instance name nor class-level name is provided.
        """
        if not self.name and not name:
            raise ValueError("Please name your node.")

        self.name = name or self.name

    @property
    def logger(self) -> logging.Logger:
        """
        Get a logger instance specific to this router.

        Returns:
            logging.Logger: A logger instance prefixed with ROUTER::{router_name}
        """
        return logging.getLogger(f"ROUTER::{self.name}")

    @abstractmethod
    def get_router(self) -> Callable[..., Any]:
        """
        Abstract method that must be implemented by all router classes.
        Should return a callable that implements the router's routing logic.

        The router function typically takes input and determines which node or path
        in the graph should handle that input based on specific conditions.

        Returns:
            callable: The router's implementation as a callable object that handles routing logic
        """
        pass
