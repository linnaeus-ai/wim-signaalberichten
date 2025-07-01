from typing import Callable, Sequence
from graph.base import BaseRouter
from graph.text_to_kg_state import TextToKGState


class ValidatorRouter(BaseRouter):
    """
    Router that determines whether a user question needs rewriting or can be used directly.

    This router evaluates the clarity and quality of the user question to decide
    if it should go through the rewrite step or proceed directly to retrieval.
    """

    name = "ValidatorRouter"

    def __init__(self) -> None:
        """
        ...
        """
        super().__init__(name=self.name)

    def get_router(self) -> Callable[[TextToKGState], Sequence[str]]:
        """
        ...
        """
        logger = self.logger

        def _router(state: TextToKGState) -> Sequence[str]:
            """
            ...
            """

            if state.validation_returncode == 0:
                logger.debug("---DECISION: JSON CORRECT---")
                return ["AddLabelsNode"]
            elif (
                state.validation_returncode == 1
                and not state.validation_max_runs_reached
            ):
                logger.debug("---DECISION: JSON INCORRECT---")
                return ["TransformToKGNode"]
            else:
                logger.debug("---DECISION: JSON INCORRECT, TOO MANY runs---")
                return ["AddLabelsNode"]

        return _router
