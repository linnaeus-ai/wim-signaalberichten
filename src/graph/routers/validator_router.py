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
            elif state.validation_infrastructure_error:
                # Exit code 2: Infrastructure error - cannot retry
                logger.debug("---DECISION: INFRASTRUCTURE ERROR - MUST TERMINATE---")
                # Raise exception to halt the entire pipeline
                raise RuntimeError(
                    f"Infrastructure error in validator: {state.validation_output[-1] if state.validation_output else 'Unknown error'}. "
                    "Cannot continue processing - check TTL files and validator setup."
                )
            elif (
                state.validation_returncode == 1
                and not state.validation_max_runs_reached
            ):
                # Exit code 1: Recoverable error - can retry
                logger.debug("---DECISION: JSON INCORRECT (RECOVERABLE)---")
                return ["TransformToKGNode"]
            else:
                # Max runs reached or unexpected return code
                logger.debug("---DECISION: JSON INCORRECT, TOO MANY runs or unexpected error---")
                return ["AddLabelsNode"]

        return _router
