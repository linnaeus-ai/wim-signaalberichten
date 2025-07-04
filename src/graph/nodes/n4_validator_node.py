import os
import subprocess
import tempfile

from graph.text_to_kg_state import TextToKGState
from graph.base import BaseNode
from langchain_core.language_models.base import BaseLanguageModel


class ValidatorNode(BaseNode):
    name: str = "ValidatorNode"

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
                f"\033[93m    → Node 4: ValidatorNode (max runs reached: {state.validation_max_runs_reached})\033[0m"
            )

            # Check if entity extraction failed
            if state.entity_extraction_failed:
                print("\033[93m    → Entity extraction failed - skipping validation\033[0m")
                state.validation_returncode = 0  # Success
                state.validation_output.append("Skipped validation due to entity extraction failure")
                return state

            # Get the JSON-LD content from state
            json_ld_content = (
                state.json_ld_contents[-1] if state.json_ld_contents else None
            )

            if not json_ld_content:
                # Fallback: try to read from file path if content not in state
                json_ld_path = os.path.abspath(state.json_ld_paths[-1])
                try:
                    with open(json_ld_path, "r") as f:
                        json_ld_content = f.read()
                except FileNotFoundError:
                    raise Exception(
                        f"Cannot find JSON-LD file at {json_ld_path} and no content in state"
                    )

            # Write content to a temporary file for validation
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp_file:
                tmp_file.write(json_ld_content)
                tmp_path = tmp_file.name

            print(f"\033[96m    → Running schema validation...\033[0m")

            try:
                # Perform Check: call the Go validator CLI
                result = subprocess.run(
                    ["./schema-validator", "-schema-file", "schemaorg-all-http.ttl", "-use-old-parser", tmp_path],
                    capture_output=True,
                    text=True,
                    cwd="src/graph/validator",
                    timeout=30,
                )
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

            # Add the validation output to the state
            state.validation_output.append(result.stdout)
            state.validation_returncode = result.returncode

            if state.validation_returncode == 0:
                print(f"\033[92m    → Validation passed!\033[0m")
            elif state.validation_returncode == 1:
                # Recoverable validation errors - can retry
                state.validation_runs += 1
                print(
                    f"\033[91m    → Validation failed with recoverable errors (attempt {state.validation_runs}/{state.validation_max_runs})\033[0m"
                )
                if result.stdout:
                    print(f"\033[91m      Error: {result.stdout.split('\n')[0][:80]}...\033[0m")

                if state.validation_runs >= state.validation_max_runs:
                    state.validation_max_runs_reached = True
                    print(f"\033[93m    → Max validation attempts reached\033[0m")
            elif state.validation_returncode == 2:
                # Infrastructure/system errors - cannot retry
                state.validation_infrastructure_error = True
                print(f"\033[91m    → Validation failed with infrastructure error (non-recoverable)\033[0m")
                if result.stdout:
                    print(f"\033[91m      Infrastructure error: {result.stdout.split('\n')[0][:80]}...\033[0m")
                print(f"\033[93m    → Skipping retries due to infrastructure failure\033[0m")
            else:
                # Unexpected return code
                print(f"\033[91m    → Validation failed with unexpected return code: {state.validation_returncode}\033[0m")

            return state

        return _node
