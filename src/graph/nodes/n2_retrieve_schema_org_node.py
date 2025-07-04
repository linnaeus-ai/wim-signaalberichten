import pickle
import sqlite3
import time
import numpy as np
from numpy.linalg import norm
from concurrent.futures import ThreadPoolExecutor, as_completed

from graph.base import BaseNode
from graph.utils.llm import azure_embeddings
from graph.utils.logging_llm_wrapper import LoggingLLMWrapper
from graph.text_to_kg_state import TextToKGState
from graph.utils.models import RetrieveSchemaOrgStructuredOutput
from graph.prompts import (
    RETRIEVE_SCHEMA_ORG_HUMAN_PROMPT,
    RETRIEVE_SCHEMA_ORG_SYSTEM_PROMPT,
)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.base import BaseLanguageModel

# from langchain_core.runnables.config import ContextThreadPoolExecutor


class RetrieveSchemaOrgNode(BaseNode):
    name: str = "RetrieveSchemaOrgNode"
    _db_path: str = "src/data/schema_definitions.db"

    def __init__(self, llm: BaseLanguageModel):
        self._embedder = azure_embeddings()
        self._embeddings = self._prepare_embeddings(self._db_path)

        # Initialize the base class with the name and LLM
        super().__init__(llm=llm)

    def _prepare_embeddings(self, db_path):
        # Open in read-only mode with URI to prevent locking issues
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT class_label, comments, embedding FROM schema_classes")
            rows = cursor.fetchall()

            embeddings = []
            for class_label, comments, embedding_blob in rows:
                embedding = pickle.loads(embedding_blob)
                embeddings.append([class_label, comments, embedding])
        finally:
            conn.close()

        return embeddings

    def _get_yaml_for_class(self, class_label):
        """Retrieve full YAML definition for a given class label"""
        # Open in read-only mode with URI to prevent locking issues
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT yaml_definition FROM schema_classes WHERE class_label = ?",
                (class_label,),
            )
            row = cursor.fetchone()
            
            if row:
                return row[0]
            return None
        finally:
            conn.close()

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def semantic_lookup_from_sql(self, class_name, description):
        """
        Search for the 5 most semantically similar schema.org types based on class name and description.

        This tool performs semantic matching against the schema.org vocabulary to find the most
        relevant types that could represent the given class.

        Args:
            class_name (str): The class name/label to find schema.org equivalents for.
            description (str): The description/context of the class to match against schema.org types.

        Returns:
            list[dict]: List of dictionaries containing 'label' and 'comment' for the 5 best matching schema.org types, ranked by semantic similarity.

        Raises:
            Exception: If no matching schema.org types are found in the database.
        """

        # 1. Embed query
        text_to_embed = f"Class: {class_name}\nDescription: {description}".strip()
        query_embedding = self._embedder.embed_query(text_to_embed)

        # Store all scores with their corresponding data
        scores = []

        # Loop through embeddings
        for row in self._embeddings:
            # Compare with each embedding
            score = self._cosine_similarity(query_embedding, row[-1])
            scores.append((score, row[0], row[1]))  # (score, class_label, comments)

        # Sort by score in descending order and get top 5
        scores.sort(key=lambda x: x[0], reverse=True)
        top_5_scores = scores[:5]

        if top_5_scores:
            # Return list of dictionaries with label and comment
            return [
                {
                    "label": class_label,
                    "comment": comments if comments else "No description available",
                }
                for _, class_label, comments in top_5_scores
            ]
        else:
            raise Exception(
                f"No matching class found for '{class_name}' with description '{description}'"
            )

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

            print("\033[93m    → Running Node 2: RetrieveSchemaOrgNode...\033[0m")

            # Check if entity extraction failed
            if state.entity_extraction_failed:
                print("\033[93m    → Entity extraction failed - skipping schema retrieval\033[0m")
                state.schema_definitions = {}
                return state

            # Define a dictionary to hold the YAML definitions
            yaml_definitions = {}

            # Get all entities from the state
            entities = state.entity_extraction_output["entities"]

            # Group entities by class to process unique classes
            class_to_entities = {}
            for entity_triple in entities:
                _, class_name, _ = entity_triple
                if class_name not in class_to_entities:
                    class_to_entities[class_name] = entity_triple

            print(
                f"\033[92m    → Found {len(class_to_entities)} unique entity classes to process\033[0m"
            )

            def process_class(entity_triple: tuple) -> tuple[str, str]:
                """Process a single class and return (class_name, yaml_definition)"""
                entity_name, cl, description = entity_triple
                max_retries = 3
                base_delay = 1  # Start with 1 second delay

                for attempt in range(max_retries):
                    try:
                        print(f"\033[96m      • Looking up schema for '{cl}': {description}\033[0m")
                        # Fake tool call to retrieve top 5 schema.org definitions
                        top_5 = self.semantic_lookup_from_sql(cl, description)
                        candidate_labels = [schema["label"] for schema in top_5]
                        print(
                            f"        Found {len(top_5)} candidate schemas: {', '.join(candidate_labels[:3])}{'...' if len(candidate_labels) > 3 else ''}"
                        )

                        # Format candidates for the prompt
                        schema_candidates = []
                        for i, schema in enumerate(top_5, 1):
                            schema_candidates.append(
                                f"{i}. {schema['label']}: {schema['comment']}"
                            )

                        # Use the LLM instance from the node (no longer hardcoded)
                        # Note: self._llm is already configured with the appropriate model
                        thread_llm = self._llm
                        
                        # First wrap with logging if db path available
                        llm_to_use = thread_llm
                        if state.db_path and state.wiki_id and state.worker_id:
                            llm_to_use = LoggingLLMWrapper(
                                base_llm=thread_llm,
                                db_path=state.db_path,
                                source_id=state.wiki_id,
                                call_name=f"n2_schema_selection_{cl}",
                                worker_id=state.worker_id
                            )

                        # Then apply structured output to the (possibly wrapped) LLM
                        # Select the best match
                        response = llm_to_use.with_structured_output(
                            RetrieveSchemaOrgStructuredOutput
                        ).invoke(
                            [
                                SystemMessage(
                                    content=RETRIEVE_SCHEMA_ORG_SYSTEM_PROMPT
                                ),
                                HumanMessage(
                                    content=RETRIEVE_SCHEMA_ORG_HUMAN_PROMPT.format(
                                        ORIGINAL_TEXT=state.text,
                                        ENTITY_NAME=entity_name,
                                        CLASS_NAME=cl,
                                        DESCRIPTION=description,
                                        SCHEMA_CANDIDATES="\n".join(schema_candidates),
                                    )
                                ),
                            ]
                        )

                        # Get the selected class from the response
                        selected_index = (
                            response.selected_number - 1
                        )  # Convert to 0-based index
                        # Validate that the selected_class matches the candidate at selected_number
                        expected_class = top_5[selected_index]["label"]
                        if response.selected_class != expected_class:
                            print(f"\033[93m        ⚠️  Warning: selected_class '{response.selected_class}' doesn't match expected '{expected_class}' at index {response.selected_number}\033[0m")
                            # Use the class from the structured output as primary source
                        selected_class = response.selected_class

                        # Get the YAML definition for the selected class
                        yaml_definition = self._get_yaml_for_class(selected_class)

                        if not yaml_definition:
                            raise Exception(
                                f"Could not find YAML definition for selected class: {selected_class}"
                            )

                        print(
                            f"\033[92m        ✓ Selected schema for '{cl}' → {selected_class}\033[0m"
                        )
                        print(f"          Reasoning: {response.reasoning}")
                        return cl, yaml_definition

                    except Exception as e:
                        error_str = str(e)
                        
                        # Check if it's an OpenAIRefusalError
                        if "OpenAIRefusalError" in str(type(e)):
                            if attempt < max_retries - 1:
                                print(f"\033[93m        ⚠️  Model refused structured output for '{cl}', retrying (attempt {attempt + 1}/{max_retries})...\033[0m")
                                time.sleep(1.0)  # Wait 1 second before retry
                                continue
                            # If max retries reached, fall through to fallback logic below
                        
                        # Check if it's a rate limit error (429)
                        elif "429" in error_str or "rate limit" in error_str.lower():
                            if attempt < max_retries - 1:
                                # Calculate exponential backoff delay
                                delay = base_delay * (2**attempt)
                                print(
                                    f"\033[93m        ⚠️  Rate limit hit for '{cl}', retrying in {delay}s (attempt {attempt + 1}/{max_retries})...\033[0m"
                                )
                                time.sleep(delay)
                                continue
                            else:
                                print(
                                    f"\033[91m        ❌ Max retries reached for '{cl}' due to rate limiting\033[0m"
                                )

                        # For any error after max retries, fallback to selecting the first class
                        print(f"\033[93m        ⚠️  Error after {max_retries} attempts for '{cl}': {e}\033[0m")
                        print(f"\033[93m        → Falling back to first candidate schema\033[0m")
                        
                        # Check if we have top_5 results to fall back to
                        if 'top_5' in locals() and top_5:
                            # Select the first class from top_5
                            selected_class = top_5[0]["label"]
                            yaml_definition = self._get_yaml_for_class(selected_class)
                            
                            if yaml_definition:
                                print(f"\033[92m        ✓ Fallback selected schema for '{cl}' → {selected_class}\033[0m")
                                return cl, yaml_definition
                            else:
                                print(f"\033[91m        ❌ Could not retrieve fallback schema for '{selected_class}'\033[0m")
                                return cl, None
                        else:
                            print(f"\033[91m        ❌ No candidates available for fallback\033[0m")
                            return cl, None

                # Should not reach here, but just in case
                return cl, None

            # Process all classes in parallel using LangChain's context-aware executor
            with ThreadPoolExecutor(
                max_workers=min(len(class_to_entities), 10)
            ) as executor:
                # Submit all tasks
                future_to_class = {
                    executor.submit(process_class, entity_triple): class_name
                    for class_name, entity_triple in class_to_entities.items()
                }

                # Collect results as they complete
                for future in as_completed(future_to_class):
                    class_name, yaml_def = future.result()
                    if yaml_def is not None:
                        yaml_definitions[class_name] = yaml_def

            # Add to the state
            state.schema_definitions = yaml_definitions
            print(f"\033[92m    → Retrieved {len(yaml_definitions)} schema definitions\033[0m")
            return state

        return _node
