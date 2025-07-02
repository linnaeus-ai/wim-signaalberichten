from pydantic import BaseModel, Field
from typing import Dict, Optional, Union, List


class TextToKGState(BaseModel):
    """
    State for the Text to Knowledge Graph (KG) process.
    """

    text: str = Field(
        ..., description="Input text to be processed into a knowledge graph."
    )
    entity_extraction_output: Optional[Dict[str, Union[List[List[str]], str]]] = Field(
        None,
        description="Output from the entity extraction process, containing summary, entities, and relations.",
    )
    schema_definitions: Optional[Dict[str, str]] = Field(
        None,
        description="Schema definitions for the entities extracted from the input text.",
    )
    json_ld_paths: Optional[List[str]] = Field(
        [],
        description="The patha to the JSON-LD representations of the knowledge graph generated from the input text.",
    )
    json_ld_contents: Optional[List[str]] = Field(
        [],
        description="The actual JSON-LD content corresponding to each path, to avoid file system race conditions.",
    )

    # JSON-LD validation fields
    validation_output: Optional[List[str]] = Field(
        [],
        description="Collection of outputs from the JSON-LD validation process.",
    )
    validation_returncode: Optional[int] = Field(
        None,
        description="Return code from the JSON-LD validation process.",
    )

    # Rerun data
    validation_runs: int = Field(
        0,
        description="Number of times the JSON-LD validation has been rerun.",
    )
    validation_max_runs: int = Field(
        5,
        description="Maximum number of allowed runs for JSON-LD validation.",
    )
    validation_max_runs_reached: bool = Field(
        False,
        description="Flag indicating whether the maximum number of runs for validation has been reached.",
    )
    validation_infrastructure_error: bool = Field(
        False,
        description="Flag indicating whether validation failed due to infrastructure/system errors (non-recoverable).",
    )
    entity_extraction_failed: bool = Field(
        False,
        description="Flag indicating whether entity extraction failed (e.g., LLM response parsing error).",
    )
    
    # Database path and metadata for logging
    db_path: Optional[str] = Field(
        None,
        description="Path to database for logging LLM calls."
    )
    worker_id: Optional[str] = Field(
        None,
        description="Worker ID for tracking which worker processed this item."
    )
    wiki_id: Optional[int] = Field(
        None,
        description="Wiki ID or other source identifier for this processing run."
    )
