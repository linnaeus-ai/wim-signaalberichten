from .transform_to_kg_system_prompt import TRANSFORM_TO_KG_SYSTEM_PROMPT
from .transform_to_kg_human_prompt import TRANSFORM_TO_KG_HUMAN_PROMPT
from .entity_extraction_system_prompt import ENTITY_EXTRACTION_SYSTEM_PROMPT
from .entity_extraction_human_prompt import ENTITY_EXTRACTION_HUMAN_PROMPT
from .retrieve_schema_org_human_prompt import RETRIEVE_SCHEMA_ORG_HUMAN_PROMPT
from .retrieve_schema_org_system_prompt import RETRIEVE_SCHEMA_ORG_SYSTEM_PROMPT
from .add_labels_human_prompt import ADD_LABELS_HUMAN_PROMPT
from .add_labels_system_prompt import ADD_LABELS_SYSTEM_PROMPT

__all__ = [
    "TRANSFORM_TO_KG_SYSTEM_PROMPT",
    "TRANSFORM_TO_KG_HUMAN_PROMPT",
    "ENTITY_EXTRACTION_SYSTEM_PROMPT",
    "ENTITY_EXTRACTION_HUMAN_PROMPT",
    "RETRIEVE_SCHEMA_ORG_HUMAN_PROMPT",
    "RETRIEVE_SCHEMA_ORG_SYSTEM_PROMPT",
    "ADD_LABELS_HUMAN_PROMPT",
    "ADD_LABELS_SYSTEM_PROMPT",
]
