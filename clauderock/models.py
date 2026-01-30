"""Model mappings from Anthropic model names to Bedrock model IDs."""

from .config import BEDROCK_REGION_PREFIX

# Default model when no match found
DEFAULT_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

# Map Anthropic model names to Bedrock model IDs (without region prefix)
MODEL_MAP = {
    # Opus 4.5
    "claude-opus-4-5-20251101": "anthropic.claude-opus-4-5-20251101-v1:0",
    # Opus 4.1
    "claude-opus-4-1-20250805": "anthropic.claude-opus-4-1-20250805-v1:0",
    # Opus 4
    "claude-opus-4-20250514": "anthropic.claude-opus-4-20250514-v1:0",
    # Sonnet 4.5
    "claude-sonnet-4-5-20250929": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    # Sonnet 4
    "claude-sonnet-4-20250514": "anthropic.claude-sonnet-4-20250514-v1:0",
    # Sonnet 3.7
    "claude-3-7-sonnet-20250219": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    # Sonnet 3.5 v2
    "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    # Sonnet 3.5 v1
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    # Haiku 4.5
    "claude-haiku-4-5-20251001": "anthropic.claude-haiku-4-5-20251001-v1:0",
    # Haiku 3.5
    "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
    # Haiku 3
    "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
    # Opus 3
    "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
    # Sonnet 3
    "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
}


def add_region_prefix(model_id: str) -> str:
    """Add region prefix to model ID for cross-region inference."""
    if BEDROCK_REGION_PREFIX:
        return f"{BEDROCK_REGION_PREFIX}.{model_id}"
    return model_id


def get_bedrock_model(model: str) -> str:
    """Map Anthropic model name to Bedrock model ID."""
    # Direct match
    if model in MODEL_MAP:
        return add_region_prefix(MODEL_MAP[model])
    # Already a Bedrock model ID (with or without prefix)
    if "anthropic." in model:
        return model
    # Partial match
    for key, value in MODEL_MAP.items():
        if key in model:
            return add_region_prefix(value)
    # Default
    return add_region_prefix(DEFAULT_MODEL)
