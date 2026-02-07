"""Model mappings from Anthropic model names to Bedrock/Copilot model IDs."""

from .config import BEDROCK_REGION_PREFIX

# Default model when no match found
DEFAULT_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

# Map Anthropic model names to Bedrock model IDs (without region prefix)
BEDROCK_MODEL_MAP = {
    # Opus 4.6
    "claude-opus-4-6-20250515": "anthropic.claude-opus-4-6-20250515-v1:0",
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
    if model in BEDROCK_MODEL_MAP:
        return add_region_prefix(BEDROCK_MODEL_MAP[model])
    # Already a Bedrock model ID (with or without prefix)
    if "anthropic." in model:
        return model
    # Partial match
    for key, value in BEDROCK_MODEL_MAP.items():
        if key in model:
            return add_region_prefix(value)
    # Default
    return add_region_prefix(DEFAULT_MODEL)


# --- Copilot Model Mappings ---

# Default Copilot model
DEFAULT_COPILOT_MODEL = "claude-sonnet-4.5"

# Map Anthropic model names to Copilot model IDs
COPILOT_MODEL_MAP = {
    # Opus 4.6
    "claude-opus-4-6-20250515": "claude-opus-4.6",
    # Opus 4.5
    "claude-opus-4-5-20251101": "claude-opus-4.5",
    # Opus 4.1
    "claude-opus-4-1-20250805": "claude-opus-4.1",
    # Opus 4
    "claude-opus-4-20250514": "claude-opus-4",
    # Sonnet 4.5
    "claude-sonnet-4-5-20250929": "claude-sonnet-4.5",
    # Sonnet 4
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    # Haiku 4.5
    "claude-haiku-4-5-20251001": "claude-haiku-4.5",
    # Sonnet 3.7
    "claude-3-7-sonnet-20250219": "claude-3.7-sonnet",
    # Sonnet 3.5 v2
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    # Haiku 3.5
    "claude-3-5-haiku-20241022": "claude-3.5-haiku",
}


# Map OpenAI-native model names to Copilot model IDs (used for /v1/chat/completions direct passthrough)
# Source: https://docs.github.com/copilot/reference/ai-models/supported-models
COPILOT_OPENAI_MODEL_MAP: dict[str, str] = {
    # Claude models (Copilot display names)
    "claude-opus-4.6": "claude-opus-4.6",
    "claude-opus-4.5": "claude-opus-4.5",
    "claude-opus-4.1": "claude-opus-4.1",
    "claude-opus-4": "claude-opus-4",
    "claude-sonnet-4.5": "claude-sonnet-4.5",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "claude-3.7-sonnet": "claude-3.7-sonnet",
    "claude-3.5-sonnet": "claude-3.5-sonnet",
    "claude-3.5-haiku": "claude-3.5-haiku",
    # GPT-5.x series
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1": "gpt-5.1",
    "gpt-5-codex": "gpt-5-codex",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5": "gpt-5",
    # GPT-4.x series
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    # Gemini models
    "gemini-3-pro": "gemini-3-pro",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    # Other models
    "grok-code-fast-1": "grok-code-fast-1",
    "raptor-mini": "raptor-mini",
}


def get_copilot_openai_model(model: str) -> str:
    """Map a model name to a Copilot-compatible model ID for OpenAI passthrough.

    Checks COPILOT_OPENAI_MODEL_MAP (direct match), then COPILOT_MODEL_MAP
    (Anthropic names like claude-sonnet-4-5-20250929), then partial matches.
    Unknown models pass through as-is.
    """
    # Direct match in OpenAI map
    if model in COPILOT_OPENAI_MODEL_MAP:
        return COPILOT_OPENAI_MODEL_MAP[model]
    # Direct match in Anthropic->Copilot map (returns Copilot display name)
    if model in COPILOT_MODEL_MAP:
        return COPILOT_MODEL_MAP[model]
    # Partial match in Anthropic->Copilot map
    for key, value in COPILOT_MODEL_MAP.items():
        if key in model:
            return value
    # Partial match in OpenAI map
    for key, value in COPILOT_OPENAI_MODEL_MAP.items():
        if key in model:
            return value
    # Unknown model: pass through as-is (let Copilot API reject if invalid)
    return model


def get_copilot_model(model: str) -> tuple[str, str]:
    """Map Anthropic model name to Copilot model ID.

    Returns (copilot_model_id, anthropic_model_name) tuple.
    The anthropic_model_name is used for response translation.
    """
    # Direct match
    if model in COPILOT_MODEL_MAP:
        return COPILOT_MODEL_MAP[model], model
    # Partial match
    for key, value in COPILOT_MODEL_MAP.items():
        if key in model:
            return value, key
    # Default
    return DEFAULT_COPILOT_MODEL, "claude-sonnet-4-5-20250929"
