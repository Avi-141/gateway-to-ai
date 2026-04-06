"""Model mappings from Anthropic model names to Bedrock/Copilot/AI Framework model IDs."""

import time
from typing import TYPE_CHECKING, Any

from .config import AI_FRAMEWORK_ENV_KEY_MAP, BEDROCK_REGION_PREFIX, COPILOT_MODELS_TTL, logger

if TYPE_CHECKING:
    from .copilot_client import CopilotBackend


def is_claude_model(model: str) -> bool:
    """Check if a model name refers to a Claude model.

    Returns True for Anthropic model names (claude-*) and Bedrock model IDs (anthropic.*).
    """
    return model.startswith("claude-") or model.startswith("claude ") or "anthropic." in model


# Default model when no match found
DEFAULT_BEDROCK_MODEL = "anthropic.claude-sonnet-4-6"

# Map Anthropic model names to Bedrock model IDs (without region prefix)
BEDROCK_MODEL_MAP = {
    # Opus 4.6
    "claude-opus-4-6-20250515": "anthropic.claude-opus-4-6-v1",
    "claude-opus-4-6": "anthropic.claude-opus-4-6-v1",
    # Sonnet 4.6
    "claude-sonnet-4-6": "anthropic.claude-sonnet-4-6",
    # Opus 4.5
    "claude-opus-4-5-20251101": "anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4-5": "anthropic.claude-opus-4-5-20251101-v1:0",
    # Opus 4.1
    "claude-opus-4-1-20250805": "anthropic.claude-opus-4-1-20250805-v1:0",
    "claude-opus-4-1": "anthropic.claude-opus-4-1-20250805-v1:0",
    # Opus 4
    "claude-opus-4-20250514": "anthropic.claude-opus-4-20250514-v1:0",
    "claude-opus-4": "anthropic.claude-opus-4-20250514-v1:0",
    # Sonnet 4.5
    "claude-sonnet-4-5-20250929": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-5": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    # Sonnet 4
    "claude-sonnet-4-20250514": "anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-sonnet-4": "anthropic.claude-sonnet-4-20250514-v1:0",
    # Sonnet 3.7
    "claude-3-7-sonnet-20250219": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-3-7-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    # Sonnet 3.5 v2
    "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    # Sonnet 3.5 v1
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    # Haiku 4.5
    "claude-haiku-4-5-20251001": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-haiku-4-5": "anthropic.claude-haiku-4-5-20251001-v1:0",
    # Haiku 3.5
    "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    # Haiku 3
    "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
    # Opus 3
    "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
    # Sonnet 3
    "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
}


def _strip_model_prefix(model: str) -> str:
    """Strip known provider prefixes from model names (e.g., 'github-copilot/gpt-5.3-codex' -> 'gpt-5.3-codex')."""
    known_prefixes = ("github-copilot/",)
    for prefix in known_prefixes:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def add_region_prefix(model_id: str) -> str:
    """Add region prefix to model ID for cross-region inference."""
    if BEDROCK_REGION_PREFIX:
        return f"{BEDROCK_REGION_PREFIX}.{model_id}"
    return model_id


def get_bedrock_model(model: str) -> str:
    """Map Anthropic model name to Bedrock model ID."""
    model = _strip_model_prefix(model)
    # Direct match
    if model in BEDROCK_MODEL_MAP:
        return add_region_prefix(BEDROCK_MODEL_MAP[model])
    # Already a Bedrock model ID — ensure region prefix is applied
    if "anthropic." in model:
        # Strip existing region prefix (e.g., "us.anthropic..." -> "anthropic...")
        # then re-add it to ensure consistency
        bare = model.split("anthropic.", 1)[-1]
        return add_region_prefix(f"anthropic.{bare}")
    # Partial match
    for key, value in BEDROCK_MODEL_MAP.items():
        if key in model:
            return add_region_prefix(value)
    # Default
    return add_region_prefix(DEFAULT_BEDROCK_MODEL)


# --- Copilot Model Mappings ---

# Default Copilot model
DEFAULT_COPILOT_MODEL = "claude-sonnet-4.6"

# Map Anthropic model names to Copilot model IDs
COPILOT_MODEL_MAP = {
    # Opus 4.6
    "claude-opus-4-6-20250515": "claude-opus-4.6",
    "claude-opus-4-6": "claude-opus-4.6",
    # Sonnet 4.6
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    # Opus 4.5
    "claude-opus-4-5-20251101": "claude-opus-4.5",
    "claude-opus-4-5": "claude-opus-4.5",
    # Opus 4.1
    "claude-opus-4-1-20250805": "claude-opus-4.1",
    "claude-opus-4-1": "claude-opus-4.1",
    # Opus 4
    "claude-opus-4-20250514": "claude-opus-4",
    "claude-opus-4": "claude-opus-4",
    # Sonnet 4.5
    "claude-sonnet-4-5-20250929": "claude-sonnet-4.5",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    # Sonnet 4
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "claude-sonnet-4": "claude-sonnet-4",
    # Haiku 4.5
    "claude-haiku-4-5-20251001": "claude-haiku-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    # Sonnet 3.7
    "claude-3-7-sonnet-20250219": "claude-3.7-sonnet",
    "claude-3-7-sonnet": "claude-3.7-sonnet",
    # Sonnet 3.5 v2
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    "claude-3-5-sonnet": "claude-3.5-sonnet",
    # Haiku 3.5
    "claude-3-5-haiku-20241022": "claude-3.5-haiku",
    "claude-3-5-haiku": "claude-3.5-haiku",
}


# Map OpenAI-native model names to Copilot model IDs (used for /v1/chat/completions direct passthrough)
# Source: https://docs.github.com/copilot/reference/ai-models/supported-models
# NOTE: This is a fallback only — when dynamic models are fetched at startup, those are used instead.
COPILOT_OPENAI_MODEL_MAP: dict[str, str] = {
    # Claude models (Copilot display names)
    "claude-opus-4.6": "claude-opus-4.6",
    "claude-opus-4.5": "claude-opus-4.5",
    "claude-opus-4.1": "claude-opus-4.1",
    "claude-sonnet-4.5": "claude-sonnet-4.5",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-haiku-4.5": "claude-haiku-4.5",
    # GPT-5.x series
    "gpt-5.4": "gpt-5.4",
    "gpt-5.3-codex": "gpt-5.3-codex",
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
    # Gemini models
    "gemini-3-pro-preview": "gemini-3-pro-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-2.5-pro": "gemini-2.5-pro",
    # Other models
    "grok-code-fast-1": "grok-code-fast-1",
    "raptor-mini": "raptor-mini",
}


# --- Dynamic Copilot Model Registry ---

# Populated at startup from the Copilot /models endpoint
_copilot_models: list[dict[str, Any]] = []
_copilot_model_ids: set[str] = set()
_copilot_model_endpoints: dict[str, list[str]] = {}
_copilot_model_limits: dict[str, int] = {}
_copilot_model_output_limits: dict[str, int] = {}
_copilot_model_context_windows: dict[str, int] = {}
_copilot_models_fetched_at: float = 0.0


def set_copilot_models(models: list[dict[str, Any]]) -> None:
    """Store models fetched from the Copilot API."""
    global _copilot_models, _copilot_model_ids, _copilot_model_endpoints, _copilot_model_limits
    global _copilot_model_output_limits, _copilot_model_context_windows, _copilot_models_fetched_at
    _copilot_models = models
    _copilot_model_ids = {m["id"] for m in models if "id" in m}
    _copilot_model_endpoints = {
        m["id"]: [ep for ep in m["supported_endpoints"] if isinstance(ep, str)]
        for m in models
        if "id" in m and isinstance(m.get("supported_endpoints"), list)
    }
    _copilot_model_limits = {}
    _copilot_model_output_limits = {}
    _copilot_model_context_windows = {}
    for m in models:
        model_id = m.get("id")
        if not model_id:
            continue
        limits = m.get("capabilities", {}).get("limits", {})
        # Use max_prompt_tokens (the actual prompt limit Copilot enforces)
        # rather than max_context_window_tokens (which includes output tokens).
        # Copilot's error fires when prompt tokens exceed max_prompt_tokens.
        limit = limits.get("max_prompt_tokens")
        if isinstance(limit, int) and limit > 0:
            _copilot_model_limits[model_id] = limit
        output_limit = limits.get("max_output_tokens")
        if isinstance(output_limit, int) and output_limit > 0:
            _copilot_model_output_limits[model_id] = output_limit
        context_window = limits.get("max_context_window_tokens")
        if isinstance(context_window, int) and context_window > 0:
            _copilot_model_context_windows[model_id] = context_window
    _copilot_models_fetched_at = time.monotonic()


async def refresh_copilot_models_if_stale(copilot_backend: "CopilotBackend") -> None:
    """Refresh the Copilot model registry if the TTL has expired.

    On failure, logs a warning and keeps the existing cached models.
    """
    global _copilot_models_fetched_at
    if time.monotonic() - _copilot_models_fetched_at < COPILOT_MODELS_TTL:
        return
    try:
        models = await copilot_backend.list_models()
        if models:
            set_copilot_models(models)
            logger.info(f"Refreshed {len(models)} models from Copilot API")
        else:
            logger.warning("Copilot API returned no models during refresh, keeping cached models")
            # Update timestamp so we don't retry immediately
            _copilot_models_fetched_at = time.monotonic()
    except Exception:
        logger.warning("Failed to refresh Copilot models, keeping cached models", exc_info=True)
        # Update timestamp so we don't retry on every request
        _copilot_models_fetched_at = time.monotonic()


def model_requires_responses_api(model_id: str) -> bool:
    """Return True if the model only supports /responses (not /chat/completions)."""
    endpoints = _copilot_model_endpoints.get(model_id, [])
    if not endpoints:
        return False
    return "/responses" in endpoints and "/chat/completions" not in endpoints


def model_supports_responses_api(model_id: str) -> bool:
    """Return True if the model supports /responses (even if it also supports /chat/completions)."""
    endpoints = _copilot_model_endpoints.get(model_id, [])
    return "/responses" in endpoints


def model_supports_messages_api(model_id: str) -> bool:
    """Return True if the model supports /v1/messages (Anthropic native format)."""
    endpoints = _copilot_model_endpoints.get(model_id, [])
    return "/v1/messages" in endpoints


def get_copilot_context_limit(model_id: str) -> int:
    """Return the Copilot context window limit for a model, or 0 if unknown."""
    return _copilot_model_limits.get(model_id, 0)


def get_copilot_output_limit(model_id: str) -> int:
    """Return the Copilot max_output_tokens for a model, or 0 if unknown."""
    return _copilot_model_output_limits.get(model_id, 0)


def get_copilot_context_window(model_id: str) -> int:
    """Return the Copilot max_context_window_tokens for a model, or 0 if unknown."""
    return _copilot_model_context_windows.get(model_id, 0)


def get_available_copilot_models() -> list[dict[str, Any]]:
    """Return the dynamically fetched Copilot models (empty if not fetched)."""
    return _copilot_models


def get_copilot_openai_model(model: str) -> str:
    """Map a model name to a Copilot-compatible model ID for OpenAI passthrough.

    When dynamic models are available (fetched from Copilot API at startup),
    checks the dynamic registry first, then COPILOT_MODEL_MAP for Anthropic
    versioned names. Uses smart fallback for Claude models not found in registry.

    When no dynamic models are available, falls back to hardcoded maps.
    """
    model = _strip_model_prefix(model)

    # Handle sentinel "default" model name (sent by Claude Code for internal requests)
    if model == "default":
        return DEFAULT_COPILOT_MODEL

    # Check dynamic registry (exact match on model id)
    if _copilot_model_ids:
        if model in _copilot_model_ids:
            return model
        # Anthropic versioned name -> Copilot display name (stable mapping)
        if model in COPILOT_MODEL_MAP:
            resolved = COPILOT_MODEL_MAP[model]
            if resolved in _copilot_model_ids:
                return resolved
        # Smart fallback: find newest available version for Claude models.
        # This runs before partial match to avoid e.g. "claude-sonnet-4" in
        # "claude-sonnet-4-6" incorrectly matching the older model.
        fallback = _find_newest_available_claude_model(model)
        if fallback:
            return fallback[0]
        # Partial match in Anthropic->Copilot map
        for key, value in COPILOT_MODEL_MAP.items():
            if key in model and value in _copilot_model_ids:
                return value
        # Not found — pass through as-is
        return model
    # No dynamic models: fall back to hardcoded maps
    if model in COPILOT_OPENAI_MODEL_MAP:
        return COPILOT_OPENAI_MODEL_MAP[model]
    if model in COPILOT_MODEL_MAP:
        return COPILOT_MODEL_MAP[model]
    for key, value in COPILOT_MODEL_MAP.items():
        if key in model:
            return value
    for key, value in COPILOT_OPENAI_MODEL_MAP.items():
        if key in model:
            return value
    return model


def _find_newest_available_claude_model(requested_model: str) -> tuple[str, str] | None:
    """Find the newest available Claude model compatible with the requested one.

    Parses the Anthropic model name (claude-{family}-{major}-{minor}[-{date}])
    and finds the best match from dynamically fetched Copilot models
    (claude-{family}-{major}.{minor}).

    Returns (copilot_model_id, original_requested_model) or None.
    """
    if not _copilot_model_ids or not requested_model.startswith("claude-"):
        return None

    parts = requested_model[7:].split("-")  # Remove "claude-" prefix
    if len(parts) < 3:
        return None

    family = parts[0]
    try:
        major = int(parts[1])
        minor = int(parts[2])
    except ValueError:
        return None

    # Find all available versions of this family in the dynamic registry
    family_prefix = f"claude-{family}-"
    available: list[tuple[int, int, str]] = []
    for model_id in _copilot_model_ids:
        if model_id.startswith(family_prefix):
            version_parts = model_id[len(family_prefix) :].split(".")
            if len(version_parts) >= 2:
                try:
                    available.append((int(version_parts[0]), int(version_parts[1]), model_id))
                except ValueError:
                    continue

    if not available:
        return None

    # Sort highest first
    available.sort(reverse=True)

    # Prefer the newest version that doesn't exceed the requested version
    for avail_major, avail_minor, model_id in available:
        if avail_major < major or (avail_major == major and avail_minor <= minor):
            return model_id, requested_model

    # Requested version is older than everything available; use the newest
    return available[0][2], requested_model


# --- AI Framework model registry ---

AI_FRAMEWORK_MODELS: dict[str, dict[str, Any]] = {
    "gpt-5-chat": {"provider": "openai", "tool_calling": True, "envs": {"dev"}},
    "gpt-5.2-chat": {"provider": "openai", "tool_calling": True, "envs": {"dev"}},
    "gpt-5.3-chat": {"provider": "openai", "tool_calling": True, "envs": {"dev"}},
    "claude-sonnet-4-6": {"provider": "anthropic", "tool_calling": True, "envs": {"dev"}},
    "mistral-medium-2508": {"provider": "openai", "tool_calling": True, "envs": "all"},
    "magistral-medium-2507": {"provider": "openai", "tool_calling": True, "envs": "all"},
    "phi4": {"provider": "Microsoft", "openai": False, "envs": "all"},
    "Phi-4-mini-instruct": {"provider": "openai", "tool_calling": False, "envs": "all"},
    "openai/gpt-oss-120b": {"provider": "openai", "tool_calling": True, "envs": "all"},
}


def is_ai_framework_model(model: str) -> bool:
    """Return True if model uses iq/<env>/<model> format."""
    return model.startswith("iq/")


def parse_ai_framework_model(model: str) -> tuple[str, str]:
    """Parse 'iq/<env>/<model>' into (env_name, model_name).

    Raises ValueError if format is invalid or environment is unknown.
    """
    parts = model.split("/", 2)
    if len(parts) < 3 or parts[0] != "iq" or not parts[1] or not parts[2]:
        raise ValueError(f"Invalid AI Framework model format: '{model}'. Expected 'iq/<env>/<model>'.")
    env_name = parts[1]
    model_name = parts[2]
    if env_name not in AI_FRAMEWORK_ENV_KEY_MAP:
        raise ValueError(
            f"Unknown AI Framework environment: '{env_name}'. "
            f"Valid environments: {', '.join(sorted(AI_FRAMEWORK_ENV_KEY_MAP))}"
        )
    return env_name, model_name


def validate_ai_framework_model(model_name: str, env_name: str) -> list[str]:
    """Return warning strings for model/environment mismatches."""
    warnings: list[str] = []
    info = AI_FRAMEWORK_MODELS.get(model_name)
    if info is None:
        warnings.append(f"AI Framework model '{model_name}' is not in the known registry — it may still work")
        return warnings
    if env_name and info["envs"] != "all" and env_name.lower() not in info["envs"]:
        warnings.append(
            f"AI Framework model '{model_name}' is only available in {info['envs']} but environment is '{env_name}'"
        )
    return warnings


def get_copilot_model(model: str) -> tuple[str, str]:
    """Map Anthropic model name to Copilot model ID.

    Returns (copilot_model_id, anthropic_model_name) tuple.
    The anthropic_model_name is used for response translation (model label only).

    When dynamic models are available, validates against them and uses smart
    fallback to find the newest compatible Claude model version.
    """
    model = _strip_model_prefix(model)

    # Handle sentinel "default" model name (sent by Claude Code for internal requests)
    if model == "default":
        return DEFAULT_COPILOT_MODEL, DEFAULT_COPILOT_MODEL
    if _copilot_model_ids:
        # --- Dynamic models available: validate all lookups against registry ---

        # Direct match in COPILOT_MODEL_MAP, target confirmed available
        if model in COPILOT_MODEL_MAP:
            target = COPILOT_MODEL_MAP[model]
            if target in _copilot_model_ids:
                return target, model
            # Target not in registry — try smart fallback, but trust the
            # hardcoded map if no better match exists (handles startup timing
            # where the registry may not have populated yet).
            fallback = _find_newest_available_claude_model(model)
            if fallback:
                return fallback
            return target, model

        # Exact match in dynamic registry (covers Copilot display names)
        if model in _copilot_model_ids:
            return model, model

        # Smart fallback: find newest available version for Claude models.
        # This runs before partial match to avoid e.g. "claude-sonnet-4" in
        # "claude-sonnet-4-6" incorrectly matching the older model.
        fallback = _find_newest_available_claude_model(model)
        if fallback:
            return fallback

        # Partial match in COPILOT_MODEL_MAP, target confirmed available
        for key, value in COPILOT_MODEL_MAP.items():
            if key in model and value in _copilot_model_ids:
                return value, key

        # Non-Claude models in COPILOT_OPENAI_MODEL_MAP, confirmed available
        if model in COPILOT_OPENAI_MODEL_MAP and COPILOT_OPENAI_MODEL_MAP[model] in _copilot_model_ids:
            return COPILOT_OPENAI_MODEL_MAP[model], model
        for key, value in COPILOT_OPENAI_MODEL_MAP.items():
            if key in model and value in _copilot_model_ids:
                return value, key

        # Non-Claude models: pass through as-is (Copilot may support them directly)
        if not is_claude_model(model):
            return model, model

        # Default (dynamic models loaded but no Claude match found)
        return DEFAULT_COPILOT_MODEL, "claude-sonnet-4-6"

    # --- No dynamic models: use hardcoded maps ---

    # Direct match in COPILOT_MODEL_MAP
    if model in COPILOT_MODEL_MAP:
        return COPILOT_MODEL_MAP[model], model
    # Partial match in COPILOT_MODEL_MAP
    for key, value in COPILOT_MODEL_MAP.items():
        if key in model:
            return value, key
    # Direct match in COPILOT_OPENAI_MODEL_MAP
    if model in COPILOT_OPENAI_MODEL_MAP:
        return COPILOT_OPENAI_MODEL_MAP[model], model
    # Partial match in COPILOT_OPENAI_MODEL_MAP
    for key, value in COPILOT_OPENAI_MODEL_MAP.items():
        if key in model:
            return value, key
    # Non-Claude models: pass through as-is
    if not is_claude_model(model):
        return model, model
    # Default
    return DEFAULT_COPILOT_MODEL, "claude-sonnet-4-6"
