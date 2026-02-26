"""Pre-flight context window guard.

Estimates input tokens before forwarding requests to backends.
If estimated tokens approach the model's context limit, the guard clamps
max_tokens so the request fits — preventing unrecoverable 500s from Copilot.

Only raises ContextWindowExceededError when there is truly no room left
(estimated input exceeds the hard context limit with zero margin).
"""

from typing import Any

from .config import CONTEXT_GUARD_THRESHOLD, logger
from .copilot_translate import estimate_input_tokens
from .errors import ContextWindowExceededError
from .models import get_copilot_context_limit, get_copilot_model, get_copilot_openai_model

# Minimum max_tokens to allow — below this the model can't produce useful output
_MIN_OUTPUT_TOKENS = 1024


def check_context_guard_anthropic(body: dict[str, Any]) -> None:
    """Guard for /v1/messages (Anthropic format).

    When estimated tokens exceed CONTEXT_GUARD_THRESHOLD of the limit:
    - If there's room for at least _MIN_OUTPUT_TOKENS: clamp max_tokens to fit
    - If there's no room: raise ContextWindowExceededError

    Does nothing if threshold disabled (<=0) or model limit unknown.
    """
    if CONTEXT_GUARD_THRESHOLD <= 0:
        return
    model = body.get("model", "")
    copilot_model, _ = get_copilot_model(model)
    context_limit = get_copilot_context_limit(copilot_model)
    if context_limit <= 0:
        return
    estimated_tokens = estimate_input_tokens(body)
    effective_limit = int(context_limit * CONTEXT_GUARD_THRESHOLD)
    if estimated_tokens <= effective_limit:
        return

    # Over the soft threshold — check if we can still fit by clamping max_tokens
    available = context_limit - estimated_tokens
    if available >= _MIN_OUTPUT_TOKENS:
        original_max = body.get("max_tokens", 0)
        clamped = min(original_max, available) if original_max else available
        body["max_tokens"] = clamped
        logger.info(
            f"Context guard clamped max_tokens: {original_max} -> {clamped} "
            f"(estimated {estimated_tokens}/{context_limit} tokens for {copilot_model})"
        )
        return

    # No room at all — reject
    logger.warning(
        f"Context guard rejected: estimated {estimated_tokens} tokens "
        f"exceeds {context_limit} limit with < {_MIN_OUTPUT_TOKENS} room "
        f"for model {copilot_model}"
    )
    raise ContextWindowExceededError(
        prompt_tokens=context_limit,
        context_limit=context_limit,
        backend="proxy",
    )


def check_context_guard_openai(body: dict[str, Any]) -> None:
    """Guard for /v1/chat/completions (OpenAI format).

    Translates to Anthropic format to estimate tokens, then clamps or rejects.
    """
    if CONTEXT_GUARD_THRESHOLD <= 0:
        return
    model = body.get("model", "")
    copilot_model = get_copilot_openai_model(model)
    context_limit = get_copilot_context_limit(copilot_model)
    if context_limit <= 0:
        return
    from .openai_translate import openai_to_anthropic_request

    anthropic_body = openai_to_anthropic_request(body)
    estimated_tokens = estimate_input_tokens(anthropic_body)
    effective_limit = int(context_limit * CONTEXT_GUARD_THRESHOLD)
    if estimated_tokens <= effective_limit:
        return

    available = context_limit - estimated_tokens
    if available >= _MIN_OUTPUT_TOKENS:
        original_max = body.get("max_tokens") or body.get("max_completion_tokens") or 0
        clamped = min(original_max, available) if original_max else available
        if "max_completion_tokens" in body:
            body["max_completion_tokens"] = clamped
        else:
            body["max_tokens"] = clamped
        logger.info(
            f"Context guard clamped max_tokens (OpenAI): {original_max} -> {clamped} "
            f"(estimated {estimated_tokens}/{context_limit} tokens for {copilot_model})"
        )
        return

    logger.warning(
        f"Context guard rejected (OpenAI): estimated {estimated_tokens} tokens "
        f"exceeds {context_limit} limit with < {_MIN_OUTPUT_TOKENS} room "
        f"for model {copilot_model}"
    )
    raise ContextWindowExceededError(
        prompt_tokens=context_limit,
        context_limit=context_limit,
        backend="proxy",
    )


def check_context_guard_responses(body: dict[str, Any]) -> None:
    """Guard for /v1/responses (Responses API format).

    Translates to Anthropic format to estimate tokens, then clamps or rejects.
    """
    if CONTEXT_GUARD_THRESHOLD <= 0:
        return
    model = body.get("model", "")
    copilot_model = get_copilot_openai_model(model)
    context_limit = get_copilot_context_limit(copilot_model)
    if context_limit <= 0:
        return
    from .responses_translate import responses_to_anthropic_request

    anthropic_body = responses_to_anthropic_request(body)
    estimated_tokens = estimate_input_tokens(anthropic_body)
    effective_limit = int(context_limit * CONTEXT_GUARD_THRESHOLD)
    if estimated_tokens <= effective_limit:
        return

    available = context_limit - estimated_tokens
    if available >= _MIN_OUTPUT_TOKENS:
        original_max = body.get("max_output_tokens") or 0
        clamped = min(original_max, available) if original_max else available
        body["max_output_tokens"] = clamped
        logger.info(
            f"Context guard clamped max_tokens (Responses): {original_max} -> {clamped} "
            f"(estimated {estimated_tokens}/{context_limit} tokens for {copilot_model})"
        )
        return

    logger.warning(
        f"Context guard rejected (Responses): estimated {estimated_tokens} tokens "
        f"exceeds {context_limit} limit with < {_MIN_OUTPUT_TOKENS} room "
        f"for model {copilot_model}"
    )
    raise ContextWindowExceededError(
        prompt_tokens=context_limit,
        context_limit=context_limit,
        backend="proxy",
    )
