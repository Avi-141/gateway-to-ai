"""FastAPI application and route handlers."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import tiktoken
from botocore.exceptions import ClientError, ReadTimeoutError
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .backend_state import BackendState, parse_backend_string
from .bedrock_client import get_bedrock_client, reset_bedrock_client
from .config import (
    BACKEND_TYPE,
    BEDROCK_REGION_PREFIX,
    COPILOT_TIMEOUT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    FALLBACK_BACKEND,
    LOG_LEVEL,
    logger,
)
from .context_guard import check_context_guard_anthropic, check_context_guard_openai, check_context_guard_responses
from .copilot_client import compute_initiator
from .copilot_translate import has_server_tools, strip_server_tools
from .copilot_usage import CopilotUsageCache
from .errors import ContextWindowExceededError, CopilotHttpError, TransientBackendError
from .models import (
    BEDROCK_MODEL_MAP,
    COPILOT_MODEL_MAP,
    COPILOT_OPENAI_MODEL_MAP,
    add_region_prefix,
    get_available_copilot_models,
    get_bedrock_model,
    get_copilot_context_window,
    get_copilot_model,
    get_copilot_openai_model,
    is_claude_model,
    model_requires_responses_api,
    model_supports_responses_api,
    set_copilot_models,
)
from .openai_translate import (
    ReverseStreamTranslator,
    anthropic_to_openai_response,
    openai_to_anthropic_request,
)
from .request_stats import request_stats
from .responses_translate import (
    AnthropicToResponsesStreamTranslator,
    anthropic_to_responses_response,
    responses_to_anthropic_request,
)
from .server_url import remove_server_url, write_server_url

# Use cl100k_base encoding (similar to Claude's tokenizer)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Get package version
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("claudegate")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# Error message for expired credentials
CREDENTIALS_EXPIRED_MSG = (
    "AWS credentials have expired. Please re-authenticate in your terminal to refresh your credentials, then retry."
)

# Backend state (initialized in lifespan)
_backend_state = BackendState(BACKEND_TYPE, FALLBACK_BACKEND)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Attach ring buffer handler to loggers (must happen here, after
    # uvicorn's dictConfig has run and replaced all handler lists)
    from .log_buffer import attach_log_buffer

    attach_log_buffer()

    primary = _backend_state.primary
    fallback = _backend_state.fallback

    # Validate fallback config
    if fallback:
        if fallback == primary:
            raise ValueError(f"FALLBACK_BACKEND cannot be the same as BACKEND_TYPE: {primary}")
        valid = {"bedrock", "copilot"}
        if fallback not in valid:
            raise ValueError(f"Invalid FALLBACK_BACKEND: {fallback}, must be one of {valid}")

    # Startup
    logger.info(f"Starting claudegate v{__version__}")
    logger.info(f"Host: {os.environ.get('CLAUDEGATE_HOST', DEFAULT_HOST)}")
    logger.info(f"Port: {os.environ.get('CLAUDEGATE_PORT', DEFAULT_PORT)}")
    logger.info(f"Backend: {primary}")
    if fallback:
        logger.info(f"Fallback: {fallback}")
    logger.info(f"Log Level: {LOG_LEVEL}")

    host = os.environ.get("CLAUDEGATE_HOST", DEFAULT_HOST)
    port = int(os.environ.get("CLAUDEGATE_PORT", str(DEFAULT_PORT)))
    write_server_url(host, port)

    # Initialize copilot if it's primary or fallback
    needs_copilot = primary == "copilot" or fallback == "copilot"
    if needs_copilot:
        from .copilot_auth import CopilotAuth, get_github_token
        from .copilot_client import CopilotBackend

        github_token = get_github_token()
        auth = CopilotAuth(github_token)
        copilot_backend = CopilotBackend(auth, timeout=COPILOT_TIMEOUT)
        logger.info("Copilot backend initialized")

        copilot_usage_cache = CopilotUsageCache(github_token)
        logger.info("Copilot usage cache initialized")

        _backend_state.set_copilot_backend(copilot_backend, copilot_usage_cache)

        models = await copilot_backend.list_models()
        if models:
            set_copilot_models(models)
            logger.info(f"Loaded {len(models)} models from Copilot API")
        else:
            logger.warning("No models fetched from Copilot API, using hardcoded maps")

    needs_bedrock = primary == "bedrock" or fallback == "bedrock"
    if needs_bedrock:
        logger.info(f"AWS Region: {os.environ.get('AWS_REGION', 'us-west-2')}")
        logger.info(f"Bedrock Region Prefix: {BEDROCK_REGION_PREFIX or '(none)'}")

    yield

    # Shutdown
    remove_server_url()
    await _backend_state.close()
    logger.info("Shutting down claudegate")


app = FastAPI(title="claudegate", version=__version__, lifespan=lifespan)


# --- Helper Functions ---


def _error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    """Return Anthropic-style error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
    )


def _openai_error_response(status_code: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    """Return OpenAI-style error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            },
        },
    )


def _validate_request(body: dict[str, Any]) -> JSONResponse | None:
    """Validate required fields in request body. Returns error response or None."""
    if "model" not in body:
        return _error_response(400, "invalid_request_error", "Missing required field: model")
    if "max_tokens" not in body:
        return _error_response(400, "invalid_request_error", "Missing required field: max_tokens")
    if "messages" not in body:
        return _error_response(400, "invalid_request_error", "Missing required field: messages")
    if not isinstance(body["messages"], list):
        return _error_response(400, "invalid_request_error", "messages must be an array")
    if not body["messages"]:
        return _error_response(400, "invalid_request_error", "messages must not be empty")
    return None


def _context_window_error_response(err: ContextWindowExceededError, max_tokens: int = 0) -> JSONResponse:
    """Return an Anthropic-format error that triggers Claude Code's auto-compaction.

    Claude Code recognises errors matching the pattern:
      "input length and `max_tokens` exceed context limit: X + Y > Z"
    and responds by compacting/summarizing context before retrying.

    When exact token counts aren't available, falls back to the raw backend
    error message rather than fabricating incorrect numbers.
    """
    if err.prompt_tokens > 0 and err.context_limit > 0:
        message = (
            f"input length and `max_tokens` exceed context limit: "
            f"{err.prompt_tokens} + {max_tokens} > {err.context_limit}, "
            f"decrease input length or `max_tokens` and try again"
        )
    else:
        # No exact numbers — use a message that still triggers compaction
        message = (
            "input length and `max_tokens` exceed context limit, decrease input length or `max_tokens` and try again"
        )
    return _error_response(400, "invalid_request_error", message)


def _openai_context_window_message(err: ContextWindowExceededError) -> str:
    """Build an OpenAI-style error message for context window exceeded."""
    if err.prompt_tokens > 0 and err.context_limit > 0:
        return f"prompt token count of {err.prompt_tokens} exceeds the limit of {err.context_limit}"
    return "prompt exceeds the model's maximum context window"


def _count_content_tokens(content: Any) -> int:
    """Count tokens in content (string or list of blocks)."""
    if isinstance(content, str):
        return len(tokenizer.encode(content))
    elif isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    total += len(tokenizer.encode(block["text"]))
                elif block.get("type") == "tool_use":
                    # Count tool name and input as JSON
                    total += len(tokenizer.encode(block.get("name", "")))
                    total += len(tokenizer.encode(json.dumps(block.get("input", {}))))
                elif block.get("type") == "tool_result":
                    # Count tool result content
                    total += _count_content_tokens(block.get("content", ""))
        return total
    return 0


# --- Bedrock Helpers ---


# Beta flag prefixes that Bedrock accepts in the anthropic_beta body field.
# Bedrock handles most features (thinking, caching, tool use) via dedicated body
# parameters, but some features still require explicit beta flags.
# Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
_BEDROCK_BETA_PREFIXES = (
    "context-1m-",  # 1M token context window (Sonnet 4.5, Sonnet 4, Opus 4.6)
    "effort-",  # Effort parameter for Opus 4.5
)


def _filter_beta_flags(flags: list[str]) -> list[str]:
    """Keep only beta flags whose prefix Bedrock recognises."""
    return [f for f in flags if any(f.startswith(p) for p in _BEDROCK_BETA_PREFIXES)]


# Bedrock tool sanitization: allowed keys at each level
_BEDROCK_TOOL_ALLOWED_KEYS = {"name", "description", "input_schema", "type", "custom", "cache_control"}
_BEDROCK_CUSTOM_TOOL_KEYS = {"name", "description", "input_schema"}

# Content block types that Bedrock does not support (Claude Code extensions)
_BEDROCK_UNSUPPORTED_CONTENT_TYPES = {"tool_reference"}


def _clean_tool_for_bedrock(tool: dict[str, Any]) -> dict[str, Any]:
    """Strip client-side metadata from a tool definition that Bedrock rejects.

    Claude Code adds extra fields like defer_loading on tools (either at the
    top level or inside the 'custom' object). Bedrock validates strictly and
    rejects unknown properties.
    """
    # Strip unknown top-level keys
    cleaned = {k: v for k, v in tool.items() if k in _BEDROCK_TOOL_ALLOWED_KEYS}
    # Strip unknown keys inside custom object
    if "custom" in cleaned:
        custom = {k: v for k, v in cleaned["custom"].items() if k in _BEDROCK_CUSTOM_TOOL_KEYS}
        if custom:
            cleaned["custom"] = custom
        else:
            del cleaned["custom"]
    return cleaned


def _clean_messages_for_bedrock(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip unsupported content block types from messages for Bedrock.

    Claude Code sends content block types like 'tool_reference' that Bedrock
    doesn't recognize. These are filtered out from all message content arrays,
    including nested content inside tool_result blocks.
    """
    cleaned = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        new_content = []
        for block in content:
            block_type = block.get("type", "")
            if block_type in _BEDROCK_UNSUPPORTED_CONTENT_TYPES:
                continue
            # Also clean nested content inside tool_result blocks
            if block_type == "tool_result" and isinstance(block.get("content"), list):
                nested = [b for b in block["content"] if b.get("type") not in _BEDROCK_UNSUPPORTED_CONTENT_TYPES]
                if nested != block["content"]:
                    block = {**block, "content": nested}
            new_content.append(block)
        cleaned.append({**msg, "content": new_content})
    return cleaned


def _build_bedrock_body(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Build Bedrock request body from Anthropic request.

    Beta flags are filtered to only those Bedrock supports. Unsupported flags
    (e.g., interleaved-thinking, claude-code, prompt-caching-scope) are dropped
    since Bedrock controls those features via dedicated body parameters.
    """
    bedrock_body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": body["max_tokens"],
        "messages": _clean_messages_for_bedrock(body["messages"]),
    }

    # Merge beta flags from header and body, filter to Bedrock-supported prefixes
    beta_flags: list[str] = []
    beta_header = request.headers.get("anthropic-beta")
    if beta_header:
        beta_flags.extend(b.strip() for b in beta_header.split(","))
    if "anthropic_beta" in body:
        beta_flags.extend(body["anthropic_beta"])
    if beta_flags:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_flags = [f for f in beta_flags if f not in seen and not seen.add(f)]  # type: ignore[func-returns-value]
        filtered = _filter_beta_flags(unique_flags)
        dropped = [f for f in unique_flags if f not in filtered]
        if dropped:
            logger.debug(f"Dropped unsupported beta flags for Bedrock: {dropped}")
        if filtered:
            bedrock_body["anthropic_beta"] = filtered

    # Optional fields - pass through all supported Anthropic API parameters
    optional_fields = [
        "system",
        "temperature",
        "top_p",
        "top_k",
        "tools",
        "tool_choice",
        "thinking",
        "stop_sequences",
        "metadata",
    ]
    for field in optional_fields:
        if field in body:
            bedrock_body[field] = body[field]

    # Strip extra fields from tools that Bedrock doesn't accept (e.g. defer_loading)
    if "tools" in bedrock_body:
        bedrock_body["tools"] = [_clean_tool_for_bedrock(t) for t in bedrock_body["tools"]]

    return bedrock_body


def _open_bedrock_stream(model: str, body: dict[str, Any]) -> dict[str, Any]:
    """Open a bedrock streaming connection. Returns the response dict.

    Raises TransientBackendError for fallback-eligible ClientErrors.
    Re-raises non-transient ClientError as-is.
    """
    try:
        bedrock = get_bedrock_client()
        return bedrock.invoke_model_with_response_stream(modelId=model, body=json.dumps(body))
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_message = e.response.get("Error", {}).get("Message", str(e))
        if error_code == "ThrottlingException":
            raise TransientBackendError(429, "rate_limit_error", error_message, "bedrock") from e
        elif error_code == "ModelTimeoutException":
            raise TransientBackendError(504, "timeout_error", error_message, "bedrock") from e
        elif error_code in ("ServiceUnavailableException", "InternalServerException"):
            raise TransientBackendError(500, "api_error", error_message, "bedrock") from e
        raise
    except ReadTimeoutError as e:
        raise TransientBackendError(504, "timeout_error", str(e), "bedrock") from e


async def _stream_bedrock_chunks(response: dict[str, Any], request_id: str = "") -> AsyncGenerator[str, None]:
    """Iterate chunks from an already-opened bedrock stream."""
    log_prefix = f"[{request_id}] " if request_id else ""
    chunk_count = 0
    try:
        for event in response["body"]:
            chunk = event.get("chunk")
            if chunk:
                data = json.loads(chunk["bytes"].decode())
                chunk_count += 1
                if chunk_count <= 3:
                    logger.debug(f"{log_prefix}Chunk {chunk_count}: {json.dumps(data)[:200]}")
                event_type = data.get("type", "unknown")
                sse_data = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                yield sse_data
                await asyncio.sleep(0)

        logger.info(f"{log_prefix}Stream complete, {chunk_count} chunks sent")
        yield "event: done\ndata: [DONE]\n\n"
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("ExpiredTokenException", "ExpiredToken"):
            logger.error(f"{log_prefix}Credentials expired during stream")
            reset_bedrock_client()
            error_text = f"\n\n\u26a0\ufe0f **Authentication Error**: {CREDENTIALS_EXPIRED_MSG}\n"
            block_start = json.dumps(
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
            )
            block_delta = json.dumps(
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": error_text}}
            )
            block_stop = json.dumps({"type": "content_block_stop", "index": 0})
            yield f"event: content_block_start\ndata: {block_start}\n\n"
            yield f"event: content_block_delta\ndata: {block_delta}\n\n"
            yield f"event: content_block_stop\ndata: {block_stop}\n\n"
        else:
            logger.error(f"{log_prefix}Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
    except Exception as e:
        logger.error(f"{log_prefix}Stream error: {e}", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"


async def _call_bedrock(
    body: dict[str, Any], request: Request, request_id: str, stream: bool
) -> JSONResponse | StreamingResponse:
    """Execute request against Bedrock backend.

    Raises TransientBackendError for fallback-eligible errors.
    Returns JSONResponse/StreamingResponse on success or non-transient error.
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    bedrock_model = get_bedrock_model(body["model"])
    initiator = compute_initiator(body)
    logger.info(
        f"{log_prefix}Request - model: {body['model']} -> {bedrock_model}, stream: {stream}, initiator: {initiator}"
    )
    logger.debug(f"{log_prefix}Request body keys: {list(body.keys())}")

    bedrock_body = _build_bedrock_body(body, request)

    if stream:
        # Two-phase: open (can raise TransientBackendError or ClientError), then iterate
        try:
            response = _open_bedrock_stream(bedrock_model, bedrock_body)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            if error_code in ("ExpiredTokenException", "ExpiredToken"):
                logger.error(f"{log_prefix}Credentials expired")
                reset_bedrock_client()
                return _error_response(401, "authentication_error", CREDENTIALS_EXPIRED_MSG)
            elif error_code == "ValidationException":
                logger.error(f"{log_prefix}Validation error: {error_message}")
                return _error_response(400, "invalid_request_error", error_message)
            elif error_code == "AccessDeniedException":
                logger.error(f"{log_prefix}Access denied: {error_message}")
                return _error_response(403, "permission_error", error_message)
            else:
                logger.error(f"{log_prefix}Bedrock error ({error_code}): {error_message}")
                return _error_response(500, "api_error", error_message)
        return StreamingResponse(
            _stream_bedrock_chunks(response, request_id),
            media_type="text/event-stream",
        )

    try:
        bedrock = get_bedrock_client()
        response = bedrock.invoke_model(modelId=bedrock_model, body=json.dumps(bedrock_body))
        result = json.loads(response["body"].read())
        logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
        return JSONResponse(content=result)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        if error_code in ("ExpiredTokenException", "ExpiredToken"):
            logger.error(f"{log_prefix}Credentials expired")
            reset_bedrock_client()
            return _error_response(401, "authentication_error", CREDENTIALS_EXPIRED_MSG)
        elif error_code == "ValidationException":
            logger.error(f"{log_prefix}Validation error: {error_message}")
            return _error_response(400, "invalid_request_error", error_message)
        elif error_code == "AccessDeniedException":
            logger.error(f"{log_prefix}Access denied: {error_message}")
            return _error_response(403, "permission_error", error_message)
        elif error_code == "ThrottlingException":
            raise TransientBackendError(429, "rate_limit_error", error_message, "bedrock") from e
        elif error_code == "ModelTimeoutException":
            raise TransientBackendError(504, "timeout_error", error_message, "bedrock") from e
        elif error_code in ("ServiceUnavailableException", "InternalServerException"):
            raise TransientBackendError(500, "api_error", error_message, "bedrock") from e
        else:
            logger.error(f"{log_prefix}Bedrock error ({error_code}): {error_message}")
            return _error_response(500, "api_error", error_message)
    except ReadTimeoutError as e:
        raise TransientBackendError(
            504, "timeout_error", "Request timed out. Try a smaller request or use streaming.", "bedrock"
        ) from e
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error: {e}", exc_info=True)
        return _error_response(500, "api_error", str(e))


def _detect_client_context_window(request: Request, copilot_model: str) -> int:
    """Detect the client's expected context window size.

    Claude Code uses the anthropic-beta header 'context-1m-...' when the user
    selects the 1M token variant (e.g. opus-4-6[1m]). When present, the client
    expects a 1,000,000-token context window. Otherwise, we use the model's
    max_context_window_tokens from the Copilot API (typically 200k).
    """
    beta_header = request.headers.get("anthropic-beta", "")
    if "context-1m" in beta_header:
        return 1_000_000
    return get_copilot_context_window(copilot_model)


async def _call_copilot(
    body: dict[str, Any],
    request: Request,
    request_id: str,
    stream: bool,
) -> JSONResponse | StreamingResponse:
    """Execute request against Copilot backend.

    Raises TransientBackendError for fallback-eligible errors.
    Raises CopilotHttpError, RuntimeError, httpx.TimeoutException for non-fallback errors.
    """
    copilot = _backend_state.copilot_backend
    if copilot is None:
        raise RuntimeError("Copilot backend not initialized")
    copilot_model, anthropic_model = get_copilot_model(body["model"])
    log_prefix = f"[{request_id}] " if request_id else ""
    initiator = compute_initiator(body)
    logger.info(
        f"{log_prefix}Request - model: {body['model']} -> {copilot_model} (copilot), "
        f"stream: {stream}, initiator: {initiator}"
    )
    client_context_window = _detect_client_context_window(request, copilot_model)
    if model_requires_responses_api(copilot_model):
        logger.info(f"{log_prefix}Routing to Responses API for {copilot_model}")
        return await copilot.handle_responses_messages(
            body, request_id, stream, copilot_model, anthropic_model, client_context_window
        )
    return await copilot.handle_messages(
        body, request_id, stream, copilot_model, anthropic_model, client_context_window
    )


async def _call_copilot_openai(body: dict[str, Any], request_id: str, stream: bool) -> JSONResponse | StreamingResponse:
    """Execute OpenAI-format request directly against Copilot (no translation).

    Raises TransientBackendError for fallback-eligible errors.
    Raises CopilotHttpError, RuntimeError, httpx.TimeoutException for non-fallback errors.
    """
    copilot = _backend_state.copilot_backend
    if copilot is None:
        raise RuntimeError("Copilot backend not initialized")
    copilot_model = get_copilot_openai_model(body["model"])
    log_prefix = f"[{request_id}] " if request_id else ""
    logger.info(
        f"{log_prefix}OpenAI passthrough - model: {body['model']} -> {copilot_model} (copilot), stream: {stream}"
    )
    if model_requires_responses_api(copilot_model):
        logger.info(f"{log_prefix}Routing to Responses API for {copilot_model}")
        return await copilot.handle_openai_responses_messages(body, request_id, stream, copilot_model)
    return await copilot.handle_openai_messages(body, request_id, stream, copilot_model)


async def _call_bedrock_for_openai(
    body: dict[str, Any], request: Request, request_id: str, stream: bool
) -> JSONResponse | StreamingResponse:
    """Execute OpenAI-format request against Bedrock with translation.

    Translates OpenAI -> Anthropic, calls Bedrock, translates response back to OpenAI.
    Raises TransientBackendError for fallback-eligible errors.
    """
    anthropic_body = openai_to_anthropic_request(body)
    result = await _call_bedrock(anthropic_body, request, request_id, stream)

    # Convert Anthropic error responses to OpenAI format
    if isinstance(result, JSONResponse) and result.status_code >= 400:
        anthropic_err = json.loads(bytes(result.body))
        err_info = anthropic_err.get("error", {})
        return _openai_error_response(
            result.status_code,
            err_info.get("message", "Unknown error"),
            err_info.get("type", "server_error"),
        )

    # Translate response from Anthropic format to OpenAI format
    if isinstance(result, StreamingResponse):
        translator = ReverseStreamTranslator(body["model"])

        async def _translate_stream() -> AsyncGenerator[str, None]:
            async for chunk in result.body_iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                translated = translator.translate_sse(text)
                if translated:
                    yield translated

        return StreamingResponse(_translate_stream(), media_type="text/event-stream")
    else:
        anthropic_resp = json.loads(bytes(result.body))
        openai_resp = anthropic_to_openai_response(anthropic_resp, body["model"])
        return JSONResponse(content=openai_resp)


# --- Responses API Helpers ---


def _validate_responses_request(body: dict[str, Any]) -> JSONResponse | None:
    """Validate required fields in Responses API request body. Returns error response or None."""
    if "model" not in body:
        return _openai_error_response(400, "Missing required field: model")
    if "input" not in body:
        return _openai_error_response(400, "Missing required field: input")
    return None


async def _call_copilot_responses(
    body: dict[str, Any], request_id: str, stream: bool
) -> JSONResponse | StreamingResponse:
    """Execute Responses API request against Copilot backend.

    Routes: if model supports /responses -> passthrough, else -> via chat.
    """
    copilot = _backend_state.copilot_backend
    if copilot is None:
        raise RuntimeError("Copilot backend not initialized")

    model = body.get("model", "")
    copilot_model = get_copilot_openai_model(model)
    log_prefix = f"[{request_id}] " if request_id else ""

    # Update model in body to the resolved copilot model
    body = {**body, "model": copilot_model}

    if model_supports_responses_api(copilot_model):
        logger.info(f"{log_prefix}Responses passthrough to {copilot_model}")
        return await copilot.handle_responses_passthrough(body, request_id, stream)
    else:
        logger.info(f"{log_prefix}Responses via chat for {copilot_model}")
        return await copilot.handle_responses_via_chat(body, request_id, stream, copilot_model)


async def _call_bedrock_for_responses(
    body: dict[str, Any], request: Request, request_id: str, stream: bool
) -> JSONResponse | StreamingResponse:
    """Execute Responses API request against Bedrock with translation.

    Translates Responses -> Anthropic, calls Bedrock, translates response back to Responses.
    """
    model = body.get("model", "")

    if not is_claude_model(model):
        return _openai_error_response(
            400,
            f"Model '{model}' is not supported on the Bedrock backend. Non-Claude models require the Copilot backend.",
        )

    anthropic_body = responses_to_anthropic_request(body)
    result = await _call_bedrock(anthropic_body, request, request_id, stream)

    # Convert Anthropic error responses to OpenAI format
    if isinstance(result, JSONResponse) and result.status_code >= 400:
        anthropic_err = json.loads(bytes(result.body))
        err_info = anthropic_err.get("error", {})
        return _openai_error_response(
            result.status_code,
            err_info.get("message", "Unknown error"),
            err_info.get("type", "server_error"),
        )

    # Translate response from Anthropic format to Responses format
    if isinstance(result, StreamingResponse):
        translator = AnthropicToResponsesStreamTranslator(model)

        async def _translate_stream() -> AsyncGenerator[str, None]:
            from .openai_translate import parse_anthropic_sse

            async for chunk in result.body_iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                sse_events = parse_anthropic_sse(text)
                for event_type, data in sse_events:
                    translated = translator.translate_event(event_type, data)
                    if translated:
                        yield translated
            # Flush remaining state
            flushed = translator.flush()
            if flushed:
                yield flushed

        return StreamingResponse(_translate_stream(), media_type="text/event-stream")
    else:
        anthropic_resp = json.loads(bytes(result.body))
        responses_resp = anthropic_to_responses_response(anthropic_resp, model)
        return JSONResponse(content=responses_resp)


# --- Route Handlers ---


@app.post("/v1/messages", response_model=None)
async def messages(request: Request) -> JSONResponse | StreamingResponse:
    """Handle chat completion requests."""
    request_id = request.headers.get("x-request-id", "")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _error_response(400, "invalid_request_error", "Invalid JSON in request body")

    if error := _validate_request(body):
        return error

    stream = body.get("stream", False)
    log_prefix = f"[{request_id}] " if request_id else ""

    # Route non-Claude models (GPT, Gemini, Grok, etc.) to Copilot when available.
    # Bedrock only supports Claude models, so non-Claude models need Copilot.
    requested_model = body["model"]
    if not is_claude_model(requested_model):
        copilot_available = _backend_state.primary == "copilot" or _backend_state.fallback == "copilot"
        if not copilot_available:
            return _error_response(
                400,
                "invalid_request_error",
                f"Model '{requested_model}' is not supported on the Bedrock backend. "
                "Non-Claude models require the Copilot backend.",
            )
        # Force route to Copilot for non-Claude models
        try:
            check_context_guard_anthropic(body)
        except ContextWindowExceededError as e:
            request_stats.record_context_guard_rejection()
            return _context_window_error_response(e, body.get("max_tokens", 0))
        try:
            return await _call_copilot(body, request, request_id, stream)
        except ContextWindowExceededError as e:
            return _context_window_error_response(e, body.get("max_tokens", 0))
        except TransientBackendError as e:
            return _error_response(e.status_code, e.error_type, e.message)
        except CopilotHttpError as e:
            return _error_response(e.status_code, "api_error", e.detail)
        except RuntimeError as e:
            logger.error(f"{log_prefix}Auth error: {e}")
            return _error_response(401, "authentication_error", str(e))
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error: {e}", exc_info=True)
            return _error_response(500, "api_error", str(e))

    # Route requests with server-side tools (e.g. web_search) appropriately.
    # Copilot doesn't support server-side tools, so route to Bedrock if available,
    # or strip them from the request if Copilot-only.
    if has_server_tools(body) and _backend_state.primary == "copilot":
        if _backend_state.fallback == "bedrock":
            logger.info(f"{log_prefix}Server-side tools detected, routing to Bedrock fallback")
            try:
                return await _call_bedrock(body, request, request_id, stream)
            except TransientBackendError as e:
                logger.warning(
                    f"{log_prefix}Bedrock fallback failed ({e.status_code}), "
                    "stripping server tools and falling back to Copilot"
                )
                body = strip_server_tools(body)
            except Exception as e:
                logger.warning(
                    f"{log_prefix}Bedrock fallback failed ({e}), stripping server tools and falling back to Copilot"
                )
                body = strip_server_tools(body)
        else:
            logger.info(f"{log_prefix}Server-side tools detected but no Bedrock fallback, stripping from request")
            body = strip_server_tools(body)

    # Map backend name to call function
    def _get_backend_caller(backend_name: str):
        if backend_name == "copilot":
            return lambda: _call_copilot(body, request, request_id, stream)
        return lambda: _call_bedrock(body, request, request_id, stream)

    # Capture current backend config for this request
    current_primary = _backend_state.primary
    current_fallback = _backend_state.fallback
    primary_call = _get_backend_caller(current_primary)
    request_stats.record_request(current_primary)

    _original_max_tokens = body.get("max_tokens")
    if current_primary == "copilot":
        try:
            check_context_guard_anthropic(body)
        except ContextWindowExceededError as e:
            request_stats.record_context_guard_rejection()
            return _context_window_error_response(e, body.get("max_tokens", 0))

    try:
        return await primary_call()
    except ContextWindowExceededError as e:
        max_tokens = body.get("max_tokens", 0)
        if current_fallback:
            request_stats.record_fallback()
            logger.warning(
                f"{log_prefix}Context window exceeded on {e.backend} "
                f"({e.prompt_tokens} > {e.context_limit}), falling back to {current_fallback}"
            )
            if _original_max_tokens is not None:
                body["max_tokens"] = _original_max_tokens
            elif "max_tokens" in body:
                del body["max_tokens"]
            fallback_call = _get_backend_caller(current_fallback)
            try:
                return await fallback_call()
            except ContextWindowExceededError:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) also exceeded context window")
                request_stats.record_error()
                return _context_window_error_response(e, max_tokens)
            except TransientBackendError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) failed with {fallback_err.status_code}")
                request_stats.record_error()
                return _context_window_error_response(e, max_tokens)
            except CopilotHttpError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) HTTP error: {fallback_err}")
                request_stats.record_error()
                return _context_window_error_response(e, max_tokens)
            except RuntimeError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) auth error: {fallback_err}")
                request_stats.record_error()
                return _context_window_error_response(e, max_tokens)
        else:
            logger.error(
                f"{log_prefix}Context window exceeded on {e.backend} "
                f"({e.prompt_tokens} > {e.context_limit}), no fallback configured"
            )
            request_stats.record_error()
            return _context_window_error_response(e, max_tokens)
    except TransientBackendError as e:
        if not current_fallback:
            logger.error(f"{log_prefix}{e.backend} transient error {e.status_code}, no fallback configured")
            request_stats.record_error()
            return _error_response(e.status_code, e.error_type, e.message)

        request_stats.record_fallback()
        logger.warning(
            f"{log_prefix}Primary ({current_primary}) failed with {e.status_code}, falling back to {current_fallback}"
        )

        if _original_max_tokens is not None:
            body["max_tokens"] = _original_max_tokens
        elif "max_tokens" in body:
            del body["max_tokens"]
        fallback_call = _get_backend_caller(current_fallback)
        try:
            return await fallback_call()
        except TransientBackendError as fallback_err:
            logger.error(f"{log_prefix}Fallback ({current_fallback}) also failed with {fallback_err.status_code}")
            request_stats.record_error()
            return _error_response(fallback_err.status_code, fallback_err.error_type, fallback_err.message)
        except CopilotHttpError as fallback_err:
            request_stats.record_error()
            return _error_response(fallback_err.status_code, "api_error", fallback_err.detail)
        except RuntimeError as fallback_err:
            request_stats.record_error()
            return _error_response(401, "authentication_error", str(fallback_err))
        except Exception as fallback_err:
            logger.error(f"{log_prefix}Fallback ({current_fallback}) unexpected error: {fallback_err}", exc_info=True)
            request_stats.record_error()
            return _error_response(500, "api_error", str(fallback_err))
    except CopilotHttpError as e:
        request_stats.record_error()
        return _error_response(e.status_code, "api_error", e.detail)
    except RuntimeError as e:
        logger.error(f"{log_prefix}Auth error: {e}")
        request_stats.record_error()
        return _error_response(401, "authentication_error", str(e))
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error: {e}", exc_info=True)
        request_stats.record_error()
        return _error_response(500, "api_error", str(e))


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Handle OpenAI-compatible chat completion requests.

    When Copilot is the backend, requests pass through directly (0 translations).
    When Bedrock is the backend, requests are translated OpenAI -> Anthropic -> Bedrock -> Anthropic -> OpenAI.
    """
    request_id = request.headers.get("x-request-id", "")
    log_prefix = f"[{request_id}] " if request_id else ""

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _openai_error_response(400, "Invalid JSON in request body")

    # Validate required fields (OpenAI format: model + messages required, max_tokens optional)
    if "model" not in body:
        return _openai_error_response(400, "Missing required field: model")
    if "messages" not in body:
        return _openai_error_response(400, "Missing required field: messages")
    if not isinstance(body["messages"], list):
        return _openai_error_response(400, "messages must be an array")
    if not body["messages"]:
        return _openai_error_response(400, "messages must not be empty")

    stream = body.get("stream", False)
    initiator = compute_initiator(body)

    logger.info(f"{log_prefix}OpenAI-compat request - model: {body['model']}, stream: {stream}, initiator: {initiator}")

    # Map backend name to call function
    # Copilot: direct passthrough (0 translations)
    # Bedrock: translated (OpenAI -> Anthropic -> Bedrock -> Anthropic -> OpenAI)
    def _get_backend_caller(backend_name: str):
        if backend_name == "copilot":
            return lambda: _call_copilot_openai(body, request_id, stream)
        return lambda: _call_bedrock_for_openai(body, request, request_id, stream)

    # Capture current backend config for this request
    current_primary = _backend_state.primary
    current_fallback = _backend_state.fallback
    primary_call = _get_backend_caller(current_primary)
    request_stats.record_request(current_primary)

    _original_max_tokens_openai = body.get("max_tokens")
    _original_max_completion_tokens = body.get("max_completion_tokens")
    if current_primary == "copilot":
        try:
            check_context_guard_openai(body)
        except ContextWindowExceededError as e:
            request_stats.record_context_guard_rejection()
            return _openai_error_response(400, _openai_context_window_message(e))

    try:
        return await primary_call()
    except ContextWindowExceededError as e:
        _ctx_msg = _openai_context_window_message(e)
        if current_fallback:
            request_stats.record_fallback()
            logger.warning(
                f"{log_prefix}Context window exceeded on {e.backend} "
                f"({e.prompt_tokens} > {e.context_limit}), falling back to {current_fallback}"
            )
            if _original_max_tokens_openai is not None:
                body["max_tokens"] = _original_max_tokens_openai
            elif "max_tokens" in body:
                del body["max_tokens"]
            if _original_max_completion_tokens is not None:
                body["max_completion_tokens"] = _original_max_completion_tokens
            elif "max_completion_tokens" in body:
                del body["max_completion_tokens"]
            fallback_call = _get_backend_caller(current_fallback)
            try:
                return await fallback_call()
            except ContextWindowExceededError:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) also exceeded context window")
                request_stats.record_error()
                return _openai_error_response(400, _ctx_msg)
            except TransientBackendError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) failed with {fallback_err.status_code}")
                request_stats.record_error()
                return _openai_error_response(400, _ctx_msg)
            except CopilotHttpError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) HTTP error: {fallback_err}")
                request_stats.record_error()
                return _openai_error_response(400, _ctx_msg)
            except RuntimeError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) auth error: {fallback_err}")
                request_stats.record_error()
                return _openai_error_response(400, _ctx_msg)
        else:
            logger.error(
                f"{log_prefix}Context window exceeded on {e.backend} "
                f"({e.prompt_tokens} > {e.context_limit}), no fallback configured"
            )
            request_stats.record_error()
            return _openai_error_response(400, _ctx_msg)
    except TransientBackendError as e:
        if not current_fallback:
            logger.error(f"{log_prefix}{e.backend} transient error {e.status_code}, no fallback configured")
            request_stats.record_error()
            return _openai_error_response(e.status_code, e.message, "server_error")

        request_stats.record_fallback()
        logger.warning(
            f"{log_prefix}Primary ({current_primary}) failed with {e.status_code}, falling back to {current_fallback}"
        )

        if _original_max_tokens_openai is not None:
            body["max_tokens"] = _original_max_tokens_openai
        elif "max_tokens" in body:
            del body["max_tokens"]
        if _original_max_completion_tokens is not None:
            body["max_completion_tokens"] = _original_max_completion_tokens
        elif "max_completion_tokens" in body:
            del body["max_completion_tokens"]
        fallback_call = _get_backend_caller(current_fallback)
        try:
            return await fallback_call()
        except TransientBackendError as fallback_err:
            logger.error(f"{log_prefix}Fallback ({current_fallback}) also failed with {fallback_err.status_code}")
            request_stats.record_error()
            return _openai_error_response(fallback_err.status_code, fallback_err.message, "server_error")
        except CopilotHttpError as fallback_err:
            request_stats.record_error()
            return _openai_error_response(fallback_err.status_code, fallback_err.detail, "server_error")
        except RuntimeError as fallback_err:
            request_stats.record_error()
            return _openai_error_response(401, str(fallback_err), "authentication_error")
        except Exception as fallback_err:
            logger.error(f"{log_prefix}Fallback ({current_fallback}) unexpected error: {fallback_err}", exc_info=True)
            request_stats.record_error()
            return _openai_error_response(500, str(fallback_err), "server_error")
    except CopilotHttpError as e:
        request_stats.record_error()
        return _openai_error_response(e.status_code, e.detail, "server_error")
    except RuntimeError as e:
        logger.error(f"{log_prefix}Auth error: {e}")
        request_stats.record_error()
        return _openai_error_response(401, str(e), "authentication_error")
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error: {e}", exc_info=True)
        request_stats.record_error()
        return _openai_error_response(500, str(e), "server_error")


@app.post("/v1/responses", response_model=None)
async def responses(request: Request) -> JSONResponse | StreamingResponse:
    """Handle Responses API requests.

    When Copilot is the backend and model supports /responses: passthrough (0 translations).
    When Copilot is the backend but model only has /chat/completions:
        Responses -> Chat Completions -> Copilot -> Chat Completions -> Responses (2 translations).
    When Bedrock is the backend:
        Responses -> Anthropic -> Bedrock -> Anthropic -> Responses (2 translations).
    """
    request_id = request.headers.get("x-request-id", "")
    log_prefix = f"[{request_id}] " if request_id else ""

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _openai_error_response(400, "Invalid JSON in request body")

    if error := _validate_responses_request(body):
        return error

    stream = body.get("stream", False)
    initiator = compute_initiator(body)

    logger.info(f"{log_prefix}Responses API request - model: {body['model']}, stream: {stream}, initiator: {initiator}")

    # Map backend name to call function
    def _get_backend_caller(backend_name: str):
        if backend_name == "copilot":
            return lambda: _call_copilot_responses(body, request_id, stream)
        return lambda: _call_bedrock_for_responses(body, request, request_id, stream)

    # Capture current backend config for this request
    current_primary = _backend_state.primary
    current_fallback = _backend_state.fallback
    primary_call = _get_backend_caller(current_primary)
    request_stats.record_request(current_primary)

    _original_max_output_tokens = body.get("max_output_tokens")
    if current_primary == "copilot":
        try:
            check_context_guard_responses(body)
        except ContextWindowExceededError as e:
            request_stats.record_context_guard_rejection()
            return _openai_error_response(400, _openai_context_window_message(e))

    try:
        return await primary_call()
    except ContextWindowExceededError as e:
        _ctx_msg = _openai_context_window_message(e)
        if current_fallback:
            request_stats.record_fallback()
            logger.warning(
                f"{log_prefix}Context window exceeded on {e.backend} "
                f"({e.prompt_tokens} > {e.context_limit}), falling back to {current_fallback}"
            )
            if _original_max_output_tokens is not None:
                body["max_output_tokens"] = _original_max_output_tokens
            elif "max_output_tokens" in body:
                del body["max_output_tokens"]
            fallback_call = _get_backend_caller(current_fallback)
            try:
                return await fallback_call()
            except ContextWindowExceededError:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) also exceeded context window")
                request_stats.record_error()
                return _openai_error_response(400, _ctx_msg)
            except TransientBackendError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) failed with {fallback_err.status_code}")
                request_stats.record_error()
                return _openai_error_response(400, _ctx_msg)
            except CopilotHttpError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) HTTP error: {fallback_err}")
                request_stats.record_error()
                return _openai_error_response(400, _ctx_msg)
            except RuntimeError as fallback_err:
                logger.error(f"{log_prefix}Fallback ({current_fallback}) auth error: {fallback_err}")
                request_stats.record_error()
                return _openai_error_response(400, _ctx_msg)
        else:
            logger.error(
                f"{log_prefix}Context window exceeded on {e.backend} "
                f"({e.prompt_tokens} > {e.context_limit}), no fallback configured"
            )
            request_stats.record_error()
            return _openai_error_response(400, _ctx_msg)
    except TransientBackendError as e:
        if not current_fallback:
            logger.error(f"{log_prefix}{e.backend} transient error {e.status_code}, no fallback configured")
            request_stats.record_error()
            return _openai_error_response(e.status_code, e.message, "server_error")

        request_stats.record_fallback()
        logger.warning(
            f"{log_prefix}Primary ({current_primary}) failed with {e.status_code}, falling back to {current_fallback}"
        )

        if _original_max_output_tokens is not None:
            body["max_output_tokens"] = _original_max_output_tokens
        elif "max_output_tokens" in body:
            del body["max_output_tokens"]
        fallback_call = _get_backend_caller(current_fallback)
        try:
            return await fallback_call()
        except TransientBackendError as fallback_err:
            logger.error(f"{log_prefix}Fallback ({current_fallback}) also failed with {fallback_err.status_code}")
            request_stats.record_error()
            return _openai_error_response(fallback_err.status_code, fallback_err.message, "server_error")
        except CopilotHttpError as fallback_err:
            request_stats.record_error()
            return _openai_error_response(fallback_err.status_code, fallback_err.detail, "server_error")
        except RuntimeError as fallback_err:
            request_stats.record_error()
            return _openai_error_response(401, str(fallback_err), "authentication_error")
        except Exception as fallback_err:
            logger.error(f"{log_prefix}Fallback ({current_fallback}) unexpected error: {fallback_err}", exc_info=True)
            request_stats.record_error()
            return _openai_error_response(500, str(fallback_err), "server_error")
    except CopilotHttpError as e:
        request_stats.record_error()
        return _openai_error_response(e.status_code, e.detail, "server_error")
    except RuntimeError as e:
        logger.error(f"{log_prefix}Auth error: {e}")
        request_stats.record_error()
        return _openai_error_response(401, str(e), "authentication_error")
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error: {e}", exc_info=True)
        request_stats.record_error()
        return _openai_error_response(500, str(e), "server_error")


@app.get("/health")
async def health(check_bedrock: bool = False, check_copilot: bool = False) -> dict[str, Any]:
    """Health check endpoint. Use ?check_bedrock=true or ?check_copilot=true for deep check."""
    result: dict[str, Any] = {"status": "ok", "version": __version__, "backend": _backend_state.primary}

    if _backend_state.fallback:
        result["fallback"] = _backend_state.fallback

    if check_bedrock and _backend_state.primary == "bedrock":
        try:
            bedrock = get_bedrock_client()
            bedrock.invoke_model(
                modelId=add_region_prefix("anthropic.claude-3-haiku-20240307-v1:0"),
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ),
            )
            result["bedrock"] = "ok"
        except Exception as e:
            result["status"] = "degraded"
            result["bedrock"] = f"error: {e}"

    if check_copilot and _backend_state.primary == "copilot" and _backend_state.copilot_backend is not None:
        try:
            token = await _backend_state.copilot_backend._auth.get_token()
            result["copilot"] = "ok" if token else "no token"
        except Exception as e:
            result["status"] = "degraded"
            result["copilot"] = f"error: {e}"

    return result


@app.get("/version")
async def get_version() -> dict[str, str]:
    """Return the current version."""
    return {"version": __version__}


def _infer_owned_by(model_id: str) -> str:
    """Infer the owned_by field from a model ID prefix."""
    if model_id.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    elif model_id.startswith("gemini-"):
        return "google"
    elif model_id.startswith("grok-"):
        return "xai"
    elif model_id.startswith("claude-"):
        return "anthropic"
    return "other"


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Return available models in OpenAI-compatible format.

    Compatible with both Open WebUI (expects object/data[].object) and
    Claude Code (only reads data[].id).
    When Copilot is the backend, uses dynamically fetched models if available,
    otherwise falls back to hardcoded maps.
    """
    if _backend_state.primary == "copilot":
        dynamic_models = get_available_copilot_models()
        if dynamic_models:
            models = []
            for m in dynamic_models:
                if "id" not in m:
                    continue
                entry = {
                    "id": m["id"],
                    "object": "model",
                    "created": m.get("created_at", 1700000000),
                    "owned_by": m.get("owned_by", _infer_owned_by(m["id"])),
                }
                limits = m.get("capabilities", {}).get("limits")
                if limits:
                    entry["limits"] = limits
                models.append(entry)
        else:
            # Fallback to hardcoded maps
            all_model_ids: dict[str, str] = {}
            copilot_ids = set(COPILOT_MODEL_MAP.values())
            for model_id in COPILOT_MODEL_MAP:
                all_model_ids[model_id] = "anthropic"
            for model_id in COPILOT_OPENAI_MODEL_MAP:
                if model_id not in all_model_ids and model_id not in copilot_ids:
                    all_model_ids[model_id] = _infer_owned_by(model_id)
            models = [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": owned_by,
                }
                for model_id, owned_by in all_model_ids.items()
            ]
    else:
        models = [
            {
                "id": model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            }
            for model_id in BEDROCK_MODEL_MAP
        ]
        # When Copilot is a fallback, also include non-Claude models from Copilot
        if _backend_state.fallback == "copilot":
            bedrock_ids = set(BEDROCK_MODEL_MAP.keys())
            dynamic_models = get_available_copilot_models()
            if dynamic_models:
                for m in dynamic_models:
                    mid = m.get("id", "")
                    if mid and not is_claude_model(mid) and mid not in bedrock_ids:
                        entry = {
                            "id": mid,
                            "object": "model",
                            "created": m.get("created_at", 1700000000),
                            "owned_by": m.get("owned_by", _infer_owned_by(mid)),
                        }
                        limits = m.get("capabilities", {}).get("limits")
                        if limits:
                            entry["limits"] = limits
                        models.append(entry)
            else:
                for model_id in COPILOT_OPENAI_MODEL_MAP:
                    if not is_claude_model(model_id) and model_id not in bedrock_ids:
                        models.append(
                            {
                                "id": model_id,
                                "object": "model",
                                "created": 1700000000,
                                "owned_by": _infer_owned_by(model_id),
                            }
                        )
    return {
        "object": "list",
        "data": models,
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> dict[str, int]:
    """Count tokens using tiktoken (cl100k_base encoding).

    When using the Copilot backend, scales the token count to match the client's
    expected context window (e.g. 200k or 1M) relative to Copilot's prompt limit.
    """
    from .models import get_copilot_context_limit

    body = await request.json()

    total_tokens = 0

    # Count system prompt tokens
    if "system" in body:
        total_tokens += _count_content_tokens(body["system"])

    # Count message tokens
    for message in body.get("messages", []):
        total_tokens += _count_content_tokens(message.get("content", ""))

    # Count tool definitions
    if "tools" in body:
        total_tokens += len(tokenizer.encode(json.dumps(body["tools"])))

    # Scale token count for Copilot backend
    if _backend_state.primary == "copilot" and "model" in body:
        copilot_model, _ = get_copilot_model(body["model"])
        client_context_window = _detect_client_context_window(request, copilot_model)
        copilot_limit = get_copilot_context_limit(copilot_model)
        if copilot_limit > 0 and client_context_window > 0:
            total_tokens = int(total_tokens * client_context_window / copilot_limit)

    return {"input_tokens": total_tokens}


@app.post("/api/event_logging/batch")
async def event_logging() -> dict[str, str]:
    """Stub for telemetry - just acknowledge."""
    return {"status": "ok"}


@app.get("/api/backend")
async def get_backend() -> dict[str, str]:
    """Return current backend configuration."""
    return {"primary": _backend_state.primary, "fallback": _backend_state.fallback}


@app.post("/api/backend")
async def set_backend(request: Request) -> JSONResponse:
    """Switch backend at runtime.

    Body: {"backend": "copilot"} or {"backend": "copilot,bedrock"}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    backend_value = body.get("backend", "")
    if not backend_value:
        return JSONResponse(status_code=400, content={"error": "Missing required field: backend"})

    try:
        primary, fallback = parse_backend_string(backend_value)
        result = await _backend_state.switch(primary, fallback)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the dashboard HTML page."""
    from .dashboard import DASHBOARD_HTML

    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/status")
async def api_status(log_level: str | None = None) -> dict[str, Any]:
    """Combined status endpoint for the dashboard.

    Returns health info, service status, available models, and recent logs.
    """
    from .log_buffer import log_buffer
    from .service import get_service_status

    # Health info (reuse logic from /health)
    health: dict[str, Any] = {"status": "ok", "version": __version__, "backend": _backend_state.primary}
    if _backend_state.fallback:
        health["fallback"] = _backend_state.fallback

    # Models (reuse logic from /v1/models), deduplicated and sorted by owner then model ID
    models_response = await list_models()
    all_models = models_response.get("data", [])
    # Prefer entries with limits (richer data) when deduplicating
    all_models.sort(key=lambda m: (0 if m.get("limits") else 1, m.get("owned_by", ""), m.get("id", "")))
    seen_ids: set[str] = set()
    models_data: list[dict[str, Any]] = []
    for m in all_models:
        mid = m.get("id", "")
        if mid not in seen_ids:
            seen_ids.add(mid)
            models_data.append(m)
    models_data.sort(key=lambda m: (m.get("owned_by", ""), m.get("id", "")))

    # Service status
    service = get_service_status()

    # Recent logs
    logs = log_buffer.get_entries(limit=200, level_filter=log_level)

    # Copilot quota (if available)
    copilot: dict[str, Any] | None = None
    if _backend_state.copilot_usage_cache is not None:
        copilot = await _backend_state.copilot_usage_cache.get()

    result: dict[str, Any] = {
        "health": health,
        "service": service,
        "models": models_data,
        "logs": logs,
    }
    if copilot is not None:
        result["copilot"] = copilot
    result["stats"] = request_stats.snapshot()
    return result


@app.post("/api/logs/clear")
async def api_logs_clear() -> dict[str, str]:
    """Clear all entries from the log buffer."""
    from .log_buffer import log_buffer

    log_buffer.clear()
    return {"status": "ok"}
