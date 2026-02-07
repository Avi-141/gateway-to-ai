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
from fastapi.responses import JSONResponse, StreamingResponse

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
from .errors import CopilotHttpError, TransientBackendError
from .models import (
    BEDROCK_MODEL_MAP,
    COPILOT_MODEL_MAP,
    COPILOT_OPENAI_MODEL_MAP,
    add_region_prefix,
    get_bedrock_model,
    get_copilot_model,
    get_copilot_openai_model,
)
from .openai_translate import (
    ReverseStreamTranslator,
    anthropic_to_openai_response,
    openai_to_anthropic_request,
)

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

# Copilot backend (initialized in lifespan if needed)
_copilot_backend = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global _copilot_backend

    # Validate fallback config
    if FALLBACK_BACKEND:
        if FALLBACK_BACKEND == BACKEND_TYPE:
            raise ValueError(f"FALLBACK_BACKEND cannot be the same as BACKEND_TYPE: {BACKEND_TYPE}")
        valid = {"bedrock", "copilot"}
        if FALLBACK_BACKEND not in valid:
            raise ValueError(f"Invalid FALLBACK_BACKEND: {FALLBACK_BACKEND}, must be one of {valid}")

    # Startup
    logger.info(f"Starting claudegate v{__version__}")
    logger.info(f"Host: {os.environ.get('HOST', DEFAULT_HOST)}")
    logger.info(f"Port: {os.environ.get('PORT', DEFAULT_PORT)}")
    logger.info(f"Backend: {BACKEND_TYPE}")
    if FALLBACK_BACKEND:
        logger.info(f"Fallback: {FALLBACK_BACKEND}")
    logger.info(f"Log Level: {LOG_LEVEL}")

    # Initialize copilot if it's primary or fallback
    needs_copilot = BACKEND_TYPE == "copilot" or FALLBACK_BACKEND == "copilot"
    if needs_copilot:
        from .copilot_auth import CopilotAuth, get_github_token
        from .copilot_client import CopilotBackend

        github_token = get_github_token()
        auth = CopilotAuth(github_token)
        _copilot_backend = CopilotBackend(auth, timeout=COPILOT_TIMEOUT)
        logger.info("Copilot backend initialized")

    needs_bedrock = BACKEND_TYPE == "bedrock" or FALLBACK_BACKEND == "bedrock"
    if needs_bedrock:
        logger.info(f"AWS Region: {os.environ.get('AWS_REGION', 'us-west-2')}")
        logger.info(f"Bedrock Region Prefix: {BEDROCK_REGION_PREFIX or '(none)'}")

    yield

    # Shutdown
    if _copilot_backend is not None:
        await _copilot_backend.close()
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


def _build_bedrock_body(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Build Bedrock request body from Anthropic request."""
    bedrock_body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": body["max_tokens"],
        "messages": body["messages"],
    }

    # Check for anthropic-beta header and convert to body field
    beta_header = request.headers.get("anthropic-beta")
    if beta_header:
        bedrock_body["anthropic_beta"] = [b.strip() for b in beta_header.split(",")]

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
        "anthropic_beta",
    ]
    for field in optional_fields:
        if field in body:
            bedrock_body[field] = body[field]

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
        logger.error(f"{log_prefix}Stream error: {e}")
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
    logger.info(f"{log_prefix}Request - model: {body['model']} -> {bedrock_model}, stream: {stream}")
    logger.debug(f"{log_prefix}Request body keys: {list(body.keys())}")

    bedrock_body = _build_bedrock_body(body, request)

    if stream:
        # Two-phase: open (can raise TransientBackendError), then iterate
        response = _open_bedrock_stream(bedrock_model, bedrock_body)
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
        logger.error(f"{log_prefix}Unexpected error: {e}")
        return _error_response(500, "api_error", str(e))


async def _call_copilot(body: dict[str, Any], request_id: str, stream: bool) -> JSONResponse | StreamingResponse:
    """Execute request against Copilot backend.

    Raises TransientBackendError for fallback-eligible errors.
    Raises CopilotHttpError, RuntimeError, httpx.TimeoutException for non-fallback errors.
    """
    if _copilot_backend is None:
        raise RuntimeError("Copilot backend not initialized")
    copilot_model, anthropic_model = get_copilot_model(body["model"])
    log_prefix = f"[{request_id}] " if request_id else ""
    logger.info(f"{log_prefix}Request - model: {body['model']} -> {copilot_model} (copilot), stream: {stream}")
    return await _copilot_backend.handle_messages(body, request_id, stream, copilot_model, anthropic_model)


async def _call_copilot_openai(body: dict[str, Any], request_id: str, stream: bool) -> JSONResponse | StreamingResponse:
    """Execute OpenAI-format request directly against Copilot (no translation).

    Raises TransientBackendError for fallback-eligible errors.
    Raises CopilotHttpError, RuntimeError, httpx.TimeoutException for non-fallback errors.
    """
    if _copilot_backend is None:
        raise RuntimeError("Copilot backend not initialized")
    copilot_model = get_copilot_openai_model(body["model"])
    log_prefix = f"[{request_id}] " if request_id else ""
    logger.info(
        f"{log_prefix}OpenAI passthrough - model: {body['model']} -> {copilot_model} (copilot), stream: {stream}"
    )
    return await _copilot_backend.handle_openai_messages(body, request_id, stream, copilot_model)


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

    # Map backend name to call function
    def _get_backend_caller(backend_name: str):
        if backend_name == "copilot":
            return lambda: _call_copilot(body, request_id, stream)
        return lambda: _call_bedrock(body, request, request_id, stream)

    primary_call = _get_backend_caller(BACKEND_TYPE)

    try:
        return await primary_call()
    except TransientBackendError as e:
        if not FALLBACK_BACKEND:
            logger.error(f"{log_prefix}{e.backend} transient error {e.status_code}, no fallback configured")
            return _error_response(e.status_code, e.error_type, e.message)

        logger.warning(
            f"{log_prefix}Primary ({BACKEND_TYPE}) failed with {e.status_code}, falling back to {FALLBACK_BACKEND}"
        )

        fallback_call = _get_backend_caller(FALLBACK_BACKEND)
        try:
            return await fallback_call()
        except TransientBackendError as fallback_err:
            logger.error(f"{log_prefix}Fallback ({FALLBACK_BACKEND}) also failed with {fallback_err.status_code}")
            return _error_response(fallback_err.status_code, fallback_err.error_type, fallback_err.message)
        except CopilotHttpError as fallback_err:
            return _error_response(fallback_err.status_code, "api_error", fallback_err.detail)
        except RuntimeError as fallback_err:
            return _error_response(401, "authentication_error", str(fallback_err))
        except Exception as fallback_err:
            logger.error(f"{log_prefix}Fallback ({FALLBACK_BACKEND}) unexpected error: {fallback_err}")
            return _error_response(500, "api_error", str(fallback_err))
    except CopilotHttpError as e:
        return _error_response(e.status_code, "api_error", e.detail)
    except RuntimeError as e:
        logger.error(f"{log_prefix}Auth error: {e}")
        return _error_response(401, "authentication_error", str(e))
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error: {e}")
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

    logger.info(f"{log_prefix}OpenAI-compat request - model: {body['model']}, stream: {stream}")

    # Map backend name to call function
    # Copilot: direct passthrough (0 translations)
    # Bedrock: translated (OpenAI -> Anthropic -> Bedrock -> Anthropic -> OpenAI)
    def _get_backend_caller(backend_name: str):
        if backend_name == "copilot":
            return lambda: _call_copilot_openai(body, request_id, stream)
        return lambda: _call_bedrock_for_openai(body, request, request_id, stream)

    primary_call = _get_backend_caller(BACKEND_TYPE)

    try:
        return await primary_call()
    except TransientBackendError as e:
        if not FALLBACK_BACKEND:
            logger.error(f"{log_prefix}{e.backend} transient error {e.status_code}, no fallback configured")
            return _openai_error_response(e.status_code, e.message, "server_error")

        logger.warning(
            f"{log_prefix}Primary ({BACKEND_TYPE}) failed with {e.status_code}, falling back to {FALLBACK_BACKEND}"
        )

        fallback_call = _get_backend_caller(FALLBACK_BACKEND)
        try:
            return await fallback_call()
        except TransientBackendError as fallback_err:
            logger.error(f"{log_prefix}Fallback ({FALLBACK_BACKEND}) also failed with {fallback_err.status_code}")
            return _openai_error_response(fallback_err.status_code, fallback_err.message, "server_error")
        except CopilotHttpError as fallback_err:
            return _openai_error_response(fallback_err.status_code, fallback_err.detail, "server_error")
        except RuntimeError as fallback_err:
            return _openai_error_response(401, str(fallback_err), "authentication_error")
        except Exception as fallback_err:
            logger.error(f"{log_prefix}Fallback ({FALLBACK_BACKEND}) unexpected error: {fallback_err}")
            return _openai_error_response(500, str(fallback_err), "server_error")
    except CopilotHttpError as e:
        return _openai_error_response(e.status_code, e.detail, "server_error")
    except RuntimeError as e:
        logger.error(f"{log_prefix}Auth error: {e}")
        return _openai_error_response(401, str(e), "authentication_error")
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error: {e}")
        return _openai_error_response(500, str(e), "server_error")


@app.get("/health")
async def health(check_bedrock: bool = False, check_copilot: bool = False) -> dict[str, Any]:
    """Health check endpoint. Use ?check_bedrock=true or ?check_copilot=true for deep check."""
    result: dict[str, Any] = {"status": "ok", "version": __version__, "backend": BACKEND_TYPE}

    if FALLBACK_BACKEND:
        result["fallback"] = FALLBACK_BACKEND

    if check_bedrock and BACKEND_TYPE == "bedrock":
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

    if check_copilot and BACKEND_TYPE == "copilot" and _copilot_backend is not None:
        try:
            token = await _copilot_backend._auth.get_token()
            result["copilot"] = "ok" if token else "no token"
        except Exception as e:
            result["status"] = "degraded"
            result["copilot"] = f"error: {e}"

    return result


@app.get("/version")
async def get_version() -> dict[str, str]:
    """Return the current version."""
    return {"version": __version__}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Return available models in OpenAI-compatible format.

    Compatible with both Open WebUI (expects object/data[].object) and
    Claude Code (only reads data[].id).
    When Copilot is the backend, includes both Claude and non-Claude models.
    """
    if BACKEND_TYPE == "copilot":
        # Combine Anthropic-name models and OpenAI-native models
        all_model_ids: dict[str, str] = {}
        copilot_ids = set(COPILOT_MODEL_MAP.values())
        for model_id in COPILOT_MODEL_MAP:
            all_model_ids[model_id] = "anthropic"
        for model_id in COPILOT_OPENAI_MODEL_MAP:
            if model_id not in all_model_ids and model_id not in copilot_ids:
                if model_id.startswith(("gpt-",)):
                    owned_by = "openai"
                elif model_id.startswith("gemini-"):
                    owned_by = "google"
                elif model_id.startswith("grok-"):
                    owned_by = "xai"
                elif model_id.startswith("claude-"):
                    owned_by = "anthropic"
                else:
                    owned_by = "other"
                all_model_ids[model_id] = owned_by
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
    return {
        "object": "list",
        "data": models,
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> dict[str, int]:
    """Count tokens using tiktoken (cl100k_base encoding)."""
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

    return {"input_tokens": total_tokens}


@app.post("/api/event_logging/batch")
async def event_logging() -> dict[str, str]:
    """Stub for telemetry - just acknowledge."""
    return {"status": "ok"}
