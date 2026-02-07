"""FastAPI application and route handlers."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

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
    LOG_LEVEL,
    logger,
)
from .models import COPILOT_MODEL_MAP, BEDROCK_MODEL_MAP, add_region_prefix, get_bedrock_model, get_copilot_model

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
    "AWS credentials have expired. Please re-authenticate in your terminal "
    "to refresh your credentials, then retry."
)

# Copilot backend (initialized in lifespan if BACKEND=copilot)
_copilot_backend = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global _copilot_backend

    # Startup
    logger.info(f"Starting claudegate v{__version__}")
    logger.info(f"Host: {os.environ.get('HOST', DEFAULT_HOST)}")
    logger.info(f"Port: {os.environ.get('PORT', DEFAULT_PORT)}")
    logger.info(f"Backend: {BACKEND_TYPE}")
    logger.info(f"Log Level: {LOG_LEVEL}")

    if BACKEND_TYPE == "copilot":
        from .copilot_auth import CopilotAuth, get_github_token
        from .copilot_client import CopilotBackend

        github_token = get_github_token()
        auth = CopilotAuth(github_token)
        _copilot_backend = CopilotBackend(auth, timeout=COPILOT_TIMEOUT)
        logger.info("Copilot backend initialized")
    else:
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


# --- Streaming ---


async def stream_response(
    model: str, body: dict[str, Any], request_id: str = ""
) -> AsyncGenerator[str, None]:
    """Handle streaming responses."""
    log_prefix = f"[{request_id}] " if request_id else ""
    try:
        logger.info(f"{log_prefix}Starting stream for model: {model}")
        bedrock = get_bedrock_client()
        response = bedrock.invoke_model_with_response_stream(
            modelId=model, body=json.dumps(body)
        )

        chunk_count = 0
        for event in response["body"]:
            chunk = event.get("chunk")
            if chunk:
                data = json.loads(chunk["bytes"].decode())
                chunk_count += 1
                if chunk_count <= 3:
                    logger.debug(f"{log_prefix}Chunk {chunk_count}: {json.dumps(data)[:200]}")
                # Include event type for proper SSE format
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
            # Reset client so next request after re-auth works
            reset_bedrock_client()
            # Inject a visible message into the stream so user sees it
            error_text = f"\n\n⚠️ **Authentication Error**: {CREDENTIALS_EXPIRED_MSG}\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': error_text}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        else:
            logger.error(f"{log_prefix}Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
    except Exception as e:
        logger.error(f"{log_prefix}Stream error: {e}")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"


# --- Route Handlers ---


@app.post("/v1/messages", response_model=None)
async def messages(request: Request) -> JSONResponse | StreamingResponse:
    """Handle chat completion requests."""
    # Get request ID for tracing
    request_id = request.headers.get("x-request-id", "")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _error_response(400, "invalid_request_error", "Invalid JSON in request body")

    # Validate required fields
    if error := _validate_request(body):
        return error

    model = body["model"]
    stream = body.get("stream", False)

    log_prefix = f"[{request_id}] " if request_id else ""

    # Dispatch to Copilot backend
    if BACKEND_TYPE == "copilot" and _copilot_backend is not None:
        copilot_model, anthropic_model = get_copilot_model(model)
        logger.info(f"{log_prefix}Request - model: {model} -> {copilot_model} (copilot), stream: {stream}")
        return await _copilot_backend.handle_messages(body, request_id, stream, copilot_model, anthropic_model)

    bedrock_model = get_bedrock_model(model)
    logger.info(f"{log_prefix}Request - model: {model} -> {bedrock_model}, stream: {stream}")
    logger.debug(f"{log_prefix}Request body keys: {list(body.keys())}")

    # Build Bedrock request body
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
        "system",  # System prompt
        "temperature",  # Sampling temperature (0.0-1.0)
        "top_p",  # Nucleus sampling
        "top_k",  # Top-k sampling
        "tools",  # Tool definitions
        "tool_choice",  # Tool selection preference
        "thinking",  # Extended thinking config
        "stop_sequences",  # Custom stop sequences
        "metadata",  # Request metadata (user_id)
        "anthropic_beta",  # Beta features (if passed in body)
    ]
    for field in optional_fields:
        if field in body:
            bedrock_body[field] = body[field]

    try:
        bedrock = get_bedrock_client()
        if stream:
            return StreamingResponse(
                stream_response(bedrock_model, bedrock_body, request_id),
                media_type="text/event-stream",
            )
        else:
            response = bedrock.invoke_model(modelId=bedrock_model, body=json.dumps(bedrock_body))
            result = json.loads(response["body"].read())
            logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
            return JSONResponse(content=result)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        if error_code in ("ExpiredTokenException", "ExpiredToken"):
            logger.error(f"{log_prefix}Credentials expired")
            # Reset client so next request after re-auth works
            reset_bedrock_client()
            return _error_response(401, "authentication_error", CREDENTIALS_EXPIRED_MSG)
        elif error_code == "ValidationException":
            logger.error(f"{log_prefix}Validation error: {error_message}")
            return _error_response(400, "invalid_request_error", error_message)
        elif error_code == "AccessDeniedException":
            logger.error(f"{log_prefix}Access denied: {error_message}")
            return _error_response(403, "permission_error", error_message)
        elif error_code == "ThrottlingException":
            logger.error(f"{log_prefix}Rate limited: {error_message}")
            return _error_response(429, "rate_limit_error", error_message)
        elif error_code == "ModelTimeoutException":
            logger.error(f"{log_prefix}Model timeout: {error_message}")
            return _error_response(504, "timeout_error", "Model took too long to respond")
        else:
            logger.error(f"{log_prefix}Bedrock error ({error_code}): {error_message}")
            return _error_response(500, "api_error", error_message)
    except ReadTimeoutError as e:
        logger.error(f"{log_prefix}Read timeout: {e}")
        return _error_response(
            504, "timeout_error", "Request timed out. Try a smaller request or use streaming."
        )
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error: {e}")
        return _error_response(500, "api_error", str(e))


@app.get("/health")
async def health(check_bedrock: bool = False, check_copilot: bool = False) -> dict[str, Any]:
    """Health check endpoint. Use ?check_bedrock=true or ?check_copilot=true for deep check."""
    result: dict[str, Any] = {"status": "ok", "version": __version__, "backend": BACKEND_TYPE}

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
    """Return available models in Anthropic API format."""
    model_map = COPILOT_MODEL_MAP if BACKEND_TYPE == "copilot" else BEDROCK_MODEL_MAP
    models = [
        {
            "id": model_id,
            "type": "model",
            "display_name": model_id.replace("-", " ").title(),
            "created_at": "2024-01-01T00:00:00Z",
        }
        for model_id in model_map.keys()
    ]
    return {
        "data": models,
        "has_more": False,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
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
