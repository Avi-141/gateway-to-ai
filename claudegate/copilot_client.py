"""GitHub Copilot HTTP client and backend."""

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import (
    COPILOT_MAX_RATE,
    COPILOT_RETRY_BASE_DELAY,
    COPILOT_RETRY_MAX,
    FALLBACK_ON_ERRORS,
    SSL_CONTEXT,
    logger,
)
from .copilot_auth import COPILOT_HEADERS, CopilotAuth
from .copilot_translate import (
    StreamTranslator,
    anthropic_to_openai_request,
    estimate_input_tokens,
    openai_to_anthropic_response,
)
from .errors import ContextWindowExceededError, CopilotHttpError, TransientBackendError
from .models import get_copilot_context_limit
from .responses_translate import (
    OpenAIChatToResponsesStreamTranslator,
    ResponsesStreamTranslator,
    ResponsesToOpenAIStreamTranslator,
    anthropic_to_responses_request,
    openai_chat_to_responses_request,
    responses_to_anthropic_response,
    responses_to_openai_chat_request,
    responses_to_openai_chat_response,
)

COPILOT_CHAT_URL = "https://api.githubcopilot.com/chat/completions"
COPILOT_RESPONSES_URL = "https://api.githubcopilot.com/responses"
COPILOT_MODELS_URL = "https://api.githubcopilot.com/models"

# Map HTTP status codes to Anthropic error types
_ERROR_TYPE_MAP: dict[int, str] = {
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
    504: "timeout_error",
}


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    """Parse retry-after header value as seconds, or None if absent/invalid."""
    val = headers.get("retry-after")
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


class TokenBucket:
    """Async token bucket rate limiter.

    Allows ``rate`` requests per 60 seconds.  Requests that exceed the rate
    wait until a token is available (queue-and-wait).
    """

    def __init__(self, rate: int):
        self._rate = rate
        self._tokens = float(rate)
        self._max_tokens = float(rate)
        self._interval = 60.0 / rate  # seconds per token
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Acquire a token.  Returns the wait time in seconds (0.0 if immediate)."""
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            # Calculate wait time until next token
            wait = self._interval * (1.0 - self._tokens)
            self._tokens = 0.0
        await asyncio.sleep(wait)
        # After waiting, consume the newly refilled token
        async with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)
        return wait

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed / self._interval)
        self._last_refill = now


# Pattern to extract token counts from Copilot's token limit error
# e.g. "prompt token count of 145794 exceeds the limit of 128000"
_TOKEN_LIMIT_PATTERN = re.compile(r"prompt token count of (\d+) exceeds the limit of (\d+)")


def _parse_token_limit_error(status_code: int, detail: str) -> ContextWindowExceededError | None:
    """Check if a Copilot error is a token limit exceeded error.

    Returns ContextWindowExceededError if matched, None otherwise.
    When the regex matches, exact token counts are extracted.
    When only the error code matches, the original detail is preserved
    so callers can pass it through rather than fabricating numbers.
    """
    if status_code != 400:
        return None
    match = _TOKEN_LIMIT_PATTERN.search(detail)
    if match:
        return ContextWindowExceededError(int(match.group(1)), int(match.group(2)), "copilot")
    # Fallback: check for the error code without parseable numbers
    if "model_max_prompt_tokens_exceeded" in detail:
        return ContextWindowExceededError(0, 0, "copilot", detail)
    return None


def _normalize_openai_response(resp: dict[str, Any], streaming: bool = False) -> dict[str, Any]:
    """Ensure a Copilot chat completions response conforms to the OpenAI spec.

    Copilot responses may be missing fields that the OpenAI spec requires,
    such as 'object', 'created', and 'index' in choices. This ensures
    downstream clients (e.g. BAML) can parse the response without errors.
    """
    if "object" not in resp:
        resp["object"] = "chat.completion.chunk" if streaming else "chat.completion"
    if "created" not in resp:
        resp["created"] = int(time.time())
    if "id" not in resp:
        resp["id"] = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    for choice in resp.get("choices", []):
        if "index" not in choice:
            choice["index"] = 0
    return resp


def _is_suggestion_mode(content: Any) -> bool:
    """Check if message content is a Claude Code suggestion mode prompt."""
    marker = "[SUGGESTION MODE:"
    if isinstance(content, str):
        return content.lstrip().startswith(marker)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text.lstrip().startswith(marker):
                    return True
    return False


def compute_initiator(body: dict[str, Any]) -> str:
    """Determine X-Initiator header value from request body."""
    # Chat Completions / Anthropic Messages format
    messages = body.get("messages")
    if messages and isinstance(messages, list):
        last = messages[-1]
        role = last.get("role", "")
        if role in ("assistant", "tool"):
            return "agent"
        # Anthropic format: tool results are sent as role "user" with
        # content blocks containing type "tool_result"
        if role == "user":
            content = last.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in content
            ):
                return "agent"
            if _is_suggestion_mode(content):
                return "agent"
        return "user"

    # Responses API format
    input_items = body.get("input")
    if isinstance(input_items, str):
        return "user"
    if input_items and isinstance(input_items, list):
        last = input_items[-1]
        if isinstance(last, dict):
            item_type = last.get("type", "")
            if item_type in ("function_call_output", "function_call"):
                return "agent"
            if last.get("role") == "assistant":
                return "agent"
        return "user"

    return "user"


class CopilotBackend:
    """Handles routing requests through GitHub Copilot API."""

    def __init__(
        self,
        auth: CopilotAuth,
        timeout: int = 300,
        retry_max: int = COPILOT_RETRY_MAX,
        retry_base_delay: float = COPILOT_RETRY_BASE_DELAY,
        max_rate: int = COPILOT_MAX_RATE,
    ):
        self._auth = auth
        self._client = httpx.AsyncClient(verify=SSL_CONTEXT, timeout=httpx.Timeout(timeout, connect=30.0))
        self._retry_max = retry_max
        self._retry_base_delay = retry_base_delay
        self._rate_limiter = TokenBucket(max_rate) if max_rate > 0 else None

    async def _get_headers(self, body: dict[str, Any] | None = None, initiator: str | None = None) -> dict[str, str]:
        """Build request headers with fresh Copilot token.

        If *initiator* is provided it is used directly as the X-Initiator
        header value; otherwise it is computed from *body*.
        """
        token = await self._auth.get_token()
        headers = {
            **COPILOT_HEADERS,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
        }
        if initiator is not None:
            headers["X-Initiator"] = initiator
        elif body is not None:
            headers["X-Initiator"] = compute_initiator(body)
        return headers

    async def _post_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        log_prefix: str,
    ) -> httpx.Response:
        """POST with rate limiting and retry-on-429.

        Only 429 is retried; other errors are returned immediately so the
        caller can raise TransientBackendError (for fallback) or CopilotHttpError.
        Timeouts are converted to TransientBackendError for fallback eligibility.
        """
        resp: httpx.Response | None = None
        for attempt in range(1 + self._retry_max):
            if self._rate_limiter:
                waited = await self._rate_limiter.acquire()
                if waited > 0:
                    logger.info(f"{log_prefix}Rate limited, waited {waited:.1f}s")

            try:
                resp = await self._client.post(url, headers=headers, json=json)
            except httpx.TimeoutException as exc:
                logger.error(f"{log_prefix}Copilot request timed out: {exc}")
                raise TransientBackendError(
                    504, "timeout_error", f"Copilot request timed out: {exc}", "copilot"
                ) from exc
            except httpx.ConnectError as exc:
                logger.error(f"{log_prefix}Copilot connection failed: {exc}")
                raise TransientBackendError(
                    502, "connection_error", f"Copilot connection failed: {exc}", "copilot"
                ) from exc

            if resp.status_code == 429 and attempt < self._retry_max:
                retry_after = _parse_retry_after(resp.headers)
                delay = retry_after or (self._retry_base_delay * (2**attempt))
                delay = min(delay, 30.0)
                logger.warning(f"{log_prefix}Copilot 429, retry {attempt + 1}/{self._retry_max} after {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            return resp
        return resp  # type: ignore[return-value]  # unreachable, satisfies type checker

    async def _open_stream_with_retry(
        self,
        url: str,
        body: dict[str, Any],
        log_prefix: str,
        initiator: str | None = None,
    ) -> tuple[httpx.Response, Any]:
        """Open a streaming POST with rate limiting and retry-on-429.

        Only the stream-open phase is retried (429 occurs before streaming starts).
        Returns (response, context_manager) on success.
        Timeouts are converted to TransientBackendError for fallback eligibility.
        """
        for attempt in range(1 + self._retry_max):
            if self._rate_limiter:
                waited = await self._rate_limiter.acquire()
                if waited > 0:
                    logger.info(f"{log_prefix}Rate limited, waited {waited:.1f}s")

            headers = await self._get_headers(body, initiator=initiator)
            stream_cm = self._client.stream("POST", url, headers=headers, json=body)
            try:
                resp = await stream_cm.__aenter__()
            except httpx.TimeoutException as exc:
                logger.error(f"{log_prefix}Copilot stream open timed out: {exc}")
                raise TransientBackendError(
                    504, "timeout_error", f"Copilot stream open timed out: {exc}", "copilot"
                ) from exc
            except httpx.ConnectError as exc:
                logger.error(f"{log_prefix}Copilot stream connection failed: {exc}")
                raise TransientBackendError(
                    502, "connection_error", f"Copilot stream connection failed: {exc}", "copilot"
                ) from exc

            if resp.status_code == 429 and attempt < self._retry_max:
                await resp.aread()
                await stream_cm.__aexit__(None, None, None)
                retry_after = _parse_retry_after(resp.headers)
                delay = retry_after or (self._retry_base_delay * (2**attempt))
                delay = min(delay, 30.0)
                logger.warning(
                    f"{log_prefix}Copilot stream 429, retry {attempt + 1}/{self._retry_max} after {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            return resp, stream_cm
        return resp, stream_cm  # type: ignore[possibly-undefined]  # unreachable

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models from the Copilot API."""
        headers = await self._get_headers()
        try:
            resp = await self._client.get(COPILOT_MODELS_URL, headers=headers)
        except Exception as e:
            logger.warning(f"Failed to fetch Copilot models: {e}")
            return []
        if resp.status_code != 200:
            logger.warning(f"Failed to fetch Copilot models: HTTP {resp.status_code}")
            return []
        data = resp.json()
        return data.get("data", [])

    def _error_response(self, status_code: int, error_type: str, message: str) -> JSONResponse:
        """Return Anthropic-style error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "type": "error",
                "error": {"type": error_type, "message": message},
            },
        )

    def _map_http_error(self, status_code: int, detail: str) -> JSONResponse:
        """Map Copilot HTTP errors to Anthropic error format."""
        if status_code == 401:
            return self._error_response(401, "authentication_error", f"Copilot authentication failed: {detail}")
        elif status_code == 403:
            return self._error_response(403, "permission_error", f"Copilot access denied: {detail}")
        elif status_code == 429:
            return self._error_response(429, "rate_limit_error", f"Copilot rate limit exceeded: {detail}")
        elif status_code == 404:
            return self._error_response(404, "not_found_error", f"Copilot model not found: {detail}")
        elif status_code >= 500:
            return self._error_response(502, "api_error", f"Copilot server error: {detail}")
        else:
            return self._error_response(status_code, "api_error", f"Copilot error: {detail}")

    async def handle_messages(
        self,
        body: dict[str, Any],
        request_id: str,
        stream: bool,
        openai_model: str,
        anthropic_model: str,
        client_context_window: int = 0,
        initiator: str | None = None,
    ) -> JSONResponse | StreamingResponse:
        """Handle a messages request by proxying through Copilot.

        Raises TransientBackendError for fallback-eligible errors (429, 5xx).
        Raises CopilotHttpError for non-transient HTTP errors.
        Auth errors (RuntimeError) and unexpected errors are still raised.
        """
        log_prefix = f"[{request_id}] " if request_id else ""

        openai_body = anthropic_to_openai_request(body, openai_model)

        if stream:
            openai_body["stream"] = True
            openai_body["stream_options"] = {"include_usage": True}
            estimated_tokens = estimate_input_tokens(body)
            copilot_context_limit = get_copilot_context_limit(openai_model)
            resp, stream_cm = await self._open_stream(openai_body, log_prefix, initiator=initiator)
            return StreamingResponse(
                self._stream_response(
                    resp,
                    stream_cm,
                    anthropic_model,
                    log_prefix,
                    estimated_tokens,
                    copilot_context_limit,
                    client_context_window,
                ),
                media_type="text/event-stream",
            )
        else:
            headers = await self._get_headers(openai_body, initiator=initiator)
            logger.info(f"{log_prefix}Copilot request to {openai_model}")
            logger.debug(f"{log_prefix}OpenAI body keys: {list(openai_body.keys())}")

            resp = await self._post_with_retry(COPILOT_CHAT_URL, headers, openai_body, log_prefix)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}Copilot error {resp.status_code}: {detail}")
                logger.error(
                    f"{log_prefix}Copilot error request payload: "
                    f"model={openai_body.get('model')}, "
                    f"messages={len(openai_body.get('messages', []))}, "
                    f"tools={len(openai_body.get('tools', []))}"
                )
                logger.debug(f"{log_prefix}Copilot error full request body: {json.dumps(openai_body)[:5000]}")
                token_err = _parse_token_limit_error(resp.status_code, detail)
                if token_err:
                    raise token_err
                if resp.status_code in FALLBACK_ON_ERRORS:
                    error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                    raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
                raise CopilotHttpError(resp.status_code, detail)

            openai_resp = resp.json()
            if not openai_resp.get("choices"):
                logger.warning(f"{log_prefix}Copilot returned empty choices: {json.dumps(openai_resp)[:500]}")
            result = openai_to_anthropic_response(openai_resp, anthropic_model)
            logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
            return JSONResponse(content=result)

    async def _open_stream(
        self, openai_body: dict[str, Any], log_prefix: str, initiator: str | None = None
    ) -> tuple[httpx.Response, Any]:
        """Open streaming connection and validate HTTP status.

        Returns (response, context_manager) on success.
        Raises TransientBackendError for fallback-eligible status codes.
        Raises CopilotHttpError for non-transient HTTP errors.
        """
        logger.info(f"{log_prefix}Starting Copilot stream for {openai_body.get('model')}")

        resp, stream_cm = await self._open_stream_with_retry(
            COPILOT_CHAT_URL, openai_body, log_prefix, initiator=initiator
        )

        if resp.status_code != 200:
            body = await resp.aread()
            detail = body.decode()[:500]
            await stream_cm.__aexit__(None, None, None)
            resp_headers = dict(resp.headers) if isinstance(resp.headers, httpx.Headers) else {}
            logger.error(
                f"{log_prefix}Copilot stream error {resp.status_code}: {detail} | response_headers={resp_headers}"
            )
            logger.error(
                f"{log_prefix}Copilot stream error request payload: "
                f"model={openai_body.get('model')}, "
                f"messages={len(openai_body.get('messages', []))}, "
                f"tools={len(openai_body.get('tools', []))}, "
                f"stream={openai_body.get('stream')}"
            )
            logger.debug(f"{log_prefix}Copilot stream error full request body: {json.dumps(openai_body)[:5000]}")
            token_err = _parse_token_limit_error(resp.status_code, detail)
            if token_err:
                raise token_err
            if resp.status_code in FALLBACK_ON_ERRORS:
                error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
            raise CopilotHttpError(resp.status_code, detail)

        return resp, stream_cm

    async def _stream_response(
        self,
        resp: httpx.Response,
        stream_cm: Any,
        anthropic_model: str,
        log_prefix: str,
        estimated_input_tokens: int = 0,
        copilot_context_limit: int = 0,
        client_context_window: int = 0,
    ) -> AsyncGenerator[str, None]:
        """Stream response from already-opened Copilot connection, translating to Anthropic SSE format."""
        translator = StreamTranslator(
            anthropic_model,
            estimated_input_tokens,
            copilot_context_limit,
            client_context_window,
        )
        chunk_count = 0

        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning(f"{log_prefix}Skipping malformed chunk: {data_str[:100]}")
                    continue

                chunk_count += 1
                if chunk_count <= 3:
                    logger.debug(f"{log_prefix}Chunk {chunk_count}: {data_str[:200]}")

                events = translator.translate_chunk(chunk)
                if events:
                    yield events
                    await asyncio.sleep(0)

            # Flush any pending state (deferred message_delta/stop)
            flush_events = translator.flush()
            if flush_events:
                yield flush_events

            logger.info(f"{log_prefix}Copilot stream complete, {chunk_count} chunks")
            yield "event: done\ndata: [DONE]\n\n"

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}Copilot stream timed out")
            error_data = json.dumps({"type": "error", "error": {"message": "Copilot stream timed out"}})
            yield f"event: error\ndata: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}Copilot stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    async def handle_openai_messages(
        self, openai_body: dict[str, Any], request_id: str, stream: bool, copilot_model: str
    ) -> JSONResponse | StreamingResponse:
        """Handle an OpenAI-format request by passing directly to Copilot (no translation).

        Raises TransientBackendError for fallback-eligible errors (429, 5xx).
        Raises CopilotHttpError for non-transient HTTP errors.
        """
        log_prefix = f"[{request_id}] " if request_id else ""

        # Override model to Copilot-compatible name
        openai_body = {**openai_body, "model": copilot_model}

        if stream:
            openai_body["stream"] = True
            openai_body["stream_options"] = {"include_usage": True}
            resp, stream_cm = await self._open_stream(openai_body, log_prefix)
            return StreamingResponse(
                self._stream_openai_response(resp, stream_cm, log_prefix),
                media_type="text/event-stream",
            )
        else:
            headers = await self._get_headers(openai_body)
            logger.info(f"{log_prefix}Copilot OpenAI passthrough to {copilot_model}")
            logger.debug(f"{log_prefix}OpenAI body keys: {list(openai_body.keys())}")

            resp = await self._post_with_retry(COPILOT_CHAT_URL, headers, openai_body, log_prefix)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}Copilot error {resp.status_code}: {detail}")
                logger.error(
                    f"{log_prefix}Copilot error request payload: "
                    f"model={openai_body.get('model')}, "
                    f"messages={len(openai_body.get('messages', []))}, "
                    f"tools={len(openai_body.get('tools', []))}"
                )
                logger.debug(f"{log_prefix}Copilot error full request body: {json.dumps(openai_body)[:5000]}")
                token_err = _parse_token_limit_error(resp.status_code, detail)
                if token_err:
                    raise token_err
                if resp.status_code in FALLBACK_ON_ERRORS:
                    error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                    raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
                raise CopilotHttpError(resp.status_code, detail)

            openai_resp = resp.json()
            _normalize_openai_response(openai_resp)
            logger.debug(f"{log_prefix}Response: {json.dumps(openai_resp)[:500]}")
            return JSONResponse(content=openai_resp)

    async def _stream_openai_response(
        self, resp: httpx.Response, stream_cm: Any, log_prefix: str
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI response as-is from Copilot (no Anthropic translation)."""
        chunk_count = 0

        try:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                # Normalize streaming chunks to ensure OpenAI spec compliance
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    try:
                        chunk = json.loads(line[6:])
                        _normalize_openai_response(chunk, streaming=True)
                        line = f"data: {json.dumps(chunk)}"
                    except json.JSONDecodeError:
                        pass
                chunk_count += 1
                if chunk_count <= 3:
                    logger.debug(f"{log_prefix}OpenAI chunk {chunk_count}: {line[:200]}")
                yield f"{line}\n\n"
                await asyncio.sleep(0)

            logger.info(f"{log_prefix}Copilot OpenAI stream complete, {chunk_count} lines")

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}Copilot stream timed out")
            error_data = json.dumps(
                {"error": {"message": "Copilot stream timed out", "type": "server_error", "param": None, "code": None}}
            )
            yield f"data: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}Copilot stream error: {e}")
            error_data = json.dumps({"error": {"message": str(e), "type": "server_error", "param": None, "code": None}})
            yield f"data: {error_data}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    # --- Responses API methods (for codex models) ---

    async def handle_responses_messages(
        self,
        body: dict[str, Any],
        request_id: str,
        stream: bool,
        responses_model: str,
        anthropic_model: str,
        client_context_window: int = 0,
        initiator: str | None = None,
    ) -> JSONResponse | StreamingResponse:
        """Handle a messages request by proxying through Copilot Responses API.

        Used for models that only support /responses (not /chat/completions).
        Translates Anthropic Messages -> Responses API -> Anthropic Messages.
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        responses_body = anthropic_to_responses_request(body, responses_model)

        if stream:
            responses_body["stream"] = True
            estimated_tokens = estimate_input_tokens(body)
            copilot_context_limit = get_copilot_context_limit(responses_model)
            resp, stream_cm = await self._open_responses_stream(responses_body, log_prefix, initiator=initiator)
            return StreamingResponse(
                self._stream_responses_response(
                    resp,
                    stream_cm,
                    anthropic_model,
                    log_prefix,
                    estimated_tokens,
                    copilot_context_limit,
                    client_context_window,
                ),
                media_type="text/event-stream",
            )
        else:
            headers = await self._get_headers(responses_body, initiator=initiator)
            logger.info(f"{log_prefix}Copilot Responses request to {responses_model}")
            logger.debug(f"{log_prefix}Responses body keys: {list(responses_body.keys())}")

            resp = await self._post_with_retry(COPILOT_RESPONSES_URL, headers, responses_body, log_prefix)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}Copilot Responses error {resp.status_code}: {detail}")
                logger.error(
                    f"{log_prefix}Copilot Responses error request payload: "
                    f"model={responses_body.get('model')}, "
                    f"input_items={len(responses_body.get('input', []))}, "
                    f"tools={len(responses_body.get('tools', []))}"
                )
                logger.debug(
                    f"{log_prefix}Copilot Responses error full request body: {json.dumps(responses_body)[:5000]}"
                )
                token_err = _parse_token_limit_error(resp.status_code, detail)
                if token_err:
                    raise token_err
                if resp.status_code in FALLBACK_ON_ERRORS:
                    error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                    raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
                raise CopilotHttpError(resp.status_code, detail)

            responses_resp = resp.json()
            result = responses_to_anthropic_response(responses_resp, anthropic_model)
            logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
            return JSONResponse(content=result)

    async def handle_openai_responses_messages(
        self,
        body: dict[str, Any],
        request_id: str,
        stream: bool,
        responses_model: str,
    ) -> JSONResponse | StreamingResponse:
        """Handle an OpenAI-format request by proxying through Copilot Responses API.

        Translates Chat Completions -> Responses API -> Chat Completions.
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        responses_body = openai_chat_to_responses_request(body, responses_model)

        if stream:
            responses_body["stream"] = True
            resp, stream_cm = await self._open_responses_stream(responses_body, log_prefix)
            return StreamingResponse(
                self._stream_responses_openai_response(resp, stream_cm, responses_model, log_prefix),
                media_type="text/event-stream",
            )
        else:
            headers = await self._get_headers(responses_body)
            logger.info(f"{log_prefix}Copilot Responses OpenAI passthrough to {responses_model}")
            logger.debug(f"{log_prefix}Responses body keys: {list(responses_body.keys())}")

            resp = await self._post_with_retry(COPILOT_RESPONSES_URL, headers, responses_body, log_prefix)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}Copilot Responses error {resp.status_code}: {detail}")
                logger.error(
                    f"{log_prefix}Copilot Responses error request payload: "
                    f"model={responses_body.get('model')}, "
                    f"input_items={len(responses_body.get('input', []))}, "
                    f"tools={len(responses_body.get('tools', []))}"
                )
                logger.debug(
                    f"{log_prefix}Copilot Responses error full request body: {json.dumps(responses_body)[:5000]}"
                )
                token_err = _parse_token_limit_error(resp.status_code, detail)
                if token_err:
                    raise token_err
                if resp.status_code in FALLBACK_ON_ERRORS:
                    error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                    raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
                raise CopilotHttpError(resp.status_code, detail)

            responses_resp = resp.json()
            result = responses_to_openai_chat_response(responses_resp, responses_model)
            logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
            return JSONResponse(content=result)

    async def _open_responses_stream(
        self, responses_body: dict[str, Any], log_prefix: str, initiator: str | None = None
    ) -> tuple[httpx.Response, Any]:
        """Open streaming connection to Responses API and validate HTTP status."""
        logger.info(f"{log_prefix}Starting Copilot Responses stream for {responses_body.get('model')}")

        resp, stream_cm = await self._open_stream_with_retry(
            COPILOT_RESPONSES_URL, responses_body, log_prefix, initiator=initiator
        )

        if resp.status_code != 200:
            body = await resp.aread()
            detail = body.decode()[:500]
            await stream_cm.__aexit__(None, None, None)
            resp_headers = dict(resp.headers) if isinstance(resp.headers, httpx.Headers) else {}
            logger.error(
                f"{log_prefix}Copilot Responses stream error {resp.status_code}: {detail}"
                f" | response_headers={resp_headers}"
            )
            logger.error(
                f"{log_prefix}Copilot Responses stream error request payload: "
                f"model={responses_body.get('model')}, "
                f"input_items={len(responses_body.get('input', []))}, "
                f"tools={len(responses_body.get('tools', []))}"
            )
            logger.debug(
                f"{log_prefix}Copilot Responses stream error full request body: {json.dumps(responses_body)[:5000]}"
            )
            token_err = _parse_token_limit_error(resp.status_code, detail)
            if token_err:
                raise token_err
            if resp.status_code in FALLBACK_ON_ERRORS:
                error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
            raise CopilotHttpError(resp.status_code, detail)

        return resp, stream_cm

    async def _stream_responses_response(
        self,
        resp: httpx.Response,
        stream_cm: Any,
        anthropic_model: str,
        log_prefix: str,
        estimated_input_tokens: int = 0,
        copilot_context_limit: int = 0,
        client_context_window: int = 0,
    ) -> AsyncGenerator[str, None]:
        """Stream Responses API events, translating to Anthropic SSE format."""
        translator = ResponsesStreamTranslator(
            anthropic_model,
            estimated_input_tokens,
            copilot_context_limit,
            client_context_window,
        )
        chunk_count = 0
        current_event_type = ""

        try:
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    current_event_type = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning(f"{log_prefix}Skipping malformed Responses chunk: {data_str[:100]}")
                    continue

                chunk_count += 1
                if chunk_count <= 3:
                    logger.debug(f"{log_prefix}Responses chunk {chunk_count}: {current_event_type} {data_str[:200]}")

                events = translator.translate_event(current_event_type, data)
                if events:
                    yield events
                    await asyncio.sleep(0)

            # Cleanup
            flush_events = translator.flush()
            if flush_events:
                yield flush_events

            logger.info(f"{log_prefix}Copilot Responses stream complete, {chunk_count} chunks")
            yield "event: done\ndata: [DONE]\n\n"

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}Copilot Responses stream timed out")
            error_data = json.dumps({"type": "error", "error": {"message": "Copilot Responses stream timed out"}})
            yield f"event: error\ndata: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}Copilot Responses stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    async def _stream_responses_openai_response(
        self, resp: httpx.Response, stream_cm: Any, model: str, log_prefix: str
    ) -> AsyncGenerator[str, None]:
        """Stream Responses API events, translating to OpenAI Chat Completions SSE format."""
        translator = ResponsesToOpenAIStreamTranslator(model)
        chunk_count = 0
        current_event_type = ""

        try:
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    current_event_type = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning(f"{log_prefix}Skipping malformed Responses chunk: {data_str[:100]}")
                    continue

                chunk_count += 1
                if chunk_count <= 3:
                    logger.debug(f"{log_prefix}Responses chunk {chunk_count}: {current_event_type} {data_str[:200]}")

                events = translator.translate_event(current_event_type, data)
                if events:
                    yield events
                    await asyncio.sleep(0)

            logger.info(f"{log_prefix}Copilot Responses OpenAI stream complete, {chunk_count} chunks")

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}Copilot Responses stream timed out")
            error_data = json.dumps(
                {
                    "error": {
                        "message": "Copilot Responses stream timed out",
                        "type": "server_error",
                        "param": None,
                        "code": None,
                    }
                }
            )
            yield f"data: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}Copilot Responses stream error: {e}")
            error_data = json.dumps({"error": {"message": str(e), "type": "server_error", "param": None, "code": None}})
            yield f"data: {error_data}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    # --- Client-facing /v1/responses endpoint methods ---

    async def handle_responses_passthrough(
        self,
        body: dict[str, Any],
        request_id: str,
        stream: bool,
    ) -> JSONResponse | StreamingResponse:
        """Handle a Responses API request by passing directly to Copilot /responses.

        Used when the model supports /responses natively (0 translations).
        All tools (function and built-in like web_search_preview, code_interpreter,
        etc.) are passed through as-is since Copilot's /responses endpoint supports them.
        """
        log_prefix = f"[{request_id}] " if request_id else ""

        if stream:
            body = {**body, "stream": True}
            resp, stream_cm = await self._open_responses_stream(body, log_prefix)
            return StreamingResponse(
                self._stream_responses_passthrough(resp, stream_cm, log_prefix),
                media_type="text/event-stream",
            )
        else:
            headers = await self._get_headers(body)
            logger.info(f"{log_prefix}Copilot Responses passthrough to {body.get('model')}")

            resp = await self._post_with_retry(COPILOT_RESPONSES_URL, headers, body, log_prefix)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}Copilot Responses error {resp.status_code}: {detail}")
                logger.error(
                    f"{log_prefix}Copilot Responses error request payload: "
                    f"model={body.get('model')}, "
                    f"input_items={len(body.get('input', []))}, "
                    f"tools={len(body.get('tools', []))}"
                )
                logger.debug(f"{log_prefix}Copilot Responses error full request body: {json.dumps(body)[:5000]}")
                token_err = _parse_token_limit_error(resp.status_code, detail)
                if token_err:
                    raise token_err
                if resp.status_code in FALLBACK_ON_ERRORS:
                    error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                    raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
                raise CopilotHttpError(resp.status_code, detail)

            return JSONResponse(content=resp.json())

    async def _stream_responses_passthrough(
        self, resp: httpx.Response, stream_cm: Any, log_prefix: str
    ) -> AsyncGenerator[str, None]:
        """Stream Responses API events as-is from Copilot (no translation).

        Reassembles SSE events from aiter_lines() which strips the blank-line
        boundaries.  Each SSE event is ``event: <type>\\ndata: <json>\\n\\n``.
        """
        chunk_count = 0
        pending_event_line: str | None = None

        try:
            async for line in resp.aiter_lines():
                if not line:
                    continue

                if line.startswith("event: "):
                    # Buffer the event-type line; it belongs to the next data line.
                    pending_event_line = line
                    continue

                # Anything else (typically "data: …") forms a complete SSE event
                # together with the buffered event line (if any).
                chunk_count += 1
                if chunk_count <= 3:
                    logger.debug(f"{log_prefix}Responses passthrough chunk {chunk_count}: {line[:200]}")

                if pending_event_line is not None:
                    yield f"{pending_event_line}\n{line}\n\n"
                    pending_event_line = None
                else:
                    yield f"{line}\n\n"
                await asyncio.sleep(0)

            logger.info(f"{log_prefix}Copilot Responses passthrough stream complete, {chunk_count} lines")

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}Copilot Responses stream timed out")
            error_data = json.dumps({"type": "error", "error": {"message": "Copilot Responses stream timed out"}})
            yield f"event: error\ndata: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}Copilot Responses stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    async def handle_responses_via_chat(
        self,
        body: dict[str, Any],
        request_id: str,
        stream: bool,
        copilot_model: str,
    ) -> JSONResponse | StreamingResponse:
        """Handle a Responses API request via Chat Completions translation.

        Used when the model only supports /chat/completions (2 translations).
        Translates Responses -> Chat Completions -> Copilot -> Chat Completions -> Responses.
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        openai_body = responses_to_openai_chat_request(body, copilot_model)

        if stream:
            openai_body["stream"] = True
            openai_body["stream_options"] = {"include_usage": True}
            resp, stream_cm = await self._open_stream(openai_body, log_prefix)
            return StreamingResponse(
                self._stream_responses_via_chat(resp, stream_cm, copilot_model, log_prefix),
                media_type="text/event-stream",
            )
        else:
            headers = await self._get_headers(openai_body)
            logger.info(f"{log_prefix}Copilot Responses via chat to {copilot_model}")

            resp = await self._post_with_retry(COPILOT_CHAT_URL, headers, openai_body, log_prefix)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}Copilot error {resp.status_code}: {detail}")
                token_err = _parse_token_limit_error(resp.status_code, detail)
                if token_err:
                    raise token_err
                if resp.status_code in FALLBACK_ON_ERRORS:
                    error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                    raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
                raise CopilotHttpError(resp.status_code, detail)

            openai_resp = resp.json()
            from .responses_translate import openai_chat_to_responses_response

            result = openai_chat_to_responses_response(openai_resp, copilot_model)
            return JSONResponse(content=result)

    async def _stream_responses_via_chat(
        self,
        resp: httpx.Response,
        stream_cm: Any,
        model: str,
        log_prefix: str,
    ) -> AsyncGenerator[str, None]:
        """Stream Chat Completions response, translating to Responses SSE format."""
        translator = OpenAIChatToResponsesStreamTranslator(model)
        chunk_count = 0

        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning(f"{log_prefix}Skipping malformed chunk: {data_str[:100]}")
                    continue

                chunk_count += 1
                if chunk_count <= 3:
                    logger.debug(f"{log_prefix}Chat->Responses chunk {chunk_count}: {data_str[:200]}")

                events = translator.translate_chunk(chunk)
                if events:
                    yield events
                    await asyncio.sleep(0)

            # Flush pending state
            flush_events = translator.flush()
            if flush_events:
                yield flush_events

            logger.info(f"{log_prefix}Copilot Responses via chat stream complete, {chunk_count} chunks")

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}Copilot stream timed out")
            error_data = json.dumps({"type": "error", "error": {"message": "Copilot stream timed out"}})
            yield f"event: error\ndata: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}Copilot stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    async def close(self) -> None:
        """Close HTTP client and auth."""
        await self._client.aclose()
        await self._auth.close()
