"""GitHub Copilot HTTP client and backend."""

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import FALLBACK_ON_ERRORS, SSL_CONTEXT, logger
from .copilot_auth import COPILOT_HEADERS, CopilotAuth
from .copilot_translate import (
    StreamTranslator,
    anthropic_to_openai_request,
    estimate_input_tokens,
    openai_to_anthropic_response,
)
from .errors import ContextWindowExceededError, CopilotHttpError, TransientBackendError
from .responses_translate import (
    ResponsesStreamTranslator,
    ResponsesToOpenAIStreamTranslator,
    anthropic_to_responses_request,
    openai_chat_to_responses_request,
    responses_to_anthropic_response,
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


class CopilotBackend:
    """Handles routing requests through GitHub Copilot API."""

    def __init__(self, auth: CopilotAuth, timeout: int = 300):
        self._auth = auth
        self._client = httpx.AsyncClient(verify=SSL_CONTEXT, timeout=httpx.Timeout(timeout, connect=30.0))

    async def _get_headers(self) -> dict[str, str]:
        """Build request headers with fresh Copilot token."""
        token = await self._auth.get_token()
        return {
            **COPILOT_HEADERS,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
        }

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
        self, body: dict[str, Any], request_id: str, stream: bool, openai_model: str, anthropic_model: str
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
            resp, stream_cm = await self._open_stream(openai_body, log_prefix)
            return StreamingResponse(
                self._stream_response(resp, stream_cm, anthropic_model, log_prefix, estimated_tokens),
                media_type="text/event-stream",
            )
        else:
            headers = await self._get_headers()
            logger.info(f"{log_prefix}Copilot request to {openai_model}")
            logger.debug(f"{log_prefix}OpenAI body keys: {list(openai_body.keys())}")

            resp = await self._client.post(COPILOT_CHAT_URL, headers=headers, json=openai_body)

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
            result = openai_to_anthropic_response(openai_resp, anthropic_model)
            logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
            return JSONResponse(content=result)

    async def _open_stream(self, openai_body: dict[str, Any], log_prefix: str) -> tuple[httpx.Response, Any]:
        """Open streaming connection and validate HTTP status.

        Returns (response, context_manager) on success.
        Raises TransientBackendError for fallback-eligible status codes.
        Raises CopilotHttpError for non-transient HTTP errors.
        """
        headers = await self._get_headers()
        logger.info(f"{log_prefix}Starting Copilot stream for {openai_body.get('model')}")

        stream_cm = self._client.stream("POST", COPILOT_CHAT_URL, headers=headers, json=openai_body)
        resp = await stream_cm.__aenter__()

        if resp.status_code != 200:
            body = await resp.aread()
            detail = body.decode()[:500]
            await stream_cm.__aexit__(None, None, None)
            logger.error(f"{log_prefix}Copilot stream error {resp.status_code}: {detail}")
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
    ) -> AsyncGenerator[str, None]:
        """Stream response from already-opened Copilot connection, translating to Anthropic SSE format."""
        translator = StreamTranslator(anthropic_model, estimated_input_tokens)
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
            headers = await self._get_headers()
            logger.info(f"{log_prefix}Copilot OpenAI passthrough to {copilot_model}")
            logger.debug(f"{log_prefix}OpenAI body keys: {list(openai_body.keys())}")

            resp = await self._client.post(COPILOT_CHAT_URL, headers=headers, json=openai_body)

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
            resp, stream_cm = await self._open_responses_stream(responses_body, log_prefix)
            return StreamingResponse(
                self._stream_responses_response(resp, stream_cm, anthropic_model, log_prefix, estimated_tokens),
                media_type="text/event-stream",
            )
        else:
            headers = await self._get_headers()
            logger.info(f"{log_prefix}Copilot Responses request to {responses_model}")
            logger.debug(f"{log_prefix}Responses body keys: {list(responses_body.keys())}")

            resp = await self._client.post(COPILOT_RESPONSES_URL, headers=headers, json=responses_body)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}Copilot Responses error {resp.status_code}: {detail}")
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
            headers = await self._get_headers()
            logger.info(f"{log_prefix}Copilot Responses OpenAI passthrough to {responses_model}")
            logger.debug(f"{log_prefix}Responses body keys: {list(responses_body.keys())}")

            resp = await self._client.post(COPILOT_RESPONSES_URL, headers=headers, json=responses_body)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}Copilot Responses error {resp.status_code}: {detail}")
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
        self, responses_body: dict[str, Any], log_prefix: str
    ) -> tuple[httpx.Response, Any]:
        """Open streaming connection to Responses API and validate HTTP status."""
        headers = await self._get_headers()
        logger.info(f"{log_prefix}Starting Copilot Responses stream for {responses_body.get('model')}")

        stream_cm = self._client.stream("POST", COPILOT_RESPONSES_URL, headers=headers, json=responses_body)
        resp = await stream_cm.__aenter__()

        if resp.status_code != 200:
            body = await resp.aread()
            detail = body.decode()[:500]
            await stream_cm.__aexit__(None, None, None)
            logger.error(f"{log_prefix}Copilot Responses stream error {resp.status_code}: {detail}")
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
    ) -> AsyncGenerator[str, None]:
        """Stream Responses API events, translating to Anthropic SSE format."""
        translator = ResponsesStreamTranslator(anthropic_model, estimated_input_tokens)
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

    async def close(self) -> None:
        """Close HTTP client and auth."""
        await self._client.aclose()
        await self._auth.close()
