"""GitHub Copilot HTTP client and backend."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import FALLBACK_ON_ERRORS, logger
from .copilot_auth import COPILOT_HEADERS, CopilotAuth
from .copilot_translate import (
    StreamTranslator,
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)
from .errors import CopilotHttpError, TransientBackendError

COPILOT_CHAT_URL = "https://api.githubcopilot.com/chat/completions"

# Map HTTP status codes to Anthropic error types
_ERROR_TYPE_MAP: dict[int, str] = {
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
    504: "timeout_error",
}


class CopilotBackend:
    """Handles routing requests through GitHub Copilot API."""

    def __init__(self, auth: CopilotAuth, timeout: int = 300):
        self._auth = auth
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0))

    async def _get_headers(self) -> dict[str, str]:
        """Build request headers with fresh Copilot token."""
        token = await self._auth.get_token()
        return {
            **COPILOT_HEADERS,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
        }

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
            resp, stream_cm = await self._open_stream(openai_body, log_prefix)
            return StreamingResponse(
                self._stream_response(resp, stream_cm, anthropic_model, log_prefix),
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
            if resp.status_code in FALLBACK_ON_ERRORS:
                error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                raise TransientBackendError(resp.status_code, error_type, detail, "copilot")
            raise CopilotHttpError(resp.status_code, detail)

        return resp, stream_cm

    async def _stream_response(
        self, resp: httpx.Response, stream_cm: Any, anthropic_model: str, log_prefix: str
    ) -> AsyncGenerator[str, None]:
        """Stream response from already-opened Copilot connection, translating to Anthropic SSE format."""
        translator = StreamTranslator(anthropic_model)
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

            # If translator never got a finish_reason, ensure cleanup
            if translator.current_block_type is not None:
                yield translator._emit_content_block_stop()

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

    async def close(self) -> None:
        """Close HTTP client and auth."""
        await self._client.aclose()
        await self._auth.close()
