"""GitHub Copilot HTTP client and backend."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import logger
from .copilot_auth import COPILOT_HEADERS, CopilotAuth
from .copilot_translate import (
    StreamTranslator,
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)

COPILOT_CHAT_URL = "https://api.githubcopilot.com/chat/completions"


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
        """Handle a messages request by proxying through Copilot."""
        log_prefix = f"[{request_id}] " if request_id else ""

        try:
            openai_body = anthropic_to_openai_request(body, openai_model)

            if stream:
                openai_body["stream"] = True
                openai_body["stream_options"] = {"include_usage": True}
                return StreamingResponse(
                    self._stream_response(openai_body, anthropic_model, log_prefix),
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
                    return self._map_http_error(resp.status_code, detail)

                openai_resp = resp.json()
                result = openai_to_anthropic_response(openai_resp, anthropic_model)
                logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
                return JSONResponse(content=result)

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}Copilot request timed out")
            return self._error_response(
                504, "timeout_error", "Copilot request timed out. Try a smaller request or use streaming."
            )
        except RuntimeError as e:
            logger.error(f"{log_prefix}Auth error: {e}")
            return self._error_response(401, "authentication_error", str(e))
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected Copilot error: {e}")
            return self._error_response(500, "api_error", str(e))

    async def _stream_response(
        self, openai_body: dict[str, Any], anthropic_model: str, log_prefix: str
    ) -> AsyncGenerator[str, None]:
        """Stream response from Copilot, translating to Anthropic SSE format."""
        translator = StreamTranslator(anthropic_model)
        chunk_count = 0

        try:
            headers = await self._get_headers()
            logger.info(f"{log_prefix}Starting Copilot stream for {openai_body.get('model')}")

            async with self._client.stream("POST", COPILOT_CHAT_URL, headers=headers, json=openai_body) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    detail = body.decode()[:500]
                    logger.error(f"{log_prefix}Copilot stream error {resp.status_code}: {detail}")
                    error_event = {
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Copilot error {resp.status_code}: {detail}"},
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                    return

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
            if translator.started:
                # Check if message_stop was already emitted
                pass

            logger.info(f"{log_prefix}Copilot stream complete, {chunk_count} chunks")
            yield "event: done\ndata: [DONE]\n\n"

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}Copilot stream timed out")
            error_data = json.dumps({"type": "error", "error": {"message": "Copilot stream timed out"}})
            yield f"event: error\ndata: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}Copilot stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"

    async def close(self) -> None:
        """Close HTTP client and auth."""
        await self._client.aclose()
        await self._auth.close()
