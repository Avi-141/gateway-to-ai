"""AI Framework HTTP client and backend."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import AI_FRAMEWORK_CHAT_PATH, FALLBACK_ON_ERRORS, logger
from .copilot_translate import (
    StreamTranslator,
    anthropic_to_openai_request,
    estimate_input_tokens,
    openai_to_anthropic_response,
)
from .errors import TransientBackendError

_ERROR_TYPE_MAP: dict[int, str] = {
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
    504: "timeout_error",
}


class CxAiClusterBackend:
    """Routes requests through the AI Framework chat completions endpoint.

    The endpoint speaks OpenAI chat completions, so we reuse the same
    Anthropic <-> OpenAI translation layer used by the Copilot backend.
    Auth is via static X-ServiceCredentials header.
    """

    def __init__(self, service_credentials: str, base_url: str, timeout: int = 300):
        self._service_credentials = service_credentials
        self._chat_url = f"{base_url.rstrip('/')}{AI_FRAMEWORK_CHAT_PATH}"
        # verify=False: corporate endpoint with internal TLS cert
        self._client = httpx.AsyncClient(verify=False, timeout=httpx.Timeout(timeout, connect=30.0))  # noqa: S501

    def _get_headers(self) -> dict[str, str]:
        return {
            "X-ServiceCredentials": self._service_credentials,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # /v1/messages  (Anthropic in -> Anthropic out)
    # ------------------------------------------------------------------

    async def handle_messages(
        self,
        body: dict[str, Any],
        request_id: str,
        stream: bool,
        anthropic_model: str,
        target_model: str,
    ) -> JSONResponse | StreamingResponse:
        """Translate Anthropic request -> OpenAI -> AI Framework -> OpenAI -> Anthropic.

        Args:
            target_model: The actual model name to send to the AI Framework API.
            anthropic_model: The original model string from the client (for response labels).

        Raises TransientBackendError for fallback-eligible errors (429, 5xx).
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        openai_body = anthropic_to_openai_request(body, target_model)

        if stream:
            openai_body["stream"] = True
            # Note: stream_options is intentionally omitted. Some AI Framework
            # models (e.g. Mistral) reject it with 422. The StreamTranslator
            # falls back to estimated token counts when usage is not provided.
            estimated_tokens = estimate_input_tokens(body)
            resp, stream_cm = await self._open_stream(openai_body, log_prefix)
            return StreamingResponse(
                self._stream_response(resp, stream_cm, anthropic_model, log_prefix, estimated_tokens),
                media_type="text/event-stream",
            )
        else:
            headers = self._get_headers()
            logger.info(f"{log_prefix}AI Framework request to {target_model}")
            resp = await self._client.post(self._chat_url, headers=headers, json=openai_body)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}AI Framework error {resp.status_code}: {detail}")
                if resp.status_code in FALLBACK_ON_ERRORS:
                    error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                    raise TransientBackendError(resp.status_code, error_type, detail, "iq-ai-cluster")
                return self._error_response(resp.status_code, detail)

            openai_resp = resp.json()
            result = openai_to_anthropic_response(openai_resp, anthropic_model)
            logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
            return JSONResponse(content=result)

    async def _open_stream(self, openai_body: dict[str, Any], log_prefix: str) -> tuple[httpx.Response, Any]:
        """Open a streaming connection and validate HTTP status."""
        headers = self._get_headers()
        logger.info(f"{log_prefix}Starting AI Framework stream for {openai_body.get('model')}")
        stream_cm = self._client.stream("POST", self._chat_url, headers=headers, json=openai_body)
        resp = await stream_cm.__aenter__()

        if resp.status_code != 200:
            body_bytes = await resp.aread()
            detail = body_bytes.decode()[:500]
            await stream_cm.__aexit__(None, None, None)
            logger.error(f"{log_prefix}AI Framework stream error {resp.status_code}: {detail}")
            if resp.status_code in FALLBACK_ON_ERRORS:
                error_type = _ERROR_TYPE_MAP.get(resp.status_code, "api_error")
                raise TransientBackendError(resp.status_code, error_type, detail, "iq-ai-cluster")
            raise RuntimeError(f"AI Framework HTTP {resp.status_code}: {detail}")

        return resp, stream_cm

    async def _stream_response(
        self,
        resp: httpx.Response,
        stream_cm: Any,
        anthropic_model: str,
        log_prefix: str,
        estimated_input_tokens: int = 0,
    ) -> AsyncGenerator[str, None]:
        """Stream response from AI Framework, translating to Anthropic SSE format."""
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

            flush_events = translator.flush()
            if flush_events:
                yield flush_events

            logger.info(f"{log_prefix}AI Framework stream complete, {chunk_count} chunks")
            yield "event: done\ndata: [DONE]\n\n"

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}AI Framework stream timed out")
            error_data = json.dumps({"type": "error", "error": {"message": "AI Framework stream timed out"}})
            yield f"event: error\ndata: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}AI Framework stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    def _error_response(self, status_code: int, detail: str) -> JSONResponse:
        if status_code == 401:
            msg = f"AI Framework auth failed: {detail}"
            error_type = "authentication_error"
        elif status_code == 403:
            msg = f"AI Framework access denied: {detail}"
            error_type = "permission_error"
        elif status_code == 404:
            msg = f"AI Framework model not found: {detail}"
            error_type = "not_found_error"
        else:
            msg = f"AI Framework error: {detail}"
            error_type = "api_error"
            status_code = 502
        return JSONResponse(
            status_code=status_code,
            content={"type": "error", "error": {"type": error_type, "message": msg}},
        )

    async def close(self) -> None:
        await self._client.aclose()
