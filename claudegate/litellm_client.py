"""LiteLLM proxy HTTP client and backend."""

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import FALLBACK_ON_ERRORS, SSL_CONTEXT, logger
from .copilot_translate import (
    StreamTranslator,
    anthropic_to_openai_request,
    estimate_input_tokens,
    openai_to_anthropic_response,
)
from .errors import ContextWindowExceededError, LiteLLMHttpError, TransientBackendError
from .responses_translate import (
    OpenAIChatToResponsesStreamTranslator,
    openai_chat_to_responses_response,
    responses_to_openai_chat_request,
)

# Map HTTP status codes to Anthropic error types
_ERROR_TYPE_MAP: dict[int, str] = {
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
    504: "timeout_error",
}

# Patterns to extract token counts from various provider errors surfaced by LiteLLM
_TOKEN_LIMIT_PATTERNS = [
    # Anthropic-style: "prompt token count of 145794 exceeds the limit of 128000"
    re.compile(r"prompt token count of (\d+) exceeds the limit of (\d+)"),
    # OpenAI-style: "maximum context length is 128000 tokens. However, your messages resulted in 145794 tokens"
    re.compile(r"maximum context length is (\d+) tokens.*?(\d+) tokens"),
]


def _parse_token_limit_error(status_code: int, detail: str) -> ContextWindowExceededError | None:
    """Check if a LiteLLM error is a token limit exceeded error.

    LiteLLM surfaces errors from various providers, each with different formats.
    Returns ContextWindowExceededError if matched, None otherwise.
    """
    if status_code != 400:
        return None
    for pattern in _TOKEN_LIMIT_PATTERNS:
        match = pattern.search(detail)
        if match:
            g1, g2 = int(match.group(1)), int(match.group(2))
            # Anthropic-style: (prompt_tokens, limit), OpenAI-style: (limit, prompt_tokens)
            if g1 > g2:
                return ContextWindowExceededError(g1, g2, "litellm")
            return ContextWindowExceededError(g2, g1, "litellm")
    # Fallback: check for common error codes without parseable numbers
    if "context_length_exceeded" in detail or "model_max_prompt_tokens_exceeded" in detail:
        return ContextWindowExceededError(0, 0, "litellm", detail)
    return None


class LiteLLMBackend:
    """Handles routing requests through a LiteLLM proxy server."""

    def __init__(self, api_base: str, api_key: str = "", timeout: int = 300):
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            verify=SSL_CONTEXT,
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    def _get_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @property
    def _chat_url(self) -> str:
        return f"{self._api_base}/chat/completions"

    @property
    def _models_url(self) -> str:
        return f"{self._api_base}/models"

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models from the LiteLLM /models endpoint."""
        headers = self._get_headers()
        try:
            resp = await self._client.get(self._models_url, headers=headers)
        except Exception as e:
            logger.warning(f"Failed to fetch LiteLLM models: {e}")
            return []
        if resp.status_code != 200:
            logger.warning(f"Failed to fetch LiteLLM models: HTTP {resp.status_code}")
            return []
        data = resp.json()
        return data.get("data", [])

    # --- /v1/messages path (Anthropic -> OpenAI -> LiteLLM -> OpenAI -> Anthropic) ---

    async def handle_messages(
        self,
        body: dict[str, Any],
        request_id: str,
        stream: bool,
        openai_model: str,
        anthropic_model: str,
        client_context_window: int = 0,
    ) -> JSONResponse | StreamingResponse:
        """Handle a messages request by proxying through LiteLLM.

        Raises TransientBackendError for fallback-eligible errors (429, 5xx).
        Raises LiteLLMHttpError for non-transient HTTP errors.
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        openai_body = anthropic_to_openai_request(body, openai_model)

        if stream:
            openai_body["stream"] = True
            openai_body["stream_options"] = {"include_usage": True}
            estimated_tokens = estimate_input_tokens(body)
            resp, stream_cm = await self._open_stream(openai_body, log_prefix)
            return StreamingResponse(
                self._stream_response(
                    resp,
                    stream_cm,
                    anthropic_model,
                    log_prefix,
                    estimated_tokens,
                    0,
                    client_context_window,
                ),
                media_type="text/event-stream",
            )
        else:
            headers = self._get_headers()
            logger.info(f"{log_prefix}LiteLLM request to {openai_model}")
            logger.debug(f"{log_prefix}OpenAI body keys: {list(openai_body.keys())}")

            resp = await self._client.post(self._chat_url, headers=headers, json=openai_body)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}LiteLLM error {resp.status_code}: {detail}")
                self._raise_for_status(resp.status_code, detail)

            openai_resp = resp.json()
            result = openai_to_anthropic_response(openai_resp, anthropic_model)
            logger.debug(f"{log_prefix}Response: {json.dumps(result)[:500]}")
            return JSONResponse(content=result)

    async def _open_stream(self, openai_body: dict[str, Any], log_prefix: str) -> tuple[httpx.Response, Any]:
        """Open streaming connection and validate HTTP status.

        Returns (response, context_manager) on success.
        Raises TransientBackendError for fallback-eligible status codes.
        Raises LiteLLMHttpError for non-transient HTTP errors.
        """
        headers = self._get_headers()
        logger.info(f"{log_prefix}Starting LiteLLM stream for {openai_body.get('model')}")

        stream_cm = self._client.stream("POST", self._chat_url, headers=headers, json=openai_body)
        resp = await stream_cm.__aenter__()

        if resp.status_code != 200:
            body = await resp.aread()
            detail = body.decode()[:500]
            await stream_cm.__aexit__(None, None, None)
            logger.error(f"{log_prefix}LiteLLM stream error {resp.status_code}: {detail}")
            self._raise_for_status(resp.status_code, detail)

        return resp, stream_cm

    async def _stream_response(
        self,
        resp: httpx.Response,
        stream_cm: Any,
        anthropic_model: str,
        log_prefix: str,
        estimated_input_tokens: int = 0,
        litellm_context_limit: int = 0,
        client_context_window: int = 0,
    ) -> AsyncGenerator[str, None]:
        """Stream response from LiteLLM, translating to Anthropic SSE format."""
        translator = StreamTranslator(
            anthropic_model,
            estimated_input_tokens,
            litellm_context_limit,
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

            logger.info(f"{log_prefix}LiteLLM stream complete, {chunk_count} chunks")
            yield "event: done\ndata: [DONE]\n\n"

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}LiteLLM stream timed out")
            error_data = json.dumps({"type": "error", "error": {"message": "LiteLLM stream timed out"}})
            yield f"event: error\ndata: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}LiteLLM stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    # --- /v1/chat/completions path (OpenAI passthrough, 0 translations) ---

    async def handle_openai_messages(
        self, openai_body: dict[str, Any], request_id: str, stream: bool, litellm_model: str
    ) -> JSONResponse | StreamingResponse:
        """Handle an OpenAI-format request by passing directly to LiteLLM (no translation).

        Raises TransientBackendError for fallback-eligible errors (429, 5xx).
        Raises LiteLLMHttpError for non-transient HTTP errors.
        """
        log_prefix = f"[{request_id}] " if request_id else ""

        # Override model to LiteLLM-compatible name
        openai_body = {**openai_body, "model": litellm_model}

        if stream:
            openai_body["stream"] = True
            openai_body["stream_options"] = {"include_usage": True}
            resp, stream_cm = await self._open_stream(openai_body, log_prefix)
            return StreamingResponse(
                self._stream_openai_response(resp, stream_cm, log_prefix),
                media_type="text/event-stream",
            )
        else:
            headers = self._get_headers()
            logger.info(f"{log_prefix}LiteLLM OpenAI passthrough to {litellm_model}")
            logger.debug(f"{log_prefix}OpenAI body keys: {list(openai_body.keys())}")

            resp = await self._client.post(self._chat_url, headers=headers, json=openai_body)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}LiteLLM error {resp.status_code}: {detail}")
                self._raise_for_status(resp.status_code, detail)

            openai_resp = resp.json()
            logger.debug(f"{log_prefix}Response: {json.dumps(openai_resp)[:500]}")
            return JSONResponse(content=openai_resp)

    async def _stream_openai_response(
        self, resp: httpx.Response, stream_cm: Any, log_prefix: str
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI response as-is from LiteLLM (no translation)."""
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

            logger.info(f"{log_prefix}LiteLLM OpenAI stream complete, {chunk_count} lines")

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}LiteLLM stream timed out")
            error_data = json.dumps(
                {"error": {"message": "LiteLLM stream timed out", "type": "server_error", "param": None, "code": None}}
            )
            yield f"data: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}LiteLLM stream error: {e}")
            error_data = json.dumps({"error": {"message": str(e), "type": "server_error", "param": None, "code": None}})
            yield f"data: {error_data}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    # --- /v1/responses path (Responses -> Chat Completions -> LiteLLM -> Chat Completions -> Responses) ---

    async def handle_responses_via_chat(
        self,
        body: dict[str, Any],
        request_id: str,
        stream: bool,
        litellm_model: str,
    ) -> JSONResponse | StreamingResponse:
        """Handle a Responses API request via Chat Completions translation.

        Translates Responses -> Chat Completions -> LiteLLM -> Chat Completions -> Responses.
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        openai_body = responses_to_openai_chat_request(body, litellm_model)

        if stream:
            openai_body["stream"] = True
            openai_body["stream_options"] = {"include_usage": True}
            resp, stream_cm = await self._open_stream(openai_body, log_prefix)
            return StreamingResponse(
                self._stream_responses_via_chat(resp, stream_cm, litellm_model, log_prefix),
                media_type="text/event-stream",
            )
        else:
            headers = self._get_headers()
            logger.info(f"{log_prefix}LiteLLM Responses via chat to {litellm_model}")

            resp = await self._client.post(self._chat_url, headers=headers, json=openai_body)

            if resp.status_code != 200:
                detail = resp.text[:500]
                logger.error(f"{log_prefix}LiteLLM error {resp.status_code}: {detail}")
                self._raise_for_status(resp.status_code, detail)

            openai_resp = resp.json()
            result = openai_chat_to_responses_response(openai_resp, litellm_model)
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

            logger.info(f"{log_prefix}LiteLLM Responses via chat stream complete, {chunk_count} chunks")

        except httpx.TimeoutException:
            logger.error(f"{log_prefix}LiteLLM stream timed out")
            error_data = json.dumps({"type": "error", "error": {"message": "LiteLLM stream timed out"}})
            yield f"event: error\ndata: {error_data}\n\n"
        except Exception as e:
            logger.error(f"{log_prefix}LiteLLM stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        finally:
            await stream_cm.__aexit__(None, None, None)

    # --- Shared helpers ---

    def _raise_for_status(self, status_code: int, detail: str) -> None:
        """Raise the appropriate error for a non-200 HTTP response.

        Raises ContextWindowExceededError, TransientBackendError, or LiteLLMHttpError.
        """
        token_err = _parse_token_limit_error(status_code, detail)
        if token_err:
            raise token_err
        if status_code in FALLBACK_ON_ERRORS:
            error_type = _ERROR_TYPE_MAP.get(status_code, "api_error")
            raise TransientBackendError(status_code, error_type, detail, "litellm")
        raise LiteLLMHttpError(status_code, detail)

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
