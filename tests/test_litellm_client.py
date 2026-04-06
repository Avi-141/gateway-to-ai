"""Tests for claudegate/litellm_client.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claudegate.errors import ContextWindowExceededError, LiteLLMHttpError, TransientBackendError
from claudegate.litellm_client import LiteLLMBackend, _parse_token_limit_error

# Token limit error payloads used across multiple tests
_ANTHROPIC_STYLE_DETAIL = (
    '{"error":{"message":"prompt token count of 145794 exceeds the limit of 128000",'
    '"code":"model_max_prompt_tokens_exceeded"}}'
)
_OPENAI_STYLE_DETAIL = (
    '{"error":{"message":"This model\'s maximum context length is 128000 tokens. '
    'However, your messages resulted in 145794 tokens","code":"context_length_exceeded"}}'
)


@pytest.fixture
def backend():
    """LiteLLMBackend with default settings."""
    return LiteLLMBackend("http://localhost:4000", api_key="test-key", timeout=30)


# --- list_models ---


class TestListModels:
    @pytest.mark.anyio
    async def test_success(self, backend):
        """Successful fetch returns model list."""
        models_data = {
            "data": [
                {"id": "anthropic/claude-sonnet-4.5", "owned_by": "anthropic"},
                {"id": "openai/gpt-4o", "owned_by": "openai"},
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = models_data

        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await backend.list_models()

        assert len(result) == 2
        assert result[0]["id"] == "anthropic/claude-sonnet-4.5"
        assert result[1]["id"] == "openai/gpt-4o"

    @pytest.mark.anyio
    async def test_http_error_returns_empty(self, backend):
        """Non-200 response returns empty list."""
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await backend.list_models()

        assert result == []

    @pytest.mark.anyio
    async def test_network_error_returns_empty(self, backend):
        """Network exception returns empty list."""
        with patch.object(backend._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("failed")):
            result = await backend.list_models()

        assert result == []

    @pytest.mark.anyio
    async def test_missing_data_key_returns_empty(self, backend):
        """Response without 'data' key returns empty list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": "list"}

        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await backend.list_models()

        assert result == []


# --- _parse_token_limit_error ---


class TestParseTokenLimitError:
    def test_matches_anthropic_style(self):
        err = _parse_token_limit_error(400, _ANTHROPIC_STYLE_DETAIL)
        assert err is not None
        assert err.prompt_tokens == 145794
        assert err.context_limit == 128000
        assert err.backend == "litellm"

    def test_matches_openai_style(self):
        err = _parse_token_limit_error(400, _OPENAI_STYLE_DETAIL)
        assert err is not None
        assert err.prompt_tokens == 145794
        assert err.context_limit == 128000
        assert err.backend == "litellm"

    def test_matches_error_code_without_numbers(self):
        detail = '{"error":{"message":"request too large","code":"context_length_exceeded"}}'
        err = _parse_token_limit_error(400, detail)
        assert err is not None
        assert err.prompt_tokens == 0
        assert err.context_limit == 0
        assert err.raw_detail == detail

    def test_matches_model_max_prompt_tokens_code(self):
        detail = '{"error":{"message":"too big","code":"model_max_prompt_tokens_exceeded"}}'
        err = _parse_token_limit_error(400, detail)
        assert err is not None
        assert err.prompt_tokens == 0
        assert err.context_limit == 0

    def test_returns_none_for_non_400(self):
        assert _parse_token_limit_error(429, _ANTHROPIC_STYLE_DETAIL) is None

    def test_returns_none_for_unrelated_400(self):
        detail = '{"error":{"message":"invalid model","code":"invalid_request"}}'
        assert _parse_token_limit_error(400, detail) is None

    def test_returns_none_for_empty_detail(self):
        assert _parse_token_limit_error(400, "") is None


# --- handle_messages ---


class TestHandleMessages:
    @pytest.mark.anyio
    async def test_non_streaming_success(self, backend):
        """Non-streaming request returns Anthropic-format response."""
        openai_resp = {
            "id": "chat-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = openai_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            resp = await backend.handle_messages(
                body, "req1", False, "anthropic/claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"
            )

        assert resp.status_code == 200
        result = json.loads(resp.body)
        assert result["content"][0]["text"] == "Hello!"
        assert result["type"] == "message"

    @pytest.mark.anyio
    async def test_non_streaming_429_raises_transient(self, backend):
        """429 raises TransientBackendError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(TransientBackendError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.status_code == 429
        assert exc_info.value.backend == "litellm"

    @pytest.mark.anyio
    async def test_non_streaming_500_raises_transient(self, backend):
        """500 raises TransientBackendError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server error"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(TransientBackendError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.status_code == 500

    @pytest.mark.anyio
    async def test_non_streaming_401_raises_litellm_http_error(self, backend):
        """401 raises LiteLLMHttpError (not transient)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "unauthorized"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(LiteLLMHttpError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.status_code == 401

    @pytest.mark.anyio
    async def test_non_streaming_token_limit_raises_context_window_error(self, backend):
        """400 with token limit exceeded raises ContextWindowExceededError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = _ANTHROPIC_STYLE_DETAIL

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(ContextWindowExceededError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.prompt_tokens == 145794
        assert exc_info.value.context_limit == 128000
        assert exc_info.value.backend == "litellm"


# --- handle_openai_messages ---


class TestHandleOpenAIMessages:
    @pytest.mark.anyio
    async def test_non_streaming_passthrough(self, backend):
        """Non-streaming OpenAI request passes through without translation."""
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = openai_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            resp = await backend.handle_openai_messages(body, "req1", False, "gpt-4o")

        assert resp.status_code == 200
        result = json.loads(resp.body)
        assert result["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.anyio
    async def test_429_raises_transient(self, backend):
        """429 on OpenAI passthrough raises TransientBackendError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(TransientBackendError) as exc_info:
                await backend.handle_openai_messages(body, "", False, "gpt-4o")

        assert exc_info.value.status_code == 429

    @pytest.mark.anyio
    async def test_token_limit_raises_context_window_error(self, backend):
        """400 with token limit on OpenAI passthrough raises ContextWindowExceededError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = _OPENAI_STYLE_DETAIL

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(ContextWindowExceededError) as exc_info:
                await backend.handle_openai_messages(body, "", False, "gpt-4o")

        assert exc_info.value.prompt_tokens == 145794
        assert exc_info.value.context_limit == 128000


# --- handle_responses_via_chat ---


class TestHandleResponsesViaChat:
    @pytest.mark.anyio
    async def test_non_streaming_success(self, backend):
        """Non-streaming Responses via chat returns Responses-format response."""
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = openai_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            }
            resp = await backend.handle_responses_via_chat(body, "req1", False, "anthropic/claude-sonnet-4-5-20250929")

        assert resp.status_code == 200
        result = json.loads(resp.body)
        assert result["object"] == "response"

    @pytest.mark.anyio
    async def test_429_raises_transient(self, backend):
        """429 on Responses via chat raises TransientBackendError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            }
            with pytest.raises(TransientBackendError):
                await backend.handle_responses_via_chat(body, "", False, "anthropic/claude-sonnet-4-5-20250929")


# --- _open_stream ---


class TestOpenStream:
    @pytest.mark.anyio
    async def test_stream_error_raises_transient(self, backend):
        """Non-200 stream open raises TransientBackendError for 5xx."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 502
        mock_resp.aread = AsyncMock(return_value=b"bad gateway")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(backend._client, "stream", return_value=mock_stream_ctx),
            pytest.raises(TransientBackendError) as exc_info,
        ):
            await backend._open_stream({"model": "m"}, "")

        assert exc_info.value.status_code == 502

    @pytest.mark.anyio
    async def test_stream_token_limit_raises_context_window(self, backend):
        """400 with token limit on stream open raises ContextWindowExceededError."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 400
        mock_resp.aread = AsyncMock(return_value=_ANTHROPIC_STYLE_DETAIL.encode())

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(backend._client, "stream", return_value=mock_stream_ctx),
            pytest.raises(ContextWindowExceededError) as exc_info,
        ):
            await backend._open_stream({"model": "m"}, "")

        assert exc_info.value.prompt_tokens == 145794

    @pytest.mark.anyio
    async def test_stream_401_raises_litellm_http_error(self, backend):
        """401 on stream open raises LiteLLMHttpError."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 401
        mock_resp.aread = AsyncMock(return_value=b"unauthorized")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(backend._client, "stream", return_value=mock_stream_ctx),
            pytest.raises(LiteLLMHttpError) as exc_info,
        ):
            await backend._open_stream({"model": "m"}, "")

        assert exc_info.value.status_code == 401


# --- close ---


class TestClose:
    @pytest.mark.anyio
    async def test_close(self, backend):
        """Close shuts down the httpx client."""
        with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
            await backend.close()

        mock_close.assert_called_once()
