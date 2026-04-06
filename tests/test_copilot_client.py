"""Tests for claudegate/copilot_client.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claudegate.copilot_client import (
    CopilotBackend,
    TokenBucket,
    _normalize_openai_response,
    _parse_retry_after,
    _parse_token_limit_error,
    compute_initiator,
)
from claudegate.errors import ContextWindowExceededError, CopilotHttpError, TransientBackendError

# Copilot token limit error payload used across multiple tests
_TOKEN_LIMIT_DETAIL = (
    '{"error":{"message":"prompt token count of 145794 exceeds the limit of 128000",'
    '"code":"model_max_prompt_tokens_exceeded"}}'
)


@pytest.fixture
def backend(mock_copilot_auth):
    """CopilotBackend with mocked auth, retry and rate limiting disabled."""
    return CopilotBackend(mock_copilot_auth, timeout=30, retry_max=0, max_rate=0)


# --- list_models ---


class TestListModels:
    @pytest.mark.anyio
    async def test_success(self, backend):
        """Successful fetch returns model list."""
        models_data = {
            "data": [
                {"id": "claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "owned_by": "anthropic"},
                {"id": "gpt-4o", "name": "GPT-4o", "owned_by": "openai"},
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = models_data

        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await backend.list_models()

        assert len(result) == 2
        assert result[0]["id"] == "claude-sonnet-4.5"
        assert result[1]["id"] == "gpt-4o"

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
    def test_matches_standard_copilot_error(self):
        detail = _TOKEN_LIMIT_DETAIL
        err = _parse_token_limit_error(400, detail)
        assert err is not None
        assert err.prompt_tokens == 145794
        assert err.context_limit == 128000
        assert err.backend == "copilot"

    def test_matches_error_code_without_numbers(self):
        detail = '{"error":{"message":"request too large","code":"model_max_prompt_tokens_exceeded"}}'
        err = _parse_token_limit_error(400, detail)
        assert err is not None
        assert err.prompt_tokens == 0
        assert err.context_limit == 0
        assert err.raw_detail == detail

    def test_returns_none_for_non_400(self):
        detail = "prompt token count of 145794 exceeds the limit of 128000"
        assert _parse_token_limit_error(429, detail) is None

    def test_returns_none_for_unrelated_400(self):
        detail = '{"error":{"message":"invalid model","code":"invalid_request"}}'
        assert _parse_token_limit_error(400, detail) is None

    def test_returns_none_for_empty_detail(self):
        assert _parse_token_limit_error(400, "") is None


# --- compute_initiator ---


class TestComputeInitiator:
    def test_chat_last_user(self):
        body = {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]}
        assert compute_initiator(body) == "user"

    def test_chat_last_assistant(self):
        body = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        assert compute_initiator(body) == "agent"

    def test_chat_last_tool(self):
        body = {"messages": [{"role": "user", "content": "hi"}, {"role": "tool", "content": "result"}]}
        assert compute_initiator(body) == "agent"

    def test_responses_string_input(self):
        body = {"input": "hello world"}
        assert compute_initiator(body) == "user"

    def test_responses_last_function_call_output(self):
        body = {"input": [{"type": "function_call_output", "call_id": "x", "output": "result"}]}
        assert compute_initiator(body) == "agent"

    def test_responses_last_function_call(self):
        body = {"input": [{"type": "function_call", "name": "fn", "arguments": "{}"}]}
        assert compute_initiator(body) == "agent"

    def test_responses_last_assistant_role(self):
        body = {"input": [{"role": "assistant", "content": "hi"}]}
        assert compute_initiator(body) == "agent"

    def test_responses_last_user_message(self):
        body = {"input": [{"role": "user", "content": "hi"}]}
        assert compute_initiator(body) == "user"

    def test_empty_messages(self):
        body = {"messages": []}
        assert compute_initiator(body) == "user"

    def test_no_messages_key(self):
        body = {"model": "gpt-4o"}
        assert compute_initiator(body) == "user"

    def test_anthropic_tool_result(self):
        body = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "fn", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            ]
        }
        assert compute_initiator(body) == "agent"

    def test_anthropic_plain_user(self):
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
        assert compute_initiator(body) == "user"

    def test_anthropic_user_string_content(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        assert compute_initiator(body) == "user"

    def test_suggestion_mode_string_content(self):
        msg = "[SUGGESTION MODE: Suggest what the user might naturally type next.]"
        body = {"messages": [{"role": "user", "content": msg}]}
        assert compute_initiator(body) == "agent"

    def test_suggestion_mode_content_blocks(self):
        msg = "[SUGGESTION MODE: Suggest what the user might naturally type next.]"
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": msg}]},
            ]
        }
        assert compute_initiator(body) == "agent"

    def test_normal_user_message_not_suggestion_mode(self):
        body = {"messages": [{"role": "user", "content": "Tell me about suggestion mode"}]}
        assert compute_initiator(body) == "user"


# --- _get_headers initiator override ---


class TestGetHeadersInitiatorOverride:
    @pytest.mark.anyio
    async def test_initiator_override_takes_precedence(self, backend):
        """When an explicit initiator is provided, _get_headers must use it."""
        # Body alone would yield "user"
        body = {"messages": [{"role": "user", "content": "hello"}]}
        headers = await backend._get_headers(body, initiator="agent")
        assert headers["X-Initiator"] == "agent"

    @pytest.mark.anyio
    async def test_initiator_override_none_falls_back_to_body(self, backend):
        """When initiator is None, _get_headers computes from body."""
        body = {"messages": [{"role": "user", "content": "hello"}]}
        headers = await backend._get_headers(body, initiator=None)
        assert headers["X-Initiator"] == "user"

    @pytest.mark.anyio
    async def test_no_body_no_initiator(self, backend):
        """When neither body nor initiator is provided, no X-Initiator header."""
        headers = await backend._get_headers()
        assert "X-Initiator" not in headers


class TestHandleMessagesTokenLimit:
    @pytest.mark.anyio
    async def test_non_streaming_token_limit_raises_context_window_error(self, backend):
        """400 with token limit exceeded raises ContextWindowExceededError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = _TOKEN_LIMIT_DETAIL

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(ContextWindowExceededError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.prompt_tokens == 145794
        assert exc_info.value.context_limit == 128000

    @pytest.mark.anyio
    async def test_streaming_token_limit_raises_context_window_error(self, backend):
        """400 with token limit in stream open raises ContextWindowExceededError."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 400
        mock_resp.aread = AsyncMock(return_value=_TOKEN_LIMIT_DETAIL.encode())

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
    async def test_non_streaming_regular_400_still_raises_copilot_http(self, backend):
        """A regular 400 (not token limit) still raises CopilotHttpError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = '{"error":{"message":"invalid model","code":"invalid_request"}}'

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(CopilotHttpError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.status_code == 400


class TestHandleOpenAIMessagesTokenLimit:
    @pytest.mark.anyio
    async def test_token_limit_raises_context_window_error(self, backend):
        """400 with token limit exceeded raises ContextWindowExceededError for OpenAI path."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = _TOKEN_LIMIT_DETAIL

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(ContextWindowExceededError) as exc_info:
                await backend.handle_openai_messages(body, "", False, "gpt-4o")

        assert exc_info.value.prompt_tokens == 145794


# --- _map_http_error ---


class TestMapHttpError:
    def test_401(self, backend):
        resp = backend._map_http_error(401, "unauthorized")
        assert resp.status_code == 401
        body = json.loads(resp.body)
        assert body["error"]["type"] == "authentication_error"

    def test_403(self, backend):
        resp = backend._map_http_error(403, "forbidden")
        assert resp.status_code == 403
        body = json.loads(resp.body)
        assert body["error"]["type"] == "permission_error"

    def test_429(self, backend):
        resp = backend._map_http_error(429, "rate limited")
        assert resp.status_code == 429
        body = json.loads(resp.body)
        assert body["error"]["type"] == "rate_limit_error"

    def test_404(self, backend):
        resp = backend._map_http_error(404, "not found")
        assert resp.status_code == 404
        body = json.loads(resp.body)
        assert body["error"]["type"] == "not_found_error"

    def test_500(self, backend):
        resp = backend._map_http_error(500, "server error")
        assert resp.status_code == 502
        body = json.loads(resp.body)
        assert body["error"]["type"] == "api_error"


# --- handle_messages ---


class TestHandleMessages:
    @pytest.mark.anyio
    async def test_non_streaming_success(self, backend):
        openai_resp = {
            "id": "chat-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = openai_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            body = {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            resp = await backend.handle_messages(body, "req1", False, "claude-sonnet-4.5", "claude-sonnet-4-5-20250929")

        assert resp.status_code == 200
        result = json.loads(resp.body)
        assert result["content"][0]["text"] == "Hello!"
        assert result["type"] == "message"
        # Verify X-Initiator header is set
        posted_headers = mock_post.call_args.kwargs["headers"]
        assert posted_headers["X-Initiator"] == "user"

    @pytest.mark.anyio
    async def test_non_streaming_transient_error_raises(self, backend):
        """429 raises TransientBackendError instead of returning response."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            with pytest.raises(TransientBackendError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.status_code == 429
        assert exc_info.value.backend == "copilot"

    @pytest.mark.anyio
    async def test_non_streaming_500_raises_transient(self, backend):
        """500 raises TransientBackendError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server error"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            with pytest.raises(TransientBackendError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.status_code == 500

    @pytest.mark.anyio
    async def test_non_streaming_401_raises_copilot_http_error(self, backend):
        """401 raises CopilotHttpError (not transient)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "unauthorized"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            with pytest.raises(CopilotHttpError) as exc_info:
                await backend.handle_messages(body, "", False, "m", "x")

        assert exc_info.value.status_code == 401

    @pytest.mark.anyio
    async def test_non_streaming_timeout_raises(self, backend):
        """Timeout is re-raised (not caught internally)."""
        with patch.object(
            backend._client, "post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")
        ):
            body = {
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            with pytest.raises(httpx.TimeoutException):
                await backend.handle_messages(body, "", False, "m", "x")

    @pytest.mark.anyio
    async def test_non_streaming_auth_error_raises(self, backend):
        """Auth errors are re-raised."""
        backend._auth.get_token.side_effect = RuntimeError("bad token")

        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(RuntimeError, match="bad token"):
            await backend.handle_messages(body, "", False, "m", "x")

    @pytest.mark.anyio
    async def test_streaming_returns_streaming_response(self, backend):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }

        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        with (
            patch.object(backend, "_open_stream", new_callable=AsyncMock, return_value=(mock_resp, AsyncMock())),
            patch.object(backend, "_stream_response") as mock_stream,
        ):

            async def fake_gen(*args, **kwargs):
                yield "event: message_start\ndata: {}\n\n"

            mock_stream.return_value = fake_gen()
            resp = await backend.handle_messages(body, "", True, "m", "x")

        assert resp.media_type == "text/event-stream"


class TestHandleMessagesModelNotSupported:
    """handle_messages retries via /responses when /chat/completions returns model_not_supported."""

    @pytest.mark.anyio
    async def test_non_streaming_retries_via_responses(self, backend):
        error_detail = '{"error":{"message":"The requested model is not supported.","code":"model_not_supported"}}'
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = error_detail

        responses_result = MagicMock()

        with (
            patch.object(backend, "_post_with_retry", new_callable=AsyncMock, return_value=mock_resp),
            patch.object(
                backend, "handle_responses_messages", new_callable=AsyncMock, return_value=responses_result
            ) as mock_responses,
        ):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            result = await backend.handle_messages(body, "req1", False, "gpt-5.1-codex", "gpt-5.1-codex")

        assert result is responses_result
        mock_responses.assert_called_once()

    @pytest.mark.anyio
    async def test_streaming_retries_via_responses(self, backend):
        error_detail = '{"error":{"message":"The requested model is not supported.","code":"model_not_supported"}}'
        mock_open_stream = AsyncMock(side_effect=CopilotHttpError(400, error_detail))
        responses_result = MagicMock()

        with (
            patch.object(backend, "_open_stream", mock_open_stream),
            patch.object(
                backend, "handle_responses_messages", new_callable=AsyncMock, return_value=responses_result
            ) as mock_responses,
        ):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            result = await backend.handle_messages(body, "", True, "gpt-5.1-codex", "gpt-5.1-codex")

        assert result is responses_result
        mock_responses.assert_called_once()

    @pytest.mark.anyio
    async def test_non_model_error_still_raises(self, backend):
        """Non model_not_supported 400 errors are not retried."""
        error_detail = '{"error":{"message":"Bad request","code":"invalid_request_body"}}'
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = error_detail

        with patch.object(backend, "_post_with_retry", new_callable=AsyncMock, return_value=mock_resp):
            body = {"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(CopilotHttpError):
                await backend.handle_messages(body, "", False, "m", "x")


# --- _open_stream ---


class TestOpenStream:
    @pytest.mark.anyio
    async def test_success(self, backend):
        """Successful stream open returns (response, context_manager)."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend._client, "stream", return_value=mock_stream_ctx):
            resp, cm = await backend._open_stream({"model": "m"}, "")

        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_transient_error_raises(self, backend):
        """429 from Copilot stream raises TransientBackendError."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 429
        mock_resp.aread = AsyncMock(return_value=b"rate limited")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(backend._client, "stream", return_value=mock_stream_ctx),
            pytest.raises(TransientBackendError) as exc_info,
        ):
            await backend._open_stream({"model": "m"}, "")

        assert exc_info.value.status_code == 429
        mock_stream_ctx.__aexit__.assert_called_once()

    @pytest.mark.anyio
    async def test_non_transient_error_raises_copilot_http(self, backend):
        """401 from Copilot stream raises CopilotHttpError."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 401
        mock_resp.aread = AsyncMock(return_value=b"unauthorized")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(backend._client, "stream", return_value=mock_stream_ctx),
            pytest.raises(CopilotHttpError) as exc_info,
        ):
            await backend._open_stream({"model": "m"}, "")

        assert exc_info.value.status_code == 401


# --- _stream_response ---


class TestStreamResponse:
    @pytest.mark.anyio
    async def test_translates_chunks(self, backend):
        """Test that streaming translates OpenAI chunks to Anthropic SSE events."""

        async def mock_aiter_lines():
            yield 'data: {"id":"c1","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}'
            yield 'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}'
            yield "data: [DONE]"

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        events = []
        async for event in backend._stream_response(mock_resp, mock_stream_ctx, "model", ""):
            events.append(event)

        all_text = "".join(events)
        assert "message_start" in all_text
        assert "text_delta" in all_text
        assert "message_stop" in all_text
        mock_stream_ctx.__aexit__.assert_called_once()

    @pytest.mark.anyio
    async def test_timeout_error(self, backend):
        """Test that timeout during streaming produces error event."""

        async def mock_aiter_lines():
            raise httpx.TimeoutException("timed out")
            yield  # noqa: RET503 - unreachable, needed to make it an async generator

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        events = []
        async for event in backend._stream_response(mock_resp, mock_stream_ctx, "model", ""):
            events.append(event)

        all_text = "".join(events)
        assert "error" in all_text
        assert "timed out" in all_text
        mock_stream_ctx.__aexit__.assert_called_once()


# --- handle_openai_messages ---


# --- _normalize_openai_response ---


class TestNormalizeOpenAIResponse:
    def test_adds_missing_index_to_choices(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
        _normalize_openai_response(resp)
        assert resp["choices"][0]["index"] == 0

    def test_preserves_existing_index(self):
        resp = {"choices": [{"index": 2, "message": {"content": "hi"}, "finish_reason": "stop"}]}
        _normalize_openai_response(resp)
        assert resp["choices"][0]["index"] == 2

    def test_adds_missing_object_field(self):
        resp = {"choices": []}
        _normalize_openai_response(resp)
        assert resp["object"] == "chat.completion"

    def test_streaming_object_type(self):
        resp = {"choices": []}
        _normalize_openai_response(resp, streaming=True)
        assert resp["object"] == "chat.completion.chunk"

    def test_adds_missing_created_and_id(self):
        resp = {"choices": []}
        _normalize_openai_response(resp)
        assert "created" in resp
        assert resp["id"].startswith("chatcmpl-")

    def test_preserves_existing_fields(self):
        resp = {
            "id": "original-id",
            "object": "chat.completion",
            "created": 12345,
            "choices": [{"index": 0, "message": {"content": "hi"}}],
        }
        _normalize_openai_response(resp)
        assert resp["id"] == "original-id"
        assert resp["created"] == 12345


# --- handle_openai_messages ---


class TestHandleOpenAIMessages:
    @pytest.mark.anyio
    async def test_non_streaming_success(self, backend):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = openai_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            resp = await backend.handle_openai_messages(body, "req1", False, "gpt-4o")

        assert resp.status_code == 200
        result = json.loads(resp.body)
        assert result["choices"][0]["message"]["content"] == "Hello!"
        # Verify model was overridden in the posted body
        posted_body = mock_post.call_args.kwargs["json"]
        assert posted_body["model"] == "gpt-4o"
        # Verify X-Initiator header is set
        posted_headers = mock_post.call_args.kwargs["headers"]
        assert posted_headers["X-Initiator"] == "user"

    @pytest.mark.anyio
    async def test_model_override(self, backend):
        """Verify the model is overridden to the copilot model name."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = openai_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            body = {"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]}
            await backend.handle_openai_messages(body, "", False, "claude-sonnet-4.5")

        posted_body = mock_post.call_args.kwargs["json"]
        assert posted_body["model"] == "claude-sonnet-4.5"

    @pytest.mark.anyio
    async def test_429_raises_transient_error(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(TransientBackendError) as exc_info:
                await backend.handle_openai_messages(body, "", False, "gpt-4o")

        assert exc_info.value.status_code == 429
        assert exc_info.value.backend == "copilot"

    @pytest.mark.anyio
    async def test_401_raises_copilot_http_error(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "unauthorized"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            with pytest.raises(CopilotHttpError) as exc_info:
                await backend.handle_openai_messages(body, "", False, "gpt-4o")

        assert exc_info.value.status_code == 401

    @pytest.mark.anyio
    async def test_streaming_returns_streaming_response(self, backend):
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}

        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        with (
            patch.object(backend, "_open_stream", new_callable=AsyncMock, return_value=(mock_resp, AsyncMock())),
            patch.object(backend, "_stream_openai_response") as mock_stream,
        ):

            async def fake_gen(*args, **kwargs):
                yield "data: {}\n\n"

            mock_stream.return_value = fake_gen()
            resp = await backend.handle_openai_messages(body, "", True, "gpt-4o")

        assert resp.media_type == "text/event-stream"

    @pytest.mark.anyio
    async def test_non_streaming_normalizes_response(self, backend):
        """Copilot responses missing OpenAI spec fields get normalized."""
        openai_resp = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = openai_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            resp = await backend.handle_openai_messages(body, "req1", False, "gpt-4o")

        result = json.loads(resp.body)
        assert result["choices"][0]["index"] == 0
        assert result["object"] == "chat.completion"
        assert "created" in result

    @pytest.mark.anyio
    async def test_does_not_mutate_original_body(self, backend):
        """Original body dict should not be modified."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = openai_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {"model": "original-model", "messages": [{"role": "user", "content": "hi"}]}
            await backend.handle_openai_messages(body, "", False, "gpt-4o")

        assert body["model"] == "original-model"


# --- _stream_openai_response ---


class TestStreamOpenAIResponse:
    @pytest.mark.anyio
    async def test_normalizes_streaming_chunks(self, backend):
        """Streaming chunks missing index/object fields get normalized."""

        async def mock_aiter_lines():
            yield 'data: {"id":"c1","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}'
            yield "data: [DONE]"

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        events = []
        async for event in backend._stream_openai_response(mock_resp, mock_stream_ctx, ""):
            events.append(event)

        # First event should have the normalized chunk
        chunk_data = json.loads(events[0].replace("data: ", "").strip())
        assert chunk_data["choices"][0]["index"] == 0
        assert chunk_data["object"] == "chat.completion.chunk"

    @pytest.mark.anyio
    async def test_forwards_lines_as_is(self, backend):
        """Lines are forwarded without Anthropic translation."""

        async def mock_aiter_lines():
            yield 'data: {"id":"c1","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}'
            yield 'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        events = []
        async for event in backend._stream_openai_response(mock_resp, mock_stream_ctx, ""):
            events.append(event)

        all_text = "".join(events)
        # Should contain raw OpenAI content, not Anthropic events
        assert "Hi" in all_text
        assert "finish_reason" in all_text
        assert "[DONE]" in all_text
        # Should NOT contain Anthropic-style events
        assert "message_start" not in all_text
        assert "text_delta" not in all_text
        mock_stream_ctx.__aexit__.assert_called_once()

    @pytest.mark.anyio
    async def test_timeout_error(self, backend):
        async def mock_aiter_lines():
            raise httpx.TimeoutException("timed out")
            yield  # noqa: RET503

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        events = []
        async for event in backend._stream_openai_response(mock_resp, mock_stream_ctx, ""):
            events.append(event)

        all_text = "".join(events)
        assert "timed out" in all_text
        assert "server_error" in all_text
        mock_stream_ctx.__aexit__.assert_called_once()

    @pytest.mark.anyio
    async def test_empty_lines_skipped(self, backend):
        async def mock_aiter_lines():
            yield ""
            yield 'data: {"id":"c1","choices":[{"delta":{"content":"Hi"}}]}'
            yield ""

        mock_resp = AsyncMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        events = []
        async for event in backend._stream_openai_response(mock_resp, mock_stream_ctx, ""):
            events.append(event)

        # Only the non-empty line should produce output
        assert len(events) == 1
        assert "Hi" in events[0]


# --- close ---


class TestClose:
    @pytest.mark.anyio
    async def test_closes_client_and_auth(self, backend):
        with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
            await backend.close()
            mock_close.assert_called_once()
            backend._auth.close.assert_called_once()


# --- _parse_retry_after ---


class TestParseRetryAfter:
    def test_valid_integer(self):
        headers = httpx.Headers({"retry-after": "5"})
        assert _parse_retry_after(headers) == 5.0

    def test_valid_float(self):
        headers = httpx.Headers({"retry-after": "2.5"})
        assert _parse_retry_after(headers) == 2.5

    def test_missing_header(self):
        headers = httpx.Headers({})
        assert _parse_retry_after(headers) is None

    def test_invalid_value(self):
        headers = httpx.Headers({"retry-after": "not-a-number"})
        assert _parse_retry_after(headers) is None


# --- TokenBucket ---


class TestTokenBucket:
    @pytest.mark.anyio
    async def test_immediate_acquires_up_to_rate(self):
        """Acquiring up to `rate` tokens should return 0.0 wait time."""
        bucket = TokenBucket(5)
        for _ in range(5):
            waited = await bucket.acquire()
            assert waited == 0.0

    @pytest.mark.anyio
    async def test_exceeding_rate_causes_wait(self):
        """Exceeding the rate should cause a non-zero wait."""
        bucket = TokenBucket(2)
        # Drain the bucket
        await bucket.acquire()
        await bucket.acquire()
        # The third should wait
        with patch("claudegate.copilot_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            waited = await bucket.acquire()
            assert waited > 0.0
            mock_sleep.assert_called()

    @pytest.mark.anyio
    async def test_refill_over_time(self):
        """After time passes, tokens refill and acquires are immediate again."""
        import time

        bucket = TokenBucket(10)
        # Drain all tokens
        for _ in range(10):
            await bucket.acquire()
        # Simulate time passing (6 seconds = 1 token at 10/min rate)
        bucket._last_refill = time.monotonic() - 6.0
        waited = await bucket.acquire()
        assert waited == 0.0

    @pytest.mark.anyio
    async def test_disabled_when_rate_zero(self, mock_copilot_auth):
        """Rate limiter should be None when max_rate=0."""
        b = CopilotBackend(mock_copilot_auth, timeout=30, retry_max=0, max_rate=0)
        assert b._rate_limiter is None


# --- _post_with_retry ---


class TestPostWithRetry:
    @pytest.fixture
    def retry_backend(self, mock_copilot_auth):
        """CopilotBackend with retry enabled, rate limiting disabled."""
        return CopilotBackend(mock_copilot_auth, timeout=30, retry_max=2, retry_base_delay=0.01, max_rate=0)

    @pytest.mark.anyio
    async def test_retry_on_429_then_success(self, retry_backend):
        """First POST returns 429, second returns 200 → success with 2 requests."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = httpx.Headers({})

        resp_200 = MagicMock()
        resp_200.status_code = 200

        with patch.object(
            retry_backend._client, "post", new_callable=AsyncMock, side_effect=[resp_429, resp_200]
        ) as mock_post:
            result = await retry_backend._post_with_retry(
                "https://example.com/api", {"Authorization": "Bearer tok"}, {"model": "test"}, "[test] "
            )
            assert result.status_code == 200
            assert mock_post.call_count == 2

    @pytest.mark.anyio
    async def test_all_retries_exhausted_returns_429(self, retry_backend):
        """All attempts return 429 → final 429 response returned (caller raises)."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = httpx.Headers({})

        with patch.object(retry_backend._client, "post", new_callable=AsyncMock, return_value=resp_429) as mock_post:
            result = await retry_backend._post_with_retry(
                "https://example.com/api", {"Authorization": "Bearer tok"}, {"model": "test"}, "[test] "
            )
            assert result.status_code == 429
            # 1 initial + 2 retries = 3
            assert mock_post.call_count == 3

    @pytest.mark.anyio
    async def test_retry_after_header_respected(self, retry_backend):
        """retry-after header value is used as delay."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = httpx.Headers({"retry-after": "1"})

        resp_200 = MagicMock()
        resp_200.status_code = 200

        with (
            patch.object(retry_backend._client, "post", new_callable=AsyncMock, side_effect=[resp_429, resp_200]),
            patch("claudegate.copilot_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await retry_backend._post_with_retry(
                "https://example.com/api", {"Authorization": "Bearer tok"}, {"model": "test"}, "[test] "
            )
            assert result.status_code == 200
            # Should have slept with the retry-after value (1.0), not the base delay
            mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.anyio
    async def test_500_not_retried(self, retry_backend):
        """500 error is not retried — returned immediately for fallback."""
        resp_500 = MagicMock()
        resp_500.status_code = 500

        with patch.object(retry_backend._client, "post", new_callable=AsyncMock, return_value=resp_500) as mock_post:
            result = await retry_backend._post_with_retry(
                "https://example.com/api", {"Authorization": "Bearer tok"}, {"model": "test"}, "[test] "
            )
            assert result.status_code == 500
            assert mock_post.call_count == 1

    @pytest.mark.anyio
    async def test_no_retry_when_max_zero(self, mock_copilot_auth):
        """With retry_max=0, 429 is returned immediately."""
        b = CopilotBackend(mock_copilot_auth, timeout=30, retry_max=0, max_rate=0)
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = httpx.Headers({})

        with patch.object(b._client, "post", new_callable=AsyncMock, return_value=resp_429) as mock_post:
            result = await b._post_with_retry(
                "https://example.com/api", {"Authorization": "Bearer tok"}, {"model": "test"}, ""
            )
            assert result.status_code == 429
            assert mock_post.call_count == 1

    @pytest.mark.anyio
    async def test_rate_limiter_called_before_request(self, mock_copilot_auth):
        """When rate limiter is enabled, acquire() is called before each POST."""
        b = CopilotBackend(mock_copilot_auth, timeout=30, retry_max=0, max_rate=10)
        resp_200 = MagicMock()
        resp_200.status_code = 200

        with (
            patch.object(b._client, "post", new_callable=AsyncMock, return_value=resp_200),
            patch.object(b._rate_limiter, "acquire", new_callable=AsyncMock, return_value=0.0) as mock_acquire,
        ):
            await b._post_with_retry("https://example.com/api", {"Authorization": "Bearer tok"}, {"model": "test"}, "")
            mock_acquire.assert_called_once()

    @pytest.mark.anyio
    async def test_delay_capped_at_30s(self, retry_backend):
        """Exponential backoff delay is capped at 30 seconds."""
        # With base_delay=0.01 and 2 retries, delay won't actually exceed 30s,
        # but test with a large base_delay to confirm capping
        b = CopilotBackend(retry_backend._auth, timeout=30, retry_max=1, retry_base_delay=50.0, max_rate=0)
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = httpx.Headers({})

        resp_200 = MagicMock()
        resp_200.status_code = 200

        with (
            patch.object(b._client, "post", new_callable=AsyncMock, side_effect=[resp_429, resp_200]),
            patch("claudegate.copilot_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await b._post_with_retry(
                "https://example.com/api", {"Authorization": "Bearer tok"}, {"model": "test"}, ""
            )
            assert result.status_code == 200
            # 50.0 * 2^0 = 50.0, capped to 30.0
            mock_sleep.assert_called_once_with(30.0)


# --- _open_stream_with_retry ---


class TestOpenStreamWithRetry:
    @pytest.fixture
    def retry_backend(self, mock_copilot_auth):
        """CopilotBackend with retry enabled, rate limiting disabled."""
        return CopilotBackend(mock_copilot_auth, timeout=30, retry_max=2, retry_base_delay=0.01, max_rate=0)

    @pytest.mark.anyio
    async def test_stream_retry_on_429_then_success(self, retry_backend):
        """Stream open: first 429, second 200 → success."""
        # First call: 429
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = httpx.Headers({})
        resp_429.aread = AsyncMock(return_value=b"rate limited")
        stream_cm_429 = AsyncMock()
        stream_cm_429.__aenter__ = AsyncMock(return_value=resp_429)
        stream_cm_429.__aexit__ = AsyncMock(return_value=False)

        # Second call: 200
        resp_200 = MagicMock()
        resp_200.status_code = 200
        stream_cm_200 = AsyncMock()
        stream_cm_200.__aenter__ = AsyncMock(return_value=resp_200)

        call_count = 0

        def fake_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stream_cm_429
            return stream_cm_200

        with patch.object(retry_backend._client, "stream", side_effect=fake_stream):
            resp, cm = await retry_backend._open_stream_with_retry(
                "https://example.com/api", {"model": "test"}, "[test] "
            )
            assert resp.status_code == 200
            assert call_count == 2

    @pytest.mark.anyio
    async def test_stream_500_not_retried(self, retry_backend):
        """Stream open: 500 is not retried, returned immediately."""
        resp_500 = MagicMock()
        resp_500.status_code = 500
        stream_cm_500 = AsyncMock()
        stream_cm_500.__aenter__ = AsyncMock(return_value=resp_500)

        call_count = 0

        def fake_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return stream_cm_500

        with patch.object(retry_backend._client, "stream", side_effect=fake_stream):
            resp, cm = await retry_backend._open_stream_with_retry(
                "https://example.com/api", {"model": "test"}, "[test] "
            )
            assert resp.status_code == 500
            assert call_count == 1


# --- handle_responses_passthrough ---


class TestHandleResponsesPassthrough:
    """Tests for handle_responses_passthrough tool passthrough behavior."""

    @pytest.mark.anyio
    async def test_server_side_tools_preserved(self, backend):
        """Built-in tools like web_search_preview are passed through to Copilot."""
        body = {
            "model": "gpt-4o",
            "input": "Search for latest news",
            "tools": [
                {"type": "web_search_preview"},
                {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
            ],
        }
        responses_resp = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Here are results."}]}],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = responses_resp

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            resp = await backend.handle_responses_passthrough(body, "req1", False)

        assert resp.status_code == 200
        # Verify tools were sent as-is (not stripped)
        posted_body = mock_post.call_args.kwargs["json"]
        assert len(posted_body["tools"]) == 2
        assert posted_body["tools"][0] == {"type": "web_search_preview"}
        assert posted_body["tools"][1]["type"] == "function"

    @pytest.mark.anyio
    async def test_function_only_tools_work(self, backend):
        """Requests with only function tools still work as before."""
        body = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [
                {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
            ],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "resp_123", "object": "response", "status": "completed", "output": []}

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            resp = await backend.handle_responses_passthrough(body, "req1", False)

        assert resp.status_code == 200
        posted_body = mock_post.call_args.kwargs["json"]
        assert len(posted_body["tools"]) == 1
        assert posted_body["tools"][0]["type"] == "function"

    @pytest.mark.anyio
    async def test_no_tools_passthrough(self, backend):
        """Requests without tools pass through fine."""
        body = {
            "model": "gpt-4o",
            "input": "Hello",
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "resp_123", "object": "response", "status": "completed", "output": []}

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            resp = await backend.handle_responses_passthrough(body, "req1", False)

        assert resp.status_code == 200
        posted_body = mock_post.call_args.kwargs["json"]
        assert "tools" not in posted_body
