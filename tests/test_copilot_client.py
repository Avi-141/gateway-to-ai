"""Tests for claudegate/copilot_client.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claudegate.copilot_client import CopilotBackend, _normalize_openai_response, _parse_token_limit_error, compute_initiator
from claudegate.errors import ContextWindowExceededError, CopilotHttpError, TransientBackendError

# Copilot token limit error payload used across multiple tests
_TOKEN_LIMIT_DETAIL = (
    '{"error":{"message":"prompt token count of 145794 exceeds the limit of 128000",'
    '"code":"model_max_prompt_tokens_exceeded"}}'
)


@pytest.fixture
def backend(mock_copilot_auth):
    """CopilotBackend with mocked auth."""
    return CopilotBackend(mock_copilot_auth, timeout=30)


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


# --- handle_messages token limit ---


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
