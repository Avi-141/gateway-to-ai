"""Tests for claudegate/copilot_client.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claudegate.copilot_client import CopilotBackend
from claudegate.errors import CopilotHttpError, TransientBackendError


@pytest.fixture
def backend(mock_copilot_auth):
    """CopilotBackend with mocked auth."""
    return CopilotBackend(mock_copilot_auth, timeout=30)


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

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
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
