"""Tests for claudegate/copilot_client.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claudegate.copilot_client import CopilotBackend


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
            "choices": [
                {"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
            ],
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
    async def test_non_streaming_http_error(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"

        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
            body = {
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            resp = await backend.handle_messages(body, "", False, "m", "x")

        assert resp.status_code == 429

    @pytest.mark.anyio
    async def test_non_streaming_timeout(self, backend):
        with patch.object(
            backend._client, "post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")
        ):
            body = {
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            resp = await backend.handle_messages(body, "", False, "m", "x")

        assert resp.status_code == 504

    @pytest.mark.anyio
    async def test_non_streaming_auth_error(self, backend):
        backend._auth.get_token.side_effect = RuntimeError("bad token")

        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        resp = await backend.handle_messages(body, "", False, "m", "x")

        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_non_streaming_unexpected_error(self, backend):
        with patch.object(backend._client, "post", new_callable=AsyncMock, side_effect=ValueError("unexpected")):
            body = {
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
            resp = await backend.handle_messages(body, "", False, "m", "x")

        assert resp.status_code == 500

    @pytest.mark.anyio
    async def test_streaming_returns_streaming_response(self, backend):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        # Mock _stream_response to yield some data
        async def mock_stream(*args, **kwargs):
            yield "event: message_start\ndata: {}\n\n"

        with patch.object(backend, "_stream_response", mock_stream):
            resp = await backend.handle_messages(body, "", True, "m", "x")

        assert resp.media_type == "text/event-stream"


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
        mock_resp.status_code = 200
        mock_resp.aiter_lines = mock_aiter_lines

        # Use async context manager
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend._client, "stream", return_value=mock_stream_ctx):
            events = []
            async for event in backend._stream_response({"model": "m"}, "model", ""):
                events.append(event)

        all_text = "".join(events)
        assert "message_start" in all_text
        assert "text_delta" in all_text
        assert "message_stop" in all_text

    @pytest.mark.anyio
    async def test_non_200_error(self, backend):
        """Test that non-200 status from Copilot produces error event."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 500
        mock_resp.aread = AsyncMock(return_value=b"server error")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend._client, "stream", return_value=mock_stream_ctx):
            events = []
            async for event in backend._stream_response({"model": "m"}, "model", ""):
                events.append(event)

        all_text = "".join(events)
        assert "error" in all_text

    @pytest.mark.anyio
    async def test_timeout_error(self, backend):
        """Test that timeout during streaming produces error event."""

        with patch.object(
            backend._client, "stream", side_effect=httpx.TimeoutException("timed out")
        ):
            events = []
            async for event in backend._stream_response({"model": "m"}, "model", ""):
                events.append(event)

        all_text = "".join(events)
        assert "error" in all_text
        assert "timed out" in all_text


# --- close ---


class TestClose:
    @pytest.mark.anyio
    async def test_closes_client_and_auth(self, backend):
        with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
            await backend.close()
            mock_close.assert_called_once()
            backend._auth.close.assert_called_once()
