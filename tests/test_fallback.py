"""Tests for cross-backend fallback logic."""

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claudegate.app import app
from claudegate.errors import TransientBackendError


@pytest.fixture
def fallback_client():
    """HTTPX async client wired to the FastAPI app (no lifespan)."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def minimal_body() -> dict[str, object]:
    return {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }


def _mock_bedrock_success():
    """Return a mock bedrock client whose invoke_model returns a valid response."""
    mock = MagicMock()
    result = {
        "type": "message",
        "content": [{"type": "text", "text": "Hello from bedrock"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock.invoke_model.return_value = {"body": BytesIO(json.dumps(result).encode())}
    return mock


def _mock_copilot_backend_success():
    """Return a mock CopilotBackend whose handle_messages returns success."""
    from fastapi.responses import JSONResponse

    mock = AsyncMock()
    result = {
        "type": "message",
        "content": [{"type": "text", "text": "Hello from copilot"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock.handle_messages.return_value = JSONResponse(content=result)
    mock.close = AsyncMock()
    return mock


class TestFallbackNonStreaming:
    """Non-streaming fallback scenarios."""

    @pytest.mark.anyio
    async def test_primary_429_fallback_succeeds(self, fallback_client, minimal_body):
        """Primary bedrock 429 -> fallback copilot succeeds."""
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "rate limited")

        mock_copilot = _mock_copilot_backend_success()

        with (
            patch("claudegate.app.BACKEND_TYPE", "bedrock"),
            patch("claudegate.app.FALLBACK_BACKEND", "copilot"),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await fallback_client.post("/v1/messages", json=minimal_body)

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"][0]["text"] == "Hello from copilot"

    @pytest.mark.anyio
    async def test_primary_500_fallback_also_fails(self, fallback_client, minimal_body):
        """Primary bedrock 500 -> fallback copilot also fails -> returns fallback error."""
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("InternalServerException", "internal error")

        mock_copilot = AsyncMock()
        mock_copilot.handle_messages.side_effect = TransientBackendError(
            503, "api_error", "copilot also down", "copilot"
        )
        mock_copilot.close = AsyncMock()

        with (
            patch("claudegate.app.BACKEND_TYPE", "bedrock"),
            patch("claudegate.app.FALLBACK_BACKEND", "copilot"),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await fallback_client.post("/v1/messages", json=minimal_body)

        assert resp.status_code == 503
        body = resp.json()
        assert body["type"] == "error"

    @pytest.mark.anyio
    async def test_primary_401_no_fallback(self, fallback_client, minimal_body):
        """Primary bedrock 401 (expired creds) -> no fallback (not transient)."""
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ExpiredTokenException", "expired")

        with (
            patch("claudegate.app.BACKEND_TYPE", "bedrock"),
            patch("claudegate.app.FALLBACK_BACKEND", "copilot"),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app.reset_bedrock_client"),
        ):
            resp = await fallback_client.post("/v1/messages", json=minimal_body)

        # 401 is not transient, so returned directly without fallback
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_no_fallback_configured(self, fallback_client, minimal_body):
        """Transient error with no fallback returns directly."""
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "rate limited")

        with (
            patch("claudegate.app.BACKEND_TYPE", "bedrock"),
            patch("claudegate.app.FALLBACK_BACKEND", ""),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
        ):
            resp = await fallback_client.post("/v1/messages", json=minimal_body)

        assert resp.status_code == 429

    @pytest.mark.anyio
    async def test_primary_succeeds_no_fallback_called(self, fallback_client, minimal_body):
        """Primary succeeds -> fallback never called."""
        mock_bedrock = _mock_bedrock_success()
        mock_copilot = _mock_copilot_backend_success()

        with (
            patch("claudegate.app.BACKEND_TYPE", "bedrock"),
            patch("claudegate.app.FALLBACK_BACKEND", "copilot"),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await fallback_client.post("/v1/messages", json=minimal_body)

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"][0]["text"] == "Hello from bedrock"
        mock_copilot.handle_messages.assert_not_called()

    @pytest.mark.anyio
    async def test_copilot_primary_bedrock_fallback(self, fallback_client, minimal_body):
        """Primary copilot 429 -> fallback bedrock succeeds."""
        mock_copilot = AsyncMock()
        mock_copilot.handle_messages.side_effect = TransientBackendError(
            429, "rate_limit_error", "rate limited", "copilot"
        )
        mock_copilot.close = AsyncMock()

        mock_bedrock = _mock_bedrock_success()

        with (
            patch("claudegate.app.BACKEND_TYPE", "copilot"),
            patch("claudegate.app.FALLBACK_BACKEND", "bedrock"),
            patch("claudegate.app._copilot_backend", mock_copilot),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
        ):
            resp = await fallback_client.post("/v1/messages", json=minimal_body)

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"][0]["text"] == "Hello from bedrock"


class TestFallbackStreaming:
    """Streaming fallback scenarios (pre-stream errors only)."""

    @pytest.mark.anyio
    async def test_streaming_pre_stream_429_fallback(self, fallback_client, minimal_body):
        """Primary bedrock stream fails pre-stream (429) -> fallback copilot stream succeeds."""
        from fastapi.responses import StreamingResponse

        # Bedrock invoke_model_with_response_stream raises ThrottlingException
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model_with_response_stream.side_effect = make_client_error(
            "ThrottlingException", "rate limited"
        )

        # Copilot returns a streaming response
        async def fake_stream():
            yield "event: message_start\ndata: {}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        mock_copilot = AsyncMock()
        mock_copilot.handle_messages.return_value = StreamingResponse(fake_stream(), media_type="text/event-stream")
        mock_copilot.close = AsyncMock()

        minimal_body["stream"] = True

        with (
            patch("claudegate.app.BACKEND_TYPE", "bedrock"),
            patch("claudegate.app.FALLBACK_BACKEND", "copilot"),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await fallback_client.post("/v1/messages", json=minimal_body)

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


def _parse_backends(env_val: str) -> list[str]:
    """Replicate the BACKEND env var parsing logic from config.py."""
    return [b.strip() for b in env_val.split(",") if b.strip()]


class TestConfigParsing:
    """Test BACKEND env var parsing."""

    def test_single_backend(self):
        """Single backend value works as before."""
        backends = _parse_backends("copilot")
        assert backends == ["copilot"]

    def test_comma_separated(self):
        """Comma-separated value produces primary and fallback."""
        backends = _parse_backends("copilot,bedrock")
        assert backends[0] == "copilot"
        assert backends[1] == "bedrock"

    def test_with_spaces(self):
        """Spaces around commas are trimmed."""
        backends = _parse_backends("copilot , bedrock")
        assert backends[0] == "copilot"
        assert backends[1] == "bedrock"

    def test_empty_trailing(self):
        """Trailing comma doesn't produce empty fallback."""
        backends = _parse_backends("bedrock,")
        assert len(backends) == 1
        assert backends[0] == "bedrock"


def make_client_error(code: str = "InternalError", message: str = "Something failed"):
    """Create a botocore ClientError with the given error code."""
    from botocore.exceptions import ClientError

    return ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="InvokeModel",
    )
