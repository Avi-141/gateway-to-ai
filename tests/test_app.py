"""Tests for claudegate/app.py."""

import json
import sys
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from claudegate.app import _count_content_tokens, _error_response, _validate_request
from claudegate.errors import CopilotHttpError, TransientBackendError
from tests.conftest import make_client_error

# Get the actual module object (not the FastAPI app re-exported by __init__)
app_module = sys.modules["claudegate.app"]


# --- _error_response ---


class TestErrorResponse:
    def test_structure(self):
        resp = _error_response(400, "invalid_request_error", "bad request")
        assert resp.status_code == 400
        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == "bad request"

    def test_different_status(self):
        resp = _error_response(500, "api_error", "oops")
        assert resp.status_code == 500


# --- _validate_request ---


class TestValidateRequest:
    def test_missing_model(self):
        resp = _validate_request({"max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]})
        assert resp is not None
        assert resp.status_code == 400

    def test_missing_max_tokens(self):
        resp = _validate_request({"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        assert resp is not None
        assert resp.status_code == 400

    def test_missing_messages(self):
        resp = _validate_request({"model": "x", "max_tokens": 100})
        assert resp is not None
        assert resp.status_code == 400

    def test_non_array_messages(self):
        resp = _validate_request({"model": "x", "max_tokens": 100, "messages": "not an array"})
        assert resp is not None
        body = json.loads(resp.body)
        assert "array" in body["error"]["message"]

    def test_empty_messages(self):
        resp = _validate_request({"model": "x", "max_tokens": 100, "messages": []})
        assert resp is not None
        body = json.loads(resp.body)
        assert "empty" in body["error"]["message"]

    def test_valid_request(self):
        resp = _validate_request({"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]})
        assert resp is None


# --- _count_content_tokens ---


class TestCountContentTokens:
    def test_string(self):
        count = _count_content_tokens("hello world")
        assert count > 0

    def test_text_blocks(self):
        count = _count_content_tokens([{"type": "text", "text": "hello"}])
        assert count > 0

    def test_tool_use(self):
        count = _count_content_tokens([{"type": "tool_use", "name": "fn", "input": {"a": 1}}])
        assert count > 0

    def test_tool_result(self):
        count = _count_content_tokens([{"type": "tool_result", "content": "result"}])
        assert count > 0

    def test_empty(self):
        assert _count_content_tokens([]) == 0
        assert _count_content_tokens(123) == 0


# --- POST /v1/messages ---


class TestMessagesRoute:
    @pytest.mark.anyio
    async def test_invalid_json(self, async_client):
        resp = await async_client.post(
            "/v1/messages",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert "Invalid JSON" in resp.json()["error"]["message"]

    @pytest.mark.anyio
    async def test_validation_error(self, async_client):
        resp = await async_client.post("/v1/messages", json={"model": "x"})
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_non_streaming_success(
        self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch
    ):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_response = {
            "body": BytesIO(
                json.dumps(
                    {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi!"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ).encode()
            )
        }
        mock_bedrock_client.invoke_model.return_value = mock_response

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 200
        body = resp.json()
        assert body["content"][0]["text"] == "Hi!"

    @pytest.mark.anyio
    async def test_streaming_returns_sse(
        self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch
    ):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        minimal_anthropic_request["stream"] = True

        # Mock streaming response
        chunk_data = json.dumps({"type": "message_start", "message": {}}).encode()
        mock_event = {"chunk": {"bytes": chunk_data}}
        mock_bedrock_client.invoke_model_with_response_stream.return_value = {"body": [mock_event]}

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.anyio
    async def test_expired_token(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = make_client_error("ExpiredTokenException", "Token expired")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_validation_exception(
        self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch
    ):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = make_client_error("ValidationException", "Bad input")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_access_denied(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = make_client_error("AccessDeniedException", "No access")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 403

    @pytest.mark.anyio
    async def test_throttling(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = make_client_error("ThrottlingException", "Rate limited")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 429

    @pytest.mark.anyio
    async def test_model_timeout(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = make_client_error("ModelTimeoutException", "Timed out")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 504

    @pytest.mark.anyio
    async def test_generic_client_error(
        self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch
    ):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = make_client_error("SomeOtherError", "Unknown")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 500

    @pytest.mark.anyio
    async def test_read_timeout_error(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = ReadTimeoutError(endpoint_url="https://bedrock.example.com")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 504

    @pytest.mark.anyio
    async def test_unexpected_exception(
        self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch
    ):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = RuntimeError("kaboom")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 500

    @pytest.mark.anyio
    async def test_non_claude_model_via_copilot(self, async_client, monkeypatch):
        """Non-Claude models (GPT, Gemini) should route correctly via /v1/messages with Copilot."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")

        mock_backend = AsyncMock()
        from fastapi.responses import JSONResponse

        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "GPT response"}],
            "model": "gpt-5.1-codex",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        mock_backend.handle_messages.return_value = JSONResponse(content=anthropic_resp)
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.post(
            "/v1/messages",
            json={
                "model": "gpt-5.1-codex",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        # Verify handle_messages was called with the correct model
        mock_backend.handle_messages.assert_called_once()
        call_kwargs = mock_backend.handle_messages.call_args
        # openai_model (copilot_model) should be "gpt-5.1-codex"
        assert call_kwargs[0][3] == "gpt-5.1-codex"  # copilot_model positional arg
        # anthropic_model should be "gpt-5.1-codex" (label for response)
        assert call_kwargs[0][4] == "gpt-5.1-codex"  # anthropic_model positional arg

    @pytest.mark.anyio
    async def test_copilot_http_error(self, async_client, monkeypatch):
        """CopilotHttpError from primary should return error response."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        mock_backend = AsyncMock()
        mock_backend.handle_messages.side_effect = CopilotHttpError(403, "Forbidden")
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["message"] == "Forbidden"

    @pytest.mark.anyio
    async def test_runtime_error_auth(self, async_client, monkeypatch):
        """RuntimeError from copilot should return 401 auth error."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        monkeypatch.setattr(app_module, "_copilot_backend", None)

        resp = await async_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_generic_exception(self, async_client, monkeypatch):
        """Unexpected exception from copilot primary should return 500."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        mock_backend = AsyncMock()
        mock_backend.handle_messages.side_effect = ValueError("unexpected")
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 500
        assert "unexpected" in resp.json()["error"]["message"]


# --- Streaming Error Handling ---


class TestStreamingErrors:
    @pytest.mark.anyio
    async def test_expired_credentials_mid_stream(self, async_client, monkeypatch):
        """Expired credentials during streaming should inject error content blocks."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")

        # Create a mock response whose body iterator raises ExpiredTokenException
        expired_error = ClientError(
            error_response={"Error": {"Code": "ExpiredTokenException", "Message": "Token expired"}},
            operation_name="InvokeModelWithResponseStream",
        )

        def failing_body():
            # Yield one good chunk, then raise
            chunk_data = json.dumps({"type": "message_start", "message": {}}).encode()
            yield {"chunk": {"bytes": chunk_data}}
            raise expired_error

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model_with_response_stream.return_value = {"body": failing_body()}

        with (
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app.reset_bedrock_client") as mock_reset,
        ):
            resp = await async_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text
        assert "content_block_start" in text
        assert "content_block_delta" in text
        assert "Authentication Error" in text
        mock_reset.assert_called_once()

    @pytest.mark.anyio
    async def test_generic_client_error_mid_stream(self, async_client, monkeypatch):
        """Non-expired ClientError during streaming should yield an error event."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")

        generic_error = ClientError(
            error_response={"Error": {"Code": "InternalServerException", "Message": "Internal error"}},
            operation_name="InvokeModelWithResponseStream",
        )

        def failing_body():
            chunk_data = json.dumps({"type": "message_start", "message": {}}).encode()
            yield {"chunk": {"bytes": chunk_data}}
            raise generic_error

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model_with_response_stream.return_value = {"body": failing_body()}

        with patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock):
            resp = await async_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text
        assert "event: error" in text

    @pytest.mark.anyio
    async def test_generic_exception_mid_stream(self, async_client, monkeypatch):
        """Non-ClientError exception during streaming should yield an error event."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")

        def failing_body():
            chunk_data = json.dumps({"type": "message_start", "message": {}}).encode()
            yield {"chunk": {"bytes": chunk_data}}
            raise RuntimeError("connection lost")

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model_with_response_stream.return_value = {"body": failing_body()}

        with patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock):
            resp = await async_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text
        assert "event: error" in text
        assert "connection lost" in text


# --- Fallback Error Handlers (/v1/messages) ---


class TestMessagesFallbackErrors:
    """Cover fallback error branches in /v1/messages (lines 464-478)."""

    @pytest.mark.anyio
    async def test_fallback_copilot_http_error(self, async_client, monkeypatch):
        """Primary transient -> fallback CopilotHttpError."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "copilot")

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "rate limited")

        mock_copilot = AsyncMock()
        mock_copilot.handle_messages.side_effect = CopilotHttpError(403, "Forbidden")

        with (
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await async_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 403
        assert resp.json()["error"]["message"] == "Forbidden"

    @pytest.mark.anyio
    async def test_fallback_runtime_error(self, async_client, monkeypatch):
        """Primary transient -> fallback RuntimeError (auth)."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "copilot")

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "rate limited")

        mock_copilot = AsyncMock()
        mock_copilot.handle_messages.side_effect = RuntimeError("Copilot backend not initialized")

        with (
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await async_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 401
        assert "authentication_error" in resp.json()["error"]["type"]

    @pytest.mark.anyio
    async def test_fallback_generic_exception(self, async_client, monkeypatch):
        """Primary transient -> fallback unexpected exception."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "copilot")

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "rate limited")

        mock_copilot = AsyncMock()
        mock_copilot.handle_messages.side_effect = ValueError("something broke")

        with (
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await async_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 500
        assert "something broke" in resp.json()["error"]["message"]


# --- Fallback Error Handlers (/v1/chat/completions) ---


class TestChatCompletionsFallbackErrors:
    """Cover fallback error branches in /v1/chat/completions (lines 524-551)."""

    @pytest.mark.anyio
    async def test_no_fallback_configured(self, async_client, monkeypatch):
        """Transient error with no fallback returns OpenAI-format error."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "")

        mock_copilot = AsyncMock()
        mock_copilot.handle_openai_messages.side_effect = TransientBackendError(
            429, "rate_limit_error", "rate limited", "copilot"
        )
        monkeypatch.setattr(app_module, "_copilot_backend", mock_copilot)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 429
        body = resp.json()
        assert "error" in body
        assert body["error"]["type"] == "server_error"

    @pytest.mark.anyio
    async def test_fallback_both_fail_transient(self, async_client, monkeypatch):
        """Primary transient -> fallback also transient."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "bedrock")

        mock_copilot = AsyncMock()
        mock_copilot.handle_openai_messages.side_effect = TransientBackendError(
            429, "rate_limit_error", "rate limited", "copilot"
        )
        monkeypatch.setattr(app_module, "_copilot_backend", mock_copilot)

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "also rate limited")

        with patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["type"] == "server_error"

    @pytest.mark.anyio
    async def test_fallback_copilot_http_error(self, async_client, monkeypatch):
        """Primary transient -> fallback CopilotHttpError (OpenAI format)."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "copilot")

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "rate limited")

        mock_copilot = AsyncMock()
        mock_copilot.handle_openai_messages.side_effect = CopilotHttpError(403, "Forbidden")

        with (
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["message"] == "Forbidden"
        assert body["error"]["type"] == "server_error"

    @pytest.mark.anyio
    async def test_fallback_runtime_error(self, async_client, monkeypatch):
        """Primary transient -> fallback RuntimeError (auth, OpenAI format)."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "copilot")

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "rate limited")

        mock_copilot = AsyncMock()
        mock_copilot.handle_openai_messages.side_effect = RuntimeError("not initialized")

        with (
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["type"] == "authentication_error"

    @pytest.mark.anyio
    async def test_fallback_generic_exception(self, async_client, monkeypatch):
        """Primary transient -> fallback unexpected exception (OpenAI format)."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "copilot")

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = make_client_error("ThrottlingException", "rate limited")

        mock_copilot = AsyncMock()
        mock_copilot.handle_openai_messages.side_effect = ValueError("something broke")

        with (
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
            patch("claudegate.app._copilot_backend", mock_copilot),
        ):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 500
        body = resp.json()
        assert "something broke" in body["error"]["message"]

    @pytest.mark.anyio
    async def test_primary_copilot_http_error(self, async_client, monkeypatch):
        """Primary CopilotHttpError (non-transient) returns OpenAI-format error."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")

        mock_copilot = AsyncMock()
        mock_copilot.handle_openai_messages.side_effect = CopilotHttpError(403, "Forbidden")
        monkeypatch.setattr(app_module, "_copilot_backend", mock_copilot)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["type"] == "server_error"

    @pytest.mark.anyio
    async def test_primary_runtime_error(self, async_client, monkeypatch):
        """Primary RuntimeError returns 401 in OpenAI format."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        monkeypatch.setattr(app_module, "_copilot_backend", None)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["type"] == "authentication_error"

    @pytest.mark.anyio
    async def test_primary_generic_exception(self, async_client, monkeypatch):
        """Primary unexpected exception returns 500 in OpenAI format."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")

        mock_copilot = AsyncMock()
        mock_copilot.handle_openai_messages.side_effect = ValueError("unexpected")
        monkeypatch.setattr(app_module, "_copilot_backend", mock_copilot)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert "unexpected" in body["error"]["message"]


# --- GET /health ---


class TestHealthRoute:
    @pytest.mark.anyio
    async def test_basic(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body

    @pytest.mark.anyio
    async def test_check_bedrock_ok(self, async_client, mock_bedrock_client, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.return_value = {
            "body": BytesIO(json.dumps({"content": [{"text": "hi"}]}).encode())
        }

        resp = await async_client.get("/health?check_bedrock=true")
        assert resp.status_code == 200
        assert resp.json()["bedrock"] == "ok"

    @pytest.mark.anyio
    async def test_check_bedrock_error(self, async_client, mock_bedrock_client, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = RuntimeError("fail")

        resp = await async_client.get("/health?check_bedrock=true")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert "error" in body["bedrock"]

    @pytest.mark.anyio
    async def test_check_copilot_ok(self, async_client, monkeypatch):
        """Deep copilot health check returns ok when token is valid."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")

        mock_auth = AsyncMock()
        mock_auth.get_token.return_value = "valid-token"
        mock_backend = AsyncMock()
        mock_backend._auth = mock_auth
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.get("/health?check_copilot=true")
        assert resp.status_code == 200
        body = resp.json()
        assert body["copilot"] == "ok"

    @pytest.mark.anyio
    async def test_check_copilot_no_token(self, async_client, monkeypatch):
        """Deep copilot health check returns 'no token' when token is None."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")

        mock_auth = AsyncMock()
        mock_auth.get_token.return_value = None
        mock_backend = AsyncMock()
        mock_backend._auth = mock_auth
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.get("/health?check_copilot=true")
        assert resp.status_code == 200
        body = resp.json()
        assert body["copilot"] == "no token"

    @pytest.mark.anyio
    async def test_check_copilot_error(self, async_client, monkeypatch):
        """Deep copilot health check returns degraded on exception."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")

        mock_auth = AsyncMock()
        mock_auth.get_token.side_effect = RuntimeError("auth failed")
        mock_backend = AsyncMock()
        mock_backend._auth = mock_auth
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.get("/health?check_copilot=true")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert "error" in body["copilot"]

    @pytest.mark.anyio
    async def test_health_with_fallback(self, async_client, monkeypatch):
        """Health check includes fallback field when configured."""
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "copilot")

        resp = await async_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["fallback"] == "copilot"


# --- GET /version ---


class TestVersionRoute:
    @pytest.mark.anyio
    async def test_returns_version(self, async_client):
        resp = await async_client.get("/version")
        assert resp.status_code == 200
        assert "version" in resp.json()


# --- GET /v1/models ---


class TestModelsRoute:
    @pytest.mark.anyio
    async def test_bedrock_models(self, async_client, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert "data" in body
        assert len(body["data"]) > 0
        assert body["data"][0]["object"] == "model"
        assert "id" in body["data"][0]

    @pytest.mark.anyio
    async def test_copilot_models(self, async_client, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) > 0

    @pytest.mark.anyio
    async def test_copilot_dynamic_models(self, async_client, monkeypatch):
        """When dynamic models are available, /v1/models returns them."""
        from claudegate.models import set_copilot_models

        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        dynamic = [
            {"id": "claude-sonnet-4.5", "owned_by": "anthropic", "created_at": 1700000001},
            {"id": "gpt-4o", "owned_by": "openai", "created_at": 1700000002},
            {"id": "gemini-2.5-pro-preview"},
        ]
        set_copilot_models(dynamic)
        try:
            resp = await async_client.get("/v1/models")
            assert resp.status_code == 200
            body = resp.json()
            ids = [m["id"] for m in body["data"]]
            assert "claude-sonnet-4.5" in ids
            assert "gpt-4o" in ids
            assert "gemini-2.5-pro-preview" in ids
            # Check owned_by inference for model without explicit owned_by
            gemini = next(m for m in body["data"] if m["id"] == "gemini-2.5-pro-preview")
            assert gemini["owned_by"] == "google"
            # Check explicit owned_by is used
            gpt = next(m for m in body["data"] if m["id"] == "gpt-4o")
            assert gpt["owned_by"] == "openai"
        finally:
            set_copilot_models([])

    @pytest.mark.anyio
    async def test_copilot_fallback_to_hardcoded(self, async_client, monkeypatch):
        """When dynamic models are empty, /v1/models falls back to hardcoded maps."""
        from claudegate.models import set_copilot_models

        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        set_copilot_models([])
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        # Should have entries from the hardcoded maps
        assert len(body["data"]) > 0


# --- POST /v1/messages/count_tokens ---


class TestCountTokensRoute:
    @pytest.mark.anyio
    async def test_basic(self, async_client):
        resp = await async_client.post(
            "/v1/messages/count_tokens",
            json={"messages": [{"role": "user", "content": "hello world"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] > 0

    @pytest.mark.anyio
    async def test_with_system(self, async_client):
        resp = await async_client.post(
            "/v1/messages/count_tokens",
            json={
                "system": "You are a helpful assistant",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] > 0

    @pytest.mark.anyio
    async def test_with_tools(self, async_client):
        resp = await async_client.post(
            "/v1/messages/count_tokens",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "fn", "description": "d", "input_schema": {}}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] > 0


# --- POST /api/event_logging/batch ---


class TestEventLoggingRoute:
    @pytest.mark.anyio
    async def test_returns_ok(self, async_client):
        resp = await async_client.post("/api/event_logging/batch", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
