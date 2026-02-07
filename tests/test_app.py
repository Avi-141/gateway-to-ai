"""Tests for claudegate/app.py."""

import json
import sys
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ReadTimeoutError

from claudegate.app import _count_content_tokens, _error_response, _validate_request
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
    async def test_non_streaming_success(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
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
    async def test_streaming_returns_sse(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
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
    async def test_validation_exception(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
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
    async def test_generic_client_error(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
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
    async def test_unexpected_exception(self, async_client, mock_bedrock_client, minimal_anthropic_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = RuntimeError("kaboom")

        resp = await async_client.post("/v1/messages", json=minimal_anthropic_request)
        assert resp.status_code == 500


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
        assert "data" in body
        assert len(body["data"]) > 0
        assert body["data"][0]["type"] == "model"
        assert "id" in body["data"][0]

    @pytest.mark.anyio
    async def test_copilot_models(self, async_client, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
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
