"""Tests for OpenAI-compatible /v1/chat/completions route."""

import json
import sys
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ReadTimeoutError

from claudegate.app import _openai_error_response
from claudegate.errors import TransientBackendError
from tests.conftest import make_client_error

app_module = sys.modules["claudegate.app"]


# --- _openai_error_response ---


class TestOpenAIErrorResponse:
    def test_structure(self):
        resp = _openai_error_response(400, "bad request")
        assert resp.status_code == 400
        body = json.loads(resp.body)
        assert body["error"]["message"] == "bad request"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] is None
        assert body["error"]["code"] is None

    def test_custom_type(self):
        resp = _openai_error_response(500, "oops", "server_error")
        body = json.loads(resp.body)
        assert body["error"]["type"] == "server_error"


# --- POST /v1/chat/completions ---


class TestChatCompletionsRoute:
    @pytest.mark.anyio
    async def test_invalid_json(self, async_client):
        resp = await async_client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert "Invalid JSON" in body["error"]["message"]

    @pytest.mark.anyio
    async def test_missing_model(self, async_client):
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400
        assert "model" in resp.json()["error"]["message"]

    @pytest.mark.anyio
    async def test_missing_messages(self, async_client):
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "x"},
        )
        assert resp.status_code == 400
        assert "messages" in resp.json()["error"]["message"]

    @pytest.mark.anyio
    async def test_non_array_messages(self, async_client):
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": "not an array"},
        )
        assert resp.status_code == 400
        assert "array" in resp.json()["error"]["message"]

    @pytest.mark.anyio
    async def test_empty_messages(self, async_client):
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": []},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["error"]["message"]

    @pytest.mark.anyio
    async def test_non_streaming_bedrock(self, async_client, mock_bedrock_client, minimal_openai_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_response = {
            "body": BytesIO(
                json.dumps(
                    {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi there!"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 3},
                    }
                ).encode()
            )
        }
        mock_bedrock_client.invoke_model.return_value = mock_response

        resp = await async_client.post("/v1/chat/completions", json=minimal_openai_request)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hi there!"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 5
        assert body["usage"]["completion_tokens"] == 3
        assert body["usage"]["total_tokens"] == 8

    @pytest.mark.anyio
    async def test_streaming_bedrock(self, async_client, mock_bedrock_client, minimal_openai_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        minimal_openai_request["stream"] = True

        # Build Anthropic SSE chunks
        chunks = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]
        mock_events = []
        for chunk in chunks:
            chunk_bytes = json.dumps(chunk).encode()
            mock_events.append({"chunk": {"bytes": chunk_bytes}})

        mock_bedrock_client.invoke_model_with_response_stream.return_value = {"body": mock_events}

        resp = await async_client.post("/v1/chat/completions", json=minimal_openai_request)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Check the SSE content contains OpenAI format
        text = resp.text
        assert "chat.completion.chunk" in text
        assert "data: [DONE]" in text

    @pytest.mark.anyio
    async def test_max_tokens_default(self, async_client, mock_bedrock_client, monkeypatch):
        """max_tokens should default to 4096 when not provided (unlike Anthropic endpoint)."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_response = {
            "body": BytesIO(
                json.dumps(
                    {
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()
            )
        }
        mock_bedrock_client.invoke_model.return_value = mock_response

        # No max_tokens in request
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4-5-20250929", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

        # Verify the bedrock call had max_tokens set
        call_args = mock_bedrock_client.invoke_model.call_args
        sent_body = json.loads(call_args.kwargs["body"])
        assert sent_body["max_tokens"] == 4096

    @pytest.mark.anyio
    async def test_error_format_is_openai(self, async_client, mock_bedrock_client, minimal_openai_request, monkeypatch):
        """Error responses should use OpenAI error format, not Anthropic."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = make_client_error("ExpiredTokenException", "Token expired")

        resp = await async_client.post("/v1/chat/completions", json=minimal_openai_request)
        body = resp.json()
        # Should be OpenAI format (error.message), not Anthropic (error.type+message at top level)
        assert "error" in body
        assert "message" in body["error"]
        assert "type" not in body or body.get("type") != "error"

    @pytest.mark.anyio
    async def test_throttling_error(self, async_client, mock_bedrock_client, minimal_openai_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = make_client_error("ThrottlingException", "Rate limited")

        resp = await async_client.post("/v1/chat/completions", json=minimal_openai_request)
        assert resp.status_code == 429

    @pytest.mark.anyio
    async def test_read_timeout_error(self, async_client, mock_bedrock_client, minimal_openai_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = ReadTimeoutError(endpoint_url="https://bedrock.example.com")

        resp = await async_client.post("/v1/chat/completions", json=minimal_openai_request)
        assert resp.status_code == 504

    @pytest.mark.anyio
    async def test_unexpected_exception(self, async_client, mock_bedrock_client, minimal_openai_request, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        mock_bedrock_client.invoke_model.side_effect = RuntimeError("kaboom")

        resp = await async_client.post("/v1/chat/completions", json=minimal_openai_request)
        assert resp.status_code == 500
        body = resp.json()
        assert "error" in body


# --- GET /v1/models (updated format) ---


class TestChatCompletionsCopilotDirect:
    """Tests for direct Copilot passthrough (0 translations) on /v1/chat/completions."""

    @pytest.mark.anyio
    async def test_copilot_direct_non_streaming(self, async_client, monkeypatch):
        """Copilot backend should call handle_openai_messages, not handle_messages."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")

        mock_backend = AsyncMock()
        from fastapi.responses import JSONResponse

        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_backend.handle_openai_messages.return_value = JSONResponse(content=openai_resp)
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Hi!"
        # Verify handle_openai_messages was called (direct path), not handle_messages
        mock_backend.handle_openai_messages.assert_called_once()
        mock_backend.handle_messages.assert_not_called()

    @pytest.mark.anyio
    async def test_gpt_model_passes_through(self, async_client, monkeypatch):
        """Non-Claude models (GPT, o-series) should work via direct Copilot path."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")

        mock_backend = AsyncMock()
        from fastapi.responses import JSONResponse

        openai_resp = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "GPT response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_backend.handle_openai_messages.return_value = JSONResponse(content=openai_resp)
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "GPT response"

    @pytest.mark.anyio
    async def test_copilot_not_initialized(self, async_client, monkeypatch):
        """When copilot backend is not initialized, should return auth error."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        monkeypatch.setattr(app_module, "_copilot_backend", None)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_fallback_copilot_primary_429_bedrock_fallback(self, async_client, mock_bedrock_client, monkeypatch):
        """Copilot primary 429 should fall back to Bedrock with translation."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "bedrock")

        mock_backend = AsyncMock()
        mock_backend.handle_openai_messages.side_effect = TransientBackendError(
            429, "rate_limit_error", "rate limited", "copilot"
        )
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        # Bedrock fallback succeeds
        mock_response = {
            "body": BytesIO(
                json.dumps(
                    {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Bedrock fallback"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 3},
                    }
                ).encode()
            )
        }
        mock_bedrock_client.invoke_model.return_value = mock_response

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Response should be in OpenAI format (translated from Bedrock's Anthropic response)
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Bedrock fallback"

    @pytest.mark.anyio
    async def test_fallback_bedrock_primary_429_copilot_fallback(self, async_client, mock_bedrock_client, monkeypatch):
        """Bedrock primary 429 should fall back to Copilot with direct path."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        monkeypatch.setattr(app_module, "FALLBACK_BACKEND", "copilot")

        mock_bedrock_client.invoke_model.side_effect = make_client_error("ThrottlingException", "Rate limited")

        mock_backend = AsyncMock()
        from fastapi.responses import JSONResponse

        openai_resp = {
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Copilot fallback"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_backend.handle_openai_messages.return_value = JSONResponse(content=openai_resp)
        monkeypatch.setattr(app_module, "_copilot_backend", mock_backend)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Copilot fallback"
        # Verify the fallback used the direct Copilot path
        mock_backend.handle_openai_messages.assert_called_once()


# --- GET /v1/models (updated format) ---


class TestModelsRouteOpenAI:
    @pytest.mark.anyio
    async def test_openai_format(self, async_client, monkeypatch):
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "bedrock")
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) > 0
        model = body["data"][0]
        assert model["object"] == "model"
        assert "id" in model
        assert model["owned_by"] == "anthropic"

    @pytest.mark.anyio
    async def test_copilot_includes_non_claude_models(self, async_client, monkeypatch):
        """Copilot backend should include GPT, Gemini, and other models."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        model_ids = [m["id"] for m in body["data"]]
        # Should include GPT, Codex, Gemini models
        assert "gpt-4o" in model_ids
        assert "gpt-5.1-codex" in model_ids
        assert "gemini-2.5-pro" in model_ids
        # Anthropic models from COPILOT_MODEL_MAP should be present
        assert "claude-sonnet-4-5-20250929" in model_ids

    @pytest.mark.anyio
    async def test_copilot_model_owned_by(self, async_client, monkeypatch):
        """Models should have correct owned_by based on provider."""
        monkeypatch.setattr(app_module, "BACKEND_TYPE", "copilot")
        resp = await async_client.get("/v1/models")
        body = resp.json()
        model_map = {m["id"]: m["owned_by"] for m in body["data"]}
        assert model_map.get("gpt-4o") == "openai"
        assert model_map.get("gpt-5.1-codex") == "openai"
        assert model_map.get("gemini-2.5-pro") == "google"
        assert model_map.get("claude-sonnet-4-5-20250929") == "anthropic"
