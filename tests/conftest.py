"""Shared test fixtures for claudegate tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claudegate.app import app


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove environment variables that affect module behaviour."""
    for var in (
        "CLAUDEGATE_BACKEND",
        "GITHUB_TOKEN",
        "AWS_REGION",
        "BEDROCK_REGION_PREFIX",
        "BEDROCK_READ_TIMEOUT",
        "CLAUDEGATE_HOST",
        "CLAUDEGATE_PORT",
        "CLAUDEGATE_LOG_LEVEL",
        "COPILOT_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def reset_bedrock_singleton():
    """Reset the Bedrock client singleton before and after each test."""
    from claudegate.bedrock_client import reset_bedrock_client

    reset_bedrock_client()
    yield
    reset_bedrock_client()


@pytest.fixture
def async_client():
    """HTTPX async client wired to the FastAPI app (no lifespan)."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def mock_bedrock_client():
    """Patch get_bedrock_client to return a MagicMock."""
    mock = MagicMock()
    with patch("claudegate.app.get_bedrock_client", return_value=mock):
        yield mock


@pytest.fixture
def minimal_anthropic_request() -> dict[str, Any]:
    """Minimal valid Anthropic Messages API request body."""
    return {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }


@pytest.fixture
def anthropic_request_with_tools() -> dict[str, Any]:
    """Request body with tool definitions and tool_choice."""
    return {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
        "tool_choice": {"type": "auto"},
    }


@pytest.fixture
def openai_chat_response() -> dict[str, Any]:
    """Sample OpenAI non-streaming chat completion response."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "claude-sonnet-4.5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def openai_streaming_chunks() -> list[dict[str, Any]]:
    """List of OpenAI SSE chunk dicts for streaming tests."""
    return [
        {
            "id": "chatcmpl-abc123",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            "usage": {"prompt_tokens": 10},
        },
        {
            "id": "chatcmpl-abc123",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-abc123",
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-abc123",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        },
    ]


@pytest.fixture
def openai_streaming_chunks_with_usage() -> list[dict[str, Any]]:
    """OpenAI streaming chunks where usage arrives in a separate final chunk.

    Models the real OpenAI pattern with stream_options.include_usage:
    1. Role init chunk
    2. Text content chunks
    3. Finish-reason chunk (no usage)
    4. Usage-only chunk (choices: [], full usage stats)
    """
    return [
        {
            "id": "chatcmpl-abc123",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-abc123",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-abc123",
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-abc123",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-abc123",
            "choices": [],
            "usage": {"prompt_tokens": 42, "completion_tokens": 3, "total_tokens": 45},
        },
    ]


@pytest.fixture
def minimal_openai_request() -> dict[str, Any]:
    """Minimal valid OpenAI Chat Completions request body."""
    return {
        "model": "claude-sonnet-4-5-20250929",
        "messages": [{"role": "user", "content": "Hello"}],
    }


@pytest.fixture
def openai_request_with_tools() -> dict[str, Any]:
    """OpenAI request body with tool definitions."""
    return {
        "model": "claude-sonnet-4-5-20250929",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }


@pytest.fixture
def mock_copilot_auth():
    """Mock CopilotAuth with get_token returning a fake token."""
    auth = AsyncMock()
    auth.get_token.return_value = "fake-copilot-token"
    auth.close = AsyncMock()
    return auth


def make_client_error(code: str = "InternalError", message: str = "Something failed"):
    """Create a botocore ClientError with the given error code."""
    from botocore.exceptions import ClientError

    return ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="InvokeModel",
    )


@pytest.fixture
def responses_api_response() -> dict[str, Any]:
    """Sample Responses API non-streaming response (text only)."""
    return {
        "id": "resp_abc123",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello there!"}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.fixture
def responses_api_tool_response() -> dict[str, Any]:
    """Sample Responses API non-streaming response with a tool call."""
    return {
        "id": "resp_tool456",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Let me check the weather."}],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location":"NYC"}',
                "call_id": "call_weather_1",
            },
        ],
        "usage": {"input_tokens": 15, "output_tokens": 25},
    }
