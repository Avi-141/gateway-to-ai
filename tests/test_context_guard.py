"""Tests for pre-flight context window guard."""

import json
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from claudegate import models as models_mod
from claudegate.app import app
from claudegate.context_guard import (
    _MIN_OUTPUT_TOKENS,
    check_context_guard_anthropic,
    check_context_guard_openai,
    check_context_guard_responses,
)
from claudegate.errors import ContextWindowExceededError
from claudegate.request_stats import request_stats

app_module = sys.modules["claudegate.app"]
_bs = app_module._backend_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_model_limits(monkeypatch, limits: dict[str, int]) -> None:
    """Set known model limits for deterministic tests.

    Model IDs should use Copilot format (dots): e.g. "claude-sonnet-4.6"
    """
    model_ids = set(limits.keys())
    monkeypatch.setattr(models_mod, "_copilot_model_limits", limits)
    monkeypatch.setattr(models_mod, "_copilot_model_ids", model_ids)


def _big_messages(n_tokens_approx: int) -> list[dict]:
    """Build a messages list that estimates to roughly n_tokens_approx tokens."""
    # Each word is ~1 token with cl100k_base; use 'hello ' repeated
    word = "hello "
    text = word * n_tokens_approx
    return [{"role": "user", "content": text}]


def _mock_copilot_success():
    """Return a mock CopilotBackend whose handle_messages returns success."""
    from fastapi.responses import JSONResponse

    mock = AsyncMock()
    result = {
        "type": "message",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock.handle_messages.return_value = JSONResponse(content=result)
    mock.handle_openai_messages.return_value = JSONResponse(
        content={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    # Responses API: handle_responses_via_chat is called when model doesn't support /responses
    responses_result = JSONResponse(
        content={
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    mock.handle_responses_passthrough.return_value = responses_result
    mock.handle_responses_via_chat.return_value = responses_result
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def guard_client():
    """HTTPX async client wired to the FastAPI app (no lifespan)."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# TestContextGuardConfig
# ---------------------------------------------------------------------------


class TestContextGuardConfig:
    def test_default_threshold_is_090(self):
        from claudegate.config import CONTEXT_GUARD_THRESHOLD

        assert CONTEXT_GUARD_THRESHOLD == 0.90

    def test_custom_threshold_from_env(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_GUARD_THRESHOLD", "0.75")
        # Must reload to pick up env change
        import importlib

        import claudegate.config

        importlib.reload(claudegate.config)
        assert claudegate.config.CONTEXT_GUARD_THRESHOLD == 0.75
        # Restore
        monkeypatch.delenv("CONTEXT_GUARD_THRESHOLD", raising=False)
        importlib.reload(claudegate.config)

    def test_zero_threshold_disables_guard(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_GUARD_THRESHOLD", "0")
        import importlib

        import claudegate.config

        importlib.reload(claudegate.config)
        assert claudegate.config.CONTEXT_GUARD_THRESHOLD == 0.0
        monkeypatch.delenv("CONTEXT_GUARD_THRESHOLD", raising=False)
        importlib.reload(claudegate.config)


# ---------------------------------------------------------------------------
# TestContextGuardStats
# ---------------------------------------------------------------------------


class TestContextGuardStats:
    def test_initial_context_guard_rejections_is_zero(self):
        from claudegate.request_stats import RequestStats

        stats = RequestStats()
        assert stats.snapshot()["context_guard_rejections"] == 0

    def test_record_context_guard_rejection(self):
        from claudegate.request_stats import RequestStats

        stats = RequestStats()
        stats.record_context_guard_rejection()
        stats.record_context_guard_rejection()
        assert stats.snapshot()["context_guard_rejections"] == 2

    def test_reset_clears_context_guard_rejections(self):
        from claudegate.request_stats import RequestStats

        stats = RequestStats()
        stats.record_context_guard_rejection()
        stats.reset()
        assert stats.snapshot()["context_guard_rejections"] == 0


# ---------------------------------------------------------------------------
# TestCheckContextGuardAnthropic (unit tests)
# ---------------------------------------------------------------------------


class TestCheckContextGuardAnthropic:
    def test_under_limit_no_exception(self, monkeypatch):
        """Requests well under the limit should pass without exception."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 1000})
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        # Should not raise
        check_context_guard_anthropic(body)

    def test_at_exact_threshold_no_exception(self, monkeypatch):
        """Requests at exactly the threshold boundary should NOT raise (boundary is >)."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        # effective_limit = int(100 * 0.90) = 90
        # We need estimated_tokens == 90 exactly. Build a body with ~90 tokens.
        # Use a single character per token approach: digits are ~1 token each
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "x"}],
        }
        # With 1-2 tokens this is well under 90, should not raise
        check_context_guard_anthropic(body)

    def test_over_threshold_raises_error(self, monkeypatch):
        """Requests over the threshold should raise ContextWindowExceededError."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        # effective_limit = int(100 * 0.90) = 90
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": _big_messages(200),
        }
        with pytest.raises(ContextWindowExceededError):
            check_context_guard_anthropic(body)

    def test_error_has_backend_proxy(self, monkeypatch):
        """Error should have backend='proxy'."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": _big_messages(200),
        }
        with pytest.raises(ContextWindowExceededError) as exc_info:
            check_context_guard_anthropic(body)
        assert exc_info.value.backend == "proxy"

    def test_error_has_correct_token_counts(self, monkeypatch):
        """Error should report prompt_tokens == context_limit for compaction."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": _big_messages(200),
        }
        with pytest.raises(ContextWindowExceededError) as exc_info:
            check_context_guard_anthropic(body)
        assert exc_info.value.prompt_tokens == 100  # equals context_limit
        assert exc_info.value.context_limit == 100

    def test_unknown_model_skips(self, monkeypatch):
        """Unknown model (limit=0) should not raise."""
        _set_model_limits(monkeypatch, {})  # No models registered
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": _big_messages(500),
        }
        # Should not raise even though messages are huge
        check_context_guard_anthropic(body)

    def test_disabled_threshold_skips(self, monkeypatch):
        """Zero threshold should skip the guard entirely."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        monkeypatch.setattr("claudegate.context_guard.CONTEXT_GUARD_THRESHOLD", 0.0)
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": _big_messages(500),
        }
        # Should not raise
        check_context_guard_anthropic(body)

    def test_large_system_prompt_triggers(self, monkeypatch):
        """A large system prompt should contribute to token count."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": "word " * 200,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with pytest.raises(ContextWindowExceededError):
            check_context_guard_anthropic(body)

    def test_tool_definitions_counted(self, monkeypatch):
        """Tool definitions should be counted in the token estimate."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        # Create a large tool definition
        tools = [
            {
                "name": "big_tool",
                "description": "word " * 200,
                "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        ]
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": tools,
        }
        with pytest.raises(ContextWindowExceededError):
            check_context_guard_anthropic(body)

    def test_images_counted_at_1600(self, monkeypatch):
        """Image blocks should add ~1600 tokens per image."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 1000})
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                    ],
                }
            ],
        }
        # 1600 tokens for image > 900 effective limit
        with pytest.raises(ContextWindowExceededError):
            check_context_guard_anthropic(body)

    def test_clamps_max_tokens_when_near_limit(self, monkeypatch):
        """When over threshold but room for output, clamp max_tokens instead of rejecting."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 50000})
        # effective_limit = int(50000 * 0.90) = 45000
        # ~46000 estimated tokens → over threshold
        # available = 50000 - 46000 = 4000 ≥ 1024 → should clamp, not raise
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 8192,
            "messages": _big_messages(46000),
        }
        # Should NOT raise — should clamp max_tokens
        check_context_guard_anthropic(body)
        assert body["max_tokens"] < 8192  # was clamped
        assert body["max_tokens"] >= _MIN_OUTPUT_TOKENS

    def test_clamp_does_not_exceed_original_max_tokens(self, monkeypatch):
        """Clamped max_tokens should not exceed the original value."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 50000})
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 2000,
            "messages": _big_messages(46000),
        }
        check_context_guard_anthropic(body)
        # available = 50000 - 46000 = 4000, but original max_tokens = 2000
        assert body["max_tokens"] == 2000  # kept original since it fits


# ---------------------------------------------------------------------------
# TestCheckContextGuardOpenAI (unit tests)
# ---------------------------------------------------------------------------


class TestCheckContextGuardOpenAI:
    def test_under_limit_no_exception(self, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 1000})
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        check_context_guard_openai(body)

    def test_over_threshold_raises_error(self, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "word " * 200}],
        }
        with pytest.raises(ContextWindowExceededError) as exc_info:
            check_context_guard_openai(body)
        assert exc_info.value.backend == "proxy"

    def test_system_role_messages_counted(self, monkeypatch):
        """System role messages in OpenAI format should be counted."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "system", "content": "word " * 200},
                {"role": "user", "content": "Hi"},
            ],
        }
        with pytest.raises(ContextWindowExceededError):
            check_context_guard_openai(body)

    def test_disabled_threshold_skips(self, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        monkeypatch.setattr("claudegate.context_guard.CONTEXT_GUARD_THRESHOLD", 0.0)
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "word " * 200}],
        }
        check_context_guard_openai(body)

    def test_clamps_max_tokens_when_near_limit(self, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 50000})
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "word " * 46000}],
        }
        check_context_guard_openai(body)
        assert body["max_tokens"] < 8192


# ---------------------------------------------------------------------------
# TestCheckContextGuardResponses (unit tests)
# ---------------------------------------------------------------------------


class TestCheckContextGuardResponses:
    def test_under_limit_no_exception(self, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 1000})
        body = {
            "model": "claude-sonnet-4-6",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
        }
        check_context_guard_responses(body)

    def test_over_threshold_raises_error(self, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        body = {
            "model": "claude-sonnet-4-6",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "word " * 200}]}],
        }
        with pytest.raises(ContextWindowExceededError) as exc_info:
            check_context_guard_responses(body)
        assert exc_info.value.backend == "proxy"

    def test_string_input_handled(self, monkeypatch):
        """String input (simple prompt) should be handled."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        body = {
            "model": "claude-sonnet-4-6",
            "input": "word " * 200,
        }
        with pytest.raises(ContextWindowExceededError):
            check_context_guard_responses(body)

    def test_disabled_threshold_skips(self, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        monkeypatch.setattr("claudegate.context_guard.CONTEXT_GUARD_THRESHOLD", 0.0)
        body = {
            "model": "claude-sonnet-4-6",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "word " * 200}]}],
        }
        check_context_guard_responses(body)

    def test_clamps_max_tokens_when_near_limit(self, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 50000})
        body = {
            "model": "claude-sonnet-4-6",
            "max_output_tokens": 8192,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "word " * 46000}]}],
        }
        check_context_guard_responses(body)
        assert body["max_output_tokens"] < 8192


# ---------------------------------------------------------------------------
# TestContextGuardIntegrationMessages (via async_client)
# ---------------------------------------------------------------------------


class TestContextGuardIntegrationMessages:
    @pytest.mark.anyio
    async def test_returns_400_with_autocompact_message(self, guard_client, monkeypatch):
        """Guard rejection should return 400 with auto-compaction message format."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": _big_messages(200),
        }
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/messages", json=body)

        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert "exceed context limit" in data["error"]["message"]
        assert "max_tokens" in data["error"]["message"]

    @pytest.mark.anyio
    async def test_passes_when_under_limit(self, guard_client, monkeypatch):
        """Requests under the limit should pass through to the backend."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100000})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/messages", json=body)

        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_skips_for_bedrock_primary(self, guard_client, monkeypatch):
        """Guard should NOT run when Bedrock is primary (Bedrock returns proper 400s)."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})

        mock_bedrock = pytest.importorskip("unittest.mock").MagicMock()
        result = {
            "type": "message",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_bedrock.invoke_model.return_value = {"body": __import__("io").BytesIO(json.dumps(result).encode())}

        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": _big_messages(200),
        }
        with (
            patch.object(_bs, "_primary", "bedrock"),
            patch.object(_bs, "_fallback", ""),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
        ):
            resp = await guard_client.post("/v1/messages", json=body)

        # Bedrock was called (not rejected by guard)
        assert resp.status_code == 200
        mock_bedrock.invoke_model.assert_called_once()

    @pytest.mark.anyio
    async def test_backend_never_called_on_rejection(self, guard_client, monkeypatch):
        """When guard rejects, the backend should never be called."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": _big_messages(200),
        }
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/messages", json=body)

        assert resp.status_code == 400
        mock_copilot.handle_messages.assert_not_called()

    @pytest.mark.anyio
    async def test_stats_counter_incremented(self, guard_client, monkeypatch):
        """Guard rejection should increment context_guard_rejections counter."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": _big_messages(200),
        }
        request_stats.reset()
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/messages", json=body)

        assert resp.status_code == 400
        assert request_stats.snapshot()["context_guard_rejections"] >= 1

    @pytest.mark.anyio
    async def test_clamps_and_forwards_when_near_limit(self, guard_client, monkeypatch):
        """When over threshold but room for output, clamp max_tokens and forward."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 50000})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 8192,
            "messages": _big_messages(46000),
        }
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/messages", json=body)

        # Request should have been forwarded (not rejected)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestContextGuardIntegrationOpenAI (via async_client)
# ---------------------------------------------------------------------------


class TestContextGuardIntegrationOpenAI:
    @pytest.mark.anyio
    async def test_returns_openai_error_format(self, guard_client, monkeypatch):
        """Guard rejection should return OpenAI error format."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "word " * 200}],
        }
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/chat/completions", json=body)

        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "exceeds" in data["error"]["message"]

    @pytest.mark.anyio
    async def test_passes_when_under_limit(self, guard_client, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100000})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/chat/completions", json=body)

        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_skips_for_bedrock_primary(self, guard_client, monkeypatch):
        """Guard should NOT run when Bedrock is primary."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})

        mock_bedrock = pytest.importorskip("unittest.mock").MagicMock()
        result = {
            "type": "message",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_bedrock.invoke_model.return_value = {"body": __import__("io").BytesIO(json.dumps(result).encode())}

        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "word " * 200}],
        }
        with (
            patch.object(_bs, "_primary", "bedrock"),
            patch.object(_bs, "_fallback", ""),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
        ):
            resp = await guard_client.post("/v1/chat/completions", json=body)

        # Bedrock was called (guard skipped)
        assert resp.status_code == 200
        mock_bedrock.invoke_model.assert_called_once()


# ---------------------------------------------------------------------------
# TestContextGuardIntegrationResponses (via async_client)
# ---------------------------------------------------------------------------


class TestContextGuardIntegrationResponses:
    @pytest.mark.anyio
    async def test_returns_openai_error_format(self, guard_client, monkeypatch):
        """Guard rejection should return OpenAI error format for Responses API."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "word " * 200}]}],
        }
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/responses", json=body)

        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "exceeds" in data["error"]["message"]

    @pytest.mark.anyio
    async def test_passes_when_under_limit(self, guard_client, monkeypatch):
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100000})
        mock_copilot = _mock_copilot_success()
        body = {
            "model": "claude-sonnet-4-6",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        }
        with (
            patch.object(_bs, "_primary", "copilot"),
            patch.object(_bs, "_copilot_backend", mock_copilot),
        ):
            resp = await guard_client.post("/v1/responses", json=body)

        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_skips_for_bedrock_primary(self, guard_client, monkeypatch):
        """Guard should NOT run when Bedrock is primary."""
        _set_model_limits(monkeypatch, {"claude-sonnet-4.6": 100})

        mock_bedrock = pytest.importorskip("unittest.mock").MagicMock()
        result = {
            "type": "message",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_bedrock.invoke_model.return_value = {"body": __import__("io").BytesIO(json.dumps(result).encode())}

        body = {
            "model": "claude-sonnet-4-6",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "word " * 200}]}],
        }
        with (
            patch.object(_bs, "_primary", "bedrock"),
            patch.object(_bs, "_fallback", ""),
            patch("claudegate.app.get_bedrock_client", return_value=mock_bedrock),
        ):
            resp = await guard_client.post("/v1/responses", json=body)

        # Bedrock was called (guard skipped)
        assert resp.status_code == 200
        mock_bedrock.invoke_model.assert_called_once()
