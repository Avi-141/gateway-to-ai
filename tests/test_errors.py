"""Tests for claudegate/errors.py."""

from claudegate.errors import ContextWindowExceededError, CopilotHttpError, TransientBackendError


class TestTransientBackendError:
    def test_attributes(self):
        err = TransientBackendError(429, "rate_limit_error", "too many requests", "bedrock")
        assert err.status_code == 429
        assert err.error_type == "rate_limit_error"
        assert err.message == "too many requests"
        assert err.backend == "bedrock"

    def test_str(self):
        err = TransientBackendError(500, "api_error", "server down", "copilot")
        assert "copilot" in str(err)
        assert "500" in str(err)
        assert "server down" in str(err)

    def test_is_exception(self):
        err = TransientBackendError(502, "api_error", "bad gateway", "bedrock")
        assert isinstance(err, Exception)


class TestCopilotHttpError:
    def test_attributes(self):
        err = CopilotHttpError(401, "unauthorized")
        assert err.status_code == 401
        assert err.detail == "unauthorized"

    def test_str(self):
        err = CopilotHttpError(403, "forbidden")
        assert "403" in str(err)
        assert "forbidden" in str(err)

    def test_is_exception(self):
        err = CopilotHttpError(404, "not found")
        assert isinstance(err, Exception)


class TestContextWindowExceededError:
    def test_attributes(self):
        err = ContextWindowExceededError(145794, 128000, "copilot")
        assert err.prompt_tokens == 145794
        assert err.context_limit == 128000
        assert err.backend == "copilot"

    def test_str(self):
        err = ContextWindowExceededError(145794, 128000, "copilot")
        assert "145794" in str(err)
        assert "128000" in str(err)
        assert "copilot" in str(err)

    def test_is_exception(self):
        err = ContextWindowExceededError(145794, 128000, "copilot")
        assert isinstance(err, Exception)
