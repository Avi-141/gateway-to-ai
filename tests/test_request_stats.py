"""Tests for request_stats module."""

from claudegate.request_stats import RequestStats


class TestRequestStats:
    def test_initial_state(self):
        stats = RequestStats()
        snap = stats.snapshot()
        assert snap == {
            "total_requests": 0,
            "requests_by_backend": {},
            "errors": 0,
            "fallbacks": 0,
        }

    def test_record_request(self):
        stats = RequestStats()
        stats.record_request("copilot")
        stats.record_request("copilot")
        stats.record_request("bedrock")

        snap = stats.snapshot()
        assert snap["total_requests"] == 3
        assert snap["requests_by_backend"] == {"copilot": 2, "bedrock": 1}

    def test_record_error(self):
        stats = RequestStats()
        stats.record_error()
        stats.record_error()
        assert stats.snapshot()["errors"] == 2

    def test_record_fallback(self):
        stats = RequestStats()
        stats.record_fallback()
        assert stats.snapshot()["fallbacks"] == 1

    def test_reset(self):
        stats = RequestStats()
        stats.record_request("copilot")
        stats.record_request("bedrock")
        stats.record_error()
        stats.record_fallback()

        stats.reset()
        snap = stats.snapshot()
        assert snap == {
            "total_requests": 0,
            "requests_by_backend": {},
            "errors": 0,
            "fallbacks": 0,
        }

    def test_snapshot_returns_copy(self):
        stats = RequestStats()
        stats.record_request("copilot")
        snap = stats.snapshot()

        # Mutating the snapshot should not affect internal state
        snap["requests_by_backend"]["copilot"] = 999
        assert stats.snapshot()["requests_by_backend"]["copilot"] == 1

    def test_mixed_operations(self):
        stats = RequestStats()
        stats.record_request("copilot")
        stats.record_request("bedrock")
        stats.record_error()
        stats.record_fallback()
        stats.record_request("copilot")
        stats.record_error()

        snap = stats.snapshot()
        assert snap["total_requests"] == 3
        assert snap["requests_by_backend"] == {"copilot": 2, "bedrock": 1}
        assert snap["errors"] == 2
        assert snap["fallbacks"] == 1
