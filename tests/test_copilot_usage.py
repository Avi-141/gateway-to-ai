"""Tests for CopilotUsageCache and /api/status copilot integration."""

import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from claudegate.app import app
from claudegate.copilot_usage import CopilotUsageCache

app_module = sys.modules["claudegate.app"]
_bs = app_module._backend_state

GITHUB_USER_RESPONSE = {
    "copilot_plan": "enterprise",
    "quota_reset_date": "2026-03-01",
    "quota_snapshots": {
        "chat": {
            "entitlement": 0,
            "remaining": 0,
            "percent_remaining": 100.0,
            "unlimited": True,
            "overage_permitted": False,
        },
        "completions": {
            "entitlement": 0,
            "remaining": 0,
            "percent_remaining": 100.0,
            "unlimited": True,
            "overage_permitted": False,
        },
        "premium_interactions": {
            "entitlement": 1000,
            "remaining": 977,
            "percent_remaining": 97.7,
            "unlimited": False,
            "overage_permitted": True,
        },
    },
}


@pytest.fixture
def mock_ssl():
    with patch("claudegate.copilot_usage.SSL_CONTEXT", None):
        yield


class TestCopilotUsageCache:
    async def test_first_call_fetches_and_caches(self, httpx_mock, mock_ssl):
        httpx_mock.add_response(json=GITHUB_USER_RESPONSE)
        cache = CopilotUsageCache("fake-token", ttl=60)
        result = await cache.get()
        assert result["plan"] == "enterprise"
        assert result["premium"]["used"] == 23
        assert result["premium"]["total"] == 1000
        assert result["premium"]["remaining"] == 977
        assert result["stale"] is False
        await cache.close()

    async def test_second_call_returns_cached(self, httpx_mock, mock_ssl):
        httpx_mock.add_response(json=GITHUB_USER_RESPONSE)
        cache = CopilotUsageCache("fake-token", ttl=60)
        await cache.get()
        await cache.get()
        assert len(httpx_mock.get_requests()) == 1
        await cache.close()

    @pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
    async def test_stale_returns_old_data(self, httpx_mock, mock_ssl):
        httpx_mock.add_response(json=GITHUB_USER_RESPONSE)
        cache = CopilotUsageCache("fake-token", ttl=0)
        await cache.get()
        httpx_mock.add_response(json={**GITHUB_USER_RESPONSE, "copilot_plan": "pro"})
        result = await cache.get()
        assert result["plan"] == "enterprise"
        assert result["stale"] is True
        await cache.close()

    async def test_negative_remaining_capped_at_zero(self, httpx_mock, mock_ssl):
        negative_response = {
            **GITHUB_USER_RESPONSE,
            "quota_snapshots": {
                **GITHUB_USER_RESPONSE["quota_snapshots"],
                "premium_interactions": {
                    "entitlement": 1000,
                    "remaining": -50,
                    "percent_remaining": 0,
                    "unlimited": False,
                    "overage_permitted": True,
                },
            },
        }
        httpx_mock.add_response(json=negative_response)
        cache = CopilotUsageCache("fake-token", ttl=60)
        result = await cache.get()
        assert result["premium"]["remaining"] == 0
        assert result["premium"]["used"] == 1050
        assert result["premium"]["percent_used"] == 105.0
        await cache.close()

    async def test_auth_error_returns_none(self, httpx_mock, mock_ssl):
        httpx_mock.add_response(status_code=401, json={"message": "Bad credentials"})
        cache = CopilotUsageCache("bad-token", ttl=60)
        assert await cache.get() is None
        await cache.close()

    @pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
    async def test_network_error_returns_stale(self, httpx_mock, mock_ssl):
        httpx_mock.add_response(json=GITHUB_USER_RESPONSE)
        cache = CopilotUsageCache("fake-token", ttl=0)
        await cache.get()
        httpx_mock.add_exception(httpx.ConnectError("connection refused"))
        result = await cache.get()
        assert result["plan"] == "enterprise"
        assert result["stale"] is True
        await cache.close()


class TestApiStatusCopilotIntegration:
    @pytest.fixture
    def async_client(self):
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def test_no_copilot_key_when_unconfigured(self, async_client):
        resp = await async_client.get("/api/status")
        assert "copilot" not in resp.json()

    async def test_includes_copilot_when_configured(self, async_client):
        mock_cache = AsyncMock()
        mock_cache.get.return_value = {
            "plan": "enterprise",
            "premium": {
                "used": 23,
                "total": 1000,
                "remaining": 977,
                "percent_used": 2.3,
                "unlimited": False,
                "overage_permitted": True,
            },
            "chat": {"unlimited": True},
            "completions": {"unlimited": True},
            "reset_date": "2026-03-01",
            "cached_at": "2026-02-20T21:09:42+00:00",
            "cache_ttl_seconds": 60,
            "stale": False,
        }
        with patch.object(_bs, "_copilot_usage_cache", mock_cache):
            resp = await async_client.get("/api/status")
        data = resp.json()
        assert data["copilot"]["plan"] == "enterprise"
        assert data["copilot"]["premium"]["used"] == 23
