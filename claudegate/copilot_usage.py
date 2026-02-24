"""Cached Copilot usage/quota data from the GitHub API."""

import asyncio
import contextlib
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from .config import SSL_CONTEXT, logger
from .copilot_auth import COPILOT_HEADERS

GITHUB_USER_URL = "https://api.github.com/copilot_internal/user"


class CopilotUsageCache:
    """Fetches and caches Copilot quota data with hybrid demand-triggered refresh.

    - First request blocks until data is fetched.
    - Subsequent requests return cached data instantly.
    - Background refresh fires every TTL seconds while there is demand.
    - When idle (no requests for one TTL period), the timer stops.
    """

    def __init__(self, github_token: str, ttl: int = 60):
        self._github_token = github_token
        self._ttl = ttl
        self._data: dict[str, Any] | None = None
        self._cached_at: float = 0
        self._last_accessed: float = 0
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None
        self._client = httpx.AsyncClient(verify=SSL_CONTEXT, timeout=httpx.Timeout(10.0, connect=5.0))

    def _is_stale(self) -> bool:
        return (time.time() - self._cached_at) >= self._ttl

    async def get(self) -> dict[str, Any] | None:
        """Return cached quota data. Blocks on first call; stale-while-revalidate after."""
        self._last_accessed = time.time()

        if self._data is not None and not self._is_stale():
            return self._data

        if self._data is not None:
            # Stale — return old data, refresh in background
            self._ensure_refresh_scheduled()
            return {**self._data, "stale": True}

        # No data at all — must wait for first fetch
        await self._refresh()
        if self._data is not None:
            self._ensure_refresh_scheduled()
        return self._data

    async def _refresh(self) -> None:
        """Fetch quota data from GitHub API under lock."""
        async with self._lock:
            # Double-check: another coroutine may have refreshed while we waited
            if not self._is_stale() and self._data is not None:
                return
            try:
                resp = await self._client.get(
                    GITHUB_USER_URL,
                    headers={
                        **COPILOT_HEADERS,
                        "Authorization": f"token {self._github_token}",
                    },
                )
                if resp.status_code == 200:
                    self._data = self._transform(resp.json())
                    self._cached_at = time.time()
                    logger.debug("Copilot usage cache refreshed")
                else:
                    logger.warning("Copilot usage fetch failed: HTTP %d", resp.status_code)
            except Exception as e:
                logger.warning("Copilot usage fetch error: %s", e)

    def _transform(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Transform GitHub API response into the /copilot/usage shape."""
        snapshots = raw.get("quota_snapshots", {})
        premium = snapshots.get("premium_interactions", {})
        total = premium.get("entitlement", 0)
        raw_remaining = premium.get("remaining", 0)
        remaining = max(raw_remaining, 0)
        used = total - raw_remaining if total > 0 else 0
        pct_used = round((used / total) * 100, 1) if total > 0 else 0.0

        return {
            "plan": raw.get("copilot_plan", "unknown"),
            "premium": {
                "used": used,
                "total": total,
                "remaining": remaining,
                "percent_used": pct_used,
                "unlimited": premium.get("unlimited", False),
                "overage_permitted": premium.get("overage_permitted", False),
            },
            "chat": {"unlimited": snapshots.get("chat", {}).get("unlimited", False)},
            "completions": {"unlimited": snapshots.get("completions", {}).get("unlimited", False)},
            "reset_date": raw.get("quota_reset_date", ""),
            "cached_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "cache_ttl_seconds": self._ttl,
            "stale": False,
        }

    def _ensure_refresh_scheduled(self) -> None:
        """Schedule a background refresh if one isn't already running.

        Safe without a lock: asyncio is single-threaded, so the check-then-assign
        below cannot be preempted between the .done() check and create_task().
        """
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._background_refresh_loop())

    async def _background_refresh_loop(self) -> None:
        """Sleep for TTL, then refresh. Repeat while there's demand; stop when idle."""
        try:
            while True:
                await asyncio.sleep(self._ttl)
                try:
                    await self._refresh()
                except Exception as e:
                    logger.warning("Copilot usage background refresh error: %s", e)
                # Stop if nobody asked during the last TTL period
                if (time.time() - self._last_accessed) >= self._ttl:
                    logger.debug("Copilot usage cache going idle (no recent requests)")
                    break
        except asyncio.CancelledError:
            raise

    async def close(self) -> None:
        """Cancel background task and close HTTP client."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
        await self._client.aclose()
        self._data = None
