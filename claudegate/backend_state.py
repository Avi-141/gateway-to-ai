"""Mutable backend state with runtime switching support."""

import asyncio
from typing import TYPE_CHECKING

from .config import COPILOT_TIMEOUT, logger

if TYPE_CHECKING:
    from .copilot_client import CopilotBackend
    from .copilot_usage import CopilotUsageCache

VALID_BACKENDS = {"copilot", "bedrock"}


def parse_backend_string(value: str) -> tuple[str, str]:
    """Parse a backend string like 'copilot' or 'copilot,bedrock' into (primary, fallback).

    Raises ValueError on invalid input.
    """
    parts = [b.strip().lower() for b in value.split(",") if b.strip()]
    if not parts:
        raise ValueError("Backend value cannot be empty")
    primary = parts[0]
    fallback = parts[1] if len(parts) > 1 else ""
    if primary not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend: {primary!r}, must be one of {VALID_BACKENDS}")
    if fallback and fallback not in VALID_BACKENDS:
        raise ValueError(f"Invalid fallback backend: {fallback!r}, must be one of {VALID_BACKENDS}")
    if fallback and fallback == primary:
        raise ValueError(f"Fallback cannot be the same as primary: {primary!r}")
    if len(parts) > 2:
        raise ValueError("At most two backends (primary,fallback) are supported")
    return primary, fallback


class BackendState:
    """Encapsulates mutable backend state with safe runtime switching.

    In-flight requests capture their backend caller in a local variable
    before executing, so a switch only affects new requests. The lock
    only serializes switch operations, not reads.
    """

    def __init__(self, primary: str, fallback: str = "") -> None:
        self._primary = primary
        self._fallback = fallback
        self._copilot_backend: CopilotBackend | None = None
        self._copilot_usage_cache: CopilotUsageCache | None = None
        self._lock = asyncio.Lock()

    @property
    def primary(self) -> str:
        return self._primary

    @property
    def fallback(self) -> str:
        return self._fallback

    @property
    def copilot_backend(self) -> "CopilotBackend | None":
        return self._copilot_backend

    @property
    def copilot_usage_cache(self) -> "CopilotUsageCache | None":
        return self._copilot_usage_cache

    def set_copilot_backend(self, backend: "CopilotBackend", cache: "CopilotUsageCache") -> None:
        """Called from lifespan() at startup to set the initialized copilot backend."""
        self._copilot_backend = backend
        self._copilot_usage_cache = cache

    async def switch(self, primary: str, fallback: str = "") -> dict:
        """Switch to a new backend configuration.

        Returns dict with 'changed', 'primary', 'fallback' keys.
        Raises ValueError on invalid input or initialization failure.
        """
        if primary not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend: {primary!r}, must be one of {VALID_BACKENDS}")
        if fallback and fallback not in VALID_BACKENDS:
            raise ValueError(f"Invalid fallback: {fallback!r}, must be one of {VALID_BACKENDS}")
        if fallback and fallback == primary:
            raise ValueError(f"Fallback cannot be the same as primary: {primary!r}")

        async with self._lock:
            if self._primary == primary and self._fallback == fallback:
                return {"changed": False, "primary": self._primary, "fallback": self._fallback}

            needs_copilot = primary == "copilot" or fallback == "copilot"
            if needs_copilot and self._copilot_backend is None:
                await self._init_copilot()

            self._primary = primary
            self._fallback = fallback
            logger.info(f"Backend switched to primary={primary}" + (f", fallback={fallback}" if fallback else ""))
            return {"changed": True, "primary": self._primary, "fallback": self._fallback}

    async def _init_copilot(self) -> None:
        """Lazily initialize the Copilot backend.

        Raises ValueError if authentication fails.
        """
        from .copilot_auth import CopilotAuth, get_github_token
        from .copilot_client import CopilotBackend
        from .copilot_usage import CopilotUsageCache
        from .models import set_copilot_models

        try:
            github_token = get_github_token()
        except Exception as e:
            raise ValueError(f"Failed to get GitHub token: {e}") from e

        auth = CopilotAuth(github_token)
        backend = CopilotBackend(auth, timeout=COPILOT_TIMEOUT)
        cache = CopilotUsageCache(github_token)

        models = await backend.list_models()
        if models:
            set_copilot_models(models)
            logger.info(f"Loaded {len(models)} models from Copilot API")
        else:
            logger.warning("No models fetched from Copilot API, using hardcoded maps")

        self._copilot_backend = backend
        self._copilot_usage_cache = cache
        logger.info("Copilot backend initialized via runtime switch")

    async def close(self) -> None:
        """Shutdown lifecycle — close copilot resources."""
        if self._copilot_usage_cache is not None:
            await self._copilot_usage_cache.close()
        if self._copilot_backend is not None:
            await self._copilot_backend.close()
