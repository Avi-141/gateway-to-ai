"""GitHub OAuth device flow and Copilot token management."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from .config import logger

# GitHub OAuth app client ID used by Copilot
COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_SCOPE = "copilot"

# Token persistence path
TOKEN_DIR = Path.home() / ".config" / "clauderock"
TOKEN_FILE = TOKEN_DIR / "github_token"

# Copilot API endpoints
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_OAUTH_TOKEN_URL = "https://github.com/login/oauth/access_token"
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

# Shared editor identification headers required by all Copilot endpoints
COPILOT_HEADERS = {
    "Editor-Version": "Neovim/0.6.1",
    "Editor-Plugin-Version": "copilot.vim/1.16.0",
    "User-Agent": "GithubCopilot/1.155.0",
    "Accept": "application/json",
}


def _load_persisted_token() -> str | None:
    """Load GitHub OAuth token from persistent storage."""
    if TOKEN_FILE.exists():
        try:
            token = TOKEN_FILE.read_text().strip()
            if token:
                logger.info("Loaded persisted GitHub token from %s", TOKEN_FILE)
                return token
        except OSError as e:
            logger.warning("Failed to read persisted token: %s", e)
    return None


def _persist_token(token: str) -> None:
    """Persist GitHub OAuth token to disk."""
    try:
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(token)
        TOKEN_FILE.chmod(0o600)
        logger.info("Persisted GitHub token to %s", TOKEN_FILE)
    except OSError as e:
        logger.warning("Failed to persist token: %s", e)


def device_flow_login() -> str:
    """Run GitHub OAuth device flow to obtain an access token.

    Displays a URL and code for the user to authorize in their browser,
    then polls until authorization is complete.
    """
    with httpx.Client() as client:
        # Request device code
        resp = client.post(
            GITHUB_DEVICE_CODE_URL,
            data={"client_id": COPILOT_CLIENT_ID, "scope": COPILOT_SCOPE},
            headers=COPILOT_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        device_code = data["device_code"]
        user_code = data["user_code"]
        verification_uri = data["verification_uri"]
        interval = data.get("interval", 5)
        expires_in = data.get("expires_in", 900)

        print()
        print("To authenticate with GitHub Copilot:")
        print(f"  1. Open {verification_uri}")
        print(f"  2. Enter code: {user_code}")
        print()
        print("Waiting for authorization...")

        # Poll for token
        deadline = time.time() + expires_in
        while time.time() < deadline:
            time.sleep(interval)
            token_resp = client.post(
                GITHUB_OAUTH_TOKEN_URL,
                data={
                    "client_id": COPILOT_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers=COPILOT_HEADERS,
            )
            token_data = token_resp.json()

            if "access_token" in token_data:
                print("Authorization successful!")
                token = token_data["access_token"]
                _persist_token(token)
                return token

            error = token_data.get("error")
            if error == "authorization_pending":
                continue
            elif error == "slow_down":
                interval += 5
            elif error == "expired_token":
                raise RuntimeError("Device code expired. Please try again.")
            elif error == "access_denied":
                raise RuntimeError("Authorization was denied by the user.")
            else:
                raise RuntimeError(f"OAuth error: {error} - {token_data.get('error_description', '')}")

    raise RuntimeError("Device flow timed out. Please try again.")


def get_github_token() -> str:
    """Get GitHub OAuth token from environment, persisted storage, or device flow."""
    # 1. Check environment variable
    env_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if env_token:
        logger.info("Using GITHUB_TOKEN from environment")
        return env_token

    # 2. Check persisted token
    persisted = _load_persisted_token()
    if persisted:
        return persisted

    # 3. Run interactive device flow
    logger.info("No GitHub token found, starting device flow authentication")
    return device_flow_login()


class CopilotAuth:
    """Manages Copilot API token lifecycle.

    Exchanges a GitHub OAuth token for a short-lived Copilot token
    and auto-refreshes before expiry.
    """

    def __init__(self, github_token: str):
        self._github_token = github_token
        self._copilot_token: str | None = None
        self._expires_at: float = 0
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient()

    async def get_token(self) -> str:
        """Get a valid Copilot token, refreshing if needed."""
        # Check with 2-minute buffer before expiry
        if self._copilot_token and time.time() < (self._expires_at - 120):
            return self._copilot_token

        async with self._lock:
            # Double-check after acquiring lock
            if self._copilot_token and time.time() < (self._expires_at - 120):
                return self._copilot_token

            try:
                return await self._refresh_token()
            except Exception as e:
                # If refresh fails but we still have a token that hasn't hard-expired,
                # fall back to it rather than crashing every request
                if self._copilot_token and time.time() < self._expires_at:
                    logger.warning(
                        "Token refresh failed (%s), using existing token (expires at %s)",
                        e,
                        self._expires_at,
                    )
                    return self._copilot_token
                raise

    async def _refresh_token(self) -> str:
        """Exchange GitHub token for a fresh Copilot token."""
        logger.info("Refreshing Copilot API token")
        resp = await self._client.get(
            COPILOT_TOKEN_URL,
            headers={
                **COPILOT_HEADERS,
                "Authorization": f"token {self._github_token}",
            },
        )

        if resp.status_code != 200:
            body = resp.text[:500]
            logger.error(
                "Copilot token endpoint returned %d: %s", resp.status_code, body
            )

        if resp.status_code == 401:
            raise RuntimeError(
                "GitHub token is invalid or expired. "
                "Please re-authenticate (delete ~/.config/clauderock/github_token and restart)."
            )
        if resp.status_code == 403:
            raise RuntimeError(
                f"Copilot token request denied (403). Response: {resp.text[:200]}"
            )

        resp.raise_for_status()
        data = resp.json()

        self._copilot_token = data["token"]
        self._expires_at = data.get("expires_at", time.time() + 1800)
        logger.info("Copilot token refreshed, expires at %s", self._expires_at)

        return self._copilot_token

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
