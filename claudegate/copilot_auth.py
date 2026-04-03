"""GitHub OAuth device flow and token management for Copilot API."""

import os
import time

import httpx

from .config import CONFIG_DIR, SSL_CONTEXT, logger

# OpenCode OAuth App client ID — impersonate OpenCode's flow
COPILOT_CLIENT_ID = "Ov23li8tweQw6odWQebz"
COPILOT_SCOPE = "read:user"

# Impersonated OpenCode version for User-Agent header
OPENCODE_VERSION = "1.3.13"

# Token persistence path
TOKEN_FILE = CONFIG_DIR / "github_token"

# GitHub OAuth endpoints
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_OAUTH_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105

# Headers matching OpenCode's Copilot plugin
COPILOT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": f"opencode/{OPENCODE_VERSION}",
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
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
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
    with httpx.Client(verify=SSL_CONTEXT) as client:
        # Request device code
        resp = client.post(
            GITHUB_DEVICE_CODE_URL,
            json={"client_id": COPILOT_CLIENT_ID, "scope": COPILOT_SCOPE},
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
                json={
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
