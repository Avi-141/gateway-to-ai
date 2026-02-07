"""Tests for claudegate/copilot_auth.py."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudegate.copilot_auth import (
    CopilotAuth,
    _load_persisted_token,
    _persist_token,
    device_flow_login,
    get_github_token,
)

# --- _load_persisted_token ---


class TestLoadPersistedToken:
    def test_file_exists(self, tmp_path, monkeypatch):
        token_file = tmp_path / "github_token"
        token_file.write_text("gho_test123")
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)
        assert _load_persisted_token() == "gho_test123"

    def test_empty_file(self, tmp_path, monkeypatch):
        token_file = tmp_path / "github_token"
        token_file.write_text("")
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)
        assert _load_persisted_token() is None

    def test_missing_file(self, tmp_path, monkeypatch):
        token_file = tmp_path / "nonexistent"
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)
        assert _load_persisted_token() is None


# --- _persist_token ---


class TestPersistToken:
    def test_creates_file(self, tmp_path, monkeypatch):
        token_dir = tmp_path / "subdir"
        token_file = token_dir / "github_token"
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_DIR", token_dir)
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)

        _persist_token("gho_abc")
        assert token_file.read_text() == "gho_abc"

    def test_creates_dir(self, tmp_path, monkeypatch):
        token_dir = tmp_path / "new" / "dir"
        token_file = token_dir / "github_token"
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_DIR", token_dir)
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)

        _persist_token("token")
        assert token_dir.exists()

    def test_sets_permissions(self, tmp_path, monkeypatch):
        token_dir = tmp_path
        token_file = token_dir / "github_token"
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_DIR", token_dir)
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)

        _persist_token("token")
        mode = token_file.stat().st_mode & 0o777
        assert mode == 0o600


# --- get_github_token ---


class TestGetGithubToken:
    def test_from_env_var(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "gho_env_token")
        assert get_github_token() == "gho_env_token"

    def test_from_persisted_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        token_file = tmp_path / "github_token"
        token_file.write_text("gho_persisted")
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)
        assert get_github_token() == "gho_persisted"

    def test_falls_through_to_device_flow(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        token_file = tmp_path / "nonexistent"
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)

        with patch("claudegate.copilot_auth.device_flow_login", return_value="gho_device") as mock_login:
            result = get_github_token()
            assert result == "gho_device"
            mock_login.assert_called_once()


# --- device_flow_login ---


class TestDeviceFlowLogin:
    def test_success_flow(self, monkeypatch):
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_DIR", MagicMock())
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", MagicMock())

        mock_client = MagicMock()
        # First call: device code request
        device_resp = MagicMock()
        device_resp.json.return_value = {
            "device_code": "dc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 0,
            "expires_in": 900,
        }
        # Second call: token poll - immediate success
        token_resp = MagicMock()
        token_resp.json.return_value = {"access_token": "gho_success"}

        mock_client.post.side_effect = [device_resp, token_resp]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("claudegate.copilot_auth.httpx.Client", return_value=mock_client),
            patch("claudegate.copilot_auth.time.sleep"),
            patch("claudegate.copilot_auth._persist_token"),
        ):
            result = device_flow_login()

        assert result == "gho_success"

    def test_slow_down(self, monkeypatch):
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_DIR", MagicMock())
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", MagicMock())

        mock_client = MagicMock()
        device_resp = MagicMock()
        device_resp.json.return_value = {
            "device_code": "dc123",
            "user_code": "ABCD",
            "verification_uri": "https://github.com/login/device",
            "interval": 0,
            "expires_in": 900,
        }
        slow_resp = MagicMock()
        slow_resp.json.return_value = {"error": "slow_down"}
        token_resp = MagicMock()
        token_resp.json.return_value = {"access_token": "gho_ok"}

        mock_client.post.side_effect = [device_resp, slow_resp, token_resp]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("claudegate.copilot_auth.httpx.Client", return_value=mock_client),
            patch("claudegate.copilot_auth.time.sleep"),
            patch("claudegate.copilot_auth._persist_token"),
        ):
            result = device_flow_login()

        assert result == "gho_ok"

    def test_expired_token_error(self, monkeypatch):
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_DIR", MagicMock())
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", MagicMock())

        mock_client = MagicMock()
        device_resp = MagicMock()
        device_resp.json.return_value = {
            "device_code": "dc123",
            "user_code": "ABCD",
            "verification_uri": "https://github.com/login/device",
            "interval": 0,
            "expires_in": 900,
        }
        error_resp = MagicMock()
        error_resp.json.return_value = {"error": "expired_token"}

        mock_client.post.side_effect = [device_resp, error_resp]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("claudegate.copilot_auth.httpx.Client", return_value=mock_client),
            patch("claudegate.copilot_auth.time.sleep"),
            pytest.raises(RuntimeError, match="expired"),
        ):
            device_flow_login()

    def test_access_denied_error(self, monkeypatch):
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_DIR", MagicMock())
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", MagicMock())

        mock_client = MagicMock()
        device_resp = MagicMock()
        device_resp.json.return_value = {
            "device_code": "dc123",
            "user_code": "ABCD",
            "verification_uri": "https://github.com/login/device",
            "interval": 0,
            "expires_in": 900,
        }
        error_resp = MagicMock()
        error_resp.json.return_value = {"error": "access_denied"}

        mock_client.post.side_effect = [device_resp, error_resp]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("claudegate.copilot_auth.httpx.Client", return_value=mock_client),
            patch("claudegate.copilot_auth.time.sleep"),
            pytest.raises(RuntimeError, match="denied"),
        ):
            device_flow_login()


# --- CopilotAuth ---


class TestCopilotAuth:
    @pytest.mark.anyio
    async def test_fresh_token_refresh(self):
        auth = CopilotAuth("gho_test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "token": "copilot_token_123",
            "expires_at": time.time() + 1800,
        }

        with patch.object(auth._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            token = await auth.get_token()

        assert token == "copilot_token_123"
        await auth.close()

    @pytest.mark.anyio
    async def test_cached_token(self):
        auth = CopilotAuth("gho_test")
        auth._copilot_token = "cached"
        auth._expires_at = time.time() + 600  # not expired

        token = await auth.get_token()
        assert token == "cached"
        await auth.close()

    @pytest.mark.anyio
    async def test_auto_refresh_on_expiry(self):
        auth = CopilotAuth("gho_test")
        auth._copilot_token = "old_token"
        auth._expires_at = time.time() - 10  # already expired

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "token": "new_token",
            "expires_at": time.time() + 1800,
        }

        with patch.object(auth._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            token = await auth.get_token()

        assert token == "new_token"
        await auth.close()

    @pytest.mark.anyio
    async def test_refresh_failure_fallback(self):
        auth = CopilotAuth("gho_test")
        auth._copilot_token = "still_valid"
        auth._expires_at = time.time() + 10  # within buffer but not hard-expired

        with patch.object(auth._client, "get", new_callable=AsyncMock, side_effect=RuntimeError("network error")):
            token = await auth.get_token()

        # Falls back to existing token
        assert token == "still_valid"
        await auth.close()

    @pytest.mark.anyio
    async def test_close(self):
        auth = CopilotAuth("gho_test")
        with patch.object(auth._client, "aclose", new_callable=AsyncMock) as mock_close:
            await auth.close()
            mock_close.assert_called_once()
