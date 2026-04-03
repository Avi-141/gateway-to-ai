"""Tests for claudegate/copilot_auth.py."""

from unittest.mock import MagicMock, patch

import pytest

from claudegate.copilot_auth import (
    COPILOT_CLIENT_ID,
    COPILOT_SCOPE,
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
        monkeypatch.setattr("claudegate.copilot_auth.CONFIG_DIR", token_dir)
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)

        _persist_token("gho_abc")
        assert token_file.read_text() == "gho_abc"

    def test_creates_dir(self, tmp_path, monkeypatch):
        token_dir = tmp_path / "new" / "dir"
        token_file = token_dir / "github_token"
        monkeypatch.setattr("claudegate.copilot_auth.CONFIG_DIR", token_dir)
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)

        _persist_token("token")
        assert token_dir.exists()

    def test_sets_permissions(self, tmp_path, monkeypatch):
        token_dir = tmp_path
        token_file = token_dir / "github_token"
        monkeypatch.setattr("claudegate.copilot_auth.CONFIG_DIR", token_dir)
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

        with (
            patch("claudegate.copilot_auth.device_flow_login", return_value="gho_device") as mock_login,
            patch("claudegate.copilot_auth.sys") as mock_sys,
        ):
            mock_sys.stdin.isatty.return_value = True
            result = get_github_token()
            assert result == "gho_device"
            mock_login.assert_called_once()

    def test_non_interactive_raises_without_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        token_file = tmp_path / "nonexistent"
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", token_file)

        with patch("claudegate.copilot_auth.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            with pytest.raises(RuntimeError, match="No GitHub token found and no interactive terminal"):
                get_github_token()


# --- device_flow_login ---


class TestDeviceFlowLogin:
    def test_success_flow(self, monkeypatch):
        monkeypatch.setattr("claudegate.copilot_auth.CONFIG_DIR", MagicMock())
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
        monkeypatch.setattr("claudegate.copilot_auth.CONFIG_DIR", MagicMock())
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
        monkeypatch.setattr("claudegate.copilot_auth.CONFIG_DIR", MagicMock())
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
        monkeypatch.setattr("claudegate.copilot_auth.CONFIG_DIR", MagicMock())
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

    def test_uses_correct_client_id_and_scope(self, monkeypatch):
        """Verify device flow uses claudegate's own OAuth App client ID and scope."""
        monkeypatch.setattr("claudegate.copilot_auth.CONFIG_DIR", MagicMock())
        monkeypatch.setattr("claudegate.copilot_auth.TOKEN_FILE", MagicMock())

        mock_client = MagicMock()
        device_resp = MagicMock()
        device_resp.json.return_value = {
            "device_code": "dc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 0,
            "expires_in": 900,
        }
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
            device_flow_login()

        # Verify the device code request used the correct client_id and scope
        first_call = mock_client.post.call_args_list[0]
        json_body = first_call.kwargs.get("json") or first_call[1].get("json")
        assert json_body is not None
        assert json_body["client_id"] == COPILOT_CLIENT_ID
        assert json_body["scope"] == COPILOT_SCOPE
        assert COPILOT_CLIENT_ID == "Ov23li8tweQw6odWQebz"
        assert COPILOT_SCOPE == "read:user"
