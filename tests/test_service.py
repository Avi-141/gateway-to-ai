"""Tests for claudegate service management (install/uninstall/status)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from claudegate.service import (
    _capture_env_vars,
    _detect_platform,
    _generate_plist,
    _generate_systemd_unit,
    _resolve_binary,
    install_service,
    service_logs,
    service_status,
    uninstall_service,
)

# -- Platform detection ------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("Darwin", "macos"),
        ("Linux", "linux"),
        ("Windows", "windows"),
        ("FreeBSD", "freebsd"),
    ],
)
def test_detect_platform(system, expected):
    with patch("claudegate.service.platform.system", return_value=system):
        assert _detect_platform() == expected


# -- Binary resolution -------------------------------------------------------


def test_resolve_binary_found():
    with patch("claudegate.service.shutil.which", return_value="/usr/local/bin/claudegate"):
        assert _resolve_binary() == "/usr/local/bin/claudegate"


def test_resolve_binary_not_found():
    with patch("claudegate.service.shutil.which", return_value=None):
        assert _resolve_binary() is None


# -- Env var capture ---------------------------------------------------------


def test_capture_env_vars():
    env = {
        "CLAUDEGATE_BACKEND": "copilot",
        "AWS_REGION": "us-east-1",
        "BEDROCK_READ_TIMEOUT": "300",
        "GITHUB_TOKEN": "ghp_abc123",
        "HOME": "/home/user",
        "PATH": "/usr/bin",
    }
    with patch.dict(os.environ, env, clear=True):
        result = _capture_env_vars()
    assert result == {
        "AWS_REGION": "us-east-1",
        "BEDROCK_READ_TIMEOUT": "300",
        "CLAUDEGATE_BACKEND": "copilot",
        "GITHUB_TOKEN": "ghp_abc123",
    }


def test_capture_env_vars_empty():
    with patch.dict(os.environ, {"HOME": "/home/user"}, clear=True):
        result = _capture_env_vars()
    assert result == {}


# -- Plist generation --------------------------------------------------------


def test_generate_plist_basic():
    plist = _generate_plist("/usr/local/bin/claudegate")
    assert '<?xml version="1.0"' in plist
    assert "<string>com.claudegate</string>" in plist
    assert "<string>/usr/local/bin/claudegate</string>" in plist
    assert "<key>NO_COLOR</key>" in plist
    assert "<key>PATH</key>" in plist
    assert "<true/>" in plist


def test_generate_plist_with_env():
    plist = _generate_plist("/usr/local/bin/claudegate", {"AWS_REGION": "us-west-2"})
    assert "<key>AWS_REGION</key>" in plist
    assert "<string>us-west-2</string>" in plist
    assert "<key>NO_COLOR</key>" in plist


def test_generate_plist_xml_escaping():
    plist = _generate_plist("/path/to/claudegate", {"KEY": "val<>&"})
    assert "<string>val&lt;&gt;&amp;</string>" in plist


def test_generate_plist_structure():
    plist = _generate_plist("/bin/claudegate")
    # Verify it's valid-looking XML (starts with declaration, has plist root)
    assert plist.startswith('<?xml version="1.0"')
    assert "<plist version" in plist
    assert plist.strip().endswith("</plist>")
    # Verify key sections exist
    assert "<key>Label</key>" in plist
    assert "<key>ProgramArguments</key>" in plist
    assert "<key>RunAtLoad</key>" in plist
    assert "<key>KeepAlive</key>" in plist
    assert "<key>StandardOutPath</key>" in plist
    assert "<key>EnvironmentVariables</key>" in plist


# -- Systemd unit generation -------------------------------------------------


def test_generate_systemd_unit_basic():
    unit = _generate_systemd_unit("/usr/local/bin/claudegate")
    assert "[Unit]" in unit
    assert "[Service]" in unit
    assert "[Install]" in unit
    assert "ExecStart=/usr/local/bin/claudegate" in unit
    assert "Environment=NO_COLOR=1" in unit
    assert "WantedBy=default.target" in unit


def test_generate_systemd_unit_with_env():
    env = {"AWS_REGION": "us-west-2", "CLAUDEGATE_BACKEND": "bedrock"}
    unit = _generate_systemd_unit("/usr/local/bin/claudegate", env)
    assert "Environment=AWS_REGION=us-west-2" in unit
    assert "Environment=CLAUDEGATE_BACKEND=bedrock" in unit
    assert "Environment=NO_COLOR=1" in unit


def test_generate_systemd_unit_sections():
    unit = _generate_systemd_unit("/bin/claudegate")
    assert "After=network.target" in unit
    assert "Type=simple" in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=5" in unit
    assert "StandardOutput=journal" in unit


# -- Install (macOS) --------------------------------------------------------


def test_install_macos_success(tmp_path):
    plist_path = tmp_path / "com.claudegate.plist"

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._resolve_binary", return_value="/usr/local/bin/claudegate"),
        patch("claudegate.service._plist_path", return_value=plist_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = install_service(capture_env=False)

    assert result == 0
    assert plist_path.exists()
    content = plist_path.read_text()
    assert "com.claudegate" in content
    mock_run.assert_called_once()


def test_install_macos_already_exists(tmp_path):
    plist_path = tmp_path / "com.claudegate.plist"
    plist_path.write_text("existing")

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._resolve_binary", return_value="/usr/local/bin/claudegate"),
        patch("claudegate.service._plist_path", return_value=plist_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = install_service(capture_env=False)

    assert result == 0
    content = plist_path.read_text()
    assert "com.claudegate" in content
    # unload (old) + load (new)
    assert mock_run.call_count == 2


def test_install_macos_with_env(tmp_path):
    plist_path = tmp_path / "com.claudegate.plist"

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._resolve_binary", return_value="/usr/local/bin/claudegate"),
        patch("claudegate.service._plist_path", return_value=plist_path),
        patch("claudegate.service.subprocess.run") as mock_run,
        patch.dict(os.environ, {"CLAUDEGATE_BACKEND": "bedrock", "AWS_REGION": "eu-west-1"}, clear=False),
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = install_service(capture_env=True)

    assert result == 0
    content = plist_path.read_text()
    assert "<key>AWS_REGION</key>" in content
    assert "<key>CLAUDEGATE_BACKEND</key>" in content


def test_install_macos_launchctl_fails(tmp_path):
    plist_path = tmp_path / "com.claudegate.plist"

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._resolve_binary", return_value="/usr/local/bin/claudegate"),
        patch("claudegate.service._plist_path", return_value=plist_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=1, stderr="load failed")
        result = install_service(capture_env=False)

    assert result == 1


# -- Install (Linux) --------------------------------------------------------


def test_install_linux_success(tmp_path):
    unit_path = tmp_path / "claudegate.service"

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._resolve_binary", return_value="/usr/local/bin/claudegate"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = install_service(capture_env=False)

    assert result == 0
    assert unit_path.exists()
    content = unit_path.read_text()
    assert "ExecStart=/usr/local/bin/claudegate" in content
    assert mock_run.call_count == 2  # daemon-reload + enable --now


def test_install_linux_already_exists(tmp_path):
    unit_path = tmp_path / "claudegate.service"
    unit_path.write_text("existing")

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._resolve_binary", return_value="/usr/local/bin/claudegate"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = install_service(capture_env=False)

    assert result == 0
    content = unit_path.read_text()
    assert "ExecStart=/usr/local/bin/claudegate" in content
    # stop (old) + daemon-reload + enable --now
    assert mock_run.call_count == 3


# -- Install (common error paths) -------------------------------------------


def test_install_binary_not_found():
    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._resolve_binary", return_value=None),
    ):
        result = install_service(capture_env=False)

    assert result == 1


def test_install_unsupported_platform():
    with (
        patch("claudegate.service._detect_platform", return_value="freebsd"),
        patch("claudegate.service._resolve_binary", return_value="/usr/local/bin/claudegate"),
    ):
        result = install_service(capture_env=False)

    assert result == 1


# -- Uninstall (macOS) -------------------------------------------------------


def test_uninstall_macos_success(tmp_path):
    plist_path = tmp_path / "com.claudegate.plist"
    plist_path.write_text("<plist/>")

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._plist_path", return_value=plist_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = uninstall_service()

    assert result == 0
    assert not plist_path.exists()


def test_uninstall_macos_not_installed(tmp_path):
    plist_path = tmp_path / "com.claudegate.plist"

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._plist_path", return_value=plist_path),
    ):
        result = uninstall_service()

    assert result == 1


# -- Uninstall (Linux) ------------------------------------------------------


def test_uninstall_linux_success(tmp_path):
    unit_path = tmp_path / "claudegate.service"
    unit_path.write_text("[Unit]")

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = uninstall_service()

    assert result == 0
    assert not unit_path.exists()


def test_uninstall_linux_not_installed(tmp_path):
    unit_path = tmp_path / "claudegate.service"

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
    ):
        result = uninstall_service()

    assert result == 1


# -- Status (macOS) ----------------------------------------------------------


def test_status_macos_running(tmp_path, capsys):
    plist_path = tmp_path / "com.claudegate.plist"
    plist_path.write_text("<plist/>")

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._plist_path", return_value=plist_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout='"PID" = 12345;', stderr="")
        result = service_status()

    assert result == 0
    out = capsys.readouterr().out
    assert "running" in out


def test_status_macos_stopped(tmp_path, capsys):
    plist_path = tmp_path / "com.claudegate.plist"
    plist_path.write_text("<plist/>")

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._plist_path", return_value=plist_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        result = service_status()

    assert result == 0
    out = capsys.readouterr().out
    assert "stopped" in out


def test_status_macos_not_installed(tmp_path, capsys):
    plist_path = tmp_path / "com.claudegate.plist"

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service._plist_path", return_value=plist_path),
    ):
        result = service_status()

    assert result == 1
    out = capsys.readouterr().out
    assert "not installed" in out


# -- Status (Linux) ----------------------------------------------------------


def test_status_linux_active(tmp_path, capsys):
    unit_path = tmp_path / "claudegate.service"
    unit_path.write_text("[Unit]")

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="active", stderr="")
        result = service_status()

    assert result == 0
    out = capsys.readouterr().out
    assert "active" in out


def test_status_linux_not_installed(tmp_path, capsys):
    unit_path = tmp_path / "claudegate.service"

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
    ):
        result = service_status()

    assert result == 1
    out = capsys.readouterr().out
    assert "not installed" in out


# -- Status (unsupported) ---------------------------------------------------


def test_status_unsupported_platform():
    with patch("claudegate.service._detect_platform", return_value="freebsd"):
        result = service_status()
    assert result == 1


# -- Logs --------------------------------------------------------------------


def test_logs_linux_follow_with_since(tmp_path):
    unit_path = tmp_path / "claudegate.service"
    unit_path.write_text("[Unit]")

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = service_logs(lines=200, follow=True, since="10m ago")

    assert result == 0
    cmd = mock_run.call_args.args[0]
    assert cmd[:5] == ["journalctl", "--user", "--unit", "claudegate.service", "--lines"]
    assert "200" in cmd
    assert "--since" in cmd
    assert "10m ago" in cmd
    assert "--follow" in cmd


def test_logs_linux_no_follow(tmp_path):
    unit_path = tmp_path / "claudegate.service"
    unit_path.write_text("[Unit]")

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = service_logs(lines=50, follow=False, since=None)

    assert result == 0
    cmd = mock_run.call_args.args[0]
    assert "--follow" not in cmd


def test_logs_linux_not_installed(tmp_path):
    unit_path = tmp_path / "claudegate.service"

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
    ):
        result = service_logs(lines=100, follow=True, since=None)

    assert result == 1


def test_logs_macos_success(tmp_path):
    log_path = tmp_path / "claudegate.log"
    log_path.write_text("line1\n")

    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service.Path") as mock_path_cls,
        patch("claudegate.service.subprocess.run") as mock_run,
    ):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__.return_value = str(log_path)
        mock_path_cls.return_value = mock_path
        mock_run.return_value = MagicMock(returncode=0)

        result = service_logs(lines=25, follow=True, since=None)

    assert result == 0
    cmd = mock_run.call_args.args[0]
    assert cmd == ["tail", "-n", "25", "-f", str(mock_path)]


def test_logs_macos_not_installed(tmp_path):
    with (
        patch("claudegate.service._detect_platform", return_value="macos"),
        patch("claudegate.service.Path") as mock_path_cls,
    ):
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_cls.return_value = mock_path

        result = service_logs(lines=100, follow=True, since=None)

    assert result == 1


def test_logs_invalid_lines():
    with patch("claudegate.service._detect_platform", return_value="linux"):
        result = service_logs(lines=0, follow=True, since=None)

    assert result == 1


def test_logs_unsupported_platform():
    with patch("claudegate.service._detect_platform", return_value="windows"):
        result = service_logs(lines=100, follow=True, since=None)

    assert result == 1


def test_logs_keyboard_interrupt_linux(tmp_path):
    unit_path = tmp_path / "claudegate.service"
    unit_path.write_text("[Unit]")

    with (
        patch("claudegate.service._detect_platform", return_value="linux"),
        patch("claudegate.service._systemd_unit_path", return_value=unit_path),
        patch("claudegate.service.subprocess.run", side_effect=KeyboardInterrupt),
    ):
        result = service_logs(lines=100, follow=True, since=None)

    assert result == 130
