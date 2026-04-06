"""Service management for claudegate autostart (install/uninstall/status)."""

import os
import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

# Environment variable prefixes to capture with --env
_ENV_PREFIXES = ("CLAUDEGATE_", "AWS_", "BEDROCK_")
_ENV_EXACT = ("GITHUB_TOKEN",)

# Service identifiers
_LAUNCHD_LABEL = "com.claudegate"
_SYSTEMD_UNIT = "claudegate.service"
_SCHTASKS_NAME = "Claudegate"


# -- Output helpers ----------------------------------------------------------


def _header(msg: str) -> None:
    print(f"\n{msg}")


def _step(msg: str) -> None:
    print(f"  {msg}", end="", flush=True)


def _ok(msg: str = "") -> None:
    suffix = f" {msg}" if msg else ""
    print(f"  done.{suffix}")


def _err(msg: str) -> None:
    print(f"  ERROR: {msg}", file=sys.stderr)


# -- Internal helpers --------------------------------------------------------


def _is_running_as_sudo() -> bool:
    """Detect if the process is running via sudo (Unix only)."""
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    return geteuid() == 0 and os.environ.get("SUDO_USER") is not None


def _detect_platform() -> str:
    s = platform.system()
    if s == "Darwin":
        return "macos"
    if s == "Linux":
        return "linux"
    if s == "Windows":
        return "windows"
    return s.lower()


def _resolve_binary() -> str | None:
    return shutil.which("claudegate")


def _capture_env_vars() -> dict[str, str]:
    captured: dict[str, str] = {}
    for key, val in sorted(os.environ.items()):
        if any(key.startswith(p) for p in _ENV_PREFIXES) or key in _ENV_EXACT:
            captured[key] = val
    return captured


# -- launchctl helpers (macOS) -----------------------------------------------


def _launchctl_domain() -> str:
    """Return the launchctl domain target for the current user (e.g. 'gui/501')."""
    return f"gui/{os.getuid()}"


def _launchctl_bootstrap(plist_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", "bootstrap", _launchctl_domain(), str(plist_path)],
        capture_output=True,
        text=True,
    )


def _launchctl_bootout(plist_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", "bootout", _launchctl_domain(), str(plist_path)],
        capture_output=True,
        text=True,
    )


# -- Plist generation (macOS) ------------------------------------------------


def _generate_plist(binary: str, env_vars: dict[str, str] | None = None) -> str:
    env_dict: dict[str, str] = {
        "HOME": str(Path.home()),
        "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin",
        "NO_COLOR": "1",
    }
    if env_vars:
        env_dict.update(env_vars)

    env_xml = ""
    for key, val in sorted(env_dict.items()):
        env_xml += f"        <key>{xml_escape(key)}</key>\n"
        env_xml += f"        <string>{xml_escape(val)}</string>\n"

    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{_LAUNCHD_LABEL}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{xml_escape(binary)}</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>/tmp/claudegate.log</string>
            <key>StandardErrorPath</key>
            <string>/tmp/claudegate.log</string>
            <key>EnvironmentVariables</key>
            <dict>
        {env_xml.rstrip()}
            </dict>
        </dict>
        </plist>
    """
    )


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCHD_LABEL}.plist"


# -- Systemd unit generation (Linux) ----------------------------------------


def _generate_systemd_unit(binary: str, env_vars: dict[str, str] | None = None) -> str:
    env_lines = "Environment=NO_COLOR=1\n"
    if env_vars:
        for key, val in sorted(env_vars.items()):
            env_lines += f"Environment={key}={val}\n"

    return textwrap.dedent(
        f"""\
        [Unit]
        Description=Claudegate - API proxy for AWS Bedrock and GitHub Copilot
        After=network.target

        [Service]
        Type=simple
        ExecStart={binary}
        Restart=on-failure
        RestartSec=5
        {env_lines.rstrip()}

        StandardOutput=journal
        StandardError=journal

        [Install]
        WantedBy=default.target
    """
    )


def _systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / _SYSTEMD_UNIT


# -- Install -----------------------------------------------------------------


def install_service(*, capture_env: bool = False) -> int:
    _header("Installing claudegate as a system service...")

    if _is_running_as_sudo():
        _err("Do not run 'claudegate install' with sudo.")
        _err("The service runs as your user and needs your user's HOME directory.")
        _err("Run without sudo:  claudegate install")
        return 1

    plat = _detect_platform()
    _step(f"Detecting platform... {plat}")
    print()

    binary = _resolve_binary()
    if not binary:
        _err("Could not find 'claudegate' on PATH. Is it installed?")
        return 1
    _step(f"Resolving binary path... {binary}")
    print()

    env_vars: dict[str, str] | None = None
    if capture_env:
        env_vars = _capture_env_vars()
        if env_vars:
            names = ", ".join(env_vars)
            _step(f"Capturing environment variables: {names}")
            print()
        else:
            _step("No CLAUDEGATE_*/AWS_*/GITHUB_TOKEN environment variables found")
            print()

    if plat == "macos":
        return _install_macos(binary, env_vars)
    if plat == "linux":
        return _install_linux(binary, env_vars)
    if plat == "windows":
        return _install_windows(binary)

    _err(f"Unsupported platform: {plat}")
    return 1


def _install_macos(binary: str, env_vars: dict[str, str] | None) -> int:
    path = _plist_path()
    if path.exists():
        _step("Existing service found, reinstalling...")
        print()
        # Bootout before overwriting (ignore errors — service may not be loaded)
        _launchctl_bootout(path)

    plist = _generate_plist(binary, env_vars)

    _step(f"Writing service file to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plist)
    _ok()

    _step("Loading service...")
    result = _launchctl_bootstrap(path)
    if result.returncode != 0:
        _err(f"launchctl bootstrap failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nDone! claudegate is now running as a system service.")
    print("Logs: /tmp/claudegate.log")
    return 0


def _install_linux(binary: str, env_vars: dict[str, str] | None) -> int:
    path = _systemd_unit_path()
    if path.exists():
        _step("Existing service found, reinstalling...")
        print()
        # Stop before overwriting
        subprocess.run(
            ["systemctl", "--user", "stop", _SYSTEMD_UNIT],
            capture_output=True,
            text=True,
        )

    unit = _generate_systemd_unit(binary, env_vars)

    _step(f"Writing service file to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit)
    _ok()

    _step("Reloading systemd...")
    result = subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"daemon-reload failed: {result.stderr.strip()}")
        return 1
    _ok()

    _step("Enabling and starting service...")
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", _SYSTEMD_UNIT],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"enable --now failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nDone! claudegate is now running as a system service.")
    print("Logs: journalctl --user -u claudegate.service -f")
    return 0


def _install_windows(binary: str) -> int:
    _step("Creating scheduled task...")
    result = subprocess.run(
        [
            "schtasks",
            "/Create",
            "/TN",
            _SCHTASKS_NAME,
            "/SC",
            "ONLOGON",
            "/TR",
            binary,
            "/F",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"schtasks failed: {result.stderr.strip()}")
        return 1
    _ok()

    _step("Starting task...")
    result = subprocess.run(
        ["schtasks", "/Run", "/TN", _SCHTASKS_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"schtasks run failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nDone! claudegate is now running as a scheduled task.")
    return 0


# -- Uninstall ---------------------------------------------------------------


def uninstall_service() -> int:
    _header("Uninstalling claudegate system service...")

    if _is_running_as_sudo():
        _err("Do not run 'claudegate uninstall' with sudo.")
        _err("The service runs as your user and needs your user's HOME directory.")
        _err("Run without sudo:  claudegate uninstall")
        return 1

    plat = _detect_platform()
    _step(f"Detecting platform... {plat}")
    print()

    if plat == "macos":
        return _uninstall_macos()
    if plat == "linux":
        return _uninstall_linux()
    if plat == "windows":
        return _uninstall_windows()

    _err(f"Unsupported platform: {plat}")
    return 1


def _uninstall_macos() -> int:
    path = _plist_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Nothing to uninstall.")
        return 1

    _step("Unloading service...")
    _launchctl_bootout(path)
    _ok()

    _step(f"Removing {path}")
    path.unlink()
    _ok()

    print("\nDone! claudegate service has been removed.")
    return 0


def _uninstall_linux() -> int:
    path = _systemd_unit_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Nothing to uninstall.")
        return 1

    _step("Stopping and disabling service...")
    subprocess.run(
        ["systemctl", "--user", "disable", "--now", _SYSTEMD_UNIT],
        capture_output=True,
        text=True,
    )
    _ok()

    _step(f"Removing {path}")
    path.unlink()
    _ok()

    _step("Reloading systemd...")
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
        text=True,
    )
    _ok()

    print("\nDone! claudegate service has been removed.")
    return 0


def _uninstall_windows() -> int:
    _step("Stopping task...")
    subprocess.run(
        ["schtasks", "/End", "/TN", _SCHTASKS_NAME],
        capture_output=True,
        text=True,
    )
    _ok()

    _step("Deleting scheduled task...")
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", _SCHTASKS_NAME, "/F"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"schtasks delete failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nDone! claudegate scheduled task has been removed.")
    return 0


# -- Status ------------------------------------------------------------------


def _launchd_pid(label: str) -> int | None:
    """Return the PID of a launchd service, or None if not running."""
    result = subprocess.run(
        ["launchctl", "list", label],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if '"PID"' in line:
            # e.g. '"PID" = 12345;'
            parts = line.split("=")
            if len(parts) == 2:
                pid_str = parts[1].strip().rstrip(";").strip()
                try:
                    return int(pid_str)
                except ValueError:
                    pass
    return None


def get_service_status() -> dict:
    """Return structured service status data for the dashboard API."""
    plat = _detect_platform()
    result: dict = {"platform": plat, "installed": False, "running": False, "service_file": None}

    if plat == "macos":
        path = _plist_path()
        if path.exists():
            result["installed"] = True
            result["service_file"] = str(path)
            result["running"] = _launchd_pid(_LAUNCHD_LABEL) is not None
    elif plat == "linux":
        path = _systemd_unit_path()
        if path.exists():
            result["installed"] = True
            result["service_file"] = str(path)
            proc = subprocess.run(
                ["systemctl", "--user", "is-active", _SYSTEMD_UNIT],
                capture_output=True,
                text=True,
            )
            result["running"] = proc.stdout.strip() == "active"
    elif plat == "windows":
        proc = subprocess.run(
            ["schtasks", "/Query", "/TN", _SCHTASKS_NAME],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            result["installed"] = True
            result["running"] = "Running" in proc.stdout

    return result


def service_status() -> int:
    plat = _detect_platform()

    if plat == "macos":
        return _status_macos()
    if plat == "linux":
        return _status_linux()
    if plat == "windows":
        return _status_windows()

    _err(f"Unsupported platform: {plat}")
    return 1


def service_logs(*, lines: int = 100, follow: bool = True, since: str | None = None) -> int:
    if lines <= 0:
        _err("--lines must be a positive integer")
        return 1

    plat = _detect_platform()

    if plat == "macos":
        return _logs_macos(lines=lines, follow=follow, since=since)
    if plat == "linux":
        return _logs_linux(lines=lines, follow=follow, since=since)

    _err(f"'logs' currently supports macOS and Linux only (detected: {plat})")
    return 1


def _logs_macos(*, lines: int, follow: bool, since: str | None) -> int:
    path = Path("/tmp/claudegate.log")  # noqa: S108
    if not path.exists():
        _err("macOS log file not found: /tmp/claudegate.log")
        _err("Install/start the service first with 'claudegate install'.")
        return 1

    tail_cmd = ["tail", "-n", str(lines)]
    if follow:
        tail_cmd.append("-f")
    tail_cmd.append(str(path))

    if since:
        _step("The --since option is not supported for macOS file logs; showing recent lines instead")
        print()

    return _stream_command(tail_cmd)


def _logs_linux(*, lines: int, follow: bool, since: str | None) -> int:
    path = _systemd_unit_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Install/start the service first with 'claudegate install'.")
        return 1

    cmd = ["journalctl", "--user", "--unit", _SYSTEMD_UNIT, "--lines", str(lines), "--no-pager"]
    if since:
        cmd.extend(["--since", since])
    if follow:
        cmd.append("--follow")
    return _stream_command(cmd)


def _stream_command(cmd: list[str]) -> int:
    _step(f"Running: {shlex.join(cmd)}")
    print()
    try:
        proc = subprocess.run(cmd)
    except KeyboardInterrupt:
        print()
        return 130
    except FileNotFoundError:
        _err(f"Required command not found: {cmd[0]}")
        return 1
    return proc.returncode


def _status_macos() -> int:
    path = _plist_path()
    if not path.exists():
        print("claudegate is not installed as a system service.")
        return 1

    print(f"Service file: {path}")
    pid = _launchd_pid(_LAUNCHD_LABEL)
    if pid is not None:
        print(f"Status: running (PID {pid})")
    else:
        print("Status: loaded but not running")
    return 0


def _status_linux() -> int:
    path = _systemd_unit_path()
    if not path.exists():
        print("claudegate is not installed as a system service.")
        return 1

    print(f"Service file: {path}")
    result = subprocess.run(
        ["systemctl", "--user", "is-active", _SYSTEMD_UNIT],
        capture_output=True,
        text=True,
    )
    state = result.stdout.strip()
    print(f"Status: {state}")
    return 0


def _status_windows() -> int:
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", _SCHTASKS_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("claudegate is not installed as a scheduled task.")
        return 1

    print(f"Scheduled task: {_SCHTASKS_NAME}")
    print(result.stdout.strip())
    return 0


# -- Start -------------------------------------------------------------------


def start_service() -> int:
    """Start the claudegate service."""
    _header("Starting claudegate service...")

    plat = _detect_platform()

    if plat == "macos":
        return _start_macos()
    if plat == "linux":
        return _start_linux()
    if plat == "windows":
        return _start_windows()

    _err(f"Unsupported platform: {plat}")
    return 1


def _start_macos() -> int:
    path = _plist_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Install the service first with 'claudegate install'.")
        return 1

    _step("Loading service...")
    result = _launchctl_bootstrap(path)
    if result.returncode != 0:
        _err(f"launchctl bootstrap failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service started.")
    return 0


def _start_linux() -> int:
    path = _systemd_unit_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Install the service first with 'claudegate install'.")
        return 1

    _step("Starting service...")
    result = subprocess.run(
        ["systemctl", "--user", "start", _SYSTEMD_UNIT],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"systemctl start failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service started.")
    return 0


def _start_windows() -> int:
    _step("Starting task...")
    result = subprocess.run(
        ["schtasks", "/Run", "/TN", _SCHTASKS_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"schtasks run failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service started.")
    return 0


# -- Stop --------------------------------------------------------------------


def stop_service() -> int:
    """Stop the claudegate service."""
    _header("Stopping claudegate service...")

    plat = _detect_platform()

    if plat == "macos":
        return _stop_macos()
    if plat == "linux":
        return _stop_linux()
    if plat == "windows":
        return _stop_windows()

    _err(f"Unsupported platform: {plat}")
    return 1


def _stop_macos() -> int:
    path = _plist_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Install the service first with 'claudegate install'.")
        return 1

    _step("Unloading service...")
    result = _launchctl_bootout(path)
    if result.returncode != 0:
        _err(f"launchctl bootout failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service stopped.")
    return 0


def _stop_linux() -> int:
    path = _systemd_unit_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Install the service first with 'claudegate install'.")
        return 1

    _step("Stopping service...")
    result = subprocess.run(
        ["systemctl", "--user", "stop", _SYSTEMD_UNIT],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"systemctl stop failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service stopped.")
    return 0


def _stop_windows() -> int:
    _step("Stopping task...")
    result = subprocess.run(
        ["schtasks", "/End", "/TN", _SCHTASKS_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"schtasks end failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service stopped.")
    return 0


# -- Restart -----------------------------------------------------------------


def restart_service() -> int:
    """Restart the claudegate service."""
    _header("Restarting claudegate service...")

    plat = _detect_platform()

    if plat == "macos":
        return _restart_macos()
    if plat == "linux":
        return _restart_linux()
    if plat == "windows":
        return _restart_windows()

    _err(f"Unsupported platform: {plat}")
    return 1


def _restart_macos() -> int:
    path = _plist_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Install the service first with 'claudegate install'.")
        return 1

    _step("Unloading service...")
    _launchctl_bootout(path)
    _ok()

    _step("Loading service...")
    result = _launchctl_bootstrap(path)
    if result.returncode != 0:
        _err(f"launchctl bootstrap failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service restarted.")
    return 0


def _restart_linux() -> int:
    path = _systemd_unit_path()
    if not path.exists():
        _err(f"Service file not found: {path}")
        _err("Install the service first with 'claudegate install'.")
        return 1

    _step("Restarting service...")
    result = subprocess.run(
        ["systemctl", "--user", "restart", _SYSTEMD_UNIT],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"systemctl restart failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service restarted.")
    return 0


def _restart_windows() -> int:
    _step("Stopping task...")
    subprocess.run(
        ["schtasks", "/End", "/TN", _SCHTASKS_NAME],
        capture_output=True,
        text=True,
    )
    _ok()

    _step("Starting task...")
    result = subprocess.run(
        ["schtasks", "/Run", "/TN", _SCHTASKS_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"schtasks run failed: {result.stderr.strip()}")
        return 1
    _ok()

    print("\nclaudegate service restarted.")
    return 0
