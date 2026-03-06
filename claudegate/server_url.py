"""Server URL discovery file management.

Writes a JSON file on startup so skills and CLI can discover the running server
without relying on environment variables or hardcoded defaults.
"""

import json
import os

from .config import CONFIG_DIR, SERVER_URL_FILE, logger


def write_server_url(host: str, port: int) -> None:
    """Write server URL and PID to the discovery file."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {"url": f"http://{host}:{port}", "pid": os.getpid()}
        SERVER_URL_FILE.write_text(json.dumps(data))
        logger.info("Wrote server URL to %s", SERVER_URL_FILE)
    except OSError as e:
        logger.warning("Failed to write server URL file: %s", e)


def remove_server_url() -> None:
    """Remove the discovery file on clean shutdown."""
    try:
        SERVER_URL_FILE.unlink(missing_ok=True)
        logger.info("Removed server URL file")
    except OSError as e:
        logger.warning("Failed to remove server URL file: %s", e)


def read_server_url() -> str | None:
    """Read the server URL from the discovery file.

    Returns the URL string or None if the file is missing/invalid.
    """
    try:
        data = json.loads(SERVER_URL_FILE.read_text())
        return data["url"]
    except Exception:
        return None
