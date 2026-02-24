"""CLI handler for the 'claudegate backend' command."""

import json
import os
import sys
import urllib.request

from .config import DEFAULT_HOST, DEFAULT_PORT


def _server_url() -> str:
    host = os.environ.get("CLAUDEGATE_HOST", DEFAULT_HOST)
    port = os.environ.get("CLAUDEGATE_PORT", str(DEFAULT_PORT))
    return f"http://{host}:{port}"


def backend_command(value: str | None) -> int:
    """Get or set the backend on the running server.

    Args:
        value: Backend value like 'copilot', 'bedrock', 'copilot,bedrock'.
               None to query current backend.

    Returns:
        Exit code (0 success, 1 error).
    """
    base = _server_url()

    if value is None:
        # GET current backend
        try:
            req = urllib.request.Request(f"{base}/api/backend")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            primary = data.get("primary", "?")
            fallback = data.get("fallback", "")
            if fallback:
                print(f"{primary},{fallback}")
            else:
                print(primary)
            return 0
        except urllib.error.URLError as e:
            print(f"Error: cannot connect to claudegate at {base} — {e.reason}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        # POST to switch backend
        try:
            body = json.dumps({"backend": value}).encode()
            req = urllib.request.Request(
                f"{base}/api/backend",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            primary = data.get("primary", "?")
            fallback = data.get("fallback", "")
            changed = data.get("changed", False)
            display = f"{primary},{fallback}" if fallback else primary
            if changed:
                print(f"Backend switched to: {display}")
            else:
                print(f"Backend already set to: {display}")
            return 0
        except urllib.error.HTTPError as e:
            try:
                err_data = json.loads(e.read().decode())
                print(f"Error: {err_data.get('error', e)}", file=sys.stderr)
            except Exception:
                print(f"Error: HTTP {e.code}", file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(f"Error: cannot connect to claudegate at {base} — {e.reason}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
