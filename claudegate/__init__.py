"""Claudegate - Anthropic API to AWS Bedrock proxy."""

import argparse
import os
import sys

from .app import __version__, app
from .config import DEFAULT_HOST, DEFAULT_PORT, LOGGING_CONFIG

__all__ = ["app", "main", "__version__"]


def _start_server() -> None:
    """Start the uvicorn server."""
    import uvicorn

    host = os.environ.get("CLAUDEGATE_HOST", DEFAULT_HOST)
    port = int(os.environ.get("CLAUDEGATE_PORT", DEFAULT_PORT))

    # Uvicorn handles SIGTERM/SIGINT gracefully
    # Lifespan context manager handles startup/shutdown logging
    # Pass LOGGING_CONFIG to unify log format between uvicorn and app
    uvicorn.run(app, host=host, port=port, log_config=LOGGING_CONFIG)


def main() -> None:
    """Entry point for the proxy server and service management commands."""
    parser = argparse.ArgumentParser(
        prog="claudegate",
        description="Anthropic API proxy for AWS Bedrock and GitHub Copilot backends",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"claudegate {__version__}",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["install", "uninstall", "start", "stop", "restart", "status", "logs"],
        default=None,
        help="service management command (omit to start the server)",
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=100,
        help="number of log lines to show before following (logs command only)",
    )
    parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        default=True,
        help="follow logs (logs command only, default: true)",
    )
    parser.add_argument(
        "--no-follow",
        action="store_false",
        dest="follow",
        help="do not follow logs after showing initial output (logs command only)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="show logs since a given time (logs command only)",
    )
    parser.add_argument(
        "--env",
        action="store_true",
        default=False,
        help="capture current CLAUDEGATE_*/AWS_*/GITHUB_TOKEN env vars into the service file (install only)",
    )

    args = parser.parse_args()

    if args.command is None:
        _start_server()
        return

    from .service import (
        install_service,
        restart_service,
        service_logs,
        service_status,
        start_service,
        stop_service,
        uninstall_service,
    )

    if args.command == "install":
        sys.exit(install_service(capture_env=args.env))
    elif args.command == "uninstall":
        sys.exit(uninstall_service())
    elif args.command == "start":
        sys.exit(start_service())
    elif args.command == "stop":
        sys.exit(stop_service())
    elif args.command == "restart":
        sys.exit(restart_service())
    elif args.command == "status":
        sys.exit(service_status())
    elif args.command == "logs":
        sys.exit(service_logs(lines=args.lines, follow=args.follow, since=args.since))
