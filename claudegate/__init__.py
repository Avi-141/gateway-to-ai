"""Claudegate - Anthropic API to AWS Bedrock proxy."""

import os

from .app import __version__, app
from .config import DEFAULT_HOST, DEFAULT_PORT, LOGGING_CONFIG

__all__ = ["app", "main", "__version__"]


def main() -> None:
    """Entry point for the proxy server."""
    import uvicorn

    host = os.environ.get("HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", DEFAULT_PORT))

    # Uvicorn handles SIGTERM/SIGINT gracefully
    # Lifespan context manager handles startup/shutdown logging
    # Pass LOGGING_CONFIG to unify log format between uvicorn and app
    uvicorn.run(app, host=host, port=port, log_config=LOGGING_CONFIG)
