"""Claudegate - API proxy supporting Anthropic, OpenAI, and Responses APIs via Bedrock and Copilot."""

import os

import typer

from .app import __version__, app
from .config import DEFAULT_HOST, DEFAULT_PORT, LOGGING_CONFIG

__all__ = ["app", "main", "__version__"]

cli = typer.Typer(
    name="claudegate",
    help="API proxy supporting Anthropic, OpenAI, and Responses APIs via AWS Bedrock and GitHub Copilot backends",
    add_completion=False,
    invoke_without_command=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"claudegate {__version__}")
        raise typer.Exit()


def _start_server() -> None:
    """Start the uvicorn server."""
    import uvicorn

    host = os.environ.get("CLAUDEGATE_HOST", DEFAULT_HOST)
    port = int(os.environ.get("CLAUDEGATE_PORT", DEFAULT_PORT))

    uvicorn.run(app, host=host, port=port, log_config=LOGGING_CONFIG)


@cli.callback()
def _default(
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """API proxy supporting Anthropic, OpenAI, and Responses APIs via AWS Bedrock and GitHub Copilot backends."""
    if ctx.invoked_subcommand is None:
        _start_server()


@cli.command()
def install(
    env: bool = typer.Option(  # noqa: B008
        False,
        "--env",
        help="Capture current CLAUDEGATE_*/AWS_*/GITHUB_TOKEN env vars into the service file.",
    ),
) -> None:
    """Install the autostart service."""
    from .service import install_service

    raise typer.Exit(install_service(capture_env=env))


@cli.command()
def uninstall() -> None:
    """Uninstall the autostart service."""
    from .service import uninstall_service

    raise typer.Exit(uninstall_service())


@cli.command()
def start() -> None:
    """Start the background service."""
    from .service import start_service

    raise typer.Exit(start_service())


@cli.command()
def stop() -> None:
    """Stop the background service."""
    from .service import stop_service

    raise typer.Exit(stop_service())


@cli.command()
def restart() -> None:
    """Restart the background service."""
    from .service import restart_service

    raise typer.Exit(restart_service())


@cli.command()
def status() -> None:
    """Show service status."""
    from .service import service_status

    raise typer.Exit(service_status())


@cli.command()
def logs(
    lines: int = typer.Option(100, "--lines", "-n", help="Number of log lines to show before following."),  # noqa: B008
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f", help="Follow logs after showing initial output."),  # noqa: B008
    since: str | None = typer.Option(None, "--since", help="Show logs since a given time."),  # noqa: B008
) -> None:
    """Show service logs."""
    from .service import service_logs

    raise typer.Exit(service_logs(lines=lines, follow=follow, since=since))


@cli.command()
def backend(
    value: str | None = typer.Argument(None, help="Backend value to set (omit to show current)."),  # noqa: B008
) -> None:
    """Get or set the active backend."""
    from .cli_backend import backend_command

    raise typer.Exit(backend_command(value))


def main() -> None:
    """Entry point for the proxy server and service management commands."""
    cli()
