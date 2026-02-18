# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed

- Default listen address changed from `0.0.0.0` to `127.0.0.1` (localhost only). Set `CLAUDEGATE_HOST=0.0.0.0` to allow remote connections.

## [0.1.0] - 2026-02-17

### Added

- Anthropic API proxy with two backends: GitHub Copilot and AWS Bedrock
- OpenAI-compatible `/v1/chat/completions` endpoint for Open WebUI and other clients
- Non-Claude model support (GPT-4o, o3-mini, etc.) via Copilot backend
- Cross-backend fallback (`CLAUDEGATE_BACKEND=copilot,bedrock`)
- Streaming and non-streaming response support
- Full tool use round-trip translation (Copilot)
- Extended thinking support (Bedrock)
- GitHub OAuth device flow for Copilot authentication
- Token counting via tiktoken
- `claudegate install` / `uninstall` / `status` CLI commands for autostart
- macOS (launchd), Linux (systemd), and Windows (schtasks) service support
- Health checks with backend connectivity verification
