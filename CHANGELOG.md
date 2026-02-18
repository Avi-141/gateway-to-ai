# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Non-Claude models (GPT, Gemini, Grok, etc.) are now accessible via `/v1/messages` (Anthropic API) when the Copilot backend is available. Previously these models were only reachable through `/v1/chat/completions`. When Bedrock is the primary backend with Copilot as fallback, non-Claude model requests route directly to Copilot. Bedrock-only configurations return a clear 400 error for unsupported models.
- `/v1/models` now includes non-Claude models when Bedrock is the primary backend with Copilot as fallback.
- OpenAI Responses API support for codex models (`gpt-5.x-codex`) that only support the `/responses` endpoint. Requests are automatically translated between Anthropic/OpenAI and Responses API formats, with full streaming support.

### Fixed

- Handle Copilot backend token limit errors (128K context window) gracefully. When the prompt exceeds the Copilot model's context limit, the proxy now returns an Anthropic-format error that triggers Claude Code's auto-compaction instead of surfacing a raw backend error. If a Bedrock fallback is configured, requests automatically retry on Bedrock (200K context) before returning an error.
- Fix model routing substring bug where `claude-sonnet-4-6` matched `claude-sonnet-4` instead of falling back to the newest available version (`claude-sonnet-4.5`). Adds a smart version fallback (`_find_newest_available_claude_model`) that parses model family and version from the dynamic Copilot registry, working for all families (opus, sonnet, haiku) without hardcoded names.

## [0.2.0] - 2026-02-18

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
