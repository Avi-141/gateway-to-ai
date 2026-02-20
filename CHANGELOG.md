# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- **Multimodal image support** — images in user messages are now properly translated between all API formats instead of being dropped. Base64 and URL image sources are supported across Anthropic (`/v1/messages`), OpenAI (`/v1/chat/completions`), and Responses (`/v1/responses`) endpoints. The Bedrock path already worked natively; this adds full image translation for Copilot and cross-format paths.
- **`/v1/responses` endpoint** — OpenAI Responses API endpoint for clients that target the newer Responses API format (e.g., [Codex CLI](https://github.com/openai/codex)). Supports streaming and non-streaming, tool use, and all backends. Copilot models that natively support `/responses` get zero-translation passthrough; others are translated automatically. Bedrock requests are translated via Anthropic format. Includes full fallback orchestration matching the other endpoints.
- **Server-side tool routing** — requests containing server-side tools (e.g. `web_search_20250305`) are automatically routed to Bedrock when Copilot is the primary backend, since Copilot does not support them. Falls back to stripping the tools if Bedrock is unavailable.
- `/v1/models` now passes through `limits` (context_window, max_prompt, max_output) from the Copilot API when available.
- **Web dashboard** — a self-contained HTML page showing server status, service info, available models with token limits, and a live log viewer with level filtering.
- **OS trust store support** — use the native OS certificate store (macOS Keychain, Windows CertStore, Linux OpenSSL) via `truststore` so corporate SSL inspection certificates (e.g. enterprise SSL inspection) are trusted automatically without manual cert installation.
- **Claude Code plugin** — `/models` skill available as a plugin. Install with `/plugin marketplace add github.com/Avi-141/gateway-to-ai` then `/plugin install models@claudegate`.

### Changed

- `claudegate install` is now idempotent — if a service is already installed, it automatically stops and reinstalls instead of erroring out. This makes upgrades simpler (`uv tool install --force ... && claudegate install`).

### Fixed

- Streaming token tracking — input token counts are now reported accurately in streaming mode via deferred usage emission and tiktoken-based estimation, enabling Claude Code's context remaining indicator and auto-compaction.
- Clamp `max_output_tokens` to a minimum of 16 for the Responses API. Claude Code occasionally sends `max_tokens: 1` (probe requests), which the Responses API rejects. This fixes errors when using codex models (`gpt-5.x-codex`).

## [0.3.0] - 2026-02-18

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
