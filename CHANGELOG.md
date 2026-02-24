# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.5.0] - 2026-02-24

### Added

- **Runtime backend switching** — switch between `copilot`, `bedrock`, or fallback configurations at runtime without restarting the server. Available via API (`POST /api/backend`), web dashboard dropdown, CLI (`claudegate backend <value>`), and Claude Code skill (`/backend`). Changes are ephemeral; the `CLAUDEGATE_BACKEND` env var remains the startup source of truth. If switching to Copilot and it wasn't initialized at startup, authentication is performed lazily.
- **`/backend` Claude Code skill** — install the claudegate plugin to switch backends from Claude Code with `/backend copilot`, `/backend bedrock,copilot`, etc.
- **`claudegate backend` CLI command** — query (`claudegate backend`) or switch (`claudegate backend copilot`) the backend on a running server.
- **`start`, `stop`, `restart` service commands** — manage the service lifecycle without a full uninstall/reinstall cycle. Works on macOS (launchctl), Linux (systemd), and Windows (schtasks).
- **Copilot premium usage quota** — `/api/status` now includes Copilot premium request usage data (plan type, used/total/remaining, percent used, reset date) fetched from the GitHub API with a 60-second stale-while-revalidate cache. The dashboard shows a color-coded progress bar (green/orange/red) and stale data indicator.
- **Token count scaling** — dynamically scale `input_tokens` and `output_tokens` reported to Claude Code so its percentage-based context tracking accurately reflects Copilot's actual capacity. Looks up each model's `max_prompt_tokens` from the Copilot `/models` API and applies a per-model scaling factor. Supports both standard 200k and 1M context window variants (detected via `anthropic-beta: context-1m` header).
- **`claudegate logs` command** — tail service logs on macOS (`tail -f /tmp/claudegate.log`) and Linux (`journalctl`). Supports `--lines` / `-n`, `--follow` / `--no-follow`, and `--since` (Linux only).
- **Request stats dashboard panel** — the web dashboard now shows a "Request Stats" panel with live counters for total requests, requests per backend, errors, and fallbacks. Stats are tracked in-memory (ephemeral, reset on restart) and included in `/api/status` under a `stats` key.

### Changed

- **Default model updated to Sonnet 4.6** — both `DEFAULT_BEDROCK_MODEL` and `DEFAULT_COPILOT_MODEL` now default to Claude Sonnet 4.6 instead of 4.5. Added Sonnet 4.6 to Bedrock and Copilot model maps. Fixed Opus 4.6 Bedrock model IDs to match actual AWS API output.

### Fixed

- **Accurate macOS service status** — `claudegate status` previously reported "running" whenever the service was *registered* with launchd, even if the process had crashed or failed to start. Now parses the PID from `launchctl list` output to distinguish between a truly running process (`running (PID 12345)`) and a loaded-but-dead service (`loaded but not running`). The `/api/status` endpoint (`get_service_status()`) uses the same logic.
- **Block `sudo` for `install`/`uninstall` commands** — running `sudo claudegate install` on macOS silently writes the plist to `/var/root/Library/LaunchAgents/` and creates a root-owned log file, both invisible to the normal user. The commands now detect `sudo` and refuse with a clear error message.
- **Filter unsupported beta flags for Bedrock** — Claude Code sends `anthropic-beta` header flags (e.g., `interleaved-thinking-*`, `context-management-*`, `prompt-caching-scope-*`) that Bedrock doesn't recognize, causing `ValidationException: invalid beta flag`. Beta flags are now filtered to a whitelist of Bedrock-supported prefixes (`context-1m-*`, `effort-*`). Unsupported flags are silently dropped.
- **Handle `ClientError` in Bedrock streaming path** — non-transient errors like `ValidationException` and `AccessDeniedException` from `_open_bedrock_stream` were not caught in the streaming branch, falling through to the generic `except Exception` handler and returning 500. These now return proper error responses (400, 403).
- **Add region prefix for pass-through Bedrock model IDs** — when Claude Code sends a raw Bedrock model ID (e.g., `anthropic.claude-haiku-4-5-20251001-v1:0`), it was returned without the cross-region inference profile prefix (`us.`), causing Bedrock to reject it with "on-demand throughput isn't supported".
- **Handle `model: "default"` sentinel** — Claude Code sends the literal string `"default"` for certain internal requests (e.g. sub-agent tool selection). This was falling through model mapping and getting rejected by Copilot with `model_not_supported`.
- Fix `/models` plugin marketplace installation — move `marketplace.json` to `.claude-plugin/marketplace.json` where Claude Code expects it, and use full HTTPS URL for GitHub Enterprise clone.

## [0.4.0] - 2026-02-20

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
