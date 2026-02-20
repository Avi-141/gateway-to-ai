# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_copilot_translate.py

# Run a single test by name
uv run pytest -k test_tool_use_round_trip

# Run tests with coverage
uv run pytest --cov=claudegate --cov-report=term-missing

# Lint and format
uv run ruff check claudegate/ tests/
uv run ruff check --fix claudegate/ tests/
uv run ruff format claudegate/ tests/

# Type checking (source only, ty has false positives on test files)
uv run ty check claudegate/

# Run all pre-commit hooks
uv run pre-commit run --all-files

# Start the server
uv run claudegate
```

## Architecture

Claudegate is a proxy server that translates Anthropic API requests to either AWS Bedrock or GitHub Copilot backends. It also exposes OpenAI-compatible APIs (`/v1/chat/completions` and `/v1/responses`) for clients like Open WebUI and newer OpenAI SDK clients.

### Two-Backend Design

The `CLAUDEGATE_BACKEND` env var (`bedrock`, `copilot`, or comma-separated `copilot,bedrock` for fallback) selects the backend:

- **Bedrock**: Near-passthrough. Bedrock speaks Anthropic format natively, so requests go through with minimal transformation.
- **Copilot**: Full bidirectional translation. Requests are translated Anthropic → OpenAI → Copilot → OpenAI → Anthropic via `copilot_translate.py`.

### Request Flow

Each path does the minimum necessary translations:

```
/v1/messages (Anthropic in):
  → Bedrock:  passthrough (0 translations)
  → Copilot:  Anthropic → OpenAI → Copilot → OpenAI → Anthropic (2 translations)

/v1/chat/completions (OpenAI in):
  → Copilot:  passthrough (0 translations) — direct via _call_copilot_openai()
  → Bedrock:  OpenAI → Anthropic → Bedrock → Anthropic → OpenAI (2 translations)

/v1/responses (Responses API in):
  → Copilot (model supports /responses):  passthrough (0 translations)
  → Copilot (model only has /chat/completions):  Responses → OpenAI → Copilot → OpenAI → Responses (2 translations)
  → Bedrock:  Responses → Anthropic → Bedrock → Anthropic → Responses (2 translations)
```

### Key Modules

- **`app.py`** — FastAPI routes, request validation, error mapping, streaming SSE, fallback orchestration
- **`errors.py`** — `TransientBackendError` (fallback-eligible: 429, 5xx) and `CopilotHttpError` (non-transient)
- **`models.py`** — Maps Anthropic model names to backend-specific IDs (Bedrock: 14 models, Copilot: 10 Claude + 21 OpenAI-native)
- **`copilot_translate.py`** — Stateless Anthropic ↔ OpenAI translation functions + `StreamTranslator` state machine that accumulates tool call arguments across chunks
- **`openai_translate.py`** — Reverse translation for `/v1/chat/completions`: OpenAI ↔ Anthropic conversion + `ReverseStreamTranslator` state machine
- **`responses_translate.py`** — Bidirectional Responses API ↔ Anthropic/OpenAI translation functions + `AnthropicToResponsesStreamTranslator` and `OpenAIChatToResponsesStreamTranslator` state machines
- **`copilot_client.py`** — Copilot HTTP client with direct OpenAI passthrough path for non-Claude models and Responses API passthrough/via-chat paths
- **`copilot_auth.py`** — Three-tier auth: env var → persisted token file → interactive OAuth device flow
- **`bedrock_client.py`** — Singleton boto3 client with `reset_bedrock_client()` for credential refresh
- **`config.py`** — All env var loading and logging setup
- **`service.py`** — Cross-platform autostart: macOS (launchd), Linux (systemd), Windows (schtasks)

### Error Handling Pattern

`TransientBackendError` triggers fallback retry on the alternate backend. Non-transient errors (`CopilotHttpError`, 400/401/403) are returned immediately. `/v1/messages` returns Anthropic error format; `/v1/chat/completions` and `/v1/responses` return OpenAI error format.

## Test Patterns

Tests use pytest-asyncio (`asyncio_mode = "auto"`), pytest-httpx for HTTP mocking, and `unittest.mock.patch` for Bedrock/auth mocking. Shared fixtures in `conftest.py` include `minimal_anthropic_request`, `minimal_openai_request`, `minimal_responses_request`, `openai_chat_response`, `openai_request_with_tools`, and `make_client_error()` for simulating botocore errors. Tests are organized in classes grouped by function.
