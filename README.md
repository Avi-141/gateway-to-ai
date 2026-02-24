# claudegate

![Version](https://img.shields.io/badge/version-0.4.0-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-UNLICENSED-green)

A lightweight proxy that translates Anthropic API requests to GitHub Copilot or AWS Bedrock, enabling Claude Code, Open WebUI, and other Anthropic or OpenAI API clients to use either backend.

> [!IMPORTANT]
> **New here?** Follow the **[Getting Started Guide](docs/getting-started.md)** for step-by-step setup instructions.

## Use Cases

- **Claude Code + GitHub Copilot** - Run Claude Code using your GitHub Copilot subscription as the backend
- **[Auto-Claude](https://github.com/AndyMik90/Auto-Claude)** - Run Auto-Claude with AWS Bedrock or GitHub Copilot
- **Any Anthropic API client** - Redirect to Bedrock or Copilot without code changes
- **[Open WebUI](https://github.com/open-webui/open-webui)** - Use Open WebUI with Bedrock or Copilot via the OpenAI-compatible API
- **[OpenAI Codex CLI](https://github.com/openai/codex)** - Run Codex CLI using your GitHub Copilot subscription or AWS Bedrock via the Responses API
- **Any OpenAI API client** - Use the `/v1/chat/completions` or `/v1/responses` endpoint with any OpenAI-format client

## Features

- Two backends: **GitHub Copilot** (default, OpenAI-compatible, auto-translated) and **AWS Bedrock** (native Anthropic format)
- Backend selected via `CLAUDEGATE_BACKEND=copilot|bedrock` environment variable
- **Cross-backend fallback**: set `CLAUDEGATE_BACKEND=copilot,bedrock` to automatically retry on the other backend when the primary returns a transient error (429, 5xx)
- Supports streaming and non-streaming responses
- Extended thinking support (Bedrock)
- Full tool use round-trip translation (Copilot)
- **Multimodal image support** — images in user messages are translated between Anthropic, OpenAI, and Responses API formats
- Token counting via tiktoken
- Model mapping from Anthropic model names to backend-specific IDs
- Anthropic-compatible error responses
- Request ID tracking (`x-request-id` header)
- Deep health checks with backend connectivity verification
- **Web dashboard** at `/` with live status, models, and log viewer
- Graceful startup/shutdown with configuration logging
- GitHub OAuth device flow for easy Copilot authentication
- **OpenAI-compatible API** (`/v1/chat/completions` and `/v1/responses`) for Open WebUI and other OpenAI-format clients
- **Non-Claude models** (GPT-4o, o3-mini, etc.) via Copilot backend's OpenAI-compatible endpoint
- **Direct Copilot passthrough** — zero-translation path for `/v1/chat/completions` when Copilot is the backend

## Supported Models

Available models are fetched dynamically from each backend at startup. View them via:

- **Web dashboard** — open `http://localhost:8080` in your browser
- **API** — `GET http://localhost:8080/v1/models`
- **Claude Code** — install the `/models` plugin (see below)

The Copilot backend supports Claude models (translated to Anthropic format) and non-Claude models like GPT, Gemini, and others (passed through via OpenAI format). The Bedrock backend supports all Claude models available in your AWS region.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows)
- **Copilot backend (default):** GitHub account with an active GitHub Copilot subscription
- **Bedrock backend:** AWS credentials configured (via `aws configure`, environment variables, or IAM role) and access to Anthropic models in AWS Bedrock

## Installation

### Install globally with uv

```bash
# Install a specific version (recommended)
uv tool install git+https://github.com/Avi-141/gateway-to-ai.git@v0.4.0

# Install latest from master
uv tool install git+https://github.com/Avi-141/gateway-to-ai.git
```

### Run without installing

```bash
# Specific version
uvx --from git+https://github.com/Avi-141/gateway-to-ai.git@v0.4.0 claudegate

# Latest from master
uvx --from git+https://github.com/Avi-141/gateway-to-ai.git claudegate
```

After installing, run `claudegate` once manually in your terminal to complete the GitHub OAuth device flow. The token is persisted to `~/.config/claudegate/github_token` so subsequent starts (including autostart via `claudegate install`) authenticate automatically.

### Upgrade

```bash
# Upgrade to a specific version
uv tool install --force git+https://github.com/Avi-141/gateway-to-ai.git@v0.4.0

# Upgrade to latest
uv tool install --force git+https://github.com/Avi-141/gateway-to-ai.git
```

### Development

```bash
git clone https://github.com/Avi-141/gateway-to-ai.git
cd claudegate
uv run claudegate
```

## Configuration

### Backend Selection

```bash
# Use GitHub Copilot (default)
export CLAUDEGATE_BACKEND="copilot"

# Use AWS Bedrock
export CLAUDEGATE_BACKEND="bedrock"

# Use Copilot with Bedrock as fallback (retries on 429, 5xx errors)
export CLAUDEGATE_BACKEND="copilot,bedrock"

# Use Bedrock with Copilot as fallback
export CLAUDEGATE_BACKEND="bedrock,copilot"
```

### Copilot Configuration

```bash
# GitHub OAuth token (if not set, interactive device flow runs at startup)
export GITHUB_TOKEN="gho_xxxxxxxxxxxx"

# Optional: HTTP timeout for Copilot requests
export COPILOT_TIMEOUT="300"  # default: 300 (seconds)
```

### Bedrock Configuration

```bash
# AWS region for Bedrock (default: us-west-2)
export AWS_REGION="us-west-2"

# AWS credentials (if not using aws configure or IAM role)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Optional: Cross-region inference prefix
export BEDROCK_REGION_PREFIX="us"  # default: us (options: us, eu, apac)

# Optional: Read timeout for slow models (Opus can be slow)
export BEDROCK_READ_TIMEOUT="300"  # default: 300 (seconds)
```

### General Configuration

```bash
# Optional: Server configuration
export CLAUDEGATE_HOST="0.0.0.0"  # default: 127.0.0.1 (localhost only)
export CLAUDEGATE_PORT="8080"     # default: 8080

# Optional: Log level (DEBUG, INFO, WARNING, ERROR)
export CLAUDEGATE_LOG_LEVEL="INFO"  # default: INFO

# Optional: Disable colored log output (https://no-color.org)
export NO_COLOR=1
```

## Usage

### Start the proxy

**With Copilot (default):**
```bash
claudegate
```

**With Bedrock:**
```bash
CLAUDEGATE_BACKEND=bedrock claudegate
```

**With fallback (Copilot primary, Bedrock fallback):**
```bash
CLAUDEGATE_BACKEND=copilot,bedrock claudegate
```

If `GITHUB_TOKEN` is not set, the proxy will run an interactive OAuth device flow at startup:

```
To authenticate with GitHub Copilot:
  1. Open https://github.com/login/device
  2. Enter code: XXXX-XXXX

Waiting for authorization...
Authorization successful!
```

The token is persisted to `~/.config/claudegate/github_token` for subsequent startups.

### Autostart as a system service

The `install` command sets up claudegate to start automatically on login and starts it immediately:

```bash
claudegate install
```

To also capture your current environment variables (`CLAUDEGATE_*`, `AWS_*`, `GITHUB_TOKEN`) into the service file:

```bash
claudegate install --env
```

This works on all platforms:
- **macOS** — creates a launchd plist in `~/Library/LaunchAgents/`
- **Linux** — creates a systemd user unit in `~/.config/systemd/user/`
- **Windows** — creates a scheduled task via `schtasks`

**Other service commands:**

```bash
# Check if the service is installed and running
claudegate status

# Stop the service
claudegate stop

# Start the service
claudegate start

# Restart the service
claudegate restart

# Tail service logs (macOS/Linux)
claudegate logs

# Show 200 lines without following
claudegate logs -n 200 --no-follow

# Linux only: show logs since a given time
claudegate logs --since "10m ago"

# Stop and remove the service
claudegate uninstall
```

### Configure Open WebUI

Point Open WebUI at claudegate as an OpenAI-compatible backend:

1. In Open WebUI **Settings > Connections**, add a new OpenAI connection:
   - **API Base URL:** `http://localhost:8080/v1`
   - **API Key:** `sk-dummy` (any value; claudegate ignores it)
2. Select a model from the list (e.g., `claude-sonnet-4-5-20250929`)
3. Start chatting

This works with any OpenAI-format client, not just Open WebUI.

### Configure Claude Code

Set these environment variables before running Claude Code:

```bash
export ANTHROPIC_API_KEY="sk-ant-dummy-key"  # Any value starting with sk-ant-
export ANTHROPIC_BASE_URL="http://localhost:8080"
```

#### /models Plugin

Install the `/models` plugin to list available models and their token limits directly in Claude Code:

```
/plugin marketplace add https://github.com/Avi-141/gateway-to-ai.git
/plugin install models@claudegate
```

Then type `/models` to see a table of available models.

### Configure Codex CLI

Set these environment variables before running [Codex CLI](https://github.com/openai/codex):

```bash
export OPENAI_API_KEY="sk-dummy"        # Any value; claudegate ignores it
export OPENAI_BASE_URL="http://localhost:8080/v1"
```

Then run Codex with any model available through the proxy:

```bash
codex --model gpt-5.3-codex
```

### Test the proxy

**Anthropic format:**
```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**OpenAI format:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**OpenAI Responses API:**
```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "input": "Hello!"
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard with live status, models, and log viewer |
| `/api/status` | GET | Combined JSON status (health, service, models, logs) |
| `/v1/messages` | POST | Anthropic Messages API endpoint |
| `/v1/chat/completions` | POST | OpenAI-compatible Chat Completions endpoint |
| `/v1/responses` | POST | OpenAI-compatible Responses API endpoint |
| `/v1/models` | GET | List available models (OpenAI-compatible format) |
| `/v1/messages/count_tokens` | POST | Count tokens in a request |
| `/health` | GET | Health check (add `?check_bedrock=true` or `?check_copilot=true` for deep check) |
| `/version` | GET | Return current version |

## Error Handling

When a fallback backend is configured (`CLAUDEGATE_BACKEND=copilot,bedrock`), transient errors (429, 5xx) on the primary backend automatically trigger a retry on the fallback. Context window exceeded errors also trigger fallback. Non-transient errors (400, 401, 403) are returned immediately. For streaming requests, fallback only works for pre-stream errors; mid-stream errors are delivered as SSE error events.

`/v1/messages` returns Anthropic-format errors; `/v1/chat/completions` and `/v1/responses` return OpenAI-format errors.

**Bedrock credentials:** The proxy detects expired AWS credentials and resets the cache. If still expired, refresh manually with `aws sso login`.

**Copilot tokens:** The proxy refreshes tokens automatically. If your GitHub OAuth token becomes invalid, delete `~/.config/claudegate/github_token` and restart to re-authenticate.
