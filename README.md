# claudegate

A lightweight proxy that translates Anthropic API requests to AWS Bedrock or GitHub Copilot, enabling Claude Code, Open WebUI, and other Anthropic or OpenAI API clients to use either backend.

## Use Cases

- **[Auto-Claude](https://github.com/AndyMik90/Auto-Claude)** - Run Auto-Claude with AWS Bedrock or GitHub Copilot
- **Any Anthropic API client** - Redirect to Bedrock or Copilot without code changes
- **GitHub Copilot users** - Use your Copilot subscription to power Claude Code
- **[Open WebUI](https://github.com/open-webui/open-webui)** - Use Open WebUI with Bedrock or Copilot via the OpenAI-compatible API
- **Any OpenAI API client** - Use the `/v1/chat/completions` endpoint with any OpenAI-format client

## Features

- Two backends: **AWS Bedrock** (native Anthropic format) and **GitHub Copilot** (OpenAI-compatible, auto-translated)
- Backend selected via `CLAUDEGATE_BACKEND=bedrock|copilot` environment variable
- **Cross-backend fallback**: set `CLAUDEGATE_BACKEND=copilot,bedrock` to automatically retry on the other backend when the primary returns a transient error (429, 5xx)
- Supports streaming and non-streaming responses
- Extended thinking support (Bedrock)
- Full tool use round-trip translation (Copilot)
- Token counting via tiktoken
- Model mapping from Anthropic model names to backend-specific IDs
- Anthropic-compatible error responses
- Request ID tracking (`x-request-id` header)
- Deep health checks with backend connectivity verification
- Graceful startup/shutdown with configuration logging
- GitHub OAuth device flow for easy Copilot authentication
- **OpenAI-compatible API** (`/v1/chat/completions`) for Open WebUI and other OpenAI-format clients
- **Non-Claude models** (GPT-4o, o3-mini, etc.) via Copilot backend's OpenAI-compatible endpoint
- **Direct Copilot passthrough** — zero-translation path for `/v1/chat/completions` when Copilot is the backend

## Supported Models

### Bedrock Backend

| Anthropic Model | Bedrock Model |
|-----------------|---------------|
| claude-opus-4-6-20250515 | us.anthropic.claude-opus-4-6-20250515-v1:0 |
| claude-opus-4-5-20251101 | us.anthropic.claude-opus-4-5-20251101-v1:0 |
| claude-opus-4-1-20250805 | us.anthropic.claude-opus-4-1-20250805-v1:0 |
| claude-opus-4-20250514 | us.anthropic.claude-opus-4-20250514-v1:0 |
| claude-sonnet-4-5-20250929 | us.anthropic.claude-sonnet-4-5-20250929-v1:0 |
| claude-sonnet-4-20250514 | us.anthropic.claude-sonnet-4-20250514-v1:0 |
| claude-3-7-sonnet-20250219 | us.anthropic.claude-3-7-sonnet-20250219-v1:0 |
| claude-3-5-sonnet-20241022 | us.anthropic.claude-3-5-sonnet-20241022-v2:0 |
| claude-3-5-sonnet-20240620 | us.anthropic.claude-3-5-sonnet-20240620-v1:0 |
| claude-haiku-4-5-20251001 | us.anthropic.claude-haiku-4-5-20251001-v1:0 |
| claude-3-5-haiku-20241022 | us.anthropic.claude-3-5-haiku-20241022-v1:0 |
| claude-3-haiku-20240307 | us.anthropic.claude-3-haiku-20240307-v1:0 |
| claude-3-opus-20240229 | us.anthropic.claude-3-opus-20240229-v1:0 |
| claude-3-sonnet-20240229 | us.anthropic.claude-3-sonnet-20240229-v1:0 |

### Copilot Backend (Claude Models)

| Anthropic Model | Copilot Model |
|-----------------|---------------|
| claude-opus-4-6-20250515 | claude-opus-4.6 |
| claude-opus-4-5-20251101 | claude-opus-4.5 |
| claude-opus-4-1-20250805 | claude-opus-4.1 |
| claude-opus-4-20250514 | claude-opus-4 |
| claude-sonnet-4-5-20250929 | claude-sonnet-4.5 |
| claude-sonnet-4-20250514 | claude-sonnet-4 |
| claude-haiku-4-5-20251001 | claude-haiku-4.5 |
| claude-3-7-sonnet-20250219 | claude-3.7-sonnet |
| claude-3-5-sonnet-20241022 | claude-3.5-sonnet |
| claude-3-5-haiku-20241022 | claude-3.5-haiku |

### Copilot Backend (Non-Claude Models)

When using the Copilot backend, non-Claude models are available via both `/v1/messages` (Anthropic format) and `/v1/chat/completions` (OpenAI format):

| Model | Provider |
|-------|----------|
| gpt-5.2-codex | OpenAI |
| gpt-5.2 | OpenAI |
| gpt-5.1-codex-max | OpenAI |
| gpt-5.1-codex-mini | OpenAI |
| gpt-5.1-codex | OpenAI |
| gpt-5.1 | OpenAI |
| gpt-5-codex | OpenAI |
| gpt-5-mini | OpenAI |
| gpt-5 | OpenAI |
| gpt-4.1 | OpenAI |
| gpt-4o | OpenAI |
| gpt-4o-mini | OpenAI |
| gemini-3-pro | Google |
| gemini-3-flash | Google |
| gemini-2.5-pro | Google |
| grok-code-fast-1 | xAI |
| raptor-mini | Other |

These models work via both endpoints when using the Copilot backend. On `/v1/chat/completions`, they pass through directly with zero format translations. On `/v1/messages`, requests are translated to OpenAI format and responses are translated back to Anthropic format. Model availability depends on your Copilot plan.

## Prerequisites

- Python 3.11+
- **Bedrock backend:** AWS credentials configured (via `aws configure`, environment variables, or IAM role) and access to Anthropic models in AWS Bedrock
- **Copilot backend:** GitHub account with an active GitHub Copilot subscription

## Installation

### Install globally with uv

```bash
# Install as a global tool
uv tool install git+https://github.com/yourusername/claudegate.git

# Run anytime
claudegate
```

### Run without installing

```bash
# One-liner using uvx
uvx --from git+https://github.com/yourusername/claudegate.git claudegate
```

### Development

```bash
git clone https://github.com/yourusername/claudegate.git
cd claudegate
uv run claudegate
```

## Configuration

### Backend Selection

```bash
# Use AWS Bedrock (default)
export CLAUDEGATE_BACKEND="bedrock"

# Use GitHub Copilot
export CLAUDEGATE_BACKEND="copilot"

# Use Copilot with Bedrock as fallback (retries on 429, 5xx errors)
export CLAUDEGATE_BACKEND="copilot,bedrock"

# Use Bedrock with Copilot as fallback
export CLAUDEGATE_BACKEND="bedrock,copilot"
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

### Copilot Configuration

```bash
# GitHub OAuth token (if not set, interactive device flow runs at startup)
export GITHUB_TOKEN="gho_xxxxxxxxxxxx"

# Optional: HTTP timeout for Copilot requests
export COPILOT_TIMEOUT="300"  # default: 300 (seconds)
```

### General Configuration

```bash
# Optional: Server configuration
export CLAUDEGATE_HOST="0.0.0.0"  # default: 0.0.0.0
export CLAUDEGATE_PORT="8080"     # default: 8080

# Optional: Log level (DEBUG, INFO, WARNING, ERROR)
export CLAUDEGATE_LOG_LEVEL="INFO"  # default: INFO
```

## Usage

### Start the proxy

**With Bedrock (default):**
```bash
claudegate
```

**With Copilot:**
```bash
CLAUDEGATE_BACKEND=copilot claudegate
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

### Run in background

**Quick (terminal session):**
```bash
claudegate &
```

**Persist after terminal close:**
```bash
nohup claudegate > ~/.claudegate.log 2>&1 &
```

**macOS (auto-start on login):**
```bash
# Edit the plist to set your AWS_REGION, then:
cp contrib/launchd/com.claudegate.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.claudegate.plist
```

**Linux (systemd user service):**
```bash
# Edit the service file to set your AWS_REGION, then:
mkdir -p ~/.config/systemd/user
cp contrib/systemd/claudegate.service ~/.config/systemd/user/
systemctl --user enable --now claudegate
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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic Messages API endpoint |
| `/v1/chat/completions` | POST | OpenAI-compatible Chat Completions endpoint |
| `/v1/models` | GET | List available models (OpenAI-compatible format) |
| `/v1/messages/count_tokens` | POST | Count tokens in a request |
| `/health` | GET | Health check (add `?check_bedrock=true` or `?check_copilot=true` for deep check) |
| `/version` | GET | Return current version |

## Supported Parameters

### Anthropic API (`/v1/messages`)

The `/v1/messages` endpoint supports all standard Anthropic API parameters:

- `model` (required) - Model identifier
- `max_tokens` (required) - Maximum tokens to generate
- `messages` (required) - Array of messages
- `system` - System prompt
- `temperature` - Sampling temperature (0.0-1.0)
- `top_p` - Nucleus sampling
- `top_k` - Top-k sampling (Bedrock only)
- `tools` - Tool definitions
- `tool_choice` - Tool selection preference
- `thinking` - Extended thinking configuration (Bedrock only)
- `stop_sequences` - Custom stop sequences
- `metadata` - Request metadata
- `stream` - Enable streaming responses
- `anthropic_beta` - Beta features list (Bedrock only)

The `anthropic-beta` header is also supported and converted to the `anthropic_beta` body field (Bedrock only).

### OpenAI-Compatible API (`/v1/chat/completions`)

The `/v1/chat/completions` endpoint supports standard OpenAI Chat Completions parameters:

- `model` (required) - Model identifier
- `messages` (required) - Array of messages (system, user, assistant, tool roles)
- `max_tokens` - Maximum tokens to generate (defaults to 4096 if omitted)
- `temperature` - Sampling temperature (0.0-1.0)
- `top_p` - Nucleus sampling
- `stop` - Custom stop sequences
- `stream` - Enable streaming responses
- `tools` - Tool/function definitions
- `tool_choice` - Tool selection (`auto`, `required`, `none`, or specific function)

When the **Copilot** backend is used, requests pass through directly to Copilot with zero format translations. This also enables non-Claude models (GPT-4o, o3-mini, etc.) that Copilot supports natively. When the **Bedrock** backend is used, requests are translated to Anthropic format, processed through Bedrock, and responses are translated back to OpenAI format.

## Error Handling

### Anthropic format (`/v1/messages`)

The `/v1/messages` endpoint returns Anthropic-compatible error responses:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Missing required field: model"
  }
}
```

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | `invalid_request_error` | Missing/invalid fields, bad JSON |
| 401 | `authentication_error` | Credentials expired or invalid |
| 403 | `permission_error` | Access denied |
| 429 | `rate_limit_error` | Rate limiting / throttling |
| 504 | `timeout_error` | Request timed out |
| 500 | `api_error` | Internal / backend errors |

### OpenAI format (`/v1/chat/completions`)

The `/v1/chat/completions` endpoint returns OpenAI-compatible error responses:

```json
{
  "error": {
    "message": "Missing required field: model",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

### Fallback behavior

**Fallback:** When a fallback backend is configured (`CLAUDEGATE_BACKEND=copilot,bedrock`), transient errors (429, 500, 502, 503, 504) on the primary backend automatically trigger a retry on the fallback. Non-transient errors (400, 401, 403) are returned immediately without fallback. For streaming requests, fallback only works for pre-stream errors (connection failures, HTTP status errors before the first chunk is sent). Mid-stream errors are delivered as SSE error events as usual.

**Bedrock:** The proxy detects expired AWS credentials and resets the credential cache. If credentials are still expired (e.g., SSO session expired), refresh them manually:

```bash
# For SSO users
aws sso login

# For assume-role users
# Re-run your assume-role command or script
```

**Copilot:** The proxy automatically refreshes Copilot tokens before expiry. If your GitHub OAuth token becomes invalid, delete `~/.config/claudegate/github_token` and restart the proxy to re-authenticate via device flow.

## Request Tracing

Pass `x-request-id` header to trace requests through logs:

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "x-request-id: my-trace-id" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4-20250514", "max_tokens": 100, "messages": [...]}'
```

Logs will include the request ID:
```
[my-trace-id] Request - model: claude-sonnet-4-20250514 -> us.anthropic.claude-sonnet-4-20250514-v1:0, stream: false
```
