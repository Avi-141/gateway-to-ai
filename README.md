# clauderock

A lightweight proxy that translates Anthropic API requests to AWS Bedrock or GitHub Copilot, enabling Claude Code and other Anthropic API clients to use either backend.

## Use Cases

- **[Auto-Claude](https://github.com/AndyMik90/Auto-Claude)** - Run Auto-Claude with AWS Bedrock or GitHub Copilot
- **Any Anthropic API client** - Redirect to Bedrock or Copilot without code changes
- **GitHub Copilot users** - Use your Copilot subscription to power Claude Code

## Features

- Two backends: **AWS Bedrock** (native Anthropic format) and **GitHub Copilot** (OpenAI-compatible, auto-translated)
- Backend selected via `BACKEND=bedrock|copilot` environment variable
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

### Copilot Backend

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

## Prerequisites

- Python 3.11+
- **Bedrock backend:** AWS credentials configured (via `aws configure`, environment variables, or IAM role) and access to Anthropic models in AWS Bedrock
- **Copilot backend:** GitHub account with an active GitHub Copilot subscription

## Installation

### Install globally with uv

```bash
# Install as a global tool
uv tool install git+https://github.com/yourusername/clauderock.git

# Run anytime
clauderock
```

### Run without installing

```bash
# One-liner using uvx
uvx --from git+https://github.com/yourusername/clauderock.git clauderock
```

### Development

```bash
git clone https://github.com/yourusername/clauderock.git
cd clauderock
uv run clauderock
```

## Project Structure

```
clauderock/
├── __init__.py           # Package entry point, exports main()
├── app.py                # FastAPI application and route handlers
├── client.py             # AWS Bedrock client management
├── config.py             # Configuration constants and logging
├── models.py             # Model mappings (Anthropic -> Bedrock/Copilot)
├── copilot_auth.py       # GitHub OAuth device flow and Copilot token management
├── copilot_client.py     # Copilot HTTP client and backend
└── copilot_translate.py  # Anthropic <-> OpenAI format translation
```

## Configuration

### Backend Selection

```bash
# Use AWS Bedrock (default)
export BACKEND="bedrock"

# Use GitHub Copilot
export BACKEND="copilot"
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
export HOST="0.0.0.0"  # default: 0.0.0.0
export PORT="8080"     # default: 8080

# Optional: Log level (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL="INFO"  # default: INFO
```

## Usage

### Start the proxy

**With Bedrock (default):**
```bash
clauderock
```

**With Copilot:**
```bash
BACKEND=copilot clauderock
```

If `GITHUB_TOKEN` is not set, the proxy will run an interactive OAuth device flow at startup:

```
To authenticate with GitHub Copilot:
  1. Open https://github.com/login/device
  2. Enter code: XXXX-XXXX

Waiting for authorization...
Authorization successful!
```

The token is persisted to `~/.config/clauderock/github_token` for subsequent startups.

### Run in background

**Quick (terminal session):**
```bash
clauderock &
```

**Persist after terminal close:**
```bash
nohup clauderock > ~/.clauderock.log 2>&1 &
```

**macOS (auto-start on login):**
```bash
# Edit the plist to set your AWS_REGION, then:
cp contrib/launchd/com.clauderock.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.clauderock.plist
```

**Linux (systemd user service):**
```bash
# Edit the service file to set your AWS_REGION, then:
mkdir -p ~/.config/systemd/user
cp contrib/systemd/clauderock.service ~/.config/systemd/user/
systemctl --user enable --now clauderock
```

### Configure Claude Code

Set these environment variables before running Claude Code:

```bash
export ANTHROPIC_API_KEY="sk-ant-dummy-key"  # Any value starting with sk-ant-
export ANTHROPIC_BASE_URL="http://localhost:8080"
```

### Test the proxy

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Main chat completion endpoint |
| `/v1/models` | GET | List available models |
| `/v1/messages/count_tokens` | POST | Count tokens in a request |
| `/health` | GET | Health check (add `?check_bedrock=true` or `?check_copilot=true` for deep check) |
| `/version` | GET | Return current version |

## Supported Parameters

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

## Error Handling

The proxy returns Anthropic-compatible error responses:

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

**Bedrock:** The proxy detects expired AWS credentials and resets the credential cache. If credentials are still expired (e.g., SSO session expired), refresh them manually:

```bash
# For SSO users
aws sso login

# For assume-role users
# Re-run your assume-role command or script
```

**Copilot:** The proxy automatically refreshes Copilot tokens before expiry. If your GitHub OAuth token becomes invalid, delete `~/.config/clauderock/github_token` and restart the proxy to re-authenticate via device flow.

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
