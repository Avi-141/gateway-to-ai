# Getting Started with Claude Code

This guide covers setting up Claude Code. It uses **claudegate**, a local proxy that routes requests through your GitHub Copilot subscription.

## 1. Request a GitHub Copilot License

Go to [https://github.com/features/copilot](https://github.com/features/copilot) and request one of:

| Plan | Cost (quarterly) | Included prompts | Overage |
|------|-------------------|------------------|---------|
| GitHub Copilot Business | $43.68 USD | 300/month | $0.03/prompt |
| GitHub Copilot Enterprise | $90.09 USD | 1,000/month | $0.03/prompt |

Wait for approval.

> **Data classification:** You can share highly confidential information in GitHub Copilot. Use the [Data Advisor](https://github.com/features/copilot) tool to help classify and categorize the data you share.

## 2. Log into GitHub

Log in at [https://github.com](https://github.com) with the GitHub account that owns your Copilot access.

## 3. Install uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installation.

## 4. Install claudegate

```bash
uv tool install git+https://github.com/Avi-141/gateway-to-ai.git@v0.5.0
```

## 5. Authenticate with GitHub Copilot

Run claudegate once to trigger the OAuth device flow:

```bash
claudegate
```

It will display a URL and a code:

```
To authenticate with GitHub Copilot:
  1. Open https://github.com/login/device
  2. Enter code: XXXX-XXXX
```

Open the URL, ensure you are logged in as the correct GitHub account, enter the code, and authorize. The token is saved to `~/.config/claudegate/github_token` for future use.

Stop claudegate with `Ctrl+C` after authentication succeeds.

## 6. Install claudegate as a service

```bash
claudegate install
```

This registers claudegate to start automatically on login:

| Platform | Mechanism |
|----------|-----------|
| macOS | launchd (`~/Library/LaunchAgents/`) |
| Linux | systemd user unit (`~/.config/systemd/user/`) |
| Windows | Scheduled task |

Manage the service with `claudegate status`, `claudegate stop`, `claudegate start`, `claudegate restart`, and `claudegate uninstall`. You can also open [http://localhost:8080](http://localhost:8080) in your browser to see the dashboard with live status, available models, and logs.

**Logs:**

```bash
claudegate logs
```

## 7. Install Claude Code

**macOS / Linux:**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://claude.ai/install.ps1 | iex
```

## 8. Configure Claude Code

Add to your shell profile (`~/.zshrc`, `~/.bashrc`):

```bash
export ANTHROPIC_BASE_URL="http://localhost:8080"
export ANTHROPIC_API_KEY="sk-ant-dummy-key"
```

Reload with `source ~/.zshrc` (or `~/.bashrc`).

**Windows (PowerShell):**
```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", "http://localhost:8080", "User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-dummy-key", "User")
```

Restart your terminal after setting these.

> The API key is a dummy value — claudegate ignores it. Authentication goes through your GitHub Copilot token.

## 9. Bootstrap Claude Code (first time only)

Claude Code shows a setup wizard on first launch that requires logging into Anthropic. Since we use claudegate instead, run this one-time bootstrap to bypass the login:

**macOS / Linux:**
```bash
CLAUDE_CODE_USE_BEDROCK=1 claude
```

**Windows (PowerShell):**
```powershell
$env:CLAUDE_CODE_USE_BEDROCK = "1"; claude
```

Walk through the setup wizard. When you see:

```
Detected a custom API key in your environment

ANTHROPIC_API_KEY: sk-ant-...sk-ant-dummy-key

Do you want to use this API key?

❯ 1. Yes
  2. No (recommended) ✔
```

Select **Yes**. Complete the remaining wizard steps, then exit Claude Code (`/exit` or `Ctrl+C`). You only need to do this once.

## 10. Run Claude Code

```bash
claude
```

## Troubleshooting

**Connection refused** — Check `claudegate status`. If stopped, run `claudegate start`.

**Authentication error (401)** — Your GitHub token was likely revoked (e.g., deauthorized in GitHub settings). Delete `~/.config/claudegate/github_token`, run `claudegate` to re-authenticate, then `claudegate restart`.

**Port conflict** — Set `CLAUDEGATE_PORT=9090` and update `ANTHROPIC_BASE_URL` to `http://localhost:9090`. Reinstall service with `claudegate uninstall && CLAUDEGATE_PORT=9090 claudegate install --env`.

**`claudegate` not found (Windows)** — Ensure `%USERPROFILE%\.local\bin` is on your PATH.

**Model not supported (`model_not_supported` 400)** — If you see `Copilot stream error 400: {"error":{"message":"The requested model is not supported.","code":"model_not_supported",...}}` in the claudegate logs, your GitHub Copilot subscription has likely lapsed. Renew it at [https://github.com/features/copilot](https://github.com/features/copilot).
