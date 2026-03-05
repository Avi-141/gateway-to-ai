# Getting Started with Codex CLI

This guide covers setting up OpenAI Codex CLI. It uses **claudegate**, a local proxy that routes requests through your GitHub Copilot subscription.

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
uv tool install git+https://github.com/Avi-141/gateway-to-ai.git@v0.6.0
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

## 7. Install Node.js

Codex CLI requires Node.js 22 or later.

**macOS (Homebrew):**
```bash
brew install node
```

**Linux:**
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**Windows:** Download and install from [https://nodejs.org](https://nodejs.org).

Verify the installation:

```bash
node --version  # should be v22+
```

## 8. Install Codex CLI

```bash
npm install -g @openai/codex
```

Verify:

```bash
codex --version
```

## 9. Configure Codex CLI

Add to your shell profile (`~/.zshrc`, `~/.bashrc`):

```bash
export OPENAI_API_KEY="sk-dummy"
export OPENAI_BASE_URL="http://localhost:8080/v1"
```

Reload with `source ~/.zshrc` (or `~/.bashrc`).

**Windows (PowerShell):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-dummy", "User")
[System.Environment]::SetEnvironmentVariable("OPENAI_BASE_URL", "http://localhost:8080/v1", "User")
```

Restart your terminal after setting these.

> The API key is a dummy value — claudegate ignores it. Authentication goes through your GitHub Copilot token.

## 10. Run Codex CLI

```bash
codex --model gpt-5.3-codex
```

Codex CLI will start in interactive mode. You can ask it to make changes to code in your current directory, run commands, and more.

**Example — ask Codex to explain a file:**
```bash
codex --model gpt-5.3-codex "explain what this project does"
```

**Example — ask Codex to write a function:**
```bash
codex --model gpt-5.3-codex "add a function to utils.py that validates email addresses"
```

## Troubleshooting

**Connection refused** — Check `claudegate status`. If stopped, run `claudegate start`.

**Authentication error (401)** — Your GitHub token was likely revoked (e.g., deauthorized in GitHub settings). Delete `~/.config/claudegate/github_token`, run `claudegate` to re-authenticate, then `claudegate restart`.

**Port conflict** — Set `CLAUDEGATE_PORT=9090` and update `OPENAI_BASE_URL` to `http://localhost:9090/v1`. Reinstall service with `claudegate uninstall && CLAUDEGATE_PORT=9090 claudegate install --env`.

**`codex` not found** — Ensure your npm global bin directory is on your PATH. Run `npm bin -g` to find it.

**Model not available** — `gpt-5.3-codex` availability depends on your Copilot plan. Check the dashboard at [http://localhost:8080](http://localhost:8080) for the full list of available models.
