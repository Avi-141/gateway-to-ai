---
description: List available models from the claudegate proxy
user-invocable: true
---

# List Available Claudegate Models

The model list has already been fetched. Here is the raw JSON:

!`python3 -c "
import urllib.request,os,json,pathlib
_f=pathlib.Path.home()/'.config'/'claudegate'/'server.json'
try: base=json.loads(_f.read_text())['url']
except Exception: base=os.environ.get('ANTHROPIC_BASE_URL','http://localhost:8080')
try:
 print(urllib.request.urlopen(base+'/v1/models',timeout=5).read().decode())
except Exception as e:
 print('CONNECTION_REFUSED')
"`

If the output is `CONNECTION_REFUSED`, tell the user that the claudegate proxy server is not running or not reachable. Suggest starting it with `claudegate` (or `uv run claudegate`) and checking the `ANTHROPIC_BASE_URL` env var.

Otherwise, display the results as a clean table with columns: Model ID, Provider, Max Input, Max Output. Sort by provider then model name. Only show the model ID, provider (owned_by field), max input tokens (limits.max_prompt_tokens if present, otherwise "—"), and max output tokens (limits.max_output_tokens if present, otherwise "—"). Omit embedding models and legacy duplicates.

After showing the table, display only this line: `Switch with: /model <model-id>` — no other commentary.
