---
description: List available models from the claudegate proxy
user-invocable: true
---

# List Available Claudegate Models

The model list has already been fetched. Here is the raw JSON:

!`python3 -c "import urllib.request,os;print(urllib.request.urlopen(os.environ.get('ANTHROPIC_BASE_URL','http://localhost:8080')+'/v1/models').read().decode())"`

Display the results as a clean table with columns: Model ID, Provider, Max Input, Max Output. Sort by provider then model name. Only show the model ID, provider (owned_by field), max input tokens (limits.max_prompt_tokens if present, otherwise "—"), and max output tokens (limits.max_output_tokens if present, otherwise "—"). Omit embedding models and legacy duplicates.

After showing the table, display only this line: `Switch with: /model <model-id>` — no other commentary.
