---
description: Show Copilot premium usage and quota status
user-invocable: true
---

# Copilot Usage Status

!`python3 -c "import urllib.request,os,json;r=json.loads(urllib.request.urlopen(os.environ.get('ANTHROPIC_BASE_URL','http://localhost:8080')+'/api/status').read().decode());print(json.dumps(r.get('copilot'),indent=2) if r.get('copilot') else 'NO_COPILOT_DATA')"`

If the output is `NO_COPILOT_DATA`, tell the user that Copilot usage data is not available (the backend may be set to bedrock-only) and suggest `/backend` to check.

Otherwise, display a clean summary:

1. **Plan**: the `plan` field
2. **Premium interactions**: a text progress bar showing `premium.used` / `premium.total` with `premium.percent_used`%. If `premium.unlimited` is true, show "Unlimited" instead of the bar.
3. **Remaining**: `premium.remaining` (note if `premium.overage_permitted` is true)
4. **Chat**: unlimited yes/no from `chat.unlimited`
5. **Completions**: unlimited yes/no from `completions.unlimited`
6. **Resets**: `reset_date`
7. If `stale` is true, add a note that the data is stale and refreshing in the background.

Keep the output compact — no extra commentary.
