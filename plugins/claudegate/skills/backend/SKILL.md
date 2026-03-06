---
description: Switch the claudegate proxy backend at runtime
user-invocable: true
---

# Switch Claudegate Backend

!`BACKEND_ARGS='$ARGUMENTS' python3 -c "
import urllib.request, os, json, sys, pathlib
_f=pathlib.Path.home()/'.config'/'claudegate'/'server.json'
try: base=json.loads(_f.read_text())['url']
except Exception: base=os.environ.get('ANTHROPIC_BASE_URL','http://localhost:8080')
args = os.environ.get('BACKEND_ARGS', '').strip()
try:
 if not args:
  print(urllib.request.urlopen(base + '/api/backend', timeout=5).read().decode())
 else:
  req = urllib.request.Request(
   base + '/api/backend',
   data=json.dumps({'backend': args}).encode(),
   headers={'Content-Type': 'application/json'},
   method='POST',
  )
  print(urllib.request.urlopen(req, timeout=5).read().decode())
except Exception as e:
 print('CONNECTION_REFUSED')
"`

If the output is `CONNECTION_REFUSED`, tell the user that the claudegate proxy server is not running or not reachable. Suggest starting it with `claudegate` (or `uv run claudegate`) and checking the `ANTHROPIC_BASE_URL` env var.

If no arguments were provided, display the current primary and fallback backend, then show usage: `/claudegate:backend <copilot|bedrock|copilot,bedrock|bedrock,copilot>`

If an argument was provided, show whether the backend was changed, and display the new primary/fallback configuration.
