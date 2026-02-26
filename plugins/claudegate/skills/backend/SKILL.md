---
description: Switch the claudegate proxy backend at runtime
user-invocable: true
---

# Switch Claudegate Backend

!`BACKEND_ARGS='$ARGUMENTS' python3 -c "
import urllib.request, os, json, sys
base = os.environ.get('ANTHROPIC_BASE_URL', 'http://localhost:8080')
args = os.environ.get('BACKEND_ARGS', '').strip()
if not args:
    print(urllib.request.urlopen(base + '/api/backend').read().decode())
else:
    req = urllib.request.Request(
        base + '/api/backend',
        data=json.dumps({'backend': args}).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    print(urllib.request.urlopen(req).read().decode())
"`

If no arguments were provided, display the current primary and fallback backend, then show usage: `/backend <copilot|bedrock|copilot,bedrock|bedrock,copilot>`

If an argument was provided, show whether the backend was changed, and display the new primary/fallback configuration.
