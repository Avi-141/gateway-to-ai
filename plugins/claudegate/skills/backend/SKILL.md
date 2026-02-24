---
description: Switch the claudegate proxy backend at runtime
user-invocable: true
---

# Switch Claudegate Backend

Arguments: $ARGUMENTS

If no arguments were provided, fetch the current backend:

!`python3 -c "import urllib.request,os;print(urllib.request.urlopen(os.environ.get('ANTHROPIC_BASE_URL','http://localhost:8080')+'/api/backend').read().decode())"`

Display the current primary and fallback backend, then show usage: `/backend <copilot|bedrock|copilot,bedrock|bedrock,copilot>`

If an argument was provided (e.g. `copilot`, `bedrock`, `copilot,bedrock`), switch the backend:

!`python3 -c "import urllib.request,os,json;r=urllib.request.urlopen(urllib.request.Request(os.environ.get('ANTHROPIC_BASE_URL','http://localhost:8080')+'/api/backend',data=json.dumps({'backend':'$ARGUMENTS'}).encode(),headers={'Content-Type':'application/json'},method='POST'));print(r.read().decode())"`

Show whether the backend was changed, and display the new primary/fallback configuration.
