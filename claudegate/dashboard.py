"""Self-contained HTML dashboard page for claudegate."""

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>claudegate dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4ed;
    --text-dim: #8b8fa3;
    --accent: #6c8aff;
    --green: #4ade80;
    --yellow: #facc15;
    --red: #f87171;
    --orange: #fb923c;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
  }
  h1 { font-size: 1.5rem; margin-bottom: 24px; }
  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .panel h2 { font-size: 1rem; margin-bottom: 12px; color: var(--accent); }
  .panel.full { grid-column: 1 / -1; }
  .kv { display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.85rem; }
  .kv .label { color: var(--text-dim); }
  .badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-green { background: rgba(74,222,128,0.15); color: var(--green); }
  .badge-red { background: rgba(248,113,113,0.15); color: var(--red); }
  .badge-yellow { background: rgba(250,204,21,0.15); color: var(--yellow); }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { text-align: left; color: var(--text-dim); font-weight: 500; padding: 6px 8px;
       border-bottom: 1px solid var(--border); }
  td { padding: 5px 8px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .log-controls { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
  .log-controls select, .log-controls label {
    font-size: 0.8rem;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 6px;
  }
  .log-controls button {
    font-size: 0.8rem;
    background: var(--bg);
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 10px;
    cursor: pointer;
    margin-left: auto;
  }
  .log-controls button:hover { color: var(--text); border-color: var(--text-dim); }
  #log-viewer {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    height: 340px;
    overflow-y: auto;
    padding: 8px;
    font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
    font-size: 0.75rem;
    line-height: 1.5;
  }
  .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-DEBUG { color: var(--text-dim); }
  .log-INFO { color: var(--text); }
  .log-WARNING { color: var(--yellow); }
  .log-ERROR { color: var(--red); }
  .log-CRITICAL { color: var(--red); font-weight: bold; }
  .error-banner {
    background: rgba(248,113,113,0.1);
    border: 1px solid var(--red);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 0.85rem;
    color: var(--red);
    display: none;
  }
  @media (max-width: 700px) { .panels { grid-template-columns: 1fr; } }
  .progress-bar {
    width: 100%;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin: 6px 0 2px;
  }
  .progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }
  .backend-switch { display: flex; gap: 6px; align-items: center; margin-top: 6px; font-size: 0.85rem; }
  .backend-switch select {
    font-size: 0.8rem;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 6px;
    max-width: 200px;
  }
  .backend-switch button {
    font-size: 0.8rem;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 4px;
    padding: 4px 12px;
    cursor: pointer;
  }
  .backend-switch button:hover { opacity: 0.85; }
  .backend-switch button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
<h1>claudegate</h1>
<div id="error-banner" class="error-banner"></div>
<div class="panels">
  <div class="panel" id="status-panel">
    <h2>Status</h2>
    <div id="status-content"><span class="label">Loading...</span></div>
    <div class="backend-switch">
      <span style="color:var(--text-dim);white-space:nowrap;margin-right:auto">Switch backend</span>
      <select id="backend-select">
        <option value="copilot">copilot</option>
        <option value="bedrock">bedrock</option>
        <option value="copilot,bedrock">copilot + bedrock fallback</option>
        <option value="bedrock,copilot">bedrock + copilot fallback</option>
      </select>
      <button id="backend-apply">Apply</button>
    </div>
  </div>
  <div class="panel" id="service-panel">
    <h2>Service</h2>
    <div id="service-content"><span class="label">Loading...</span></div>
  </div>
  <div class="panel" id="copilot-panel" style="display:none">
    <h2>Copilot Usage</h2>
    <div id="copilot-content"></div>
  </div>
  <div class="panel full" id="models-panel">
    <h2>Models</h2>
    <div id="models-content"><span class="label">Loading...</span></div>
  </div>
  <div class="panel full">
    <h2>Logs</h2>
    <div class="log-controls">
      <label for="log-level">Level:</label>
      <select id="log-level">
        <option value="">ALL</option>
        <option value="DEBUG">DEBUG</option>
        <option value="INFO" selected>INFO</option>
        <option value="WARNING">WARNING</option>
        <option value="ERROR">ERROR</option>
      </select>
      <label><input type="checkbox" id="auto-scroll" checked> Auto-scroll</label>
      <button id="clear-logs" title="Clear all logs">Clear</button>
    </div>
    <div id="log-viewer"></div>
  </div>
</div>
<script>
(function() {
  const statusEl = document.getElementById('status-content');
  const serviceEl = document.getElementById('service-content');
  const copilotPanel = document.getElementById('copilot-panel');
  const copilotEl = document.getElementById('copilot-content');
  const modelsEl = document.getElementById('models-content');
  const logViewer = document.getElementById('log-viewer');
  const levelSelect = document.getElementById('log-level');
  const autoScroll = document.getElementById('auto-scroll');
  const errorBanner = document.getElementById('error-banner');
  const backendSelect = document.getElementById('backend-select');
  const backendApply = document.getElementById('backend-apply');

  document.getElementById('clear-logs').addEventListener('click', function() {
    fetch('/api/logs/clear', { method: 'POST' })
      .then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        logViewer.innerHTML = '';
      })
      .catch(function(err) {
        errorBanner.textContent = 'Failed to clear logs: ' + err.message;
        errorBanner.style.display = 'block';
      });
  });

  backendApply.addEventListener('click', function() {
    backendApply.disabled = true;
    fetch('/api/backend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend: backendSelect.value })
    })
      .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
      .then(function(res) {
        backendApply.disabled = false;
        if (!res.ok) {
          errorBanner.textContent = 'Backend switch failed: ' + (res.data.error || 'unknown error');
          errorBanner.style.display = 'block';
        } else {
          errorBanner.style.display = 'none';
          fetchStatus();
        }
      })
      .catch(function(err) {
        backendApply.disabled = false;
        errorBanner.textContent = 'Backend switch failed: ' + err.message;
        errorBanner.style.display = 'block';
      });
  });

  function badge(ok, yesText, noText) {
    return ok
      ? '<span class="badge badge-green">' + yesText + '</span>'
      : '<span class="badge badge-red">' + noText + '</span>';
  }

  function kv(label, value) {
    return '<div class="kv"><span class="label">' + label + '</span><span>' + value + '</span></div>';
  }

  function renderStatus(d) {
    statusEl.innerHTML =
      kv('Version', d.health.version || '?') +
      kv('Status', badge(d.health.status === 'ok', 'healthy', d.health.status)) +
      kv('Backend', d.health.backend || '?') +
      kv('Fallback', d.health.fallback || 'none');
    // Sync dropdown with current backend
    var current = d.health.backend || '';
    if (d.health.fallback) current += ',' + d.health.fallback;
    backendSelect.value = current;
  }

  function renderService(d) {
    const s = d.service;
    serviceEl.innerHTML =
      kv('Platform', s.platform) +
      kv('Installed', badge(s.installed, 'yes', 'no')) +
      kv('Running', badge(s.running, 'yes', 'no')) +
      kv('Service file', s.service_file || 'n/a');
  }

  function renderCopilot(d) {
    if (!d.copilot) {
      copilotPanel.style.display = 'none';
      return;
    }
    copilotPanel.style.display = '';
    const c = d.copilot;
    const p = c.premium;
    const pct = p.percent_used || 0;
    const barColor = pct >= 90 ? 'var(--red)' : pct >= 70 ? 'var(--orange)' : 'var(--green)';
    const staleTag = c.stale ? ' <span class="badge badge-yellow">stale</span>' : '';
    copilotEl.innerHTML =
      kv('Plan', c.plan + staleTag) +
      kv('Premium', p.used + ' / ' + p.total + ' (' + pct + '%)') +
      '<div class="progress-bar"><div class="progress-fill" style="width:' +
        pct + '%;background:' + barColor + '"></div></div>' +
      kv('Remaining', String(p.remaining)) +
      kv('Chat', c.chat.unlimited ? 'unlimited' : 'limited') +
      kv('Completions', c.completions.unlimited ? 'unlimited' : 'limited') +
      kv('Reset', c.reset_date || '?');
  }

  function fmt(n) {
    if (n == null) return '-';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(0) + 'k';
    return String(n);
  }

  function renderModels(d) {
    if (!d.models || d.models.length === 0) {
      modelsEl.innerHTML = '<span class="label">No models available</span>';
      return;
    }
    let html = '<table><tr><th>Model ID</th><th>Owner</th>'
      + '<th>Context</th><th>Max Input</th><th>Max Output</th></tr>';
    d.models.forEach(function(m) {
      var lim = m.limits || {};
      html += '<tr><td>' + m.id + '</td><td>' + (m.owned_by || '?') + '</td>'
        + '<td>' + fmt(lim.max_context_window_tokens) + '</td>'
        + '<td>' + fmt(lim.max_prompt_tokens) + '</td>'
        + '<td>' + fmt(lim.max_output_tokens) + '</td></tr>';
    });
    html += '</table>';
    modelsEl.innerHTML = html;
  }

  function renderLogs(logs) {
    logViewer.innerHTML = '';
    logs.forEach(function(entry) {
      const div = document.createElement('div');
      div.className = 'log-line log-' + entry.level;
      div.textContent = entry.timestamp.substring(11, 19) + ' ' + entry.level.padEnd(8) + ' ' + entry.message;
      logViewer.appendChild(div);
    });
    if (autoScroll.checked) {
      logViewer.scrollTop = logViewer.scrollHeight;
    }
  }

  function fetchStatus() {
    const level = levelSelect.value;
    const url = '/api/status' + (level ? '?log_level=' + level : '');
    fetch(url)
      .then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function(d) {
        errorBanner.style.display = 'none';
        renderStatus(d);
        renderService(d);
        renderCopilot(d);
        renderModels(d);
        renderLogs(d.logs || []);
      })
      .catch(function(err) {
        errorBanner.textContent = 'Failed to fetch status: ' + err.message;
        errorBanner.style.display = 'block';
      });
  }

  fetchStatus();
  setInterval(fetchStatus, 5000);
  levelSelect.addEventListener('change', fetchStatus);
})();
</script>
</body>
</html>
"""
