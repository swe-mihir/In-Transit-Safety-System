"""
dashboard_server.py — Live Safety Dashboard Backend
====================================================
Runs a Flask-SocketIO server on port 5050.
pipeline_master.py calls push_decision() to send results here,
which are then broadcast to all connected browsers via WebSocket.

Install:
    pip install flask flask-socketio
"""

from flask import Flask, render_template_string, send_from_directory
from flask_socketio import SocketIO
import os, time, threading

app = Flask(__name__)
app.config["SECRET_KEY"] = "intransit-safety-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── In-memory log (last 50 decisions) ────────────────────────────────────────
_log = []
_log_lock = threading.Lock()
_stats = {"total": 0, "normal": 0, "risky": 0, "dangerous": 0}

# ── HTML Dashboard (single-file, served at /) ────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>InTransit Safety Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  :root {
    --bg:        #09090f;
    --surface:   #111118;
    --border:    #1e1e2e;
    --text:      #e2e2f0;
    --muted:     #555570;
    --normal:    #00e5a0;
    --risky:     #ffb300;
    --dangerous: #ff3b5c;
    --accent:    #5c6cff;
    --glow-n:    0 0 24px #00e5a055;
    --glow-r:    0 0 24px #ffb30055;
    --glow-d:    0 0 30px #ff3b5c88;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Grid noise overlay ── */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .wrap { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 32px 24px; }

  /* ── Header ── */
  header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 40px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 24px;
  }
  .logo { font-family: 'Syne', sans-serif; }
  .logo h1 { font-size: 1.6rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); }
  .logo span { font-size: .75rem; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }
  .live-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 6px 14px;
    font-size: .72rem;
    letter-spacing: .08em;
    color: var(--muted);
    text-transform: uppercase;
  }
  .live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--muted);
    transition: background .3s;
  }
  .live-dot.active { background: var(--normal); box-shadow: var(--glow-n); animation: pulse 1.6s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── Big status card ── */
  #status-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 40px;
    transition: border-color .4s, box-shadow .4s;
    min-height: 180px;
    position: relative;
    overflow: hidden;
  }
  #status-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 20% 50%, var(--card-glow, transparent) 0%, transparent 60%);
    pointer-events: none;
    transition: background .5s;
  }
  .status-icon {
    font-size: 4rem;
    line-height: 1;
    flex-shrink: 0;
    transition: transform .3s;
  }
  .status-icon.pop { animation: pop .3s ease; }
  @keyframes pop { 0%{transform:scale(1)} 50%{transform:scale(1.25)} 100%{transform:scale(1)} }

  .status-info { flex: 1; }
  .status-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -.03em;
    line-height: 1;
    margin-bottom: 8px;
    transition: color .3s;
  }
  .status-sub { font-size: .75rem; color: var(--muted); letter-spacing: .05em; }

  .score-ring {
    flex-shrink: 0;
    width: 90px; height: 90px;
    position: relative;
  }
  .score-ring svg { transform: rotate(-90deg); }
  .score-ring circle { fill: none; stroke-width: 6; }
  .ring-bg { stroke: var(--border); }
  .ring-fill { stroke-dasharray: 245; stroke-dashoffset: 245; transition: stroke-dashoffset .6s ease, stroke .4s; }
  .score-text {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
  }
  .score-val { font-size: 1.3rem; line-height: 1; }
  .score-lbl { font-size: .55rem; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; margin-top: 2px; }

  /* ── Metrics row ── */
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }
  .metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
  }
  .metric-lbl { font-size: .68rem; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; }
  .metric.normal .metric-val  { color: var(--normal); }
  .metric.risky .metric-val   { color: var(--risky); }
  .metric.dangerous .metric-val { color: var(--dangerous); }

  /* ── Bottom grid ── */
  .bottom { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

  /* ── ASR panel ── */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }
  .panel-title {
    font-size: .65rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }
  .panel-title::after { content:''; flex:1; height:1px; background:var(--border); }

  #asr-text {
    font-size: .9rem;
    line-height: 1.7;
    color: var(--text);
    min-height: 48px;
    word-break: break-word;
  }
  .asr-meta { margin-top: 12px; font-size: .68rem; color: var(--muted); display: flex; gap: 16px; }
  .conf-bar-wrap { margin-top: 8px; height: 3px; background: var(--border); border-radius: 2px; }
  .conf-bar { height: 100%; border-radius: 2px; background: var(--accent); transition: width .5s ease; }

  /* ── Model scores ── */
  .model-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: .75rem; }
  .model-name { width: 90px; color: var(--muted); }
  .model-bars { flex: 1; display: flex; gap: 4px; }
  .mbar { height: 20px; border-radius: 3px; transition: flex .5s ease; display: flex; align-items: center; justify-content: center; font-size: .6rem; font-weight: 500; overflow: hidden; }
  .mbar.normal    { background: #00e5a020; color: var(--normal); border: 1px solid var(--normal)44; }
  .mbar.risky     { background: #ffb30020; color: var(--risky);  border: 1px solid var(--risky)44; }
  .mbar.dangerous { background: #ff3b5c20; color: var(--dangerous); border: 1px solid var(--dangerous)44; }

  /* ── Log ── */
  #log-list { list-style: none; max-height: 260px; overflow-y: auto; }
  #log-list::-webkit-scrollbar { width: 4px; }
  #log-list::-webkit-scrollbar-track { background: transparent; }
  #log-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .log-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: .72rem;
    animation: slideIn .3s ease;
  }
  @keyframes slideIn { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }
  .log-item:last-child { border-bottom: none; }
  .log-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
  .log-dot.NORMAL    { background: var(--normal); }
  .log-dot.RISKY     { background: var(--risky); }
  .log-dot.DANGEROUS { background: var(--dangerous); }
  .log-time { color: var(--muted); width: 60px; flex-shrink: 0; }
  .log-label { font-weight: 500; width: 80px; flex-shrink: 0; }
  .log-label.NORMAL    { color: var(--normal); }
  .log-label.RISKY     { color: var(--risky); }
  .log-label.DANGEROUS { color: var(--dangerous); }
  .log-score { color: var(--muted); margin-left: auto; }
  .log-text { color: var(--muted); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* ── Waiting state ── */
  .waiting {
    color: var(--muted);
    font-size: .8rem;
    letter-spacing: .04em;
    animation: blink 2s infinite;
  }
  @keyframes blink { 0%,100%{opacity:.5} 50%{opacity:1} }

  /* ── Responsive ── */
  @media (max-width: 700px) {
    .metrics { grid-template-columns: repeat(2, 1fr); }
    .bottom  { grid-template-columns: 1fr; }
    #status-card { flex-direction: column; align-items: flex-start; gap: 20px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <!-- Header -->
  <header>
    <div class="logo">
      <h1>InTransit Safety Monitor</h1>
      <span>Real-time in-vehicle audio analysis</span>
    </div>
    <div class="live-pill">
      <div class="live-dot" id="live-dot"></div>
      <span id="live-label">Waiting</span>
    </div>
  </header>

  <!-- Big status -->
  <div id="status-card">
    <div class="status-icon" id="status-icon">🔇</div>
    <div class="status-info">
      <div class="status-label" id="status-label" style="color:var(--muted)">WAITING</div>
      <div class="status-sub" id="status-sub">No audio chunks processed yet</div>
    </div>
    <div class="score-ring">
      <svg width="90" height="90" viewBox="0 0 90 90">
        <circle class="ring-bg" cx="45" cy="45" r="39"/>
        <circle class="ring-fill" id="ring-fill" cx="45" cy="45" r="39" stroke="var(--muted)"/>
      </svg>
      <div class="score-text">
        <span class="score-val" id="score-val">—</span>
        <span class="score-lbl">Score</span>
      </div>
    </div>
  </div>

  <!-- Metrics -->
  <div class="metrics">
    <div class="metric">
      <div class="metric-val" id="m-total">0</div>
      <div class="metric-lbl">Chunks Processed</div>
    </div>
    <div class="metric normal">
      <div class="metric-val" id="m-normal">0</div>
      <div class="metric-lbl">Normal</div>
    </div>
    <div class="metric risky">
      <div class="metric-val" id="m-risky">0</div>
      <div class="metric-lbl">Risky</div>
    </div>
    <div class="metric dangerous">
      <div class="metric-val" id="m-dangerous">0</div>
      <div class="metric-lbl">Dangerous</div>
    </div>
  </div>

  <!-- Bottom grid -->
  <div class="bottom">

    <!-- Left: ASR + Model scores -->
    <div style="display:flex;flex-direction:column;gap:12px;">
      <div class="panel">
        <div class="panel-title">Transcription</div>
        <div id="asr-text"><span class="waiting">Listening for audio…</span></div>
        <div class="asr-meta">
          <span>Confidence: <span id="asr-conf">—</span></span>
          <span>Quality: <span id="asr-quality">—</span></span>
        </div>
        <div class="conf-bar-wrap"><div class="conf-bar" id="conf-bar" style="width:0%"></div></div>
      </div>

      <div class="panel">
        <div class="panel-title">Model Outputs</div>
        <div class="model-row">
          <span class="model-name">Sentiment</span>
          <div class="model-bars" id="bars-a">
            <div class="mbar normal"  style="flex:1">—</div>
            <div class="mbar risky"   style="flex:1">—</div>
            <div class="mbar dangerous" style="flex:1">—</div>
          </div>
        </div>
        <div class="model-row">
          <span class="model-name">YAMNet</span>
          <div class="model-bars" id="bars-b">
            <div class="mbar normal"  style="flex:1">—</div>
            <div class="mbar risky"   style="flex:1">—</div>
            <div class="mbar dangerous" style="flex:1">—</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right: Log -->
    <div class="panel">
      <div class="panel-title">Decision Log</div>
      <ul id="log-list">
        <li class="log-item"><span class="waiting" style="padding:8px 0">No decisions yet</span></li>
      </ul>
    </div>

  </div>
</div>

<script>
const socket = io();

// ── colour maps ──
const COLOR = { NORMAL: 'var(--normal)', RISKY: 'var(--risky)', DANGEROUS: 'var(--dangerous)' };
const ICON  = { NORMAL: '✅', RISKY: '⚠️', DANGEROUS: '🚨' };
const GLOW  = { NORMAL: '#00e5a020', RISKY: '#ffb30015', DANGEROUS: '#ff3b5c18' };
const BORD  = { NORMAL: 'var(--normal)', RISKY: 'var(--risky)', DANGEROUS: 'var(--dangerous)' };

let stats = { total: 0, normal: 0, risky: 0, dangerous: 0 };

socket.on('connect', () => {
  document.getElementById('live-dot').classList.add('active');
  document.getElementById('live-label').textContent = 'Connected';
});
socket.on('disconnect', () => {
  document.getElementById('live-dot').classList.remove('active');
  document.getElementById('live-label').textContent = 'Disconnected';
});

socket.on('decision', (d) => {
  const label = d.final_label.toUpperCase();
  const score = d.final_score;

  // ── Live pill ──
  document.getElementById('live-dot').classList.add('active');
  document.getElementById('live-label').textContent = 'Live';

  // ── Status card ──
  const card = document.getElementById('status-card');
  card.style.borderColor = BORD[label];
  card.style.boxShadow = label === 'DANGEROUS' ? '0 0 40px #ff3b5c33' : 'none';
  card.style.setProperty('--card-glow', GLOW[label]);

  const icon = document.getElementById('status-icon');
  icon.textContent = ICON[label];
  icon.classList.remove('pop');
  void icon.offsetWidth; // reflow
  icon.classList.add('pop');

  document.getElementById('status-label').textContent = label;
  document.getElementById('status-label').style.color = COLOR[label];
  document.getElementById('status-sub').textContent =
    `Score: ${score.toFixed(3)} · A: ${d.model_a_label?.toUpperCase()} · B: ${d.model_b_label?.toUpperCase()} · Escalated: ${d.escalated ? 'YES' : 'NO'}`;

  // ── Score ring ──
  const pct = score;
  const offset = 245 - (pct * 245);
  const ring = document.getElementById('ring-fill');
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = COLOR[label];
  document.getElementById('score-val').textContent = score.toFixed(2);
  document.getElementById('score-val').style.color = COLOR[label];

  // ── Stats ──
  stats.total++;
  stats[label.toLowerCase()]++;
  document.getElementById('m-total').textContent = stats.total;
  document.getElementById('m-normal').textContent = stats.normal;
  document.getElementById('m-risky').textContent = stats.risky;
  document.getElementById('m-dangerous').textContent = stats.dangerous;

  // ── ASR ──
  if (d.asr) {
    const text = d.asr.text || '(no speech detected)';
    document.getElementById('asr-text').textContent = text;
    const conf = d.asr.confidence ?? 0;
    document.getElementById('asr-conf').textContent = conf.toFixed(3);
    document.getElementById('asr-quality').textContent = d.asr.quality ?? '—';
    document.getElementById('conf-bar').style.width = (conf * 100) + '%';
  }

  // ── Model bars ──
  function updateBars(id, out) {
    if (!out) return;
    const wrap = document.getElementById(id);
    const bars = wrap.querySelectorAll('.mbar');
    const vals = [out.normal ?? 0, out.risky ?? 0, out.dangerous ?? 0];
    const total = vals.reduce((a,b) => a+b, 0) || 1;
    bars[0].style.flex = vals[0] / total * 10;
    bars[1].style.flex = vals[1] / total * 10;
    bars[2].style.flex = vals[2] / total * 10;
    bars[0].textContent = (vals[0]*100).toFixed(0)+'%';
    bars[1].textContent = (vals[1]*100).toFixed(0)+'%';
    bars[2].textContent = (vals[2]*100).toFixed(0)+'%';
  }
  updateBars('bars-a', d.output_a);
  updateBars('bars-b', d.output_b);

  // ── Log ──
  const ul = document.getElementById('log-list');
  if (ul.querySelector('.waiting')) ul.innerHTML = '';
  const li = document.createElement('li');
  li.className = 'log-item';
  const now = new Date().toLocaleTimeString('en-IN', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  li.innerHTML = `
    <span class="log-dot ${label}"></span>
    <span class="log-time">${now}</span>
    <span class="log-label ${label}">${label}</span>
    <span class="log-text">${d.asr?.text?.slice(0,40) || '—'}</span>
    <span class="log-score">${score.toFixed(3)}</span>
  `;
  ul.insertBefore(li, ul.firstChild);
  if (ul.children.length > 50) ul.removeChild(ul.lastChild);
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return DASHBOARD_HTML

# ── Public API called by pipeline_master ─────────────────────────────────────

def push_decision(fusion_result: dict, asr_result: dict = None,
                  output_a: dict = None, output_b: dict = None):
    """Call this from pipeline_master._on_decision() to send data to the dashboard."""
    payload = {**fusion_result}
    if asr_result:
        payload["asr"] = {
            "text":       asr_result.get("text", ""),
            "confidence": asr_result.get("confidence", 0),
            "quality":    asr_result.get("quality", ""),
        }
    if output_a:
        clean_a = {k: v for k, v in output_a.items() if k in {"normal","risky","dangerous"}}
        payload["output_a"] = clean_a
    if output_b:
        clean_b = {k: v for k, v in output_b.items() if k in {"normal","risky","dangerous"}}
        payload["output_b"] = clean_b

    # emit directly — no app_context needed with flask-socketio threading mode
    socketio.emit("decision", payload, namespace="/")

    with _log_lock:
        _log.insert(0, payload)
        if len(_log) > 50:
            _log.pop()
        _stats["total"] += 1
        _stats[fusion_result.get("final_label", "normal")] += 1


def start_dashboard(port: int = 5050):
    """Start the dashboard server in a background daemon thread."""
    print(f"[Dashboard] Starting on http://localhost:{port}")
    t = threading.Thread(
        target=lambda: socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True),
        daemon=True,
        name="DashboardServer",
    )
    t.start()
    time.sleep(2)   # ← wait for server to bind before pipeline starts
    return t