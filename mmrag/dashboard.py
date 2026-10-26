"""
mmrag/dashboard.py
=====================

The LIVE COST-PER-TICKET DASHBOARD: one self-contained HTML page (inline CSS +
vanilla JS, no CDN — works offline) that polls /api/metrics every 2 seconds.

Security note: ticket content ultimately comes from customers and models, so
every value is rendered with `textContent` / DOM APIs — never `innerHTML`.
A ticket summary containing <script> must display as text, not execute.
(tests/test_api_dashboard.py asserts the page never uses innerHTML.)
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multimodal RAG Engine - Cost per Ticket</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #ffffff; --text: #1a1d21; --muted: #667085; --border: #e4e7ec;
    --accent: #4c6ef5; --accent2: #f76707; --good: #2f9e44; --warn: #e67700; --bad: #c92a2a;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #111418; --card: #1a1f26; --text: #e8eaed; --muted: #98a2b3; --border: #2a313b; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 20px; background: var(--bg); color: var(--text);
         font: 14px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin-bottom: 16px; }
  .banner { background: #fff3bf; color: #664d03; border: 1px solid #ffe066; padding: 8px 12px;
            border-radius: 6px; margin-bottom: 16px; display: none; }
  .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  .kpi .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
  .kpi .note { color: var(--muted); font-size: 12px; }
  .row { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin-bottom: 16px; }
  h2 { font-size: 14px; margin: 0 0 10px; }
  .bar { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
  .bar .name { width: 90px; color: var(--muted); font-size: 12px; }
  .bar .track { flex: 1; background: var(--border); border-radius: 4px; height: 14px; overflow: hidden; }
  .bar .fill { height: 100%; background: var(--accent); }
  .bar .fill.alt { background: var(--accent2); }
  .bar .num { width: 84px; text-align: right; font-variant-numeric: tabular-nums; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); font-size: 12px; }
  th { color: var(--muted); font-weight: 500; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; border: 1px solid var(--border); }
  .pill.ok { color: var(--good); } .pill.needs_human_review { color: var(--warn); } .pill.failed { color: var(--bad); }
  .muted { color: var(--muted); }
  svg { width: 100%; height: 90px; }
</style>
</head>
<body>
<h1>Multimodal RAG Engine &mdash; cost per ticket</h1>
<div class="sub">Live view &middot; refreshes every 2s &middot; <span id="updated">connecting...</span></div>
<div class="banner" id="demo">DEMO MODE: providers are deterministic fakes and every cost shown is SYNTHETIC. It demonstrates the plumbing, not real prices or model quality.</div>

<div class="grid" id="kpis"></div>

<div class="row">
  <div class="card"><h2>Cost by stage</h2><div id="byStage"></div></div>
  <div class="card"><h2>Cost by channel</h2><div id="byChannel"></div></div>
  <div class="card"><h2>Cost per ticket (recent)</h2><div id="series"></div><div class="muted" id="seriesNote"></div></div>
</div>

<div class="row">
  <div class="card"><h2>Reliability &amp; knowledge base</h2><div id="reliability"></div></div>
</div>

<div class="card"><h2>Recent tickets</h2>
  <table><thead><tr><th>Ticket</th><th>Status</th><th>Channels</th><th>Category</th><th>Urgency</th><th>Cost</th><th>Latency</th><th>Flags</th></tr></thead>
  <tbody id="rows"></tbody></table>
</div>

<script>
const $ = (id) => document.getElementById(id);
const usd = (v) => "$" + Number(v).toFixed(v < 0.01 ? 6 : 4);

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (cls) node.className = cls;
  return node;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function kpi(label, value, note) {
  const card = el("div", null, "card kpi");
  card.appendChild(el("div", label, "label"));
  card.appendChild(el("div", value, "value"));
  card.appendChild(el("div", note || "", "note"));
  return card;
}

function bars(target, data, alt) {
  clear(target);
  const entries = Object.entries(data);
  if (!entries.length) { target.appendChild(el("div", "no data yet", "muted")); return; }
  const max = Math.max(...entries.map(([, v]) => v)) || 1;
  for (const [name, value] of entries) {
    const row = el("div", null, "bar");
    row.appendChild(el("div", name, "name"));
    const track = el("div", null, "track");
    const fill = el("div", null, alt ? "fill alt" : "fill");
    fill.style.width = (value / max * 100).toFixed(1) + "%";
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("div", usd(value), "num"));
    target.appendChild(row);
  }
}

function sparkline(target, values) {
  clear(target);
  if (values.length < 2) { target.appendChild(el("div", "need 2+ tickets", "muted")); return; }
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 300 90");
  svg.setAttribute("preserveAspectRatio", "none");
  const max = Math.max(...values) || 1;
  const pts = values.map((v, i) => (i / (values.length - 1) * 300).toFixed(1) + "," + (85 - v / max * 78).toFixed(1));
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts.join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "#4c6ef5");
  line.setAttribute("stroke-width", "2");
  svg.appendChild(line);
  target.appendChild(svg);
}

function render(m) {
  $("demo").style.display = m.demo_mode ? "block" : "none";
  const k = $("kpis"); clear(k);
  k.appendChild(kpi("Tickets", m.tickets_total, "rejected inputs: " + m.rejected_total));
  k.appendChild(kpi("Avg cost / ticket", usd(m.avg_cost_per_ticket_usd), "p95 " + usd(m.p95_cost_usd)));
  k.appendChild(kpi("Total cost", usd(m.total_cost_usd), "cap/ticket " + usd(m.budget_per_ticket_usd)));
  k.appendChild(kpi("Latency p50 / p95", Math.round(m.p50_latency_ms) + " / " + Math.round(m.p95_latency_ms) + " ms", "end-to-end"));
  k.appendChild(kpi("Needs review", (m.needs_review_rate * 100).toFixed(1) + "%", JSON.stringify(m.status_counts)));
  k.appendChild(kpi("Estimated-price share", (m.estimated_price_share * 100).toFixed(1) + "%", "priced with the default entry"));

  bars($("byStage"), m.cost_by_stage, false);
  bars($("byChannel"), m.cost_by_channel, true);
  sparkline($("series"), m.cost_series);
  $("seriesNote").textContent = m.cost_series.length + " most recent tickets";

  const rel = $("reliability"); clear(rel);
  const r = m.reliability;
  rel.appendChild(el("div", "model calls: " + r.calls + " | retries: " + r.retries + " | fallbacks: " + r.fallbacks + " | failures: " + r.failures));
  rel.appendChild(el("div", "tokens in/out: " + m.tokens.input + " / " + m.tokens.output + " | audio: " + m.tokens.audio_seconds + "s", "muted"));
  const kb = m.knowledge_base;
  rel.appendChild(el("div", "knowledge base: " + kb.chunks + " chunks from " + kb.sources + " sources " + JSON.stringify(kb.by_modality), "muted"));

  const body = $("rows"); clear(body);
  for (const t of m.recent_tickets) {
    const tr = document.createElement("tr");
    tr.appendChild(el("td", t.ticket_id));
    const status = document.createElement("td");
    status.appendChild(el("span", t.status, "pill " + t.status));
    tr.appendChild(status);
    tr.appendChild(el("td", t.channels.join("+") || "-"));
    tr.appendChild(el("td", t.category || "-"));
    tr.appendChild(el("td", t.urgency || "-"));
    tr.appendChild(el("td", usd(t.cost_usd)));
    tr.appendChild(el("td", Math.round(t.latency_ms) + " ms"));
    tr.appendChild(el("td", t.flags.join(", ") || "-", "muted"));
    body.appendChild(tr);
  }
}

async function refresh() {
  try {
    const response = await fetch("/api/metrics");
    render(await response.json());
    $("updated").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (err) {
    $("updated").textContent = "disconnected - retrying";
  }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""
