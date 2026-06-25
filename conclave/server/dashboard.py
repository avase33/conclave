"""Render an execution trace into a self-contained HTML dashboard.

No external assets. The trace JSON is embedded and rendered client-side as a
timeline of agent, LLM and tool events, with per-type counts and token totals.
"""

from __future__ import annotations

import json
from typing import Any

_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conclave trace</title>
<style>
:root{--bg:#0b0e14;--panel:#141925;--ink:#e6edf3;--muted:#8b98a9;--line:#26304180;
--blue:#5e86ff;--green:#7df0c0;--amber:#ffb454;--red:#ff6b6b;--purple:#b78bff;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:30px 22px 80px}
h1{font-size:22px;margin:0 0 2px}.sub{color:var(--muted);font-size:13px;margin:0 0 22px}
.logo{display:inline-flex;width:34px;height:34px;border-radius:9px;align-items:center;
justify-content:center;background:linear-gradient(135deg,var(--blue),var(--purple));
color:#0b0e14;font-weight:800;margin-right:10px;vertical-align:middle}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.card .v{font-size:24px;font-weight:700;margin-top:4px}
.ev{display:flex;gap:12px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;
margin-bottom:8px;background:var(--panel)}
.dot{width:10px;height:10px;border-radius:50%;margin-top:6px;flex:none}
.ev .type{font-weight:600;font-size:13px}.ev .meta{color:var(--muted);font-size:12.5px;
font-family:ui-monospace,Menlo,monospace;word-break:break-word}
.t{color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums;margin-left:auto;flex:none}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:26px}
</style></head><body><div class="wrap">
<h1><span class="logo">C</span>Conclave trace</h1>
<p class="sub" id="sub"></p>
<div class="cards" id="cards"></div>
<div id="events"></div>
<footer>Conclave multi-agent framework · offline trace viewer</footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const COLOR={run_start:'var(--purple)',run_end:'var(--purple)',agent_start:'var(--blue)',
agent_end:'var(--blue)',llm_call:'var(--green)',tool_call:'var(--amber)',
tool_result:'var(--amber)',step:'var(--blue)',handoff:'var(--purple)',note:'var(--muted)',error:'var(--red)'};
const ev=D.events||[];
const counts={};ev.forEach(e=>counts[e.type]=(counts[e.type]||0)+1);
document.getElementById('sub').textContent=`${ev.length} events`;
const cards=[['Events',ev.length],['LLM calls',counts.llm_call||0],
['Tool calls',counts.tool_call||0],['Errors',counts.error||0]];
document.getElementById('cards').innerHTML=cards.map(([k,v])=>
`<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
const t0=ev.length?ev[0].timestamp:0;
document.getElementById('events').innerHTML=ev.map(e=>{
  const d=e.data||{};const meta=Object.entries(d).map(([k,v])=>`${k}=${esc(JSON.stringify(v))}`).join('  ');
  const dt=((e.timestamp-t0)).toFixed(3)+'s';
  return `<div class="ev"><span class="dot" style="background:${COLOR[e.type]||'var(--muted)'}"></span>
  <div style="min-width:0"><div class="type">${esc(e.type)}</div><div class="meta">${meta}</div></div>
  <span class="t">+${dt}</span></div>`;
}).join('')||'<p class="sub">No events recorded.</p>';
</script></body></html>
"""


def render_dashboard(trace: dict[str, Any]) -> str:
    payload = json.dumps(trace, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__DATA__", payload)
