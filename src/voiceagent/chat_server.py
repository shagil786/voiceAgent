# src/voiceagent/chat_server.py
"""Shared bits for the demo HTTP server (kept importable/testable)."""

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>VoiceAgent demo</title>
<style>
body{font-family:system-ui;max-width:720px;margin:40px auto;padding:0 16px;color:#1a1a1a}
textarea{width:100%;min-height:70px;font-size:15px;padding:8px;border:1px solid #ccc;border-radius:6px}
button{margin-top:8px;padding:10px 18px;font-size:15px;background:#0b5;color:#fff;border:0;border-radius:6px;cursor:pointer}
pre{background:#f5f5f5;padding:12px;border-radius:6px;white-space:pre-wrap}
label{display:block;margin-top:10px;font-size:14px}
</style></head><body>
<h1>VoiceAgent</h1>
<p>Type a support query in English, Hindi, or Hinglish. You'll see the reply, the
proposed action, and the policy decision with reasons.</p>
<textarea id="q" placeholder="e.g. Bhai mera order abhi tak nahi aaya, order id ORD-55671 hai"></textarea>
<label><input type="checkbox" id="auth"> authenticated session</label>
<button onclick="go()">Send</button>
<pre id="out">—</pre>
<script>
async function go(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const auth=document.getElementById('auth').checked;
  const r=await fetch('/api/turn',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:q,authenticated:auth})});
  const d=await r.json();
  let t='[agent] '+d.reply+'\n[action] '+(d.action||'none')+'  [policy] '+(d.decision||'n/a');
  if(d.executed) t+='  [tool: EXECUTED]';
  (d.reasons||[]).forEach(x=>t+='\n   · '+x);
  document.getElementById('out').textContent=t;
}
</script></body></html>"""


def build_html() -> str:
    return PAGE
