# src/voiceagent/chat_server.py
"""Shared bits for the demo HTTP server (kept importable/testable): the demo
page plus the per-client rate limiter the server wires in."""
import os
import threading
import time
from collections import defaultdict, deque
from typing import Mapping

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


# --- per-client rate limiting -------------------------------------------------
# Fixed-window counters keyed by client IP, guarded by a lock (HTTPServer is
# single-threaded today, but the guard makes the limiter safe if the server
# ever moves to ThreadingHTTPServer). Endpoints behind the limiter are the
# ones that do real work (/api/turn runs the full LLM turn; /api/history hits
# SQLite) — static pages are not limited. Key memory is bounded per key
# (<= max_events timestamps); unique keys (IPs) expire lazily on next touch,
# which is fine for a demo — a reverse proxy with unbounded key cardinality
# would need a sweep, not a demo necessity.

class RateLimiter:
    """Fixed-window per-key rate limiter. `allow(key)` consumes one slot and
    returns True while the key has budget in the current window; hits beyond
    the limit return False until the window rolls over. Keys are client IPs
    (via `_client_ip`); `max_events`/`window_s` bound memory and reset lag —
    idle windows expire lazily on next touch. Deliberately stdlib-only."""

    def __init__(self, max_events: int, window_s: float = 60.0):
        self.max_events = max_events
        self.window_s = window_s
        # Identity mode: False = socket peer (client-supplied X-Forwarded-For
        # NOT trusted); True = XFF leftmost hop (only behind a trusted proxy).
        self.trust_proxy = False
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        """Consume one event for `key`; True when within budget for the
        current window, False once the key is over its limit."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window_s
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_events:
                return False
            hits.append(now)
            return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        """Whole seconds (min 1) until the key's oldest in-window hit expires
        — used for the Retry-After header on 429 responses."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return 1
            return max(1, int(hits[0] + self.window_s - now) + 1)


def _client_ip(handler, trust_proxy: bool = False) -> str:
    """Extract the client identity for rate limiting. X-Forwarded-For is
    CLIENT-SUPPLIED: trusting it when NOT behind a proxy lets an attacker
    rotate the header per request and bypass the limiter entirely (measured:
    20/20 through a 3/min limit). Trust is therefore OPT-IN via
    VOICEAGENT_TRUST_PROXY=true — set it ONLY when the server sits behind a
    trusted reverse proxy that overwrites/sets XFF (then the leftmost hop is
    the real client). Default (no trust): the socket peer address."""
    if trust_proxy:
        fwd = handler.headers.get("X-Forwarded-For")
        if fwd:
            return fwd.split(",")[0].strip()
    return handler.client_address[0]


def rate_limiter_from_env(env: Mapping[str, str] | None = None) -> RateLimiter | None:
    """Build the demo server's rate limiter from env (None disables limiting).
    Defaults: 30 turn-relevant requests/min per client IP (the demo's UI,
    with retries, needs nowhere near that). VOICEAGENT_TRUST_PROXY=true opts
    into X-Forwarded-For identity (set it only behind a trusted reverse
    proxy); the default keys the limiter on the socket peer."""
    e = os.environ if env is None else env
    limit = e.get("VOICEAGENT_HTTP_RATE_LIMIT", "").strip()
    if not limit:
        return None
    try:
        max_events = int(limit)
        window_s = float(e.get("VOICEAGENT_HTTP_RATE_WINDOW_S", "60"))
    except ValueError:
        return None
    if max_events <= 0 or window_s <= 0:
        return None
    trust = e.get("VOICEAGENT_TRUST_PROXY", "").strip().lower() in ("1", "true", "yes")
    limiter = RateLimiter(max_events, window_s)
    limiter.trust_proxy = trust
    return limiter
