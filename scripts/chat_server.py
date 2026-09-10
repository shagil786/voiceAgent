"""VoiceAgent demo HTTP server (stdlib only).
Usage: python scripts/chat_server.py [port] [host]   (default 8000, 127.0.0.1)
Open http://127.0.0.1:8000 in a browser. Containers pass 0.0.0.0 as host so a
published port is reachable from outside the container namespace.

Hardened for exposure: per-client rate limiting on the API endpoints (see
src/voiceagent/chat_server.rate_limiter_from_env) and a socket timeout so a
stalled client cannot hold the (single-threaded) server forever.
"""
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.chat import run_turn
from voiceagent.chat_server import (RateLimiter, _client_ip, build_html,
                                    rate_limiter_from_env)
from voiceagent.memory import SQLiteMemory, public_dict
from voiceagent.runtime import build_orchestrator as _runtime_build_orchestrator

ORCH = None
MEMORY: SQLiteMemory | None = None
RATE_LIMITER: RateLimiter | None = None
DEFAULT_CONV_ID = "demo-http"
# Per-socket-operation timeout: a client that opens a connection and never
# finishes its request (slowloris) must not pin the single-threaded server.
REQUEST_TIMEOUT_S = 15.0


def _build_live_orchestrator():
    """Build the governed Orchestrator (the only production brain) — same wiring
    the LiveKit worker and REPL use. Returns None when no frontier brain is set;
    main() fails fast in that case so the demo never serves an ungoverned path."""
    return _runtime_build_orchestrator()


def load_dotenv(path: Path) -> None:
    """Same .env loader every other script entry point uses — chat_server was
    the odd one out, so running it per README never saw .env config."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Handler(BaseHTTPRequestHandler):
    timeout = REQUEST_TIMEOUT_S  # per-socket-op timeout (BaseHTTPRequestHandler)

    def _rate_limited(self) -> bool:
        """Consume one slot for this client on the work-doing API endpoints.
        When over budget, answer 429 (with Retry-After) and return True."""
        if RATE_LIMITER is None:
            return False
        key = _client_ip(self, trust_proxy=RATE_LIMITER.trust_proxy)
        if RATE_LIMITER.allow(key):
            return False
        retry_after = RATE_LIMITER.retry_after(key)
        body = json.dumps({"error": "rate limit exceeded",
                           "retry_after_s": retry_after}).encode()
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            body = build_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/history":
            if self._rate_limited():
                return
            params = urllib.parse.parse_qs(query)
            conv_id = params.get("conv_id", [DEFAULT_CONV_ID])[0]
            turns = MEMORY.history(conv_id) if MEMORY is not None else []
            self._json([public_dict(t) for t in turns])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/turn":
            self.send_error(404)
            return
        if self._rate_limited():
            return
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        text = payload.get("text", "")
        auth = bool(payload.get("authenticated", False))
        conv_id = str(payload.get("conv_id", DEFAULT_CONV_ID))
        if not text:
            self._json({"error": "empty text"}, 400)
            return
        out = run_turn(ORCH, text, authenticated=auth, conv_id=conv_id,
                       memory=MEMORY)
        self._json(out)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the console clean


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    # Loopback-only by default; containers pass 0.0.0.0 so a published port
    # maps through (a 127.0.0.1 bind inside a container is unreachable).
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    Path("data/out").mkdir(parents=True, exist_ok=True)
    # Chat transcripts persist here (full turn text) — override the path
    # per deployment and purge it on schedule (retention.purge_expired
    # covers VOICEAGENT_CHAT_MEMORY_DB; unset keeps this default file).
    MEMORY = SQLiteMemory(os.environ.get("VOICEAGENT_CHAT_MEMORY_DB")
                          or "data/out/memory.db")
    RATE_LIMITER = rate_limiter_from_env()
    ORCH = _build_live_orchestrator()
    if not os.environ.get("VOICEAGENT_TENANT"):
        # One line, not a gate: the built-in Acme deployment is a demo
        # default and must never impersonate a real deployment silently.
        print("WARNING: no VOICEAGENT_TENANT set — serving the built-in "
              "demo deployment (Acme); set VOICEAGENT_TENANT for a real "
              "deployment", file=sys.stderr)
    if ORCH is None:
        print("ERROR: VOICEAGENT_FRONTIER_URL not set — the demo server serves "
              "the governed Orchestrator; set a frontier (see .env.example).",
              file=sys.stderr)
        sys.exit(2)
    print(f"VoiceAgent governed demo at http://{host}:{port}  (Ctrl-C to stop)")
    if RATE_LIMITER is not None:
        print(f"rate limit: {RATE_LIMITER.max_events} req/"
              f"{RATE_LIMITER.window_s:.0f}s per client IP on /api/*")
    else:
        print("rate limit: DISABLED (set VOICEAGENT_HTTP_RATE_LIMIT to enable)")
    HTTPServer((host, port), Handler).serve_forever()
