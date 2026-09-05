"""VoiceAgent demo HTTP server (stdlib only).
Usage: python scripts/chat_server.py [port]   (default 8000)
Open http://127.0.0.1:8000 in a browser.
"""
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.chat import run_turn
from voiceagent.chat_server import build_html
from voiceagent.memory import SQLiteMemory, public_dict
from voiceagent.runtime import build_orchestrator as _runtime_build_orchestrator

ORCH = None
MEMORY: SQLiteMemory | None = None
DEFAULT_CONV_ID = "demo-http"


def _build_live_orchestrator():
    """Build the governed Orchestrator (the only production brain) — same wiring
    the LiveKit worker and REPL use. Returns None when no frontier brain is set;
    main() fails fast in that case so the demo never serves an ungoverned path."""
    return _runtime_build_orchestrator()


class Handler(BaseHTTPRequestHandler):
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
    Path("data/out").mkdir(parents=True, exist_ok=True)
    MEMORY = SQLiteMemory("data/out/memory.db")
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
    print(f"VoiceAgent governed demo at http://127.0.0.1:{port}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
