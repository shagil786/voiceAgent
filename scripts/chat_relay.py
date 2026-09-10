#!/usr/bin/env python3
"""scripts/chat_relay.py — text bridge to the SAME agent a phone call runs.

Local HTTP server (127.0.0.1:8095) exposing the governed runtime
orchestrator (frontier brain + tenant bundle + real ERP via
VOICEAGENT_ERP_URL) so you can talk to the agent from chat without
spending money on calls. Same brain/tools/policy/ERP as the LiveKit
worker; only ASR/TTS/telephony are absent (text in, text out).

POST /turn  {"message": "...", "session": "webchat-1"}
  -> {"reply": "...", "actions": [...], "goodbye": bool}

Boot warms the intent classifier so the FIRST turn is fast (no 17s cliff).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s:%(name)s:%(message)s")

from voiceagent.dotenv import load_dotenv  # single source; call only inside main(), never at module level
ORCH = None  # built in __main__ (importing this module must stay side-effect
# free: no .env leak, no model builds — same rule as scripts/chat_server.py)

def build():
    from voiceagent.runtime import build_orchestrator
    if not os.environ.get("VOICEAGENT_ERP_URL"):
        print("ERROR: VOICEAGENT_ERP_URL not set (real ERP required)")
        sys.exit(2)
    orch = build_orchestrator()
    if orch is None:
        print("ERROR: no frontier brain configured")
        sys.exit(2)
    try:  # warm the intent sidecar classifier (two ST models ~16s)
        from voiceagent.memory import _sidecar_classifier
        orch._intent_classifier = _sidecar_classifier()
    except Exception:
        logging.getLogger("chat_relay").warning("classifier warmup failed", exc_info=True)
    try:
        orch.brain.client.chat(
            [{"role": "user", "content": "Reply with the single word: ready."}],
            tools=None)
    except Exception:
        pass
    return orch

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # quiet
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            message = (body.get("message") or "").strip()
            session = (body.get("session") or "webchat-default").strip()
        except Exception:
            self._send(400, {"error": "bad json"})
            return
        if not message:
            self._send(400, {"error": "empty message"})
            return
        assert ORCH is not None  # set in __main__ before serve_forever
        result = ORCH.handle_turn(session, message[:2000])
        acts = getattr(result, "actions", None) or []
        goodbye = any(a.get("action") == "end_call" and a.get("ok")
                      for a in acts)
        self._send(200, {"reply": result.reply, "actions": acts,
                         "goodbye": goodbye})

    def _send(self, code: int, payload) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    ORCH = build()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8095
    print("chat_relay on http://127.0.0.1:%d (orchestrator ready)" % port,
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
