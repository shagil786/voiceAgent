#!/usr/bin/env python3
"""LiveKit inbound worker: serve webhooks, spawn one session thread per room.

Usage:
    .venv/bin/python scripts/livekit_worker.py [--port 8080]

Secrets (LIVEKIT_URL/KEY/SECRET) come from `.env` via `RuntimeConfig`;
never logged, never in code. One daemon thread per `call-*` room runs
`run_room_session` (greet → loop → hangup); the webhook thread never blocks.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.config import load_config
from voiceagent.telephony.inbound import run_room_session, webhook_handler

logger = logging.getLogger("livekit_worker")


def build_deps():
    """Assemble the governed Orchestrator for room sessions.

    The SAME brain every other entry point uses (see voiceagent.runtime):
    frontier proposal + governed tool runner + policy engine + Deployment. We do
    NOT return None — if no frontier brain is configured the worker must refuse
    to start, because an inbound call with `orchestrator=None` would crash on the
    first `handle_turn`. `main()` also fails fast before binding the socket.

    `language` is the deployment's known query language (telephony trunk config);
    None = blind ASR (whisper small) which never auto-routes to the Indic
    engine. Override per trunk via VOICEAGENT_DEFAULT_LANG.
    """
    from voiceagent.runtime import build_orchestrator

    orchestrator = build_orchestrator()
    language = os.environ.get("VOICEAGENT_DEFAULT_LANG") or None
    return {"orchestrator": orchestrator, "session_id": None, "language": language}


def make_server(config, join_room) -> BaseHTTPRequestHandler:
    handler_fn = webhook_handler(config, join_room)

    class WebhookHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8", "replace")
            sig = self.headers.get("Authorization", "")
            ok = handler_fn(body, sig)
            self.send_response(200 if ok else 404)
            self.end_headers()
            self.wfile.write(b"ok" if ok else b"ignored")

        def log_message(self, fmt, *args) -> None:
            logger.info(fmt, *args)

    return WebhookHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config()
    deps = build_deps()
    if deps.get("orchestrator") is None:
        print("ERROR: VOICEAGENT_FRONTIER_URL not set — the LiveKit worker "
              "cannot serve calls without a governed brain. See .env.example.",
              file=sys.stderr)
        sys.exit(2)

    def join_room(room_name: str) -> None:
        session_deps = dict(deps)
        session_deps["session_id"] = room_name
        t = threading.Thread(
            target=run_room_session,
            args=(room_name, config, session_deps),
            name=f"room-{room_name}",
            daemon=True,
        )
        t.start()
        logger.info("spawned session thread for room %s", room_name)

    handler_cls = make_server(config, join_room)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler_cls)
    logger.info("webhook listening on :%d (prefix %r)", args.port, config.livekit_room_prefix)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutdown requested")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
