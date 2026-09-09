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


def load_dotenv(path: Path) -> None:
    """Tiny stdlib .env loader (repo pattern): KEY=value lines, '#' comments,
    optional quotes. Never overrides existing environment variables."""
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
    # Warm the frontier connection: the first real call must not pay TCP/TLS
    # + provider cold-start (observed 18.8s brain spikes). Fail-open — a
    # warmup failure means the provider is down; the call path reports it.
    try:
        orchestrator.brain.client.chat(
            [{"role": "user", "content": "Reply with the single word: ready."}],
            tools=None)
        logger.info("frontier warmup ok")
    except Exception:
        logger.warning("frontier warmup failed (continuing)", exc_info=True)
    # Warm the ASR engine (whisper-small CPU load is ~100s on first use —
    # never make the first caller pay it).
    try:
        from voiceagent.asr import warmup_asr
        warmup_asr()
        logger.info("asr warmup ok")
    except Exception:
        logger.warning("asr warmup failed (continuing)", exc_info=True)
    # Warm the intent sidecar classifier (two SentenceTransformer models,
    # ~16s CPU load on the FIRST live turn — the observed 17.5s mute after
    # the greeting). Load at boot and hand the orchestrator the warm copy.
    try:
        from voiceagent.memory import _sidecar_classifier
        orchestrator._intent_classifier = _sidecar_classifier()
        logger.info("intent classifier warmup ok")
    except Exception:
        logger.warning("intent classifier warmup failed (continuing)",
                       exc_info=True)
    language = os.environ.get("VOICEAGENT_DEFAULT_LANG") or None
    # Declared greeting (tenant data): instant pickup line, no brain roundtrip.
    greeting = getattr(orchestrator, "greeting", "") or ""
    return {"orchestrator": orchestrator, "session_id": None,
            "language": language, "greeting": greeting}


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
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    config = load_config()
    deps = build_deps()
    if not os.environ.get("VOICEAGENT_TENANT"):
        # One line, not a gate: the built-in Acme deployment is a demo
        # default and must never impersonate a real deployment silently.
        print("WARNING: no VOICEAGENT_TENANT set — serving the built-in "
              "demo deployment (Acme); set VOICEAGENT_TENANT for a real "
              "deployment", file=sys.stderr)
    if deps.get("orchestrator") is None:
        print("ERROR: VOICEAGENT_FRONTIER_URL not set — the LiveKit worker "
              "cannot serve calls without a governed brain. See .env.example.",
              file=sys.stderr)
        sys.exit(2)

    if not os.environ.get("VOICEAGENT_ERP_URL") \
            and not os.environ.get("VOICEAGENT_ALLOW_MOCK_ERP"):
        print("ERROR: VOICEAGENT_ERP_URL not set — live calls must be backed "
              "by a REAL ERP backend, never the in-memory MockERP fixture. "
              "Run scripts/erp_server.py and set VOICEAGENT_ERP_URL in .env "
              "(or set VOICEAGENT_ALLOW_MOCK_ERP=1 to explicitly serve "
              "non-real data for offline tests only).",
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
