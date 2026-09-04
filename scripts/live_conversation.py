#!/usr/bin/env python3
"""Live conversation runner — the Orchestrator with a REAL frontier brain.

Usage:
    VOICEAGENT_FRONTIER_URL=https://api.groq.com/openai/v1 \
    VOICEAGENT_FRONTIER_MODEL=llama-3.3-70b-versatile \
    VOICEAGENT_FRONTIER_KEY=*** \
    PYTHONPATH=src .venv/bin/python scripts/live_conversation.py [--scripted]

.env is auto-loaded from the repo root (never committed). --scripted runs
the canonical 4-turn reschedule conversation; otherwise an interactive
REPL (type 'exit' to quit). Every turn prints: brain latency, tool calls,
policy verdicts, and the agent's reply.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.runtime import build_orchestrator as _runtime_build_orchestrator


def load_dotenv(path: Path) -> None:
    """Tiny stdlib .env loader: KEY=VALUE lines, '#' comments, optional quotes."""
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


def build_orchestrator():
    # Single assembly seam: the governed brain lives in voiceagent.runtime so the
    # LiveKit worker and this REPL share the exact same wiring. Fail fast when
    # the frontier brain isn't configured.
    orch = _runtime_build_orchestrator()
    if orch is None:
        print("ERROR: VOICEAGENT_FRONTIER_URL not set (see .env.example).",
              file=sys.stderr)
        sys.exit(2)
    return orch


SCRIPTED_TURNS = [
    "Hi, where is my order ORD-4821?",
    "Actually, can you reschedule it to tomorrow?",
    "Yes please, go ahead.",
    "Great, thanks!",
]


def run_scripted(orch: Orchestrator) -> None:
    print("=== scripted conversation (session: live-001) ===\n")
    for i, text in enumerate(SCRIPTED_TURNS, 1):
        t0 = time.perf_counter()
        res = orch.handle_turn("live-001", text, authenticated=True)
        print(f"[turn {i}] customer: {text}")
        print(f"    agent  : {res.reply}")
        for a in res.actions:
            print(f"    action : {a.get('tool')} verdict={a.get('verdict')} "
                  f"ok={a.get('ok')} value={json.dumps(a.get('value'))[:120]}")
        print(f"    brain  : {res.brain_latency_s:.2f}s  "
              f"tool_calls={res.raw_tool_calls}  wall={time.perf_counter()-t0:.2f}s")
        print()
    print("=== done — check decision log + MockERP state above ===")


def run_repl(orch: Orchestrator) -> None:
    print("=== live REPL (session: live-001) — type 'exit' to quit ===")
    session = "live-001"
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"exit", "quit"}:
            break
        res = orch.handle_turn(session, text, authenticated=True)
        print(f"agent> {res.reply}")
        for a in res.actions:
            print(f"  [{a.get('tool')} verdict={a.get('verdict')} ok={a.get('ok')}]")
        print(f"  (brain {res.brain_latency_s:.2f}s)")


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    orch = build_orchestrator()
    if "--scripted" in sys.argv:
        run_scripted(orch)
    else:
        run_repl(orch)
