# tests/test_demo_warning.py — serving a fictional business silently is a
# defect: with no VOICEAGENT_TENANT the entry points serve the built-in Acme
# demo deployment and must say so (one stderr line; a warning, not a gate —
# demo ergonomics are preserved). Exercised via subprocess with the frontier
# unset so both servers fail fast offline.
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("VOICEAGENT_FRONTIER_URL", "VOICEAGENT_TENANT")}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                          capture_output=True, text=True, env=env, cwd=ROOT,
                          timeout=120)


def test_chat_server_warns_when_serving_demo_default():
    r = _run("chat_server.py")
    assert r.returncode != 0  # no frontier configured: fails fast as before
    assert "no VOICEAGENT_TENANT set" in r.stderr
    assert "demo deployment (Acme)" in r.stderr

def test_livekit_worker_warns_when_serving_demo_default():
    r = _run("livekit_worker.py")
    assert r.returncode != 0
    assert "no VOICEAGENT_TENANT set" in r.stderr
    assert "demo deployment (Acme)" in r.stderr
