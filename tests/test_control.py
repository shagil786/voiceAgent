# tests/test_control.py — agent-side control API (console seam).
"""The control surface the separate voiceagent-console repo will call:
bearer-auth fail-closed, dashboard reads over the real stores, and the
wizard preview/deploy endpoints over the deploy pipeline. Fully offline:
stub-free local HTTP server + temp sqlite files; no network."""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from voiceagent.decisionlog import DecisionEntry, SqliteDecisionLog
from voiceagent.memory import IntentMemoryStore
from voiceagent.control import server_from_env


@pytest.fixture()
def ctl(tmp_path: Path):
    audit = tmp_path / "audit.sqlite"
    mem = tmp_path / "memory.sqlite"
    log = SqliteDecisionLog(audit)
    log.record(DecisionEntry(ts="2026-09-09T03:00:00", conv_id="C-1",
                             action="cancel_order", verdict="ALLOW",
                             reasons=["ok"], amount=100.0,
                             authenticated=True))
    log.record(DecisionEntry(ts="2026-09-09T03:01:00", conv_id="C-2",
                             action="fraud", verdict="ESCALATE",
                             reasons=["high risk"]))
    m = IntentMemoryStore(str(mem))
    m.record_rating("default", "C-1", 8.0, "fast")
    cls = server_from_env({
        "VOICEAGENT_CONTROL_TOKEN": "tok",
        "VOICEAGENT_AUDIT_DB": str(audit),
        "VOICEAGENT_MEMORY_DB": str(mem),
        "VOICEAGENT_DEPLOY_ROOT": str(tmp_path / "deploy"),
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def req(method, path, body=None, token="tok"):
        r = urllib.request.Request(
            base + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(r) as resp:
            return json.loads(resp.read())

    yield req, tmp_path
    httpd.shutdown()
    httpd.server_close()


def test_auth_fails_closed(ctl):
    req, _ = ctl
    with pytest.raises(urllib.error.HTTPError) as ei:
        req("GET", "/api/control/summary", token="wrong")
    assert ei.value.code == 401
    with pytest.raises(urllib.error.HTTPError):
        req("GET", "/api/control/summary", token=None)


def test_status_reports_stores(ctl):
    req, _ = ctl
    st = req("GET", "/api/control/status")
    assert st["ok"] is True
    assert st["audit_db"].endswith("audit.sqlite")
    assert st["memory_db"].endswith("memory.sqlite")


def test_summary_aggregates(ctl):
    req, _ = ctl
    s = req("GET", "/api/control/summary")
    assert s["calls"] == 2
    assert s["verdicts"] == {"ALLOW": 1, "ESCALATE": 1}
    assert s["escalation_rate"] == 0.5
    assert s["ratings"] == 1
    assert s["avg_rating_10"] == 8.0


def test_calls_and_ratings_endpoints(ctl):
    req, _ = ctl
    calls = req("GET", "/api/control/calls")["calls"]
    assert len(calls) == 2
    assert calls[0]["conv_id"] == "C-2"  # newest first
    assert calls[0]["verdict"] == "ESCALATE"
    rated = req("GET", "/api/control/ratings")["ratings"]
    assert rated[0]["rating"] == 8.0 and rated[0]["comment"] == "fast"


def test_preview_compiles_without_writes(ctl):
    req, tmp_path = ctl
    out = req("POST", "/api/control/onboard/preview", {
        "source": {"text": "Sunrise Dental Clinic offers root canals and "
                           "cleanings from 9am-6pm weekdays."},
        "interview": {"offering": "dental clinic",
                      "top_asks": ["booking", "price"]}})
    names = [t["name"] for t in out["tools"]]
    assert "escalate_to_human" in names
    assert any("book" in n or n == "booking" for n in names)
    assert len(out["knowledge"]) >= 1
    assert "preview only" in out["note"]
    # nothing was written anywhere under the deploy root
    deploy_root = tmp_path / "deploy"
    assert not deploy_root.exists() or not any(deploy_root.iterdir())


def test_preview_requires_a_source(ctl):
    req, _ = ctl
    with pytest.raises(urllib.error.HTTPError) as ei:
        req("POST", "/api/control/onboard/preview", {"source": {}})
    assert ei.value.code == 400


def test_cors_defaults_to_wildcard_for_local_dev(ctl):
    # req helper hides raw headers; the class attr is the contract here
    # (live echo behavior is pinned by the restricted test below).
    from voiceagent.control import ControlServer
    assert ControlServer.cors_origins == "*"


def test_cors_restricted_echoes_allowed_origin(tmp_path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from voiceagent.control import server_from_env
    cls = server_from_env({"VOICEAGENT_CONTROL_TOKEN": "tok",
                           "VOICEAGENT_CORS_ORIGINS": "http://127.0.0.1:8321"})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        r = urllib.request.Request(base + "/api/control/status", method="OPTIONS",
                                   headers={"Origin": "http://127.0.0.1:8321"})
        with urllib.request.urlopen(r) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:8321"
        r2 = urllib.request.Request(base + "/api/control/status", method="OPTIONS",
                                    headers={"Origin": "https://evil.example"})
        with urllib.request.urlopen(r2) as resp2:
            assert resp2.headers.get("Access-Control-Allow-Origin") is None
    finally:
        httpd.shutdown()
        httpd.server_close()
