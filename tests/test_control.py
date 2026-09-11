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


def _stage_deploy(root, deploy_id, text="Sunrise Vet treats dogs. Cancel with VST-1042."):
    # Offline deploy staging: compile + save bundle + materialize tenant,
    # mirroring the deploy endpoint without the HTTP layer.
    from voiceagent.deploy.bundle import save_bundle, write_live
    from voiceagent.deploy.compiler import compile_bundle
    from voiceagent.deploy.materialize import materialize_tenant_bundle
    chunks = [{"source": "owner_paste", "text": text}]
    bundle = compile_bundle(deploy_id, chunks, {"offering": "vet"})
    d = root / deploy_id
    # Canonical layout, mirroring control.deploy_bundle:
    # save_bundle(bundle, deploy_dir / "bundle.json").
    save_bundle(bundle, d / "bundle.json")
    write_live(str(d), "v1")
    tenant = materialize_tenant_bundle(
        {"tools": [], "intents": {}, "entities": None, "policies": {}},
        {"offering": "vet"}, chunks, deploy_id, d / "tenant")
    assert tenant["ok"], tenant["errors"]
    return d


def test_deploys_history_lists_staged_and_live(ctl):
    req, tmp_path = ctl
    root = tmp_path / "deploy"
    _stage_deploy(root, "aaa")
    _stage_deploy(root, "bbb")
    from voiceagent.deploy.bundle import write_live_deploy
    write_live_deploy(root, "bbb")
    out = req("GET", "/api/control/deploys")
    assert out["live"] == "bbb"
    ids = [d["deploy_id"] for d in out["deploys"]]
    assert ids == ["bbb", "aaa"] or sorted(ids) == ["aaa", "bbb"]
    assert all(d["tenant_ok"] for d in out["deploys"])


def test_rollback_repoints_live_after_reverify(ctl):
    req, tmp_path = ctl
    root = tmp_path / "deploy"
    _stage_deploy(root, "aaa")
    _stage_deploy(root, "bbb")
    from voiceagent.deploy.bundle import write_live_deploy, read_live_deploy
    write_live_deploy(root, "bbb")
    out = req("POST", "/api/control/deploy/rollback", {"deploy_id": "aaa"})
    assert out["ok"] is True, (out.get("tenant_errors"), out.get("summary"))
    assert out["deploy_id"] == "aaa"
    assert read_live_deploy(root) == "aaa"


def test_rollback_refuses_bad_targets(ctl):
    import urllib.error
    req, tmp_path = ctl
    root = tmp_path / "deploy"
    _stage_deploy(root, "aaa")
    for bad in ["nope", "../evil", "", "aaa/../bbb"]:
        try:
            req("POST", "/api/control/deploy/rollback", {"deploy_id": bad})
        except urllib.error.HTTPError as e:
            assert e.code == 400
        else:
            raise AssertionError(f"rollback accepted {bad!r}")
    # already-live refuses too
    from voiceagent.deploy.bundle import write_live_deploy
    write_live_deploy(root, "aaa")
    try:
        req("POST", "/api/control/deploy/rollback", {"deploy_id": "aaa"})
    except urllib.error.HTTPError as e:
        assert e.code == 400
    else:
        raise AssertionError("rollback accepted the live deploy")


def test_rollback_fails_closed_on_broken_tenant(ctl):
    req, tmp_path = ctl
    root = tmp_path / "deploy"
    d = _stage_deploy(root, "aaa")
    (d / "tenant" / "tenant.json").write_text("not: [valid", encoding="utf-8")
    out = req("POST", "/api/control/deploy/rollback", {"deploy_id": "aaa"})
    assert out["ok"] is False
    assert out["tenant_errors"]
    from voiceagent.deploy.bundle import read_live_deploy
    assert read_live_deploy(root) is None


def test_viewer_role_reads_but_cannot_mutate(tmp_path):
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from voiceagent.control import server_from_env
    cls = server_from_env({"VOICEAGENT_CONTROL_TOKEN": "adm",
                           "VOICEAGENT_VIEW_TOKEN": "view",
                           "VOICEAGENT_DEPLOY_ROOT": str(tmp_path / "deploy")})

    def raw(method, path, token, body=None):
        r = urllib.request.Request(
            base + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, out = raw("GET", "/api/control/status", "view")
        assert code == 200 and out["role"] == "viewer"
        code, out = raw("GET", "/api/control/status", "adm")
        assert code == 200 and out["role"] == "admin"
        code, _ = raw("GET", "/api/control/summary", "view")
        assert code == 200
        code, out = raw("POST", "/api/control/deploy/rollback", "view",
                        {"deploy_id": "x"})
        assert code == 403 and "view-only" in out["error"]
        code, _ = raw("POST", "/api/control/onboard/deploy", "view",
                      {"deploy_id": "x"})
        assert code == 403
        code, _ = raw("GET", "/api/control/summary", "wrong")
        assert code == 401
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_no_view_token_means_single_operator(ctl):
    # The shared ctl fixture configures no view token: anything but the
    # control token is unauthorized, and status reports admin.
    import urllib.error
    req, _ = ctl
    try:
        req("GET", "/api/control/summary", token="view")
    except urllib.error.HTTPError as e:
        assert e.code == 401
    else:
        raise AssertionError("view token accepted without configuration")
