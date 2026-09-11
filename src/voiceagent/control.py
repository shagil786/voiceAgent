# src/voiceagent/control.py — agent-side control API for the voiceagent console.
"""The thin control surface a SEPARATE console repo (wizard + dashboard) talks to.

Design contract (2026-09-09 product discussion):
- voiceAgent stays the governed runtime + the authoritative tenant-bundle
  schema. The console is a different repo, different lifecycle. The ONLY
  seam between them is this HTTP API — the console never touches agent
  files.
- Everything here reuses existing, tested seams:
    * dashboard reads: SqliteDecisionLog (VOICEAGENT_AUDIT_DB),
      the ratings table in memory.py (VOICEAGENT_MEMORY_DB)
    * wizard compile: deploy/ ingest + compiler (deterministic, no LLM)
    * wizard deploy: deploy/ selfcheck + go_live (mechanical, gated)
- Fail-closed: every /api/control/* request requires the bearer token
  (VOICEAGENT_CONTROL_TOKEN). No token configured => control routes refuse
  (401) — an unauthenticated control plane is a worse failure than no plane.
- No new dependencies: stdlib http.server, matching scripts/chat_server.py.

Endpoints (all JSON):
    GET  /api/control/status                     server + data paths health
    GET  /api/control/calls?limit=50             recent governed decisions
    GET  /api/control/ratings?limit=50           recent caller ratings
    GET  /api/control/summary                    calls/verdicts/rating/escalation
    POST /api/control/onboard/preview            {source:{url|text}, interview}
                                                 -> compiled bundle (NO writes)
    POST /api/control/onboard/deploy             {deploy_id} -> self-check + live
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Mapping

# --- data access (all through existing stores, read-mostly) --------------------

class _SafeRow(dict):
    """dict subclass so json.dumps handles non-str keys from sqlite rows."""

    def __missing__(self, key):  # pragma: no cover - defensive
        return str(key)


def _rows(db: Path | None, sql: str, args: tuple = ()) -> list[dict]:
    """Read rows from a sqlite file (created if absent by its owner store).
    None db => [] (the endpoint reports the store is not configured)."""
    if db is None or not db.exists():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        out = [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()
    return out


def calls(audit_db: Path | None, limit: int = 50) -> list[dict]:
    rows = _rows(audit_db, "SELECT ts, conv_id, action, verdict, reasons, "
                            "amount, authenticated FROM decision_log "
                            "ORDER BY id DESC LIMIT ?", (max(1, limit),))
    for r in rows:
        try:
            r["reasons"] = json.loads(r.get("reasons") or "[]")
        except (TypeError, json.JSONDecodeError):
            r["reasons"] = []
    return rows


def ratings(memory_db: Path | None, limit: int = 50) -> list[dict]:
    return _rows(memory_db, "SELECT tenant, session_id, ts, rating, comment "
                            "FROM ratings ORDER BY rowid DESC LIMIT ?",
                 (max(1, limit),))


def conversation_spines(audit_db: Path | None,
                         limit_convs: int = 12) -> list[dict]:
    """Group the decision log into per-conversation spines (ordered newest
    first) for the quality judge — the platform's authoritative record."""
    rows = calls(audit_db, limit=100000)
    by_conv: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        cid = r.get("conv_id") or "unknown"
        if cid not in by_conv:
            by_conv[cid] = []
            order.append(cid)
        by_conv[cid].append(r)
    out = []
    for cid in order[-limit_convs:]:
        out.append({"conv_id": cid, "rows": by_conv[cid][-20:]})
    return list(reversed(out))


def summary(audit_db: Path | None, memory_db: Path | None) -> dict:
    rows = calls(audit_db, limit=100000)
    verdicts = Counter(r["verdict"] for r in rows)
    by_conv = {r["conv_id"] for r in rows}
    escalations = sum(1 for r in rows if r["verdict"] == "ESCALATE")
    rated = ratings(memory_db, limit=100000)
    avg_rating = (sum(float(r["rating"]) for r in rated) / len(rated)
                  if rated else None)
    return {
        "calls": len(rows),
        "conversations": len(by_conv),
        "verdicts": dict(verdicts),
        "escalation_rate": (escalations / len(rows)) if rows else None,
        "ratings": len(rated),
        "avg_rating_10": avg_rating,
    }


# --- wizard: compile preview (NO writes) ----------------------------------------

def compile_preview(source: dict, interview: dict) -> dict:
    """Ingest + compile a bundle preview from {url} or {text}. Pure:
    writes nothing, runs no self-checks. The deterministic compiler always
    runs; the frontier brain additionally DRAFTS tools/intents/entities/
    evals when configured (silent fallback otherwise). The console renders
    this for the owner's review — including gap `questions` the owner must
    answer; approval happens at deploy."""
    from voiceagent.deploy.draft import preview_bundle
    from voiceagent.deploy.ingest import fetch_site, ingest_owner_paste
    url = (source or {}).get("url") or ""
    text = (source or {}).get("text") or ""
    if not url and not text:
        raise ValueError("source requires 'url' or 'text'")
    crawled: list[dict] = []
    pasted: list[dict] = []
    if text:
        pasted.append(ingest_owner_paste(text))
    if url:
        crawled = fetch_site(url)  # allowlist empty; scoped crawl
    chunks = list(pasted) + crawled
    out = preview_bundle("preview", chunks, interview or {})
    out["note"] += " — preview only, nothing was written or approved"
    return out


def deploy_bundle(deploy_dir: str | Path, bundle, version: str = "v1") -> dict:
    """Self-check + mechanical go-live for an already-approved bundle.
    Mirrors the deploy/ CLI gate: writes only on full pass."""
    from voiceagent.deploy.bundle import save_bundle
    from voiceagent.deploy.selfcheck import go_live, run_self_checks
    d = Path(deploy_dir)
    d.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, d / "bundle.json")
    results = run_self_checks(bundle, live_spot=True)
    live = go_live(str(d), version, results)
    return {"checks": results, "live": live,
            "summary": f"{sum(1 for r in results if r['passed'])}/"
                       f"{len(results)} passed"}


# --- HTTP layer ----------------------------------------------------------------

class ControlServer(BaseHTTPRequestHandler):
    """JSON control-plane server. Requires Authorization: Bearer <token>
    matching VOICEAGENT_CONTROL_TOKEN on every /api/control/* request."""
    audit_db: Path | None = None
    memory_db: Path | None = None
    deploy_root: Path | None = None
    token: str | None = None
    cors_origins: str = "*"  # set from VOICEAGENT_CORS_ORIGINS at startup

    def log_message(self, *a):  # silence per-request stderr
        pass

    # -- helpers -------------------------------------------------------------

    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _cors(self) -> None:
        # The console app runs on a different origin (dev :5173/:5174, dist
        # on its own port). The bearer token is the access gate; CORS is
        # scoped to simple requests (Authorization + Content-Type) — no
        # cookies, no credentials mode. Operators lock this down with
        # VOICEAGENT_CORS_ORIGINS (comma-separated; "*" keeps local dev).
        origin = (self.headers.get("Origin") or "").strip()
        allowed = [o.strip() for o in (self.cors_origins or "*").split(",") if o.strip()]
        if "*" in allowed or origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin if "*" not in allowed and origin else "*")
        self.send_header("Access-Control-Allow-Headers",
                         "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")

    def do_OPTIONS(self):  # noqa: N802 - CORS preflight
        self.send_response(204)  # status line FIRST, then headers
        self._cors()
        self.end_headers()

    def _authorized(self) -> bool:
        if not self.token:  # fail closed: no token configured
            return False
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.token}"

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise ValueError("body must be JSON")
        return body if isinstance(body, dict) else {}

    # -- routes --------------------------------------------------------------

    def do_GET(self):  # noqa: N802 (http.server API)
        if not self._authorized():
            self._send(401, {"error": "unauthorized — set "
                                      "VOICEAGENT_CONTROL_TOKEN on the agent"})
            return
        from urllib.parse import parse_qs, urlsplit
        path, _, qs = self.path.partition("?")
        q = parse_qs(qs)
        try:
            if path == "/api/control/status":
                self._send(200, {
                    "ok": True,
                    "audit_db": str(self.audit_db) if self.audit_db else None,
                    "memory_db": str(self.memory_db) if self.memory_db else None,
                    "deploy_root": str(self.deploy_root)
                                   if self.deploy_root else None,
                })
            elif path == "/api/control/calls":
                limit = int(q.get("limit", ["50"])[0])
                self._send(200, {"calls": calls(self.audit_db, limit)})
            elif path == "/api/control/ratings":
                limit = int(q.get("limit", ["50"])[0])
                self._send(200, {"ratings": ratings(self.memory_db, limit)})
            elif path == "/api/control/summary":
                self._send(200, summary(self.audit_db, self.memory_db))
            elif path == "/api/control/scores":
                from voiceagent.quality import build_judge_from_env, judge_conversation
                judge = build_judge_from_env()
                conv = q.get("conv")
                if conv:
                    spines = conversation_spines(self.audit_db, limit_convs=500)
                    target = next((s for s in spines if s["conv_id"] == conv[0]), None)
                    if target is None:
                        self._send(404, {"error": f"no conversation {conv[0]}"})
                        return
                    out = judge_conversation(target["rows"], judge)
                    out["conv_id"] = conv[0]
                    self._send(200, out)
                    return
                scored = []
                for s in conversation_spines(self.audit_db, limit_convs=12):
                    r = judge_conversation(s["rows"], judge)
                    r["conv_id"] = s["conv_id"]
                    scored.append(r)
                self._send(200, {"scores": scored})
            elif path == "/api/control/deploys":
                from voiceagent.deploy.bundle import list_deploys, read_live_deploy
                root = self.deploy_root or Path("data/deploy")
                self._send(200, {"deploys": list_deploys(root),
                                 "live": read_live_deploy(root)})
            else:
                self._send(404, {"error": f"no such endpoint {path}"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):  # noqa: N802 (http.server API)
        if not self._authorized():
            self._send(401, {"error": "unauthorized — set "
                                      "VOICEAGENT_CONTROL_TOKEN on the agent"})
            return
        path = self.path
        try:
            body = self._read_json()
            if path == "/api/control/onboard/preview":
                out = compile_preview(body.get("source", {}),
                                      body.get("interview", {}))
                self._send(200, out)
            elif path == "/api/control/onboard/deploy":
                # Deploy from the payload the console re-sends with its
                # approval + deploy_id. The preview pipeline (deterministic
                # compile + brain drafts) runs again so approval and deploy
                # see identical surfaces; the approved draft then
                # MATERIALIZES into a verified tenant bundle beside the
                # deploy versions. Writes only on full pass, both gates.
                interview = body.get("interview", {})
                deploy_id = str(body.get("deploy_id") or "deployed")
                source = body.get("source", {})
                from voiceagent.deploy.compiler import compile_bundle
                from voiceagent.deploy.draft import preview_bundle
                from voiceagent.deploy.ingest import fetch_site, ingest_owner_paste
                from voiceagent.deploy.materialize import materialize_tenant_bundle
                crawled: list[dict] = []
                pasted: list[dict] = []
                if source.get("text"):
                    pasted.append(ingest_owner_paste(source["text"]))
                if source.get("url"):
                    crawled = fetch_site(source["url"])
                chunks = list(pasted) + crawled
                preview = preview_bundle(deploy_id, chunks, interview)
                bundle = compile_bundle(deploy_id, chunks, interview)
                root = self.deploy_root or Path("data/deploy")
                tenant_name = str(body.get("tenant_name") or deploy_id)
                tenant = materialize_tenant_bundle(
                    {"tools": preview.get("tools", []),
                     "intents": preview.get("intents", {}),
                     "entities": preview.get("entities"),
                     "policies": preview.get("policies", {})},
                    interview, chunks, tenant_name,
                    root / deploy_id / "tenant")
                out = deploy_bundle(root / deploy_id, bundle)
                checks = list(out.get("checks", []))
                checks.append({
                    "name": "tenant bundle verifies",
                    "passed": bool(tenant["ok"]),
                    "detail": ("tenant/" + " + ".join(tenant["files"][:4]) +
                               ("…" if len(tenant["files"]) > 4 else ""))
                    if tenant["ok"] else "; ".join(tenant["errors"][:3]),
                })
                if not tenant["ok"]:
                    out["live"] = False
                out["checks"] = checks
                out["tenant"] = tenant
                # Summary counts EVERYTHING the dashboard renders, including
                # the tenant gate — "10/10 passed" beside a failed tenant
                # check was a self-contradicting deploy report.
                passed = sum(1 for c in checks if c.get("passed"))
                out["summary"] = f"{passed}/{len(checks)} passed"
                if tenant.get("skipped"):
                    out["summary"] += (" (tools skipped, no params: " +
                                       ", ".join(tenant["skipped"][:3]) + ")")
                if out.get("live") and tenant.get("ok"):
                    from voiceagent.deploy.bundle import write_live_deploy as _wld
                    _wld(root, deploy_id)
                self._send(200, out)
            elif path == "/api/control/deploy/rollback":
                from voiceagent.deploy.bundle import (
                    load_bundle, safe_deploy_id, write_live_deploy,
                )
                from voiceagent.deploy.materialize import verify_files
                from voiceagent.deploy.selfcheck import run_self_checks
                target = safe_deploy_id(body.get("deploy_id"))
                if not target:
                    raise ValueError("deploy_id missing or invalid")
                root = self.deploy_root or Path("data/deploy")
                d = root / target
                # Canonical layout: bundle files live in the "bundle.json"
                # subdir (see control.deploy_bundle) - a bare file there is
                # not a deploy.
                bdir = d / "bundle.json"
                if not bdir.is_dir() or not (bdir / "bundle.json").is_file():
                    raise ValueError(f"unknown deploy {target!r}")
                from voiceagent.deploy.bundle import read_live_deploy as _rld
                if _rld(root) == target:
                    raise ValueError(f"{target!r} is already live")
                bundle = load_bundle(str(d / "bundle.json"))
                checks = run_self_checks(bundle)
                tenant_dir = d / "tenant"
                tenant_files = {
                    str(p.relative_to(tenant_dir)): p.read_text(encoding="utf-8")
                    for p in sorted(tenant_dir.rglob("*")) if p.is_file()
                } if tenant_dir.is_dir() else {}
                tenant_errors = verify_files(tenant_files) if tenant_files else ["no tenant bundle staged"]
                ok = (len(checks) >= 10 and all(c.get("passed") for c in checks)
                      and not tenant_errors)
                if ok:
                    write_live_deploy(root, target)
                self._send(200, {"ok": ok, "deploy_id": target, "checks": checks,
                                 "tenant_errors": tenant_errors,
                                 "tenant_dir": str(tenant_dir),
                                 "summary": f"{sum(1 for c in checks if c.get('passed'))}/{len(checks)} passed"})
            else:
                self._send(404, {"error": f"no such endpoint {path}"})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def server_from_env(env: Mapping[str, str] | None = None):
    """Class-level config from env (injectable for tests). None => os.environ."""
    e = os.environ if env is None else env
    ControlServer.token = (e.get("VOICEAGENT_CONTROL_TOKEN") or "").strip() or None
    audit = (e.get("VOICEAGENT_AUDIT_DB") or "").strip()
    memory = (e.get("VOICEAGENT_MEMORY_DB") or "").strip()
    deploy_root = (e.get("VOICEAGENT_DEPLOY_ROOT") or "").strip()
    ControlServer.audit_db = Path(audit) if audit else None
    ControlServer.memory_db = Path(memory) if memory else None
    ControlServer.deploy_root = Path(deploy_root) if deploy_root else None
    ControlServer.cors_origins = (e.get("VOICEAGENT_CORS_ORIGINS") or "*").strip() or "*"
    return ControlServer
