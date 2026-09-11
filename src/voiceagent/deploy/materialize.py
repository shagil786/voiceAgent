# src/voiceagent/deploy/materialize.py — approved preview -> live tenant bundle.
"""The last mile of onboarding: drafted surfaces (deploy/draft.py) become an
on-disk tenant bundle (data/tenants/<name>/ or a deploy-dir staging copy)
that validate_tenant.py passes and the worker can serve.

Posture: WRITE NOTHING unless the bundle verifies. Content builds in memory,
materializes into a temp dir, runs the REAL validator
(scripts/validate_tenant.py::validate — the same gate CI runs), and only
then moves into place. An invalid draft returns errors; the dashboard shows
them and the owner re-previews with answers. Approval stays human: this
module runs inside the deploy endpoint AFTER the owner approves, and the
written bundle's proposals.yaml carries status approved (committing it, in
git-ops deployments, IS the approval record per ADR-005).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


def _parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [a.strip() for a in value.split(",") if a.strip()]
    return []


def tenant_persona(interview: dict) -> dict:
    offering = str((interview or {}).get("offering") or "local business")
    handoffs = _parse_list((interview or {}).get("handoff_triggers"))
    never = _parse_list((interview or {}).get("never_promise"))
    greeting = str((interview or {}).get("greeting") or "").strip()
    persona: dict = {
        "role": f"the front-desk voice assistant for {offering[:120]}",
        "tone": "concise, no invented facts",
    }
    if handoffs:
        persona["handoff_triggers"] = handoffs
    if never:
        persona["never_say"] = never
    if greeting:
        persona["greeting"] = greeting
    return persona


def tenant_languages(interview: dict, chunks: list[dict]) -> list[str]:
    langs = _parse_list((interview or {}).get("languages"))
    if langs:
        return langs[:8]
    # Auto-serve what the content detects (English always serves).
    from voiceagent.langid import detect_language
    seen: list[str] = []
    for c in chunks or []:
        t = str((c or {}).get("text") or "")
        if len(t) > 40:
            code = detect_language(t)
            if code not in seen:
                seen.append(code)
    if "en" in seen:
        seen.remove("en")
    return ["en"] + seen[:4]


# Platform-owned safety valves: every deployment wires these from the
# platform surface (tools.yaml), so they are NEVER tenant proposals.
PLATFORM_VALVES = frozenset(
    {"escalate_to_human", "end_call", "record_feedback"})


def _tool_params(t: dict) -> list[str]:
    """Params for a drafted/v1 tool: the `params` list, else the v1
    `parameters.properties` keys (the two bundle generations disagree on
    shape; the boundary translates)."""
    params = t.get("params")
    if isinstance(params, list) and params:
        return [str(p) for p in params]
    props = t.get("parameters")
    if isinstance(props, dict):
        inner = props.get("properties")
        if isinstance(inner, dict) and inner:
            return [str(k) for k in inner]
    return []


def build_tenant_files(surfaces: dict, interview: dict,
                       chunks: list[dict], tenant_name: str
                       ) -> tuple[dict[str, str], list[str]]:
    """Draft surfaces + interview -> ({relative_path: file_text}, skipped).
    Pure: no disk, no validation — verify_bundle() decides. Skipped names
    parameter-less non-valve tools the proposals gate cannot take."""
    import yaml
    iv = interview or {}
    drafted = surfaces or {}
    files: dict[str, str] = {}

    persona = tenant_persona(iv)
    langs = tenant_languages(iv, chunks)
    persona["languages"] = langs
    currency = str(iv.get("currency") or "$")
    files["tenant.json"] = json.dumps(
        {"name": tenant_name, "persona": persona, "currency": currency},
        indent=2, ensure_ascii=False) + "\n"

    intents = drafted.get("intents") or {}
    if isinstance(intents, dict):
        for action, phrases in intents.items():
            clean = [str(p) for p in (phrases or []) if str(p).strip()][:8]
            if action and clean:
                files[f"intents/{action}.yaml"] = yaml.safe_dump(
                    clean, allow_unicode=True, sort_keys=False)

    entities = drafted.get("entities") or {}
    shapes = (entities.get("record_ids")
              if isinstance(entities, dict) else None) or []
    if shapes:
        files["entities.yaml"] = (
            "# Tenant-declared record-ID shapes (onboarded — drafted from\n"
            "# the owner's content, approved in the dashboard).\n"
            + yaml.safe_dump({"record_ids": shapes}, allow_unicode=True,
                             sort_keys=False))

    tools = drafted.get("tools") or []
    candidates = [t for t in tools
                  if isinstance(t, dict) and t.get("name")
                  and t.get("name") not in PLATFORM_VALVES]
    proposals = [t for t in candidates if _tool_params(t)]
    skipped = [str(t.get("name")) for t in candidates if not _tool_params(t)]
    if proposals:
        entries = []
        for t in proposals:
            entries.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "params": _tool_params(t),
                "action": t.get("action", t["name"]),
                "operation": t.get("operation", t["name"]),
                "operation_params": dict(t.get("operation_params", {})),
                "resource_type": t.get("resource_type"),
                "id_param": t.get("id_param"),
                "filter_param": t.get("filter_param"),
                "filter_key": t.get("filter_key"),
                "preconditions": list(t.get("preconditions", [])),
                "facts": list(t.get("facts", [])),
                "side_effects": bool(t.get("side_effects", False)),
                "risk_class": t.get("risk_class", "read"),
                "provenance": "ai",
                "status": "approved",
            })
        files["proposals.yaml"] = (
            "# Tenant tools — drafted from the owner's content, APPROVED in\n"
            "# the dashboard (approval = deploy). Committing this file in\n"
            "# git-ops setups IS the human approval record (ADR-005).\n"
            + yaml.safe_dump({"proposals": entries}, allow_unicode=True,
                             sort_keys=False))

    policies: dict[str, dict] = {"escalate_to_human": {"allow": True}}
    for t in proposals:
        action = t.get("action", t["name"])
        mutating = t.get("risk_class") == "mutating"
        policies[str(action)] = ({"require_auth": True} if mutating
                                 else {"allow": True})
    for a, r in ((drafted.get("policies") or {}).items()
                 if isinstance(drafted.get("policies"), dict) else []):
        if isinstance(r, dict) and r.get("require_approval"):
            policies[str(a)] = {"require_auth": True}
    files["policies.yaml"] = yaml.safe_dump(
        policies, allow_unicode=True, sort_keys=False)

    texts = [str((c or {}).get("text") or "").strip() for c in chunks or []]
    texts = [t[:4000] for t in texts if t][:12]
    for i, t in enumerate(texts):
        files[f"knowledge/{i:02d}.md"] = t + "\n"
    return files, skipped


def _load_validator():
    """scripts/validate_tenant.py::validate, loaded by path (scripts are
    entry points, not a package — same exec-by-path precedent as tests)."""
    path = Path(__file__).resolve().parents[3] / "scripts" / \
        "validate_tenant.py"
    spec = importlib.util.spec_from_file_location(
        "onboard_validate_tenant", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate


def verify_files(files: dict[str, str]) -> list[str]:
    """Write files to a temp bundle dir and run the real validator."""
    validate = _load_validator()
    tmp = Path(tempfile.mkdtemp(prefix="tenant-verify-"))
    try:
        for rel, text in files.items():
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return [str(e) for e in validate(tmp)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def materialize_tenant_bundle(surfaces: dict, interview: dict,
                              chunks: list[dict], tenant_name: str,
                              out_dir: str | Path) -> dict:
    """Build, verify, and write the tenant bundle. Returns
    {ok, files: [rel paths], skipped: [tool names], errors: []}. Writes
    NOTHING unless the bundle verifies clean."""
    safe = "".join(c for c in str(tenant_name)
                   if c.isalnum() or c in ("-", "_")).strip("-_") or "tenant"
    files, skipped = build_tenant_files(surfaces, interview, chunks, safe)
    errors = verify_files(files)
    if errors:
        return {"ok": False, "files": [], "skipped": skipped,
                "errors": errors, "tenant": safe}
    root = Path(out_dir)
    if root.exists():
        return {"ok": False, "files": [], "skipped": skipped, "errors": [
            f"refusing to overwrite existing {root}"], "tenant": safe}
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return {"ok": True, "files": sorted(files), "skipped": skipped,
            "errors": [], "tenant": safe}
