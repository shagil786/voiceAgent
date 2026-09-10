# src/voiceagent/deploy/draft.py — brain-assisted bundle drafting.
"""Turn ingested business content into GENERATABLE tenant surfaces.

The v1 compiler (compiler.py) is deterministic slot-filling: it can only
repackage what the interview already says. This module lets the frontier
brain READ the crawled site + owner paste and draft what the interview
didn't state:

- tools: operations with resource_type/id_param (the draft_from_api_spec
  shape in proposals.py) — PROPOSED, never approved here;
- intents: 3-5 caller exemplars per drafted tool;
- entities: record-ID shapes spotted in the text (booking refs, order
  numbers) in the tenant entities.yaml schema;
- policies: mutating operations default to require_approval;
- evals: self-check turns drawn from the real content;
- questions: gaps NO draft can fill — the dashboard asks the owner
  (ERP hookup, served languages, ID examples, greeting, handoff contact).

Posture (same as proposals): the brain DRAFTS, data DECLARES, humans
APPROVE. Every drafted surface is validated before it leaves this module
(validate_proposal for tools, regex compile for ID shapes, non-empty
strings elsewhere); anything invalid is dropped and becomes a question
instead of a silent bad draft. No frontier configured (or a brain failure)
falls back to the v1 compiler + the deterministic gap questions — the
preview NEVER fails for lack of a model.
"""
from __future__ import annotations

import json
import re

from voiceagent.deploy.compiler import compile_bundle

_MAX_CHARS = 6000  # chunk text budget per draft call (prompt cap)
_MAX_TOKENS = 2000


def _chunk_text(chunks: list[dict]) -> str:
    out: list[str] = []
    for c in chunks or []:
        t = str((c or {}).get("text") or "").strip()
        if t:
            out.append(t)
    text = "\n\n".join(out)[:_MAX_CHARS]
    return text


DRAFT_PROMPT = """You onboard businesses onto a voice support agent. Read the business content below and draft the agent's data surfaces as ONE JSON object (no prose, no markdown fences):

{"tools": [{"tool_name": str, "description": str, "params": [str], "action": str, "operation": str, "operation_params": {}, "resource_type": str|null, "id_param": str|null, "filter_param": str|null, "filter_key": str|null, "preconditions": [str], "facts": [str], "side_effects": bool, "risk_class": "read"|"mutating"}],
 "intents": {"<tool_action>": ["caller phrase", ...3-5 each]},
 "entities": {"record_ids": [{"code": str, "digit_pattern": str (ONE regex, group(1)=digits), "prefix_pattern": str|null, "bare_digits": bool, "min_digits": int, "max_digits": int}]},
 "policies": {"<action>": {"require_approval": bool}},
 "evals": [{"user": str (a real caller question from the content)}],
 "questions": [{"id": str, "prompt": str, "why": str, "kind": "text"|"choice"|"multi", "options": [str], "answer_key": str}]}

Rules: operations read or mutate ONE resource type; id_param names the reference callers state; record-ID codes are 2-4 uppercase letters and digit_patterns MUST contain a capture group; mutating operations require_approval=true; questions cover ONLY what the content does not state (backend hookup, served languages, reference-number examples, opening greeting, human handoff contact, risky promises to confirm). Max 8 tools, 6 questions.

INTERVIEW: %s

CONTENT:
%s"""


def _parse_draft(raw: str) -> dict:
    """First JSON object in the reply; {} when unparseable (fail-open)."""
    if not raw:
        return {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _valid_tools(ops: object) -> tuple[list[dict], list[str]]:
    """Operations -> validated proposal dicts; invalid ones come back as
    dropped tool_names (each becomes a dashboard question)."""
    from voiceagent.deploy.bundle import TOOL_STATES
    from voiceagent.proposals import (PROPOSED, validate_proposal,
                                      draft_from_api_spec)
    if not isinstance(ops, list):
        return [], []
    spec = {"operations": [o for o in ops if isinstance(o, dict)
                           and o.get("tool_name") and o.get("operation")]}
    try:
        proposals = draft_from_api_spec(spec, provenance="ai",
                                        status=PROPOSED)
    except Exception:
        return [], [str(o.get("tool_name", "?")) for o in spec["operations"]]
    good, dropped = [], []
    for prop, op in zip(proposals, spec["operations"]):
        errs = validate_proposal(prop)
        if errs:
            dropped.append(prop.name)
            continue
        good.append({
            "name": prop.name, "description": prop.description,
            "params": list(prop.params), "action": prop.action,
            "operation": prop.operation,
            "operation_params": dict(prop.operation_params),
            "resource_type": prop.resource_type, "id_param": prop.id_param,
            "filter_param": prop.filter_param, "filter_key": prop.filter_key,
            "preconditions": list(prop.preconditions),
            "facts": list(prop.facts), "side_effects": prop.side_effects,
            # Deploy bundles speak uppercase states; proposals speak
            # lowercase — the boundary translates, never the human.
            "risk_class": prop.risk_class, "state": TOOL_STATES[0],
            "policy_action": prop.action})
    return good, dropped


def _valid_entities(ent: object) -> tuple[dict | None, bool]:
    """entities.yaml shape or None; False validity mints a question."""
    if not isinstance(ent, dict):
        return None, True
    shapes = ent.get("record_ids") or []
    if not isinstance(shapes, list) or not shapes:
        return None, True
    out: list[dict] = []
    for s in shapes:
        if not isinstance(s, dict):
            return None, False
        code = str(s.get("code") or "").upper()
        digit = str(s.get("digit_pattern") or "")
        if not code or not digit:
            return None, False
        try:
            rx = re.compile(digit, re.IGNORECASE)
        except re.error:
            return None, False
        if rx.groups < 1:
            return None, False
        prefix = s.get("prefix_pattern")
        if prefix:
            try:
                re.compile(str(prefix), re.IGNORECASE)
            except re.error:
                return None, False
        try:
            lo = int(s.get("min_digits", 4))
            hi = int(s.get("max_digits", 10))
        except (TypeError, ValueError):
            return None, False
        out.append({"code": code, "digit_pattern": digit,
                    "prefix_pattern": str(prefix) if prefix else None,
                    "bare_digits": bool(s.get("bare_digits", False)),
                    "min_digits": lo, "max_digits": hi})
    return {"record_ids": out}, True


def _valid_intents(intents: object, tool_actions: set[str]) -> dict:
    """action -> 3-5 non-empty caller phrases, keyed ONLY to drafted tools
    (stray keys are dropped — exemplars must attach to a surface)."""
    if not isinstance(intents, dict):
        return {}
    out: dict[str, list[str]] = {}
    for action, phrases in intents.items():
        if action not in tool_actions or not isinstance(phrases, list):
            continue
        clean = [str(p).strip() for p in phrases if str(p).strip()][:5]
        if len(clean) >= 2:
            out[str(action)] = clean
    return out


def gap_questions(chunks: list[dict], interview: dict,
                  draft: dict) -> list[dict]:
    """Deterministic gap detectors: what NO draft can fill, the dashboard
    must ask. Each question names the interview key its answer fills."""
    text = _chunk_text(chunks).lower()
    iv = interview or {}
    drafted_tools = draft.get("tools") or []
    qs: list[dict] = []

    def ask(qid: str, prompt: str, why: str, kind: str = "text",
            options: list | None = None, answer_key: str = "") -> None:
        qs.append({"id": qid, "prompt": prompt, "why": why, "kind": kind,
                   "options": options or [], "answer_key": answer_key or qid})

    needs_ids = any(t.get("id_param") for t in drafted_tools)
    if needs_ids and not (draft.get("entities") or {}).get("record_ids"):
        ask("id_example",
            "What does a customer reference number look like? (e.g. APT-1042)",
            "Tools need the reference callers state; no ID shape was found "
            "in your content.", answer_key="id_example")
    if not iv.get("offering"):
        ask("offering", "What does your business offer, in one line?",
            "Names the agent's role and persona.", answer_key="offering")
    if not iv.get("top_asks"):
        ask("top_asks",
            "What are the 3-5 things customers call about most?",
            "Seeds the intent exemplars and self-check evals.",
            answer_key="top_asks")
    if not iv.get("handoff_triggers"):
        ask("handoff_triggers",
            "When should the agent hand off to a human? (e.g. angry caller, "
            "amounts over ₹10,000)",
            "No call ends in a dead end — handoff triggers are mandatory.",
            answer_key="handoff_triggers")
    if not iv.get("greeting"):
        ask("greeting", "What should the agent say first when it picks up?",
            "The opening line is tenant data, never a platform default.",
            answer_key="greeting")
    if not iv.get("languages"):
        from voiceagent.langid import detect_language
        seen: set[str] = set()
        for c in chunks or []:
            t = str((c or {}).get("text") or "")
            if len(t) > 40:
                seen.add(detect_language(t))
        seen.discard("en")
        hint = (f" Detected in your content: {sorted(seen)}."
                if seen else "")
        ask("languages",
            "Which languages should the agent serve? (e.g. en, hi, hinglish)"
            + hint,
            "Voices, ASR routes, and reply-language guards follow this list.",
            kind="multi", answer_key="languages")
    if not iv.get("erp_url"):
        ask("erp_url",
            "Where should the agent look up orders/bookings? (system URL, "
            "or 'none' for answers-only mode)",
            "Without a backend hookup the agent answers from knowledge "
            "only — tools stay PROPOSED.", answer_key="erp_url")
    risky = [w for w in ("guarantee", "guaranteed", "100% refund",
                         "instant refund", "same day")
             if w in text]
    if risky and not iv.get("never_promise"):
        ask("never_promise",
            f"Your content promises {risky[0]!r} — should the agent repeat "
            "that, or soften it?",
            "The agent never over-promises without explicit approval.",
            answer_key="never_promise")
    return qs


def draft_bundle_surfaces(chunks: list[dict], interview: dict,
                          llm=None) -> dict:
    """Draft generatable surfaces + gap questions. llm=None resolves the
    frontier from env; inject a stub in tests. Returns
    {drafted, tools, intents, entities, policies, evals, questions}."""
    text = _chunk_text(chunks)
    iv = dict(interview or {})
    questions: list[dict] = []
    draft: dict = {"tools": [], "intents": {}, "entities": None,
                   "policies": {}, "evals": []}

    if llm is None:
        try:
            from voiceagent.llm import build_llm_from_env
            llm = build_llm_from_env()
        except Exception:
            llm = None

    if llm is not None and text:
        try:
            raw = llm.generate(
                DRAFT_PROMPT % (json.dumps(iv)[:1500], text),
                max_tokens=_MAX_TOKENS)
            parsed = _parse_draft(raw)
        except Exception:
            parsed = {}
        if parsed:
            tools, dropped = _valid_tools(parsed.get("tools"))
            draft["tools"] = tools
            for name in dropped:
                questions.append({
                    "id": f"tool_{name}", "prompt":
                    f"The draft tool {name!r} failed validation — describe "
                    "what it should do in one line?",
                    "why": "Invalid drafts never ship; your words replace "
                           "the broken proposal.", "kind": "text",
                    "options": [], "answer_key": f"tool_{name}"})
            actions = {t["action"] for t in tools}
            draft["intents"] = _valid_intents(parsed.get("intents"), actions)
            entities, ent_ok = _valid_entities(parsed.get("entities"))
            draft["entities"] = entities
            if not ent_ok:
                questions.append({
                    "id": "id_example", "prompt":
                    "The drafted ID pattern was invalid — what does a "
                    "customer reference look like? (e.g. APT-1042)",
                    "why": "Tools need the reference callers state.",
                    "kind": "text", "options": [],
                    "answer_key": "id_example"})
            pols = parsed.get("policies")
            if isinstance(pols, dict):
                draft["policies"] = {
                    str(a): {"require_approval": bool(
                        (r or {}).get("require_approval", True))}
                    for a, r in pols.items()
                    if isinstance(r, dict)} or {
                    t["action"]: {"require_approval": t["risk_class"] ==
                                  "mutating"} for t in tools}
            else:
                draft["policies"] = {
                    t["action"]: {"require_approval": t["risk_class"] ==
                                  "mutating"} for t in tools}
            evals = parsed.get("evals")
            if isinstance(evals, list):
                draft["evals"] = [
                    {"user": str(e.get("user", "")).strip()}
                    for e in evals if isinstance(e, dict)
                    and str(e.get("user", "")).strip()][:10]
            draft["questions_hint"] = [
                q for q in (parsed.get("questions") or [])
                if isinstance(q, dict) and q.get("id") and q.get("prompt")]
    drafted = bool(draft["tools"] or draft["intents"] or draft["entities"])
    questions = list(draft.pop("questions_hint", [])) + questions
    questions += gap_questions(chunks, iv, draft)
    seen: set[str] = set()
    merged: list[dict] = []
    for q in questions:
        qid = str(q.get("id") or "")
        if qid and qid not in seen:
            seen.add(qid)
            merged.append(q)
    draft["questions"] = merged[:12]
    draft["drafted"] = drafted
    return draft


def preview_bundle(deploy_id: str, chunks: list[dict], interview: dict,
                   llm=None) -> dict:
    """v1 deterministic bundle + brain-drafted surfaces + gap questions —
    the compile_preview payload the dashboard renders. Nothing is written,
    nothing approved; the brain path degrades to v1 silently."""
    bundle = compile_bundle(deploy_id, chunks, interview)
    surfaces = draft_bundle_surfaces(chunks, interview, llm=llm)
    tools = [{
        "name": t["name"], "state": "PROPOSED",
        "description": t.get("description", ""),
        "policy_action": t.get("policy_action", t["name"]),
        "params": t.get("params", []),
        "resource_type": t.get("resource_type"),
        "id_param": t.get("id_param"),
        "risk_class": t.get("risk_class", "read"),
    } for t in surfaces["tools"]] or [
        {"name": t.name, "state": t.state, "description": t.description,
         "policy_action": t.policy_action} for t in bundle.tools]
    policies = dict(bundle.policies)
    policies.update(surfaces["policies"])
    evals = ([{"name": f"draft-{i + 1:02d}", "turns": 1,
               "user": e["user"]} for i, e in enumerate(surfaces["evals"])]
             or [{"name": e.name, "turns": len(e.turns)}
                 for e in bundle.evals])
    return {
        "deploy_id": bundle.deploy_id, "spec": bundle.spec,
        "knowledge": bundle.knowledge, "tools": tools,
        "policies": policies, "evals": evals,
        "intents": surfaces["intents"], "entities": surfaces["entities"],
        "questions": surfaces["questions"], "drafted": surfaces["drafted"],
        "note": ("brain-drafted — review everything before approving"
                 if surfaces["drafted"] else
                 "template preview — no frontier configured, answer the "
                 "questions and re-preview for a drafted proposal"),
    }
