"""Deterministic compiler: sources + interview -> Bundle. No LLM calls in v1;
slot-filling + templates only (brain-assisted extraction is a later upgrade)."""
from __future__ import annotations

import re
from voiceagent.deploy.bundle import (
    Bundle, EvalCheck, ToolEntry, SCHEMA_VERSION)

_STOP = {"price", "cost", "book", "slot", "visit", "loan", "plan",
        "possession", "refund", "status", "cancel", "timing", "contact"}


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "ask"


def compile_bundle(deploy_id: str, chunks: list[dict], interview: dict) -> Bundle:
    offering = interview.get("offering", "")
    top_asks = list(interview.get("top_asks", []))[:5]
    never = list(interview.get("never_promise", []))
    handoffs = list(interview.get("handoff_triggers", []))
    knowledge = [c for c in chunks if c.get("text")]
    tools = [ToolEntry(name="escalate_to_human",
                       description="Hand off to a human with full context",
                       parameters={"type": "object", "properties": {}},
                       state="PROPOSED", policy_action="escalate_to_human",
                       scopes=[])]
    for ask in top_asks:
        name = _slug(ask)
        if name in {t.name for t in tools} or name in _STOP and len(name) < 4:
            name = f"ask_{name}"
        tools.append(ToolEntry(
            name=name, description=f"Handle: {ask}",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string"}}},
            state="PROPOSED", policy_action=name, scopes=["read"]))
    policies: dict = {"escalate_to_human": {"allow": True}}
    for t in tools[1:]:
        policies[t.policy_action] = {"require_approval": True}
    spec = {"role": offering[:200], "tone": "concise, no invented facts",
            "patterns": ["answer", "qualify", "follow_up", "draft_action"],
            "never_promise": never, "handoff_triggers": handoffs,
            "disclosures": ["I am an AI assistant; a human signs off actions."]}
    evals = [EvalCheck(name=f"selfcheck-{i+1:02d}",
                       turns=[{"user": top_asks[i % len(top_asks)]}] if top_asks else [{"user": "Hello"}],
                       assert_={"contains": top_asks[i % len(top_asks)][:12]} if top_asks else {"contains": "Hello"})
             for i in range(10)]
    return Bundle(schema_version=SCHEMA_VERSION, deploy_id=deploy_id,
                  spec=spec, tools=tools, policies=policies,
                  knowledge=knowledge, evals=evals)
