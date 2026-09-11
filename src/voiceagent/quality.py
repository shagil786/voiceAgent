# src/voiceagent/quality.py — LLM-judge quality scoring for governed conversations.
"""Quality scoring for the console dashboard (ADR-005 sibling).

WHAT IS SCORED
--------------
The platform's authoritative record of a conversation is its DECISION SPINE:
the governed actions the agent proposed, the verdicts the spine returned,
and the reasons it gave. quality.py scores exactly that record — never a
fabricated transcript. When full turn transcripts become a stored artifact,
the same rubric extends to them.

THE JUDGE
---------
An LLM-judge (any OpenAI-compatible model, the same OpenAICompatLLM the
brain uses) receives the decision spine and returns STRICT JSON:
    {"tool_choice": 0-10, "verdict_quality": 0-10, "reasoning": "...",
     "escalation_judgment": 0-10, "overall": 0-10}
Fallback: when no LLM is configured, a deterministic HEURISTIC scores the
spine from hard signals (precondition failures, denied verdicts, missing
reasons) — labeled "heuristic" so the console never shows a model opinion
as if it were measured.

Honesty contract: scores always carry their source ("llm-judge" or
"heuristic"); a judge that returns unparseable output falls back to the
heuristic and says so. No invented numbers.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping

JUDGE_SYSTEM = """You are a strict QA reviewer for a voice-agent platform.
You receive the DECISION SPINE of one customer conversation: the sequence of
governed tool calls, their verdicts (ALLOW / DENY / ESCALATE /
REQUIRE_AUTH), and the platform's stated reasons.

Score the QUALITY OF THE GOVERNED DECISIONS on 0-10 each:
- tool_choice: were the proposed tools appropriate for a support flow?
- verdict_quality: are the verdicts consistent with the stated reasons?
- escalation_judgment: is escalating (or not) the right call here?

Return ONLY strict JSON, no prose:
{"tool_choice": <int>, "verdict_quality": <int>, "escalation_judgment": <int>, "overall": <int>, "reasoning": "<one sentence>"}"""

_HEURISTIC_NEGATIVE = {"DENY", "REQUIRE_AUTH"}


def _spine_text(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        reasons = r.get("reasons") or []
        reason = "; ".join(reasons) if isinstance(reasons, list) else str(reasons)
        lines.append(f"- {r.get('action')} -> {r.get('verdict')} ({reason})")
    return "\n".join(lines)


def heuristic_score(rows: list[dict]) -> dict:
    """Deterministic fallback when no LLM judge is configured. Transparent
    rules over the spine — never presented as a model opinion."""
    if not rows:
        return {"source": "heuristic", "overall": None,
                "tool_choice": None, "verdict_quality": None,
                "escalation_judgment": None,
                "reasoning": "no decisions recorded"}
    n = len(rows)
    reasons_ok = sum(1 for r in rows if r.get("reasons"))
    escalations = sum(1 for r in rows if str(r.get("verdict", "")).upper() == "ESCALATE")
    blocked = sum(1 for r in rows if str(r.get("verdict", "")).upper() in _HEURISTIC_NEGATIVE)
    # transparent rubric: reasons present (+), balanced escalation (not 0, not all)
    tool_choice = 8 if reasons_ok == n else 5
    verdict_quality = 7 if reasons_ok >= max(1, n - 1) else 4
    esc_judg = 8 if 0 < escalations < n else (5 if n == 1 else 6)
    overall = round((tool_choice + verdict_quality + esc_judg) / 3, 1)
    return {
        "source": "heuristic", "overall": overall,
        "tool_choice": tool_choice, "verdict_quality": verdict_quality,
        "escalation_judgment": esc_judg,
        "reasoning": (f"{n} decisions; {reasons_ok}/{n} with stated reasons; "
                      f"{escalations} escalation(s), {blocked} blocked"),
    }


def _parse_scores(text: str) -> dict | None:
    """Extract the strict-JSON score object from judge output. Returns None
    when the model didn't follow the contract (caller falls back)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    keys = ("tool_choice", "verdict_quality", "escalation_judgment", "overall")
    if not all(k in j for k in keys):
        return None
    out = {}
    for k in keys:
        try:
            v = float(j[k])
        except (TypeError, ValueError):
            return None
        out[k] = max(0.0, min(10.0, v))
    out["reasoning"] = str(j.get("reasoning", ""))[:300]
    return out


def judge_conversation(rows: list[dict], llm: Any) -> dict:
    """LLM-judge over the decision spine. Falls back to heuristic_score when
    the judge is unavailable or returns unparseable output — and labels the
    source either way so the console never misrepresents a score."""
    if llm is None:
        out = heuristic_score(rows)
        return out
    try:
        prompt = _spine_text(rows)
        llm.chat_template(JUDGE_SYSTEM, "Decision spine:", prompt)
        raw = llm.generate(prompt, max_tokens=220)
        scores = _parse_scores(raw)
    except Exception:  # noqa: BLE001 - any judge failure degrades to heuristic
        return heuristic_score(rows)
    if scores is None:
        out = heuristic_score(rows)
        out["reasoning"] = "judge output unparseable — heuristic fallback. " + out["reasoning"]
        return out
    overall = round((scores["tool_choice"] + scores["verdict_quality"]
                     + scores["escalation_judgment"]) / 3, 1)
    axes = {k: scores[k] for k in ("tool_choice", "verdict_quality", "escalation_judgment")}
    return {"source": "llm-judge", "overall": overall, **axes,
            "reasoning": scores.get("reasoning", "")}


def build_judge_from_env(env: Mapping[str, str] | None = None) -> Any | None:
    """Build the judge LLM from the standard frontier env (same brain the
    agent uses). Returns None when the frontier isn't configured — callers
    then fall back to the heuristic."""
    import os
    e = os.environ if env is None else env
    base = (e.get("VOICEAGENT_FRONTIER_URL") or "").strip()
    if not base:
        return None
    from voiceagent.llm import OpenAICompatLLM
    return OpenAICompatLLM(
        base, e.get("VOICEAGENT_FRONTIER_MODEL", "gpt-4o-mini"),
        api_key=e.get("VOICEAGENT_FRONTIER_KEY") or None, timeout=25.0)
