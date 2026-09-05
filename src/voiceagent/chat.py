# src/voiceagent/chat.py
"""Single HTTP/CLI turn entry point that works for BOTH brains.

The governed `Orchestrator` is the only production brain. `run_turn` accepts any
object exposing `handle_turn(session_id, user_text, *, authenticated)` — the
Orchestrator fits that contract exactly — so the chat server, LiveKit worker, and
CLI all route through one shape. A `memory` store, when attached, still records
the exchange so the conversation-length cap and the /api/history endpoint keep
working; the brain's own blackboard/durable memory is separate from this audit
view.
"""
from __future__ import annotations

import inspect

from voiceagent.agent import extract_required_references
from voiceagent.memory import Turn, now_ts


# Deterministic end-of-conversation reply once a conversation reaches the
# turn cap: no LLM call, no action, no policy evaluation.
CAP_REPLY = ("This conversation has reached its length limit — connecting "
             "you to a human agent.")


def _handle_accepts(agent, kwarg: str) -> bool:
    """True when agent.handle_turn accepts the given kwarg (or **kwargs).

    The governed Orchestrator is the production brain and exposes handle_turn;
    legacy agents may only expose the older `handle`. In the latter case we call
    the legacy signature (no kwarg), which run_turn still supports.
    """
    if not hasattr(agent, "handle_turn"):
        return False
    try:
        sig = inspect.signature(agent.handle_turn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.name == kwarg or param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def run_turn(agent, user_text: str, authenticated: bool = False,
             amount: float | None = None, conv_id: str = "",
             memory=None, max_turns_per_conv: int = 50,
             customer_id: str | None = None) -> dict:
    """Run one customer turn through the governed brain and return the reply,
    the proposed action, and the policy decision for display.

    `amount` is a LEGACY-ONLY input: the governed Orchestrator derives policy
    amounts from tool-call params, so it is never forwarded to `handle_turn`
    (the real Orchestrator rejects it). With a `memory` store attached, the
    user/agent turns are recorded so the cap and /api/history stay live.
    """
    if memory is not None and len(memory.history(conv_id)) >= max_turns_per_conv:
        return {"reply": CAP_REPLY, "action": None,
                "decision": "ESCALATE",
                "reasons": ["conversation length cap reached"],
                "executed": False, "tool_result": None,
                "directive": None}

    governed = hasattr(agent, "handle_turn")

    if memory is not None:
        memory.append(conv_id, Turn(ts=now_ts(), role="user", text=user_text,
                                    refs=extract_required_references(user_text)))
    # Legacy replay seam: with a memory store attached, the legacy Agent
    # receives the recent transcript INCLUDING the just-appended user turn;
    # the governed Orchestrator owns its own per-session blackboard history,
    # so it is never handed one here.
    history = (memory.history(conv_id, last_n=8)
               if memory is not None and not governed else None)

    if governed:
        kwargs: dict = {"authenticated": authenticated}
        if _handle_accepts(agent, "contact_alias") and customer_id:
            kwargs["contact_alias"] = customer_id
        res = agent.handle_turn(conv_id, user_text, **kwargs)
    else:
        # Legacy Agent: `handle(user_text, *, authenticated, amount, conv_id,
        # history)`.
        kwargs = {"authenticated": authenticated, "conv_id": conv_id}
        if amount is not None:
            kwargs["amount"] = amount
        if history is not None:
            kwargs["history"] = history
        res = agent.handle(user_text, **kwargs)

    if memory is not None:
        if governed:
            primary = res.actions[0] if res.actions else None
            memory.append(conv_id, Turn(
                ts=now_ts(), role="agent", text=res.reply,
                action=primary.get("action") if primary else None,
                verdict=primary.get("verdict") if primary else None))
        else:
            memory.append(conv_id, Turn(
                ts=now_ts(), role="agent", text=res.text,
                action=res.action,
                verdict=res.decision.verdict if res.decision else None))

    # The governed Orchestrator returns a TurnResult with `.actions`; a legacy
    # Agent returns an AgentResult with `.action`/`.decision` directly.
    if governed:
        primary = res.actions[0] if res.actions else None
        return {
            "reply": res.reply,
            "action": (primary["tool"] if primary else None),
            "decision": (primary["verdict"] if primary else "none"),
            "reasons": primary.get("reasons", []) if primary else [],
            "executed": bool(primary and primary.get("ok")),
            "tool_result": (primary.get("value") if primary else None),
            "directive": None,
        }
    return {
        "reply": res.text,
        "action": res.action,
        "decision": res.decision.verdict if res.decision else None,
        "reasons": res.decision.reasons if res.decision else [],
        "executed": False,
        "tool_result": None,
        "directive": None,
    }
