# src/voiceagent/chat.py
from __future__ import annotations

from voiceagent.agent import extract_required_references
from voiceagent.memory import Turn, now_ts

# Deterministic end-of-conversation reply once a conversation reaches the
# turn cap: no LLM call, no action, no policy evaluation.
CAP_REPLY = ("This conversation has reached its length limit — connecting "
             "you to a human agent.")


def run_turn(agent, user_text: str, authenticated: bool = False,
             amount: float | None = None, conv_id: str = "",
             memory=None, max_turns_per_conv: int = 50) -> dict:
    """Single entry point for CLI and HTTP: run one customer turn and return
    the reply plus the policy decision for display.

    With a ConversationMemory attached, the user turn is recorded first, the
    last 8 recorded turns are replayed to the agent, and the agent's reply is
    recorded after. At the cap, a deterministic human-handoff reply is
    returned without calling the agent or appending further turns. With
    memory=None the pre-M4a code path is unchanged (the history kwarg is not
    passed at all, so legacy agent duck types keep working)."""
    if memory is not None:
        if len(memory.history(conv_id)) >= max_turns_per_conv:
            return {"reply": CAP_REPLY, "action": None,
                    "decision": "ESCALATE",
                    "reasons": ["conversation length cap reached"]}
        memory.append(conv_id, Turn(ts=now_ts(), role="user",
                                    text=user_text,
                                    refs=extract_required_references(user_text)))
        res = agent.handle(user_text, authenticated=authenticated,
                           amount=amount, conv_id=conv_id,
                           history=memory.history(conv_id, last_n=8))
        memory.append(conv_id, Turn(ts=now_ts(), role="agent", text=res.text,
                                    action=res.action,
                                    verdict=res.decision.verdict
                                    if res.decision else None))
    else:
        res = agent.handle(user_text, authenticated=authenticated,
                           amount=amount, conv_id=conv_id)
    return {
        "reply": res.text,
        "action": res.action,
        "decision": res.decision.verdict if res.decision else None,
        "reasons": res.decision.reasons if res.decision else [],
    }
