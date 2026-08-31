# src/voiceagent/chat.py
from __future__ import annotations


def run_turn(agent, user_text: str, authenticated: bool = False,
             amount: float | None = None, conv_id: str = "") -> dict:
    """Single entry point for CLI and HTTP: run one customer turn and return
    the reply plus the policy decision for display."""
    res = agent.handle(user_text, authenticated=authenticated,
                       amount=amount, conv_id=conv_id)
    return {
        "reply": res.text,
        "action": res.action,
        "decision": res.decision.verdict if res.decision else None,
        "reasons": res.decision.reasons if res.decision else [],
    }
