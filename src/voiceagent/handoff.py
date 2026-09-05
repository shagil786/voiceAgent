# src/voiceagent/handoff.py
from __future__ import annotations

from dataclasses import dataclass, field
from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult
from voiceagent.entities import Entities
from voiceagent.tenant import DEFAULT_CURRENCY


@dataclass
class HandoffBundle:
    conv_id: str
    user_text: str
    reply: str
    action: str | None
    decision: str | None
    decision_reasons: list[str]
    retrieved: list[dict]
    amount: float | None = None
    order_id: str | None = None
    authenticated: bool = False


def build_handoff(conv: Conversation, res: AgentResult,
                  entities: Entities | None = None) -> HandoffBundle:
    ents = entities or Entities()
    return HandoffBundle(
        conv_id=conv.id,
        user_text=conv.user_text,
        reply=res.text,
        action=res.action,
        decision=res.decision.verdict if res.decision else None,
        decision_reasons=res.decision.reasons if res.decision else [],
        retrieved=res.retrieved,
        amount=ents.amount,
        order_id=ents.order_id,
        authenticated=conv.authenticated,
    )


def handoff_markdown(h: HandoffBundle, currency: str = DEFAULT_CURRENCY) -> str:
    # The amount symbol is the deployment's currency (tenant data where a
    # tenant is available; the platform default otherwise).
    lines = [
        f"# Handoff — {h.conv_id}",
        f"- **Action:** {h.action or 'none'}",
        f"- **Policy decision:** {h.decision or 'n/a'}",
        f"- **Authenticated:** {h.authenticated}",
        f"- **Amount:** {currency}{h.amount:,.0f}" if h.amount else "- **Amount:** n/a",
        f"- **Order:** {h.order_id or 'n/a'}",
        "",
        "## Customer said",
        h.user_text,
        "",
        "## Agent replied",
        h.reply,
        "",
        "## Why",
    ]
    lines += [f"- {r}" for r in h.decision_reasons] or ["- (no decision recorded)"]
    lines += ["", "## Retrieved context"]
    lines += [f"- [{r.get('section', '')}] {r.get('text', '')}" for r in h.retrieved]
    return "\n".join(lines)