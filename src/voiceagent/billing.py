# src/voiceagent/billing.py
from __future__ import annotations

from voiceagent.evaluator import EvalRow
from voiceagent.decisionlog import DecisionLog

PRICE_PER_RESOLVED_RS = 8.0
ESCALATED = "ESCALATE"


def compute_billing(rows: list[EvalRow], decision_log: DecisionLog,
                    price_per_resolved_rs: float = PRICE_PER_RESOLVED_RS) -> dict:
    """Per-resolved-conversation pricing. Billable = resolved AND not escalated
    (escalated conversations are handled by a human, so they're free per the
    spec's pricing rule; unresolved are also free)."""
    resolved_ids = {r.conv_id for r in rows if r.resolved}
    resolved = len(resolved_ids)
    escalated = 0
    billable_ids: set[str] = set()
    for entry in decision_log.entries():
        if entry.verdict == ESCALATED:
            escalated += 1
        elif entry.conv_id in resolved_ids:
            billable_ids.add(entry.conv_id)
    billable = len(billable_ids)
    return {
        "total": len(rows),
        "resolved": resolved,
        "escalated": escalated,
        "billable": billable,
        "revenue_rs": round(billable * price_per_resolved_rs, 2),
        "price_per_resolved_rs": price_per_resolved_rs,
    }