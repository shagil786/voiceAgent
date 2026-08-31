# src/voiceagent/evaluator.py
from __future__ import annotations

from dataclasses import dataclass, field
from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult


@dataclass
class EvalRow:
    conv_id: str
    resolved: bool
    grounded: bool
    wrong_action: bool
    hallucinated_facts: list[str]
    latency_s: float


@dataclass
class EvalSummary:
    resolution_rate: float
    grounded_rate: float
    wrong_action_rate: float
    hallucination_rate: float
    avg_latency_s: float
    n: int

    def as_dict(self) -> dict:
        return {
            "resolution_rate": self.resolution_rate,
            "grounded_rate": self.grounded_rate,
            "wrong_action_rate": self.wrong_action_rate,
            "hallucination_rate": self.hallucination_rate,
            "avg_latency_s": self.avg_latency_s,
            "n": self.n,
        }


def score_conversation(conv: Conversation, res: AgentResult) -> EvalRow:
    # Escalation rows (fraud, high_value_refund, etc.) resolve correctly when
    # the policy engine ESCALATEs to a human — the right real-world outcome,
    # not a failure.
    if conv.escalate and getattr(res, "decision", None) is not None:
        resolved = res.decision.verdict == "ESCALATE"
        return EvalRow(conv_id=conv.id, resolved=resolved, grounded=True,
                       wrong_action=False, hallucinated_facts=[],
                       latency_s=res.latency_s)

    action_ok = res.action == conv.expected_action
    facts_ok = all(f in res.text for f in conv.key_facts)
    resolved = action_ok and facts_ok

    retrieved_text = "\n".join(r["text"] for r in res.retrieved)
    # A key_fact the customer stated in their own query is not a
    # hallucination when echoed back, even if absent from retrieval.
    hallucinated = [f for f in conv.key_facts if f in res.text
                    and f not in retrieved_text
                    and f not in conv.user_text]
    grounded = len(hallucinated) == 0

    return EvalRow(
        conv_id=conv.id,
        resolved=resolved,
        grounded=grounded,
        wrong_action=bool(res.action) and res.action != conv.expected_action,
        hallucinated_facts=hallucinated,
        latency_s=res.latency_s,
    )


def aggregate(rows: list[EvalRow]) -> EvalSummary:
    n = len(rows)
    if n == 0:
        return EvalSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    def rate(pred):
        return sum(1 for r in rows if pred(r)) / n
    return EvalSummary(
        resolution_rate=rate(lambda r: r.resolved),
        grounded_rate=rate(lambda r: r.grounded),
        wrong_action_rate=rate(lambda r: r.wrong_action),
        hallucination_rate=rate(lambda r: len(r.hallucinated_facts) > 0),
        avg_latency_s=sum(r.latency_s for r in rows) / n,
        n=n,
    )
