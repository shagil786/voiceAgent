# tests/test_billing.py
from voiceagent.billing import compute_billing
from voiceagent.evaluator import EvalRow
from voiceagent.decisionlog import DecisionLog, DecisionEntry

def _row(conv_id, resolved):
    return EvalRow(conv_id=conv_id, resolved=resolved, grounded=True,
                   wrong_action=False, hallucinated_facts=[], latency_s=0.4)

def test_billing_counts_non_escalated_resolved():
    log = DecisionLog()
    log.record(DecisionEntry(ts="t", conv_id="c1", action="refund",
                             verdict="ALLOW", reasons=[]))
    log.record(DecisionEntry(ts="t", conv_id="c2", action="fraud",
                             verdict="ESCALATE", reasons=[]))
    rows = [_row("c1", True), _row("c2", True), _row("c3", False)]
    b = compute_billing(rows, log)
    assert b["total"] == 3
    assert b["resolved"] == 2
    assert b["escalated"] == 1
    assert b["billable"] == 1   # c1 only (c2 escalated = free, c3 unresolved = free)
    assert b["revenue_rs"] == 8.0

def test_billing_custom_price():
    log = DecisionLog()
    log.record(DecisionEntry(ts="t", conv_id="c1", action="refund",
                             verdict="ALLOW", reasons=[]))
    rows = [_row("c1", True)]
    b = compute_billing(rows, log, price_per_resolved_rs=12.0)
    assert b["revenue_rs"] == 12.0
