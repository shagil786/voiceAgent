# tests/test_evaluator.py
from voiceagent.evaluator import score_conversation, aggregate
from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult

def _conv(action="refund", facts=("ORD-1",), escalate=False):
    return Conversation(id="c1", language="en", intent="refund",
                        user_text="refund ORD-1", expected_action=action,
                        key_facts=list(facts), escalate=escalate)

def _res(text="refund for ORD-1 done", action="refund",
         retrieved=("Refund processed for ORD-1",)):
    return AgentResult(text=text, action=action,
                       retrieved=[{"text": t} for t in retrieved],
                       latency_s=0.5)

def test_resolution_requires_action_and_facts():
    good = score_conversation(_conv(), _res())
    assert good.resolved and good.grounded
    bad_action = score_conversation(_conv(), _res(action="cancel_order"))
    assert not bad_action.resolved
    missing_fact = score_conversation(
        _conv(facts=("ORD-1", "ORD-2")), _res(text="refund for ORD-1 done"))
    assert not missing_fact.resolved

def test_hallucination_flags_facts_missing_from_retrieval():
    # Fact ORD-9 is NOT in user_text, so if the agent asserts it and it's
    # absent from retrieval, it's a genuine hallucination.
    conv = Conversation(id="c3", language="en", intent="refund",
                        user_text="please refund my order",
                        expected_action="refund",
                        key_facts=["ORD-9"], escalate=False)
    res = AgentResult(text="refund for ORD-9 done", action="refund",
                      retrieved=[{"text": "nothing here"}],
                      latency_s=0.5)
    row = score_conversation(conv, res)
    assert not row.grounded
    assert len(row.hallucinated_facts) >= 1
    assert "ORD-9" in row.hallucinated_facts

def test_aggregate_computes_rates():
    rows = [
        score_conversation(_conv(), _res()),
        score_conversation(_conv(), _res(action="cancel_order")),
    ]
    s = aggregate(rows)
    assert s.resolution_rate == 0.5
    assert s.grounded_rate == 1.0
    assert s.avg_latency_s == 0.5
    assert s.n == 2

def test_echoed_customer_fact_is_not_hallucination():
    # Ruling: key_fact in agent text that the customer stated is correct,
    # not a hallucination — even if missing from retrieval.
    conv = Conversation(id="c2", language="en", intent="order_status",
                        user_text="where is my order ORD-1",
                        expected_action="order_status",
                        key_facts=["ORD-1"], escalate=False)
    res = AgentResult(text="Your order ORD-1 is shipped.", action="order_status",
                      retrieved=[{"text": "Orders are shipped within 3 days."}],
                      latency_s=0.3)
    row = score_conversation(conv, res)
    assert row.grounded
    assert row.hallucinated_facts == []
