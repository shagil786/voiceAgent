# tests/test_handoff.py
from voiceagent.handoff import build_handoff, handoff_markdown
from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult
from voiceagent.policy import Decision
from voiceagent.entities import Entities

def test_build_handoff_serializes_all_fields():
    conv = Conversation(id="c1", language="en", intent="refund",
                        user_text="refund my order ORD-5", expected_action="refund",
                        key_facts=["ORD-5"], authenticated=True)
    res = AgentResult(text="Your refund for ORD-5 is being processed.",
                      action="refund",
                      retrieved=[{"id": "a", "text": "Refunds take 5-7 days.",
                                  "section": "Refunds", "score": 0.9}],
                      latency_s=0.4,
                      decision=Decision("ALLOW", ["refund allowed by policy"]))
    ents = Entities(amount=1000.0, order_id="ORD-5")
    h = build_handoff(conv, res, ents)
    assert h.conv_id == "c1"
    assert h.action == "refund"
    assert h.decision == "ALLOW"
    assert h.decision_reasons == ["refund allowed by policy"]
    assert h.amount == 1000.0
    assert h.order_id == "ORD-5"
    assert h.authenticated is True
    assert len(h.retrieved) == 1

def test_handoff_markdown_contains_key_fields():
    conv = Conversation(id="c1", language="en", intent="refund",
                        user_text="refund my order ORD-5", expected_action="refund",
                        key_facts=["ORD-5"], authenticated=True)
    res = AgentResult(text="Your refund for ORD-5 is being processed.",
                      action="refund", retrieved=[], latency_s=0.4,
                      decision=Decision("ESCALATE", ["above threshold"]))
    md = handoff_markdown(build_handoff(conv, res, Entities(amount=25000.0)))
    assert "c1" in md
    assert "ESCALATE" in md
    assert "Your refund for ORD-5 is being processed." in md
    assert "above threshold" in md

def test_handoff_amount_symbol_follows_currency():
    conv = Conversation(id="c2", language="en", intent="refund",
                        user_text="refund ORD-7", expected_action="refund",
                        key_facts=["ORD-7"])
    res = AgentResult(text="ok", action="refund", retrieved=[], latency_s=0.1)
    h = build_handoff(conv, res, Entities(amount=25000.0))
    # Platform default symbol is "$"; a tenant caller passes its currency.
    assert "- **Amount:** $25,000" in handoff_markdown(h)
    assert "- **Amount:** ₹25,000" in handoff_markdown(h, currency="₹")
