# tests/test_policy.py
from voiceagent.policy import PolicyEngine, PolicyContext, Decision

def _engine():
    policies = {
        "escalate": ["fraud", "legal", "chargeback"],
        "refund": {"require_auth": True, "max_without_approval": 5000},
        "order_status": {"allow": True},
    }
    return PolicyEngine(policies)

def test_unknown_action_denied():
    d = _engine().evaluate("delete_account")
    assert d.verdict == "DENY"
    assert any("no policy" in r.lower() for r in d.reasons)

def test_order_status_allowed():
    d = _engine().evaluate("order_status")
    assert d.verdict == "ALLOW"

def test_refund_requires_auth():
    d = _engine().evaluate("refund", PolicyContext(amount=1000, authenticated=False))
    assert d.verdict == "REQUIRE_AUTH"

def test_refund_high_value_requires_human():
    d = _engine().evaluate("refund", PolicyContext(amount=20000, authenticated=True))
    assert d.verdict == "REQUIRE_HUMAN_APPROVAL"

def test_refund_within_limit_allowed_when_authed():
    d = _engine().evaluate("refund", PolicyContext(amount=1000, authenticated=True))
    assert d.verdict == "ALLOW"

def test_escalate_intents_escalate():
    for action in ["fraud", "legal", "chargeback"]:
        d = _engine().evaluate(action)
        assert d.verdict == "ESCALATE"
