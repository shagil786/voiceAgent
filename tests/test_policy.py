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


# ---------------------------------------------------------------------------
# M6a: data-driven conditional escalation — escalate_when in YAML policies
# against PolicyContext.signals. Sentiment/state routing is DATA, not code.
# ---------------------------------------------------------------------------

def test_escalate_when_signal_matches():
    eng = PolicyEngine({"complaint": {"allow": True,
                                      "escalate_when": {"frustrated": True}}})
    ctx = PolicyContext(signals={"frustrated": True, "frustration_level": "high"})
    d = eng.evaluate("complaint", ctx)
    assert d.verdict == "ESCALATE"
    assert "frustrated" in str(d.reasons)

def test_escalate_when_signal_absent_allows():
    eng = PolicyEngine({"complaint": {"allow": True,
                                      "escalate_when": {"frustrated": True}}})
    d = eng.evaluate("complaint", PolicyContext(signals={"frustrated": False}))
    assert d.verdict == "ALLOW"

def test_escalate_when_no_signals_context_allows():
    eng = PolicyEngine({"complaint": {"allow": True,
                                      "escalate_when": {"frustrated": True}}})
    assert eng.evaluate("complaint", PolicyContext()).verdict == "ALLOW"

def test_escalate_when_precedes_require_auth():
    # A frustrated customer should reach a human WITHOUT the bot first
    # demanding authentication.
    eng = PolicyEngine({"complaint": {"require_auth": True,
                                      "escalate_when": {"frustrated": True}}})
    d = eng.evaluate("complaint", PolicyContext(signals={"frustrated": True}))
    assert d.verdict == "ESCALATE"


# ---------------------------------------------------------------------------
# Currency is tenant data: the REQUIRE_HUMAN_APPROVAL reason is fed back to
# the brain and shown to customers, so it must use the tenant's symbol, not a
# hardcoded one. The platform default is "$" (USA/UK-first market).
# ---------------------------------------------------------------------------

def test_platform_default_currency_is_dollar():
    eng = PolicyEngine({"refund": {"max_without_approval": 5000}})
    d = eng.evaluate("refund", PolicyContext(amount=20000))
    assert d.verdict == "REQUIRE_HUMAN_APPROVAL"
    assert any("$20,000" in r and "$5,000" in r for r in d.reasons)
    assert not any("₹" in r for r in d.reasons)

def test_engine_currency_appears_in_amount_reason():
    eng = PolicyEngine({"refund": {"max_without_approval": 5000}},
                       currency="₹")
    d = eng.evaluate("refund", PolicyContext(amount=20000))
    assert any("₹20,000" in r and "₹5,000" in r for r in d.reasons)
    assert not any("$" in r for r in d.reasons)
