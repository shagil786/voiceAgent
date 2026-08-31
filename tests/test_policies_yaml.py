# tests/test_policies_yaml.py
from voiceagent.policy import PolicyEngine, PolicyContext

def test_real_policy_file_semantics():
    from voiceagent.policy import load_policies
    policies = load_policies("data/policies/policies.yaml")
    eng = PolicyEngine(policies)
    assert eng.evaluate("fraud").verdict == "ESCALATE"
    assert eng.evaluate("high_value_refund", PolicyContext(authenticated=True)).verdict == "ESCALATE"
    assert eng.evaluate("refund", PolicyContext(amount=1000, authenticated=True)).verdict == "ALLOW"
    assert eng.evaluate("refund", PolicyContext(amount=20000, authenticated=True)).verdict == "REQUIRE_HUMAN_APPROVAL"
    assert eng.evaluate("refund", PolicyContext(amount=1000, authenticated=False)).verdict == "REQUIRE_AUTH"
    assert eng.evaluate("order_status").verdict == "ALLOW"
    assert eng.evaluate("not_a_real_action").verdict == "DENY"
