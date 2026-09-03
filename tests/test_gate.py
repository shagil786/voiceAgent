import pytest
from voiceagent.deploy.bundle import load_bundle
from voiceagent.deploy import gate

GOLDEN = "data/deployments/_example/v1"

def test_lifecycle_proposed_approved_connected():
    b = load_bundle(GOLDEN)
    b = gate.approve_tool(b, "fetch_status")
    assert gate.tool_state(b, "fetch_status") == "APPROVED"
    probe = {"auth_ok": True,
             "benign_call": {"request": "GET /status?limit=1",
                             "response": {"ok": True}},
             "api_key": "sk-live-123"}
    b = gate.record_dry_run(b, "fetch_status", probe, confirmed_by="owner")
    assert gate.tool_state(b, "fetch_status") == "CONNECTED"
    stored = gate.get_dry_run(b, "fetch_status")
    assert stored["api_key"] == "[REDACTED]"
    assert stored["auth_ok"] is True
    stored["benign_call"]["request"] = "MUTATED"
    fresh = gate.get_dry_run(b, "fetch_status")
    assert fresh["benign_call"]["request"] == "GET /status?limit=1"

def test_dry_run_requires_auth_and_confirmation():
    b = load_bundle(GOLDEN)
    b = gate.approve_tool(b, "fetch_status")
    with pytest.raises(ValueError):
        gate.record_dry_run(b, "fetch_status",
                            {"auth_ok": False, "benign_call": {}}, confirmed_by="owner")
    with pytest.raises(ValueError):
        gate.record_dry_run(b, "fetch_status",
                            {"auth_ok": True, "benign_call": {"request": "x", "response": "y"}},
                            confirmed_by="")

def test_scope_widening_resets_to_approved():
    b = load_bundle(GOLDEN)
    b = gate.approve_tool(b, "fetch_status")
    b = gate.record_dry_run(b, "fetch_status",
                            {"auth_ok": True, "benign_call": {"request": "r", "response": "s"}},
                            confirmed_by="owner")
    b2 = gate.widen_scope(b, "fetch_status", ["read", "write"])
    assert gate.tool_state(b2, "fetch_status") == "APPROVED"
    assert gate.get_dry_run(b2, "fetch_status") is None

def test_dry_run_requires_approved_first():
    b = load_bundle(GOLDEN)
    probe = {"auth_ok": True,
             "benign_call": {"request": "r", "response": "s"}}
    with pytest.raises(ValueError):
        gate.record_dry_run(b, "fetch_status", probe, confirmed_by="owner")
    b = gate.approve_tool(b, "fetch_status")
    b = gate.record_dry_run(b, "fetch_status", probe, confirmed_by="owner")
    assert gate.tool_state(b, "fetch_status") == "CONNECTED"

def test_approve_tool_unknown_name_raises():
    with pytest.raises(ValueError):
        gate.approve_tool(load_bundle(GOLDEN), "nope")

def test_runner_blocks_unconnected_tool():
    from voiceagent.tools import GovernedToolRunner, ToolGateway
    from voiceagent.policy import PolicyEngine, PolicyContext
    from voiceagent.decisionlog import DecisionLog
    log = DecisionLog()
    runner = GovernedToolRunner(ToolGateway(),
                                PolicyEngine({"order_status": {"allow": True}}),
                                decision_log=log)
    out = runner.run("order_status", PolicyContext(authenticated=True),
                     "fetch_status",
                     {"order_id": "ORD-4821"},
                     conv_id="conv-gate-1",
                     tool_states={"fetch_status": "PROPOSED"})
    assert out.executed is False
    assert out.decision_verdict == "BLOCKED_UNCONNECTED"
    assert out.result is None
    assert len(log.entries()) == 1
    assert log.entries()[-1].verdict == "BLOCKED_UNCONNECTED"
