# tests/test_tools.py — Sprint A WS2: Tool Gateway, MockERP, GovernedToolRunner.
import pytest

from voiceagent.tools import (GovernedToolRunner, MockERP, ToolGateway,
                              ToolResult)
from voiceagent.policy import PolicyEngine, PolicyContext


def test_mockerp_seed_data():
    erp = MockERP()
    o = erp.get_order("ORD-4821")
    assert o["status"] == "CONFIRMED" and o["amount"] == 1299.0
    assert erp.orders_for_customer("CUST-001") == ["ORD-4821", "ORD-7734"]

def test_fetch_order_status():
    gw = ToolGateway()
    r = gw.execute("fetch_order_status", {"order_id": "ORD-7734"})
    assert r.ok and r.value["status"] == "SHIPPED"
    assert r.value["tracking_url"] == "https://track.fake/7734"

def test_precondition_blocks_cancelling_shipped_order():
    gw = ToolGateway()
    r = gw.execute("cancel_order", {"order_id": "ORD-7734", "reason": "late"})
    assert not r.ok
    assert "precondition_failed" in r.error and "SHIPPED" in r.error
    # the order was NOT mutated
    assert gw.erp.get_order("ORD-7734")["status"] == "SHIPPED"

def test_cancel_confirmed_order_succeeds():
    gw = ToolGateway()
    r = gw.execute("cancel_order", {"order_id": "ORD-4821", "reason": "changed my mind"})
    assert r.ok and r.value["status"] == "CANCELLED"

def test_idempotency_key_deduplicates():
    gw = ToolGateway()
    r1 = gw.execute("initiate_refund",
                    {"order_id": "ORD-4821", "amount": 1299.0, "reason": "damaged"},
                    idempotency_key="refund-4821-1")
    r2 = gw.execute("initiate_refund",
                    {"order_id": "ORD-4821", "amount": 1299.0, "reason": "damaged"},
                    idempotency_key="refund-4821-1")
    assert r1.ok and r2.ok and r2.idempotent_replay
    assert len(gw.erp.refunds) == 1  # no double refund

def test_different_keys_execute_twice():
    gw = ToolGateway()
    for i, key in enumerate(("k1", "k2"), 1):
        gw.execute("initiate_refund",
                   {"order_id": "ORD-4821", "amount": 100.0, "reason": "x"},
                   idempotency_key=key)
    assert len(gw.erp.refunds) == 2

def test_missing_params_rejected():
    gw = ToolGateway()
    r = gw.execute("cancel_order", {"order_id": "ORD-4821"})
    assert not r.ok and "missing_params" in r.error

def test_unknown_tool_and_unknown_order():
    gw = ToolGateway()
    assert not gw.execute("nope", {}).ok
    assert not gw.execute("cancel_order",
                          {"order_id": "ORD-0000", "reason": "x"}).ok

def test_graceful_timeout_on_backend_failure():
    erp = MockERP()
    gw = ToolGateway(erp=erp)
    erp.fail_next = True
    r = gw.execute("cancel_order", {"order_id": "ORD-4821", "reason": "x"})
    assert not r.ok and "backend_timeout" in r.error
    assert gw.erp.get_order("ORD-4821")["status"] == "CONFIRMED"  # no mutation

def test_yaml_spec_override():
    p = pytest.Path = None  # placeholder to keep flake quiet
    import tempfile, yaml
    from pathlib import Path as P
    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "tools.yaml"
        f.write_text(yaml.safe_dump({"tools": {
            "cancel_order": {"params": ["order_id", "reason"],
                             "preconditions": [{"field": "status", "op": "not_in",
                                                "value": ["SHIPPED", "DELIVERED", "CONFIRMED"]}]}}}))
        gw = ToolGateway.from_yaml(f)
        r = gw.execute("cancel_order", {"order_id": "ORD-4821", "reason": "x"})
        assert not r.ok and "precondition_failed" in r.error  # CONFIRMED now blocked too

def test_yaml_unknown_tool_rejected():
    import tempfile, yaml
    from pathlib import Path as P
    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "tools.yaml"
        f.write_text(yaml.safe_dump({"tools": {"nuke_db": {"params": []}}}))
        with pytest.raises(ValueError):
            ToolGateway.from_yaml(f)

def test_governed_runner_blocks_on_deny_and_logs():
    from voiceagent.decisionlog import DecisionLog
    log = DecisionLog()
    runner = GovernedToolRunner(ToolGateway(),
                                PolicyEngine({"refund": {"require_auth": True}}),
                                decision_log=log)
    out = runner.run("refund", PolicyContext(authenticated=False),
                     "initiate_refund",
                     {"order_id": "ORD-4821", "amount": 100.0, "reason": "x"})
    assert out.decision_verdict == "REQUIRE_AUTH"
    assert not out.executed and out.result is None
    assert "blocked by policy" in " ".join(out.reasons)
    assert len(log.entries()) == 1  # blocked attempts are audited too

def test_governed_runner_executes_on_allow():
    from voiceagent.decisionlog import DecisionLog
    log = DecisionLog()
    erp = MockERP()
    runner = GovernedToolRunner(ToolGateway(erp=erp),
                                PolicyEngine({"cancel_order": {"allow": True}}),
                                decision_log=log)
    out = runner.run("cancel_order", PolicyContext(authenticated=True),
                     "cancel_order",
                     {"order_id": "ORD-4821", "reason": "changed my mind"},
                     idempotency_key="c-1", conv_id="conv-9")
    assert out.executed and out.result.value["status"] == "CANCELLED"
    assert any("executed" in r for r in out.reasons)
    entry = log.entries()[-1]
    assert entry.verdict == "ALLOW" and entry.action == "cancel_order"

def test_governed_runner_high_value_refund_requires_human():
    runner = GovernedToolRunner(ToolGateway(), PolicyEngine({}))
    out = runner.run("high_value_refund", PolicyContext(authenticated=True),
                     "initiate_refund",
                     {"order_id": "ORD-7734", "amount": 6500.0, "reason": "x"})
    assert out.decision_verdict == "ESCALATE"
    assert not out.executed


def test_governed_runner_blocks_unknown_tool_when_states_passed():
    from voiceagent.decisionlog import DecisionLog
    log = DecisionLog()
    erp = MockERP()
    runner = GovernedToolRunner(ToolGateway(erp=erp),
                                PolicyEngine({"cancel_order": {"allow": True}}),
                                decision_log=log)
    out = runner.run("cancel_order", PolicyContext(authenticated=True),
                     "cancel_order",
                     {"order_id": "ORD-4821", "reason": "x"},
                     conv_id="conv-unknown-1",
                     tool_states={"other": "CONNECTED"})
    assert out.decision_verdict == "BLOCKED_UNCONNECTED"
    assert not out.executed and out.result is None
    assert len(log.entries()) == 1
    assert log.entries()[-1].verdict == "BLOCKED_UNCONNECTED"
    assert erp.get_order("ORD-4821")["status"] == "CONFIRMED"
