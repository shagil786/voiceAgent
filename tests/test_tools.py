# tests/test_tools.py — Sprint A WS2: Tool Gateway, MockERP, GovernedToolRunner.
import pytest

from voiceagent.tools import (DEFAULT_TOOL_SPECS, GovernedToolRunner, MockERP, ToolGateway,
                              ToolResult)
from voiceagent.policy import PolicyEngine, PolicyContext


def test_mockerp_seed_data():
    erp = MockERP()
    o = erp.get_order("ORD-4821")
    assert o["status"] == "CONFIRMED" and o["amount"] == 1299.0
    assert erp.orders_for_customer("CUST-001") == ["ORD-4821", "ORD-7734"]

def test_mockerp_satisfies_support_backend_protocol():
    # The gateway's ERP binding surface is a documented Protocol; the demo
    # fixture satisfies it structurally (no inheritance).
    from voiceagent.tools import SupportBackend
    assert isinstance(MockERP(), SupportBackend)

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


# ---------------------------------------------------------------------------
# Sprint A3: ToolSpec carries its reply CONTRACT as data (`facts`) — the
# customer-visible guarantees the echo guardrail enforces — declared in code
# defaults and overridable from a tenant bundle's tools.yaml.
# ---------------------------------------------------------------------------

def test_toolspec_has_facts_field():
    from voiceagent.tools import ToolSpec
    assert ToolSpec(params=()).facts == ()
    assert ToolSpec(params=(), facts=("order",)).facts == ("order",)


def test_default_tool_specs_declare_contract_facts():
    from voiceagent.tools import DEFAULT_TOOL_SPECS
    assert DEFAULT_TOOL_SPECS["fetch_order_status"].facts == ("order",)
    # "delivery" deliberately is NOT a standalone spec fact: the historical
    # echo-guard group is ["order", "delivery"] under first-match-per-group
    # semantics (it lives in the demo delivery_eta contract spec), and a
    # second "delivery" group would double-force the keyword.
    assert DEFAULT_TOOL_SPECS["reschedule_delivery"].facts == ()
    assert DEFAULT_TOOL_SPECS["initiate_refund"].facts == ("refund",)
    assert DEFAULT_TOOL_SPECS["escalate_to_human"].facts == ()


def test_yaml_loader_accepts_facts():
    import tempfile, yaml
    from pathlib import Path as P
    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "tools.yaml"
        f.write_text(yaml.safe_dump({"tools": {
            "cancel_order": {"facts": ["cancel", "cancellation"]}}}))
        gw = ToolGateway.from_yaml(f)
        assert gw.specs["cancel_order"].facts == ("cancel", "cancellation")


def test_yaml_loader_rejects_bad_facts_type():
    import tempfile, yaml
    from pathlib import Path as P
    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "tools.yaml"
        f.write_text(yaml.safe_dump({"tools": {
            "cancel_order": {"facts": "cancel"}}}))
        with pytest.raises(ValueError, match="facts"):
            ToolGateway.from_yaml(f)


def test_specs_with_yaml_facts_merges_without_touching_other_specs():
    import tempfile, yaml
    from pathlib import Path as P
    from voiceagent.tools import DEFAULT_TOOL_SPECS, specs_with_yaml_facts
    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "tools.yaml"
        f.write_text(yaml.safe_dump({"tools": {
            "fetch_order_status": {"facts": ["account"]}}}))
        specs = specs_with_yaml_facts(f)
        # Declared facts override; everything else keeps its base spec
        # (params AND preconditions intact).
        assert specs["fetch_order_status"].facts == ("account",)
        assert specs["fetch_order_status"].params == \
            DEFAULT_TOOL_SPECS["fetch_order_status"].params
        assert specs["cancel_order"] == DEFAULT_TOOL_SPECS["cancel_order"]


def test_specs_with_yaml_facts_rejects_unknown_tool():
    import tempfile, yaml
    from pathlib import Path as P
    from voiceagent.tools import specs_with_yaml_facts
    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "tools.yaml"
        f.write_text(yaml.safe_dump({"tools": {
            "not_a_tool": {"facts": ["x"]}}}))
        with pytest.raises(ValueError, match="unknown tool"):
            specs_with_yaml_facts(f)


def test_spec_facts_union_helper():
    from voiceagent.tools import ToolSpec, spec_facts
    specs = {"a": ToolSpec(params=(), facts=("x", "y")),
             "b": ToolSpec(params=(), facts=("y", "z"))}
    assert spec_facts(specs) == ["x", "y", "z"]  # declaration order, deduped


def test_order_lookup_by_phone_suffix_and_normalization():
    gw = ToolGateway()
    r = gw.execute("order_lookup", {"phone": "9876543210"})
    assert r.ok and [o["order_id"] for o in r.value] == ["ORD-4821", "ORD-7734"]
    r2 = gw.execute("order_lookup", {"phone": "+91-98765-43210"})
    assert r2.ok and len(r2.value) == 2


def test_order_lookup_unknown_phone_returns_empty_not_error():
    gw = ToolGateway()
    r = gw.execute("order_lookup", {"phone": "5550001111"})
    assert r.ok and r.value == []


def test_order_lookup_spec_declares_facts_and_params():
    spec = DEFAULT_TOOL_SPECS["order_lookup"]
    assert spec.params == ("phone",)
    assert spec.facts == ("order",)


def test_get_order_matches_loose_id_shapes():
    erp = MockERP()
    assert erp.get_order("ORD4821")["order_id"] == "ORD-4821"
    assert erp.get_order("ord-4821")["order_id"] == "ORD-4821"
    assert erp.get_order("ORD-4821")["order_id"] == "ORD-4821"
    assert erp.get_order("ORD9999") is None


def test_tool_metadata_lives_on_spec_not_runtime():
    """Adding a tool = binding + spec in ONE place. runtime.py's brain surface
    is a pure derivation — no per-tool edits in runtime, ever."""
    from voiceagent.tools import DEFAULT_TOOL_SPECS

    assert DEFAULT_TOOL_SPECS["order_lookup"].side_effects is False
    assert DEFAULT_TOOL_SPECS["escalate_to_human"].side_effects is True
    assert "phone number" in DEFAULT_TOOL_SPECS["order_lookup"].description
    assert DEFAULT_TOOL_SPECS["cancel_order"].description == ""  # generic ok


# ---------------------------------------------------------------------------
# Task D2: light param-type validation at the governed boundary. ToolSpec
# carries an optional `param_types` map (values: string|number|integer|
# boolean; undeclared params default to "string"). Safe coercions happen
# (amount "200" -> 200.0); impossible ones are rejected with
# `invalid_param: <name>` BEFORE the ERP is touched.
# ---------------------------------------------------------------------------

def test_refund_spec_declares_amount_as_number():
    spec = DEFAULT_TOOL_SPECS["initiate_refund"]
    assert spec.param_types == {"amount": "number"}


def test_param_coercion_amount_string_to_number():
    gw = ToolGateway()
    r = gw.execute("initiate_refund",
                   {"order_id": "ORD-4821", "amount": "200", "reason": "damaged"})
    assert r.ok
    assert r.value["amount"] == 200.0
    assert len(gw.erp.refunds) == 1


def test_param_bad_type_rejected_with_invalid_param_error():
    gw = ToolGateway()
    r = gw.execute("initiate_refund",
                   {"order_id": "ORD-4821", "amount": "abc", "reason": "damaged"})
    assert not r.ok
    assert r.error == "invalid_param: amount"
    assert gw.erp.refunds == []  # never reached the ERP


def test_param_validation_happens_before_order_fetch():
    """A bad-typed param must fail at the boundary, not leak an
    order_not_found (or any ERP call) first."""
    gw = ToolGateway()
    r = gw.execute("initiate_refund",
                   {"order_id": "ORD-9999", "amount": "abc", "reason": "x"})
    assert r.error == "invalid_param: amount"


def test_missing_params_still_flagged_when_param_types_present():
    gw = ToolGateway()
    r = gw.execute("initiate_refund", {"order_id": "ORD-4821",
                                       "amount": "100", "reason": None})
    assert not r.ok and "missing_params" in r.error


def test_undeclared_params_default_to_string_coercion():
    """Params without a param_types entry are treated as string: numeric
    values are safely coerced (the brain may quote a number), never
    rejected."""
    gw = ToolGateway()
    r = gw.execute("escalate_to_human", {"reason": 12345})
    assert r.ok
    assert gw.erp.handoffs[0]["reason"] == "12345"


def test_integer_and_boolean_param_types_coerce():
    from voiceagent.tools import ToolSpec
    gw = ToolGateway(specs={
        "initiate_refund": ToolSpec(
            params=("order_id", "amount", "reason", "line", "waive"),
            param_types={"amount": "number", "line": "integer",
                         "waive": "boolean"}),
    })
    # number accepts ints; integer coerces "3"; boolean coerces "true" —
    # all validated before the ERP binding runs.
    r = gw.execute("initiate_refund",
                   {"order_id": "ORD-4821", "amount": 5, "reason": "x",
                    "line": "3", "waive": "true"})
    assert r.ok and r.value["amount"] == 5.0
    # fractional string is not an integer
    r2 = gw.execute("initiate_refund",
                    {"order_id": "ORD-4821", "amount": 5, "reason": "x",
                     "line": "3.5", "waive": "true"})
    assert not r2.ok and r2.error == "invalid_param: line"
    # boolean rejects non-boolean strings
    r3 = gw.execute("initiate_refund",
                    {"order_id": "ORD-4821", "amount": 5, "reason": "x",
                     "line": "3", "waive": "maybe"})
    assert not r3.ok and r3.error == "invalid_param: waive"
    # a boolean is NOT a number/integer (bool is an int subclass — exclude)
    r4 = gw.execute("initiate_refund",
                    {"order_id": "ORD-4821", "amount": True, "reason": "x",
                     "line": "3", "waive": "true"})
    assert not r4.ok and r4.error == "invalid_param: amount"


def test_governed_runner_surfaces_invalid_param_error():
    """The runner path (orchestrator -> GovernedToolRunner -> gateway) sees
    the boundary rejection as a plain failed ToolResult."""
    from voiceagent.decisionlog import DecisionLog
    log = DecisionLog()
    runner = GovernedToolRunner(ToolGateway(),
                                PolicyEngine({"refund": {"allow": True}}),
                                decision_log=log)
    out = runner.run("refund", PolicyContext(authenticated=True),
                     "initiate_refund",
                     {"order_id": "ORD-4821", "amount": "abc", "reason": "x"})
    assert out.decision_verdict == "ALLOW"
    assert not out.executed and out.result.ok is False
    assert out.result.error == "invalid_param: amount"


def test_idempotency_key_not_poisoned_by_invalid_param():
    """A failed validation must not cache against the idempotency key: the
    same key with valid params must still execute (only ok=True is cached)."""
    erp = MockERP()
    gw = ToolGateway(erp=erp)
    bad = gw.execute("initiate_refund",
                     {"order_id": "ORD-4821", "amount": "inf", "reason": "cold"},
                     idempotency_key="k1")
    assert not bad.ok
    good = gw.execute("initiate_refund",
                      {"order_id": "ORD-4821", "amount": "200", "reason": "cold"},
                      idempotency_key="k1")
    assert good.ok
    assert erp.refunds, "refund must have executed on the retry"


def test_nonfinite_and_underscore_numbers_rejected():
    gw = ToolGateway()
    for v in ("inf", "-inf", "nan", "1_000"):
        r = gw.execute("initiate_refund",
                       {"order_id": "ORD-4821", "amount": v, "reason": "x"})
        assert not r.ok and r.error == "invalid_param: amount", v


def test_end_call_and_feedback_tools_governed():
    gw = ToolGateway()
    ok = gw.execute("end_call", {"reason": "caller said thanks and bye"})
    assert ok.ok and ok.value["call_ended"] is True
    r = gw.execute("record_feedback", {"rating": "8", "comment": "fast"})
    assert r.ok and r.value["rating"] == 8.0
    bad = gw.execute("record_feedback", {"rating": "11"})
    assert not bad.ok and "out_of_range" in bad.error
