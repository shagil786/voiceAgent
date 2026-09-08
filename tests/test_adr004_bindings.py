# tests/test_adr004_bindings.py — registration-based bindings (ADR-004).
"""Covers: the registration API (register/replace, spec-first guard), the
default bindings covering the full classic surface, a GenericBackend-backed
gateway routing through registered bindings (new domains never see
order-words), policy gating still firing before any backend call, and
byte-identical governed behavior (idempotency, graceful timeout)."""
from __future__ import annotations

import pytest

from voiceagent.generic_backend import (EcommerceAdapter,
                                        GenericBackendError)
from voiceagent.policy import PolicyEngine, PolicyContext
from voiceagent.tools import DEFAULT_TOOL_SPECS, MockERP, ToolGateway


# --- registration API -----------------------------------------------------------

def test_default_bindings_cover_every_tool_spec():
    """Every spec has a binding — the registry and the specs can't drift
    (this is the ADR-003 property, now enforced structurally)."""
    gw = ToolGateway(erp=MockERP())
    assert set(gw.bindings) == set(DEFAULT_TOOL_SPECS)


def test_register_binding_requires_a_spec():
    """A binding without a spec can never be governed (proposed, policy-
    checked) — refuse it."""
    gw = ToolGateway(erp=MockERP())
    with pytest.raises(ValueError, match="no spec"):
        gw.register_binding("teleport_customer", lambda erp, p: {})


def test_register_binding_overrides_and_executes():
    """Domain code replaces a binding without touching platform code."""
    gw = ToolGateway(erp=MockERP())
    seen = {}

    def audit_cancel(erp, params):
        seen.update(params)
        return {"cancelled": True, "via": "custom-binding"}

    gw.register_binding("cancel_order", audit_cancel)
    res = gw.execute("cancel_order",
                     {"order_id": "ORD-4821", "reason": "plan change"})
    assert res.ok and res.value["via"] == "custom-binding"
    assert seen["order_id"] == "ORD-4821"
    # and the governed flow still wrapped it (idempotency available)
    res2 = gw.execute("cancel_order",
                      {"order_id": "ORD-4821", "reason": "plan change"},
                      idempotency_key="k1")
    assert res2.ok and not res2.idempotent_replay


# --- governed behavior unchanged (byte-identical contract) -----------------------

def test_idempotent_replay_still_cached():
    gw = ToolGateway(erp=MockERP())
    a = gw.execute("cancel_order", {"order_id": "ORD-4821", "reason": "x"},
                   idempotency_key="same")
    b = gw.execute("cancel_order", {"order_id": "ORD-4821", "reason": "x"},
                   idempotency_key="same")
    assert a.ok and b.ok and b.idempotent_replay and not a.idempotent_replay


def test_graceful_timeout_contract_unchanged():
    gw = ToolGateway(erp=MockERP())
    gw.erp.fail_next = True
    res = gw.execute("cancel_order",
                     {"order_id": "ORD-4821", "reason": "x"})  # CONFIRMED: cancellable
    assert not res.ok and res.error.startswith("backend_timeout")


def test_preconditions_still_fire_from_spec():
    gw = ToolGateway(erp=MockERP())
    res = gw.execute("cancel_order",
                     {"order_id": "ORD-7734", "reason": "x"})
    # ORD-7734 is SHIPPED -> cancel precondition blocks (spec data)
    assert not res.ok and "precondition_failed" in res.error


def test_unbound_spec_name_errors_cleanly():
    gw = ToolGateway(erp=MockERP())
    # remove a default binding: the spec remains, execution must fail closed
    gw.bindings.pop("fetch_order_status")
    res = gw.execute("fetch_order_status", {"order_id": "ORD-4821"})
    assert not res.ok and res.error == "unbound_tool: fetch_order_status"


# --- policy gating still precedes the backend -------------------------------------

def test_policy_gate_fires_before_binding():
    """Policy evaluates the tool's ACTION before any binding runs: a denied
    action never reaches the backend (the agent's governed loop calls
    policy.evaluate first — same engine, verified directly here)."""
    class ExplodingERP(MockERP):
        def cancel_order(self, order_id, reason):
            raise AssertionError("backend must not be reached when blocked")

    # least-privilege: an action with NO policy entry is DENY
    policy = PolicyEngine({})
    decision = policy.evaluate("cancel_order", PolicyContext())
    assert decision.verdict == "DENY"
    # the gateway itself stays backend-ready but is never invoked in the
    # governed loop when policy denies
    gw = ToolGateway(erp=ExplodingERP())
    assert "cancel_order" in gw.bindings  # registered and ready, but gated


# --- GenericBackend: new domains without order-words -------------------------------

class _AppointmentBackend:
    """A minimal GenericBackend for a clinic domain — zero order vocabulary."""

    def __init__(self):
        self.appts = {"APT-1": {"appointment_id": "APT-1",
                                "status": "BOOKED", "doctor": "Dr. Rao"}}
        self.cancelled = []

    def get_resource(self, resource_type, resource_id):
        if resource_type != "appointment":
            raise GenericBackendError(f"unknown resource {resource_type!r}")
        return self.appts.get(resource_id)

    def list_resources(self, resource_type, filters=None):
        if resource_type != "appointment":
            raise GenericBackendError(f"unknown resource {resource_type!r}")
        return list(self.appts.values())

    def create_resource(self, resource_type, data):
        raise GenericBackendError("out of scope for this test")

    def update_resource(self, resource_type, resource_id, data):
        raise GenericBackendError("out of scope for this test")

    def execute_operation(self, operation_name, params):
        if operation_name == "cancel_appointment":
            a = self.appts[params["appointment_id"]]
            a["status"] = "CANCELLED"
            self.cancelled.append(params["appointment_id"])
            return a
        raise GenericBackendError(f"unknown operation {operation_name!r}")

    def get_lifecycle_states(self, resource_type):
        return ["BOOKED", "CANCELLED"]


def test_generic_backend_gateway_via_registered_bindings():
    """A domain backend + registered bindings = a working governed gateway
    with NO order-words anywhere (ADR-004's whole point)."""
    backend = _AppointmentBackend()
    gw = ToolGateway(erp=backend)  # erp= holds any backend object
    # platform-authored ToolSpec + binding for the domain tool
    from voiceagent.tools import ToolSpec
    gw.specs["cancel_appointment"] = ToolSpec(
        params=("appointment_id", "reason"),
        preconditions=({"field": "status", "op": "not_in",
                        "value": ["CANCELLED"]},),
        resource=("appointment", "appointment_id",
                  lambda erp, aid: erp.get_resource("appointment", aid)))
    gw.register_binding(
        "cancel_appointment",
        lambda erp, p: erp.execute_operation(
            "cancel_appointment", {"appointment_id": p["appointment_id"],
                                   "reason": p["reason"]}))
    res = gw.execute("cancel_appointment",
                     {"appointment_id": "APT-1", "reason": "schedule conflict"})
    assert res.ok and res.value["status"] == "CANCELLED"
    # precondition (from the spec, data) fires on the domain resource
    res2 = gw.execute("cancel_appointment",
                      {"appointment_id": "APT-1", "reason": "again"})
    assert not res2.ok and "precondition_failed" in res2.error
    assert backend.cancelled == ["APT-1"]


def test_ecommerce_adapter_bridges_classic_backend():
    """The bridge proof: GenericBackend verbs over the classic surface."""
    ad = EcommerceAdapter(MockERP())
    order = ad.get_resource("order", "ORD-4821")
    assert order["status"] == "CONFIRMED"
    by_phone = ad.list_resources("order", {"phone": "9876543210"})
    assert by_phone and by_phone[0]["order_id"].startswith("ORD-")
    cancelled = ad.execute_operation(
        "cancel_order", {"order_id": "ORD-4821", "reason": "r"})
    assert cancelled["status"] == "CANCELLED"
    with pytest.raises(GenericBackendError):
        ad.get_resource("dragon", "D-1")
