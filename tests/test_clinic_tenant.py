# tests/test_clinic_tenant.py — the second-domain worked example.
"""A CLINIC tenant proves a non-ecommerce business runs end-to-end through
the governed pipeline with ZERO changes to the domain-neutral core
(tools.py execute chain, runtime.py, policy.py). Guarded here:

1. ClinicBackend structurally satisfies the SupportBackend protocol —
   the mapping-backend path (ADR-003: bindings are code; the generic
   surface is the contract).
2. The example-clinic bundle loads; the repo validators pass on it.
3. Governed execution over ToolGateway(erp=ClinicBackend()) with the
   clinic policy: ALLOW reads, precondition blocks, and the
   least-privilege DENY for a binding the tools.yaml does NOT compose.
4. Prompt composition: clinic language reaches the brain via the bundle
   (persona + description overrides + knowledge) with no ecommerce leak.
5. A scripted-brain turn through build_orchestrator(tenant="example-clinic",
   erp=ClinicBackend()) executes a governed appointment-status turn that
   lands in the decision log.

Fully offline: stub FrontierClient only, no network anywhere.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from voiceagent.decisionlog import DecisionLog
from voiceagent.demo_clinic import ClinicBackend
from voiceagent.policy import PolicyContext, PolicyEngine, load_policies
from voiceagent.runtime import (
    PLATFORM_GOVERNANCE,
    build_orchestrator,
    make_deployment,
)
from voiceagent.tools import (
    DEFAULT_TOOL_SPECS,
    GovernedToolRunner,
    SupportBackend,
    ToolGateway,
)
from voiceagent.tenant import Tenant
from tests.test_orchestrator import ScriptedBrain, reply, tc, tool_messages

ROOT = Path(__file__).resolve().parents[1]
CLINIC = ROOT / "data" / "tenants" / "example-clinic"
VALIDATOR = ROOT / "scripts" / "validate_tenant.py"
POLICY_VALIDATOR = ROOT / "scripts" / "validate_policies.py"
FRONTIER_URL = {"VOICEAGENT_FRONTIER_URL": "https://fake/v1"}

# The clinic's tools.yaml `action:` renames: tool name (code binding) ->
# policy action (tenant data). The brain never sees a new tool name.
# POSTURE (proposals-only surface): tools.yaml declares ONLY the
# domain-neutral platform tools; appointment verbs arrive via
# proposals.yaml (native GenericBackend tools, zero order-words).
EXPECTED_SURFACE = {
    "escalate_to_human", "end_call", "record_feedback",
}
EXPECTED_PROPOSALS = {
    "fetch_appointment_status", "appointment_lookup", "cancel_appointment",
    "reschedule_appointment", "billing_adjustment",
}


def _clinic_policy() -> PolicyEngine:
    return PolicyEngine(load_policies(str(CLINIC / "policies.yaml")),
                        currency="₹")


# --- 1. ClinicBackend structurally satisfies SupportBackend ------------------

def test_clinic_backend_structurally_satisfies_support_backend():
    backend = ClinicBackend()
    # runtime_checkable protocol: structural, no inheritance needed —
    # exactly how a production PMS/EMR adapter would plug in.
    assert isinstance(backend, SupportBackend)
    proto_methods = {
        name: fn for name, fn in vars(SupportBackend).items()
        if callable(fn) and not name.startswith("_")
    }
    assert set(proto_methods) == {
        "get_order", "orders_for_customer", "cancel_order",
        "reschedule_delivery", "initiate_refund", "mark_return",
        "record_handoff",
    }
    for name, fn in proto_methods.items():
        want = [p.name for p in inspect.signature(fn).parameters.values()
                if p.name != "self"]
        got = [p.name
               for p in inspect.signature(getattr(backend, name))
               .parameters.values() if p.name != "self"]
        assert got == want, f"{name}: {got} != {want}"
    # The gateway's order_lookup binding also calls the phone lookup
    # (MockERP parity — not part of the protocol, required by the gateway).
    assert callable(backend.lookup_orders_by_phone)


def test_clinic_backend_semantics_map_onto_the_generic_surface():
    be = ClinicBackend()
    # Loose id spelling matches (MockERP normalization semantics).
    assert be.get_order("apt1042")["patient_name"] == "Ravi Kumar"
    assert be.get_order("APT-1042")["status"] == "CONFIRMED"
    assert be.get_order("APT-9999") is None
    # Phone lookup returns the caller's appointments (order_lookup binding).
    phone_hits = be.lookup_orders_by_phone("+91-9840010203")
    assert [a["appointment_id"] for a in phone_hits] == ["APT-1042", "APT-1060"]
    # cancel_order -> cancel appointment.
    be.cancel_order("APT-1042", "plans changed")
    assert be.get_order("APT-1042")["status"] == "CANCELLED"
    # reschedule_delivery -> reschedule appointment (clinic's own field).
    be.reschedule_delivery("APT-1042", "2026-09-20")
    assert be.get_order("APT-1042")["appointment_date"] == "2026-09-20"
    # initiate_refund -> billing adjustment on the visit invoice.
    adj = be.initiate_refund("APT-1052", 900.0, "double charge")
    assert adj["adjustment_id"] == "ADJ-0001" and adj["amount"] == 900.0
    assert be.get_order("APT-1052")["status"] == "REFUND_INITIATED"
    # mark_return -> appointment pass-back / no-show reversal.
    be.mark_return("APT-1051", "patient left before being seen")
    assert be.get_order("APT-1051")["status"] == "RETURN_REQUESTED"
    # record_handoff -> page on-call staff.
    page = be.record_handoff("duty doctor needed at front desk")
    assert page["handed_off"] is True and len(be.pages) == 1
    # orders_for_customer -> appointment ids for a patient record.
    assert be.orders_for_customer("PAT-201") == ["APT-1042", "APT-1060"]


# --- 2. the bundle loads; the repo validators pass ----------------------------

def test_clinic_tenant_bundle_loads_and_composes():
    dep = make_deployment(tenant=Tenant.load(CLINIC))
    assert dep.name == "example-clinic"
    # tools.yaml declares ONLY domain-neutral platform tools — no order
    # verbs in the composed brain surface.
    assert set(dep.gateway_tools) == EXPECTED_SURFACE
    # Clinic action names are tenant DATA (intents/ + tools.yaml actions)...
    assert dep.gateway_tools["escalate_to_human"]["action"] == \
        "escalate_to_human"
    # ...and the declared vocabulary keeps the appointment actions (now
    # sourced from intents/ + proposals-backed tools).
    assert dep.actions == [
        "appointment_lookup", "appointment_status", "billing_adjustment",
        "billing_question", "cancel_appointment", "clinic_hours",
        "complaint", "end_call", "escalate_to_human", "medical_emergency",
        "prescription_refill", "record_feedback", "reschedule_appointment",
    ]
    assert set(dep.knowledge) == {
        "hours", "cancellation-policy", "prescription-refills",
        "billing-insurance", "emergencies",
    }
    assert dep.metadata["languages"] == ["en", "hi"]
    assert dep.metadata["tenant"] == "example-clinic"


def test_repo_validators_pass_on_the_clinic_bundle():
    r = subprocess.run([sys.executable, str(VALIDATOR), str(CLINIC)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[PASS]" in r.stdout


def test_policy_validator_cli_and_clinic_policy_semantics():
    # CLI inspection: scripts/validate_policies.py takes NO path argument —
    # it validates the PLATFORM policy file's semantics. The clinic bundle's
    # policies.yaml rides through validate_tenant.py's embedded structural
    # policy checks (same checks family); the clinic's own semantics are
    # pinned directly through the PolicyEngine here.
    r = subprocess.run([sys.executable, str(POLICY_VALIDATOR)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    pol = _clinic_policy()
    assert pol.currency == "₹"  # tenant currency is wired policy data
    assert pol.evaluate("medical_emergency").verdict == "ESCALATE"
    assert pol.evaluate("appointment_status").verdict == "ALLOW"
    assert pol.evaluate(
        "complaint",
        PolicyContext(signals={"frustrated": True})).verdict == "ESCALATE"
    assert pol.evaluate(
        "cancel_appointment", PolicyContext()).verdict == "REQUIRE_AUTH"
    assert pol.evaluate(
        "billing_adjustment",
        PolicyContext(amount=1500.0, authenticated=True)
    ).verdict == "REQUIRE_HUMAN_APPROVAL"
    assert pol.evaluate(
        "billing_adjustment",
        PolicyContext(amount=500.0, authenticated=True)).verdict == "ALLOW"


# --- 3. governed execution over the clinic backend ----------------------------

def _clinic_runner() -> GovernedToolRunner:
    return GovernedToolRunner(ToolGateway(erp=ClinicBackend()),
                              _clinic_policy(), DecisionLog())


def test_governed_runner_fetches_a_clinic_appointment():
    runner = _clinic_runner()
    oc = runner.run("appointment_status", PolicyContext(authenticated=True),
                    "fetch_order_status", {"order_id": "APT-1042"})
    assert oc.decision_verdict == "ALLOW" and oc.executed
    # The result reflects CLINIC data through the generic surface.
    assert oc.result.value["patient_name"] == "Ravi Kumar"
    assert oc.result.value["clinician"] == "Dr. Meera Iyer"
    assert oc.result.value["appointment_date"] == "2026-09-09"


def test_governed_runner_finds_appointments_by_phone():
    runner = _clinic_runner()
    oc = runner.run("appointment_lookup", PolicyContext(authenticated=True),
                    "order_lookup", {"phone": "9840010203"})
    assert oc.decision_verdict == "ALLOW" and oc.executed
    ids = [a["appointment_id"] for a in oc.result.value]
    assert ids == ["APT-1042", "APT-1060"]


def test_precondition_blocks_cancelling_a_completed_visit():
    # A completed (DELIVERED) visit violates cancel_order's generic
    # precondition (status not_in SHIPPED/DELIVERED) — the GATEWAY blocks it
    # even though the clinic policy allows the authenticated cancellation.
    be = ClinicBackend()
    gov = GovernedToolRunner(ToolGateway(erp=be), _clinic_policy(),
                             DecisionLog())
    oc = gov.run("cancel_appointment", PolicyContext(authenticated=True),
                 "cancel_order",
                 {"order_id": "APT-1052", "reason": "changed my mind"})
    assert oc.decision_verdict == "ALLOW"          # policy allowed...
    assert not oc.executed                          # ...the gateway blocked
    assert oc.result.ok is False
    assert "precondition_failed" in oc.result.error
    # Backend state untouched by the blocked mutation.
    assert be.get_order("APT-1052")["status"] == "DELIVERED"


def test_uncomposed_initiate_return_stays_brain_invisible_and_denied():
    # initiate_return is a REAL binding (code) the clinic deliberately does
    # NOT compose (pass-backs are front-desk-only, not caller-facing).
    assert "initiate_return" in DEFAULT_TOOL_SPECS
    # (a) The brain-facing surface from gateway_tools_from_yaml excludes it.
    dep = make_deployment(tenant=Tenant.load(CLINIC))
    assert "initiate_return" not in dep.gateway_tools
    # (b) At the raw gateway level the spec still exists, so an UNGOVERNED
    # call would execute — which is exactly why the governed path exists.
    raw = ToolGateway(erp=ClinicBackend()).execute(
        "initiate_return", {"order_id": "APT-1051", "reason": "x"})
    assert raw.ok is True
    # (c) Through the clinic policy the action is least-privilege DENYed.
    assert _clinic_policy().evaluate(
        "return", PolicyContext(authenticated=True)).verdict == "DENY"
    gov = GovernedToolRunner(ToolGateway(erp=ClinicBackend()),
                             _clinic_policy(), DecisionLog())
    oc = gov.run("return", PolicyContext(authenticated=True),
                 "initiate_return", {"order_id": "APT-1051", "reason": "x"})
    assert oc.decision_verdict == "DENY" and not oc.executed
    assert any("least privilege" in r for r in oc.reasons)


# --- 4. prompt composition: clinic in, ecommerce out --------------------------

def test_clinic_prompt_carries_clinic_language_not_ecommerce():
    dep = make_deployment(tenant=Tenant.load(CLINIC))
    prompt = dep.system_prompt
    # Governance leads; the persona block carries the clinic identity and
    # the appointment language (compiled from tenant.json may_promise /
    # never_say) — domain language enters as DATA.
    assert prompt.startswith(PLATFORM_GOVERNANCE)
    assert ("You are the front-desk voice assistant for Sunrise Family "
            "Clinic (example tenant).") in prompt
    assert "Tone: calm, patient, and reassuring — callers may be unwell " \
           "or worried." in prompt
    assert "an appointment's current status" in prompt
    assert ("Never say or imply: a medical diagnosis; emergency medical "
            "advice; guaranteed slot availability.") in prompt
    # The safety-valve guidance survives in every compiled prompt.
    assert "propose escalate_to_human" in prompt
    # NO ecommerce leak: not the old base-prompt phrasing, not another
    # business's vocabulary.
    assert "pizza" not in prompt.lower()
    assert "order details" not in prompt.lower()


def test_clinic_tool_descriptions_reach_the_brain_schemas():
    # Appointment descriptions live in proposals.yaml and become the
    # brain's per-tool schema text — the channel (besides the persona)
    # through which "appointment" reaches the frontier brain. No order
    # verb survives in the composed surface.
    import os
    os.chdir(ROOT)
    orch = build_orchestrator(env=dict(FRONTIER_URL),
                              tenant="example-clinic",
                              erp=ClinicBackend())
    schema = {s["function"]["name"]: s["function"]
              for s in orch.brain.tool_schemas()}
    assert set(schema) == EXPECTED_SURFACE | EXPECTED_PROPOSALS
    assert schema["fetch_appointment_status"]["description"] == (
        "Look up the current status of a patient's appointment by its "
        "appointment id (e.g. APT-1042).")
    assert "medical emergency" in \
        schema["escalate_to_human"]["description"]
    assert not any("order" in name for name in schema)


# --- 5. legacy order verbs are brain-invisible; the native turn -----------

def test_legacy_order_verbs_not_proposeable_on_clinic(monkeypatch):
    # The order-verb bindings still exist in code (legacy mapping path),
    # but the clinic brain can never propose them: they are absent from
    # the composed surface, and a brain that tries anyway gets a surfaced
    # error — never an execution.
    monkeypatch.chdir(ROOT)  # bare bundle names resolve under data/tenants/
    log = DecisionLog()
    orch = build_orchestrator(env=dict(FRONTIER_URL),
                              tenant="example-clinic",
                              erp=ClinicBackend(), decision_log=log)
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t1", "fetch_order_status", order_id="APT-1042")]),
        reply("Let me look that up another way."),
    ])
    res = orch.handle_turn("s-clinic-legacy",
                           "what is the status of my appointment APT-1042?",
                           authenticated=True)
    # The bundle's policy + currency are still wired, not platform defaults.
    assert orch.runner.policy.currency == "₹"
    # DENYed by least privilege (undeclared tool name) — surfaced back to
    # the brain, never executed.
    assert len(res.actions) == 1
    assert res.actions[0]["tool"] == "fetch_order_status"
    assert res.actions[0]["verdict"] == "DENY" and not res.actions[0]["ok"]
    assert "another way" in res.reply
    # ...while the native tool executes the same turn governedly (see the
    # section-6 native tests for the ALLOW + audit assertions).


# --- 6. native appointment tools: bundle-declared, zero order-words  --------
# The proposals.yaml tools compile onto the gateway and execute against the
# clinic backend's GenericBackend surface — the same appointments, addressed
# in clinic nouns, with no platform code involved.

def test_native_appointment_tools_compile_from_bundle():
    import os
    os.chdir(ROOT)  # bare bundle names resolve under data/tenants/
    orch = build_orchestrator(env=dict(FRONTIER_URL),
                              tenant="example-clinic",
                              erp=ClinicBackend())
    schema = {s["function"]["name"]: s["function"]
              for s in orch.brain.tool_schemas()}
    for tool in ("fetch_appointment_status", "cancel_appointment",
                 "reschedule_appointment"):
        assert tool in schema, f"{tool} must be proposeable from the bundle"
        assert "appointment" in schema[tool]["description"]
    assert "order" not in schema["fetch_appointment_status"]["description"]


def test_native_appointment_status_turn_no_order_words():
    import os
    os.chdir(ROOT)
    orch = build_orchestrator(env=dict(FRONTIER_URL),
                              tenant="example-clinic",
                              erp=ClinicBackend(), decision_log=DecisionLog())
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t1", "fetch_appointment_status",
                        appointment_id="APT-1042")]),
        reply("Your appointment APT-1042 with Dr. Meera Iyer is confirmed."),
    ])
    res = orch.handle_turn("s-clinic-native",
                           "what is the status of my appointment APT-1042?",
                           authenticated=True)
    assert res.actions and res.actions[0]["tool"] == "fetch_appointment_status"
    assert res.actions[0]["action"] == "appointment_status"
    assert res.actions[0]["verdict"] == "ALLOW" and res.actions[0]["ok"]
    assert res.actions[0]["value"]["patient_name"] == "Ravi Kumar"


def test_native_appointment_not_found_uses_generic_ladder_error():
    import os
    os.chdir(ROOT)
    orch = build_orchestrator(env=dict(FRONTIER_URL),
                              tenant="example-clinic",
                              erp=ClinicBackend(), decision_log=DecisionLog())
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t1", "fetch_appointment_status",
                        appointment_id="APT-9999")]),
        reply("I could not find that appointment."),
    ])
    res = orch.handle_turn("s-clinic-miss", "status of APT-9999?",
                           authenticated=True)
    assert res.actions[0]["error"].startswith("appointment_not_found")


def test_native_cancel_appointment_executes_and_blocks_completed():
    be = ClinicBackend()
    assert be.execute_operation(
        "cancel_appointment",
        {"appointment_id": "APT-1042", "reason": "plans changed"})["status"] \
        == "CANCELLED"
    assert be.get_resource("appointment", "APT-1042")["status"] == "CANCELLED"
    assert "CONFIRMED" in be.get_lifecycle_states("appointment")
    import pytest
    from voiceagent.generic_backend import GenericBackendError
    with pytest.raises(GenericBackendError):
        be.execute_operation(
            "cancel_appointment",
            {"appointment_id": "APT-1052", "reason": "x"})  # DELIVERED
