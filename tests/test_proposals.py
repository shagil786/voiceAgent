# tests/test_proposals.py — hybrid tool proposals + human gate (ADR-005).
"""Covers the proposal pipeline: hybrid authorship (AI-drafted AND operator-
authored), the mandatory human approval gate (PROPOSED/REJECTED never
register; APPROVED compiles onto ADR-004 machinery), validation, risk-class
sanity, and that compiled tools behave identically to hand-written domain
tools (preconditions, governed execution, no order-words leak)."""
from __future__ import annotations

import pytest

from voiceagent.demo_repairs import RepairsBackend, build_repairs_gateway
from voiceagent.proposals import (APPROVED, PROPOSED, REJECTED, ToolProposal,
                                  compile_approved, draft_from_api_spec,
                                  validate_proposal)


def _approved(name="cancel_visit", description="Cancel a booking.",
              params=("booking_id", "reason"), action="cancel_booking",
              operation="cancel_booking", resource_type="booking",
              id_param="booking_id", side_effects=True,
              risk_class="mutating", provenance="operator",
              status=APPROVED, preconditions=()):
    return ToolProposal(name=name, description=description,
                        params=params, action=action, operation=operation,
                        resource_type=resource_type, id_param=id_param,
                        side_effects=side_effects, risk_class=risk_class,
                        provenance=provenance, status=status,
                        preconditions=preconditions)


# --- the gate ---------------------------------------------------------------

def test_proposed_and_rejected_never_register():
    gw = build_repairs_gateway(RepairsBackend())
    drafts = draft_from_api_spec({"operations": [{
        "operation": "cancel_booking", "tool_name": "ai_cancel",
        "description": "AI draft.", "params": ["booking_id", "reason"],
        "side_effects": True, "risk_class": "mutating"}]})
    assert all(p.status == PROPOSED for p in drafts)
    rejected = _approved(name="op_rejected", status=REJECTED)
    reg = compile_approved(gw, gw.erp, drafts + [rejected])
    assert reg == []  # nothing registered
    assert "ai_cancel" not in gw.specs and "op_rejected" not in gw.specs


def test_approved_registers_and_executes():
    gw = build_repairs_gateway(RepairsBackend())
    prop = _approved(name="cancel_visit")
    reg = compile_approved(gw, gw.erp, [prop])
    assert reg == ["cancel_visit"]
    res = gw.execute("cancel_visit",
                     {"booking_id": "BK-3002", "reason": "changed plan"})
    assert res.ok and res.value["status"] == "CANCELLED"


def test_hybrid_both_channels_register_after_human_approval():
    """The hybrid: AI drafts + operator authors; BOTH need the human gate;
    after approval both compile to working governed tools."""
    gw = build_repairs_gateway(RepairsBackend())
    op = _approved(name="op_move", params=("booking_id", "new_date"),
                   action="reschedule_booking", operation="reschedule_booking")
    ai = draft_from_api_spec({"operations": [{
        "operation": "reschedule_booking", "tool_name": "ai_move",
        "description": "AI draft.", "params": ["booking_id", "new_date"],
        "side_effects": True, "risk_class": "mutating"}]})
    # human approves the AI draft
    ai_approved = _approved(name="ai_move", params=("booking_id", "new_date"),
                            action="reschedule_booking",
                            operation="reschedule_booking",
                            provenance="ai")
    reg = compile_approved(gw, gw.erp, [op] + ai + [ai_approved])
    assert reg == ["op_move", "ai_move"]
    assert gw.execute("ai_move", {"booking_id": "BK-3001",
                                  "new_date": "2026-09-17"}).ok
    assert gw.execute("op_move", {"booking_id": "BK-3002",
                                  "new_date": "2026-09-18"}).ok


def test_compile_refuses_invalid_approved_proposal():
    gw = build_repairs_gateway(RepairsBackend())
    bad = _approved(name="Bad Name!", description="x", params=("a",))
    with pytest.raises(ValueError, match="invalid tool name"):
        compile_approved(gw, gw.erp, [bad])


# --- validation ---------------------------------------------------------------

def test_validate_proposal_catches_shape_errors():
    errs = validate_proposal(_approved(name="Bad Name!"))
    assert any("invalid tool name" in e for e in errs)
    errs2 = validate_proposal(_approved(status="weird"))
    assert any("invalid status" in e for e in errs2)
    errs3 = validate_proposal(_approved(provenance="bot"))
    assert any("provenance" in e for e in errs3)
    errs4 = validate_proposal(_approved(risk_class="read", side_effects=True))
    assert any("risk_class" in e for e in errs4)
    errs5 = validate_proposal(_approved(id_param="nope"))
    assert any("id_param" in e for e in errs5)
    assert validate_proposal(_approved()) == []


# --- governed behavior parity ----------------------------------------------------

def test_compiled_tool_preconditions_fire():
    """A compiled tool must enforce preconditions exactly like a
    hand-written domain tool (resource-verb fetch, block before mutation)."""
    gw = build_repairs_gateway(RepairsBackend())
    prop = _approved(name="cancel_visit", preconditions=(
        {"field": "status", "op": "not_in",
         "value": ["IN_PROGRESS", "DONE"]},))
    compile_approved(gw, gw.erp, [prop])
    backend = gw.erp
    before = len(backend.cancellations)
    # BK-3003 is IN_PROGRESS -> blocked, nothing mutated
    res = gw.execute("cancel_visit",
                     {"booking_id": "BK-3003", "reason": "x"})
    assert not res.ok and "precondition_failed" in res.error
    assert len(backend.cancellations) == before
    # BK-3001 BOOKED -> allowed
    res2 = gw.execute("cancel_visit",
                      {"booking_id": "BK-3001", "reason": "plan change"})
    assert res2.ok and res2.value["status"] == "CANCELLED"


def test_compiled_tool_surface_has_no_order_words():
    gw = build_repairs_gateway(RepairsBackend())
    compile_approved(gw, gw.erp, [
        _approved(name="reschedule_visit", params=("booking_id", "new_date"),
                  action="reschedule_booking",
                  operation="reschedule_booking")])
    names = set(gw.specs) & set(gw.bindings)
    assert "reschedule_visit" in names
    assert not any(w in n for n in names
                   for w in ("order", "delivery", "refund", "return"))


def test_graceful_timeout_through_compiled_tool():
    gw = build_repairs_gateway(RepairsBackend())
    compile_approved(gw, gw.erp, [_approved(name="cancel_visit")])
    gw.erp.fail_next = True
    res = gw.execute("cancel_visit",
                     {"booking_id": "BK-3001", "reason": "x"})
    assert not res.ok and res.error.startswith("backend_timeout")


def test_escalate_valve_survives_proposal_surface():
    gw = build_repairs_gateway(RepairsBackend())
    compile_approved(gw, gw.erp, [_approved(name="cancel_visit")])
    assert "escalate_to_human" in set(gw.specs) & set(gw.bindings)
