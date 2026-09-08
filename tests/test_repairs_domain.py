# tests/test_repairs_domain.py — GenericBackend worked example (ADR-004).
"""Handy Repairs & Services proves the FORWARD domain path (ADR-004): a
non-ecommerce business on GenericBackend with REGISTERED booking-named
tools — zero order-words, zero noun mapping, zero platform-tools edits.

Guarded here:
1. RepairsBackend structurally satisfies GenericBackend (resource verbs).
2. The built surface contains ONLY the domain's booking verbs + the
   escalate_to_human safety valve — no fetch_order_status/cancel_order/...
   anywhere (surface = what the brain can propose AND the gateway execute).
3. Governed execution: resource-verb precondition fetch (a dispatched/
   finished booking cannot cancel), graceful timeout, not-found ladder.
4. ToolSpec.resource routes domain records for preconditions.
5. Policy-style gating holds (least privilege: unknown tool absent).
Fully offline.
"""
from __future__ import annotations

import pytest

from voiceagent.demo_repairs import RepairsBackend, build_repairs_gateway
from voiceagent.generic_backend import GenericBackend, GenericBackendError
from voiceagent.tools import ToolGateway

ORDER_WORDS = ("order", "delivery", "refund", "return", "shipped",
               "delivered")


def _surface(gw: ToolGateway) -> set[str]:
    return set(gw.specs) & set(gw.bindings)


# --- 1. GenericBackend structural -----------------------------------------------

def test_repairs_backend_is_generic_backend():
    assert isinstance(RepairsBackend(), GenericBackend)


def test_backend_lifecycle_and_operations():
    b = RepairsBackend()
    assert b.get_lifecycle_states("booking") == ["BOOKED", "IN_PROGRESS",
                                                 "DONE", "CANCELLED"]
    booking = b.get_resource("booking", "BK-3001")
    assert booking is not None and booking["status"] == "BOOKED"
    assert b.get_resource("booking", "bk-3001") is not None  # shape match
    assert b.get_resource("booking", "BK-NOPE") is None
    cancelled = b.execute_operation(
        "cancel_booking", {"booking_id": "BK-3001", "reason": "r"})
    assert cancelled["status"] == "CANCELLED"
    with pytest.raises(GenericBackendError, match="unsupported operation"):
        b.execute_operation("no_such_op", {"booking_id": "BK-3001"})
    with pytest.raises(GenericBackendError, match="unsupported resource"):
        b.get_resource("widget", "W-1")
    with pytest.raises(GenericBackendError, match="not supported"):
        b.create_resource("booking", {})


# --- 2. the surface is pure-domain -------------------------------------------------

def test_surface_has_no_order_words():
    gw = build_repairs_gateway()
    names = _surface(gw)
    assert names == {"booking_lookup", "cancel_booking",
                     "escalate_to_human", "fetch_booking_status",
                     "reschedule_booking"}
    assert not any(w in n for n in names for w in ORDER_WORDS)


def test_classic_tools_absent_and_unexecutable():
    gw = build_repairs_gateway()
    # the classic surface must be structurally absent, not just hidden
    assert "fetch_order_status" not in gw.specs
    assert "fetch_order_status" not in gw.bindings
    assert "cancel_order" not in gw.specs
    # and even a hand-forged execute on an absent tool fails closed
    res = gw.execute("fetch_order_status", {"order_id": "ORD-1"})
    assert not res.ok and res.error == "unknown_tool: fetch_order_status"


# --- 3. governed execution ----------------------------------------------------------

def test_fetch_booking_status_read():
    gw = build_repairs_gateway()
    res = gw.execute("fetch_booking_status", {"booking_id": "BK-3001"})
    assert res.ok and res.value["service"] == "AC repair"
    assert res.value["status"] == "BOOKED"


def test_not_found_ladder():
    gw = build_repairs_gateway()
    res = gw.execute("fetch_booking_status", {"booking_id": "BK-NOPE"})
    assert not res.ok and res.error == "booking_not_found: BK-NOPE"


def test_precondition_blocks_dispatched_cancel():
    # BK-3003 is IN_PROGRESS (technician on site) -> cancel precondition
    # must block BEFORE any backend mutation (resource-verb fetch fired)
    gw = build_repairs_gateway()
    backend = gw.erp
    before = len(backend.cancellations)
    res = gw.execute("cancel_booking",
                     {"booking_id": "BK-3003", "reason": "x"})
    assert not res.ok and "precondition_failed" in res.error
    assert len(backend.cancellations) == before  # nothing mutated


def test_cancel_and_reschedule_mutations():
    gw = build_repairs_gateway()
    r = gw.execute("cancel_booking",
                   {"booking_id": "BK-3001", "reason": "changed plan"})
    assert r.ok and r.value["status"] == "CANCELLED"
    r2 = gw.execute("reschedule_booking",
                    {"booking_id": "BK-3002", "new_date": "2026-09-12"})
    assert r2.ok and r2.value["window_date"] == "2026-09-12"
    assert gw.erp.bookings["BK-3002"]["status"] == "BOOKED"


def test_graceful_timeout_unchanged():
    gw = build_repairs_gateway()
    gw.erp.fail_next = True
    res = gw.execute("fetch_booking_status", {"booking_id": "BK-3001"})
    assert not res.ok and res.error.startswith("backend_timeout")


def test_booking_lookup_by_phone():
    gw = build_repairs_gateway()
    res = gw.execute("booking_lookup", {"phone": "9812340003"})
    assert res.ok and [b["booking_id"] for b in res.value] == ["BK-3003"]


def test_idempotency_still_caches():
    gw = build_repairs_gateway()
    a = gw.execute("cancel_booking",
                   {"booking_id": "BK-3001", "reason": "r"},
                   idempotency_key="k")
    b = gw.execute("cancel_booking",
                   {"booking_id": "BK-3001", "reason": "r"},
                   idempotency_key="k")
    assert a.ok and b.ok and b.idempotent_replay and not a.idempotent_replay


def test_escalate_valve_present_and_governed():
    gw = build_repairs_gateway()
    assert "escalate_to_human" in _surface(gw)
    res = gw.execute("escalate_to_human", {"reason": "caller upset"})
    assert res.ok and res.value  # backend record_handoff returns a dict
