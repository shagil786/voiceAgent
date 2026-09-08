# src/voiceagent/demo_repairs.py — the GenericBackend worked example (ADR-004).
"""A HOME-SERVICES tenant on the NEW GenericBackend path (ADR-004).

Handy Repairs & Services — bookings for a technician visit — runs
end-to-end on the SAME ToolGateway / PolicyEngine / Orchestrator as the
ecommerce demo and the clinic, but through the FORWARD domain path:

  - backend: RepairsBackend implements GenericBackend (resource verbs:
    get_resource/list_resources/execute_operation over the "booking"
    resource type) — NO order-words anywhere, NO noun mapping onto
    SupportBackend (contrast demo_clinic, which rides the classic
    SupportBackend path by design).
  - surface: the tools are REGISTERED with domain names the brain hears
    (fetch_booking_status, cancel_booking, reschedule_booking) via
    ToolGateway.register_binding + ToolSpec.resource — ADR-004's
    registration mechanism, not a platform tools.py edit.

WHY CODE, STILL (ADR-003/004): bindings are code. A new business's tools
arrive as one small code module (this one) + a tenant bundle (data/tenants/
example-repairs/), not as platform changes and not as tenant-invented
runtime bindings. What ADR-004 removes is the SEMANTIC MAPPING and the
if/elif chain — a GenericBackend domain module is a straight translation of
the business's own verbs, and registration makes the surface explicit and
governed. Dynamic agent-created tools remain rejected (see ADR-004).

Lifecycle (booking `status` field) — the platform only requires the
precondition vocabulary the domain DECLARES in its ToolSpecs:
    BOOKED -> technician visit scheduled, upcoming
    IN_PROGRESS -> technician on site / job underway
    DONE -> job completed
    CANCELLED -> booking cancelled
(There is no platform lifecycle vocabulary; the demo defaults'
CONFIRMED/SHIPPED/DELIVERED words are the ECOMMERCE DEMO's contract data,
not the platform's — a GenericBackend domain declares its own states in
its own specs, as this module does.)

RESOURCE MAP:
    get_resource("booking", id)        -> booking by id (or None)
    list_resources("booking", {phone}) -> bookings for a caller's phone
    execute_operation("cancel_booking", {booking_id, reason})      -> cancel
    execute_operation("reschedule_booking", {booking_id, new_date}) -> move
    get_lifecycle_states("booking")    -> the states above (introspection
                                          is available to callers that ask)
"""
from __future__ import annotations

import copy
from typing import Any

from voiceagent.generic_backend import (EcommerceAdapter,  # noqa: F401
                                        GenericBackendError)
from voiceagent.tools import ToolGateway, ToolSpec

# --- the backend -------------------------------------------------------------

_BOOKING_STATES = ["BOOKED", "IN_PROGRESS", "DONE", "CANCELLED"]


def _booking_shape(booking_id: str) -> str:
    """Alphanumeric-uppercase comparison shape (MockERP's _id_shape idea)."""
    return "".join(ch for ch in str(booking_id) if ch.isalnum()).upper()


_SEED_BOOKINGS: dict[str, dict] = {
    "BK-3001": {
        "booking_id": "BK-3001",
        "customer": "Anil Verma",
        "phone": "+919812340001",
        "service": "AC repair",
        "status": "BOOKED",
        "technician": None,
        "window_date": "2026-09-10",
        "window_slot": "09:00-13:00",
        "address": "22 MG Road, Pune",
    },
    "BK-3002": {
        "booking_id": "BK-3002",
        "customer": "Sana Khan",
        "phone": "+919812340002",
        "service": "Plumbing leak",
        "status": "BOOKED",
        "technician": "Rahul (tech id T-07)",
        "window_date": "2026-09-10",
        "window_slot": "14:00-18:00",
        "address": "4 Lake View, Pune",
    },
    "BK-3003": {
        "booking_id": "BK-3003",
        "customer": "Dev Patel",
        "phone": "+919812340003",
        "service": "Washing machine",
        "status": "IN_PROGRESS",
        "technician": "Imran (tech id T-12)",
        "window_date": "2026-09-09",
        "window_slot": "09:00-13:00",
        "address": "7 Park Street, Pune",
    },
    "BK-3004": {
        "booking_id": "BK-3004",
        "customer": "Nina Roy",
        "phone": "+919812340004",
        "service": "Geyser repair",
        "status": "DONE",
        "technician": "Imran (tech id T-12)",
        "window_date": "2026-09-08",
        "window_slot": "09:00-13:00",
        "address": "15 Hill Road, Pune",
    },
}


def _phone_digits(phone: Any) -> str:
    return "".join(ch for ch in str(phone) if ch.isdigit())


class RepairsBackend:
    """A home-services booking backend implementing GenericBackend —
    resource-verb surface, zero order-words (the ADR-004 forward path)."""

    def __init__(self) -> None:
        self.bookings: dict[str, dict] = copy.deepcopy(_SEED_BOOKINGS)
        self.cancellations: list[dict] = []
        self.fail_next = False  # failure injection, MockERP-style

    def _check_live(self) -> None:
        if self.fail_next:
            self.fail_next = False
            raise TimeoutError("repairs backend timed out")

    # --- GenericBackend surface --------------------------------------------

    def get_resource(self, resource_type: str, resource_id: str) -> dict | None:
        self._check_live()
        if resource_type != "booking":
            raise GenericBackendError(
                f"unsupported resource_type {resource_type!r} "
                f"(expected 'booking')")
        want = _booking_shape(resource_id)
        for k, v in self.bookings.items():
            if _booking_shape(k) == want:
                return copy.deepcopy(v)
        return None

    def list_resources(self, resource_type: str,
                       filters: dict[str, Any] | None = None) -> list[dict]:
        self._check_live()
        if resource_type != "booking":
            raise GenericBackendError(
                f"unsupported resource_type {resource_type!r} "
                f"(expected 'booking')")
        filters = filters or {}
        want = _phone_digits(filters.get("phone", ""))
        if not want:
            raise GenericBackendError(
                "list_resources('booking') requires a 'phone' filter")
        out = []
        for b in self.bookings.values():
            have = _phone_digits(b.get("phone", ""))
            if have and (have == want or have.endswith(want)
                         or want.endswith(have)):
                out.append(copy.deepcopy(b))
        return out

    def create_resource(self, resource_type: str, data: dict) -> dict:
        raise GenericBackendError(
            "create_resource is not supported: bookings are placed through "
            "the company's own funnel, not the support agent")

    def update_resource(self, resource_type: str, resource_id: str,
                        data: dict) -> dict:
        raise GenericBackendError(
            "update_resource is not supported: booking mutations go "
            "through governed operations (cancel/reschedule)")

    def execute_operation(self, operation_name: str, params: dict) -> dict:
        self._check_live()
        params = params or {}
        booking = self.get_resource("booking", params["booking_id"])
        if booking is None:
            raise GenericBackendError(
                f"booking_not_found: {params['booking_id']}")
        if operation_name == "cancel_booking":
            if booking["status"] in ("IN_PROGRESS", "DONE"):
                raise GenericBackendError(
                    f"cannot cancel a {booking['status']} booking "
                    f"(technician already dispatched/finished)")
            booking["status"] = "CANCELLED"
            booking["cancel_reason"] = params["reason"]
            self.cancellations.append({"booking_id": booking["booking_id"],
                                       "reason": params["reason"]})
            return booking
        if operation_name == "reschedule_booking":
            if booking["status"] in ("IN_PROGRESS", "DONE"):
                raise GenericBackendError(
                    f"cannot reschedule a {booking['status']} booking")
            booking["window_date"] = params["new_date"]
            booking["status"] = "BOOKED"
            return booking
        raise GenericBackendError(
            f"unsupported operation {operation_name!r} (cancel_booking, "
            f"reschedule_booking)")

    def get_lifecycle_states(self, resource_type: str) -> list[str]:
        if resource_type != "booking":
            raise GenericBackendError(
                f"unsupported resource_type {resource_type!r}")
        return list(_BOOKING_STATES)

    # escalate_to_human's executor (the always-required safety valve,
    # ADR-003) pages the company's dispatch staff — a governed handoff.
    def record_handoff(self, reason: str) -> dict:
        return {"handoff": True, "reason": reason,
                "routed_to": "dispatch supervisor"}


# --- the governed surface: registered domain tools (ADR-004) --------------------

def build_repairs_gateway(erp: Any | None = None) -> ToolGateway:
    """A ToolGateway whose brain-visible tools are the BOOKING verbs —
    registered domain names + specs, no order-words, no platform edit.
    The classic demo tools are ABSENT from this surface entirely: the
    gateway is built on an EMPTY spec base (only the always-required
    safety valve escalate_to_human is carried over), so the demo's
    fetch_order_status / cancel_order / ... can never be proposed or
    executed for this domain. Registration order matters: register the
    domain tools on the empty base, then add the safety valve."""
    gw = ToolGateway(erp=erp or RepairsBackend(), specs={})

    gw.specs["fetch_booking_status"] = ToolSpec(
        params=("booking_id",),
        facts=("booking",),
        side_effects=False,
        action="booking_status",
        description="Fetch the current status of a technician booking by "
                    "its booking id (e.g. BK-3001).",
        resource=("booking", "booking_id",
                  lambda erp, bid: erp.get_resource("booking", bid)))
    gw.register_binding(
        "fetch_booking_status",
        lambda erp, p: erp.get_resource("booking", p["booking_id"]))

    gw.specs["booking_lookup"] = ToolSpec(
        params=("phone",),
        facts=("booking",),
        side_effects=False,
        action="booking_lookup",
        description="Find a caller's technician bookings by the phone "
                    "number they booked with — use when the caller does "
                    "not know their booking id.")
    gw.register_binding(
        "booking_lookup",
        lambda erp, p: erp.list_resources("booking", {"phone": p["phone"]}))

    gw.specs["cancel_booking"] = ToolSpec(
        params=("booking_id", "reason"),
        side_effects=True,
        action="cancel_booking",
        description="Cancel a caller's upcoming technician booking "
                    "(params: booking_id, reason). Bookings with a "
                    "technician already dispatched or finished cannot be "
                    "cancelled.",
        preconditions=({"field": "status", "op": "not_in",
                        "value": ["IN_PROGRESS", "DONE"]},),
        resource=("booking", "booking_id",
                  lambda erp, bid: erp.get_resource("booking", bid)))
    gw.register_binding(
        "cancel_booking",
        lambda erp, p: erp.execute_operation(
            "cancel_booking", {"booking_id": p["booking_id"],
                               "reason": p["reason"]}))

    gw.specs["reschedule_booking"] = ToolSpec(
        params=("booking_id", "new_date"),
        side_effects=True,
        action="reschedule_booking",
        description="Move a caller's upcoming technician booking to a new "
                    "date (params: booking_id, new_date).",
        preconditions=({"field": "status", "op": "not_in",
                        "value": ["IN_PROGRESS", "DONE"]},),
        resource=("booking", "booking_id",
                  lambda erp, bid: erp.get_resource("booking", bid)))
    gw.register_binding(
        "reschedule_booking",
        lambda erp, p: erp.execute_operation(
            "reschedule_booking", {"booking_id": p["booking_id"],
                                   "new_date": p["new_date"]}))

    # escalate_to_human must always be proposeable (ADR-003 safety valve).
    # With an empty spec base the classic demo surface is gone, so the
    # safety valve is registered EXPLICITLY from the platform defaults.
    from voiceagent.tools import DEFAULT_TOOL_SPECS
    gw.specs["escalate_to_human"] = DEFAULT_TOOL_SPECS["escalate_to_human"]
    gw.bindings["escalate_to_human"] = (
        lambda erp, p: erp.record_handoff(p["reason"]))
    return gw
