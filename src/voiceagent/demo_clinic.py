# src/voiceagent/demo_clinic.py — the second-domain worked example backend.
"""A CLINIC as a tenant, proving the governed pipeline is domain-neutral:
Sunrise Family Clinic's front desk runs end-to-end on the SAME ToolGateway,
PolicyEngine, and Orchestrator the ecommerce demo uses — zero changes to
platform code (tools.py / runtime.py / policy.py are untouched by this
module; it is data + a backend adapter, nothing else).

WHY A MAPPING-BACKEND IS THE SANCTIONED SECOND-DOMAIN PATH (ADR-003):
bindings are code, and the generic surface IS the contract. A new business
does not add tool names to the platform's ToolGateway.execute chain — it
writes a backend adapter that structurally implements SupportBackend and
MAPS its domain onto the seven generic verbs, then declares its domain
language as DATA: intents/ exemplars, tools.yaml description overrides, and
policies.yaml action rules. The brain hears "appointment" through the
tenant bundle (persona, descriptions, knowledge); the platform only ever
sees fetch_order_status / cancel_order / reschedule_delivery / ...

THE GENERIC STATUS VOCABULARY IS PLATFORM CONTRACT DATA: preconditions on
DEFAULT_TOOL_SPECS (and any tenant tools.yaml) evaluate the record's
`status` field against the state words CONFIRMED / SHIPPED / DELIVERED /
CANCELLED / RETURN_REQUESTED / REFUND_INITIATED. A tenant backend therefore
stores its lifecycle in those words — that is a data contract, not an
ecommerce leak; this module maps clinic lifecycle onto it explicitly:

    CONFIRMED         -> appointment booked, upcoming
    SHIPPED           -> patient checked in; the visit is underway
    DELIVERED         -> visit completed
    CANCELLED         -> appointment cancelled
    RETURN_REQUESTED  -> visit passed back to scheduling (no-show reversal)
    REFUND_INITIATED  -> billing adjustment initiated

CLINIC -> GENERIC VERB MAP (all seven protocol methods, plus the gateway's
phone-lookup): get_order -> appointment lookup by id; orders_for_customer
-> appointment ids for a patient record; cancel_order -> cancel
appointment; reschedule_delivery -> reschedule appointment (writes the
clinic's own `appointment_date` field — record contents are backend data);
initiate_refund -> billing adjustment on a visit invoice; mark_return ->
appointment pass-back / no-show reversal; record_handoff -> page on-call
staff; lookup_orders_by_phone -> appointments for a caller's phone number.

Demo/fixture placement follows the demo_data.py precedent: this module is
committed example content, never imported by core; a production clinic
would implement the same protocol against its PMS/EMR over HTTP and pass
it via ToolGateway(erp=...) / build_orchestrator(erp=...).
"""
from __future__ import annotations

import copy
import time

# The seeded appointment records (clinic data lives HERE, in the adapter —
# the same placement discipline as MockERP's bundle fixture: backend records
# are the backend's own shape; only `status` is platform contract data).


def _apt_shape(appointment_id: str) -> str:
    """Comparison shape for appointment IDs — the same normalization idea as
    tools.MockERP._id_shape: alphanumeric uppercase, so 'APT1042',
    'apt-1042' and 'APT-1042' all match one appointment."""
    return "".join(ch for ch in str(appointment_id) if ch.isalnum()).upper()


_SEED_APPOINTMENTS: dict[str, dict] = {
    "APT-1042": {
        "appointment_id": "APT-1042",
        "patient_id": "PAT-201",
        "patient_name": "Ravi Kumar",
        "status": "CONFIRMED",
        "clinician": "Dr. Meera Iyer",
        "department": "Cardiology",
        "appointment_date": "2026-09-09",
        "appointment_time": "10:30",
        "location": "Sunrise Family Clinic, Indiranagar",
        "fee_due": 500.0,
    },
    "APT-1043": {
        "appointment_id": "APT-1043",
        "patient_id": "PAT-202",
        "patient_name": "Meena Sharma",
        "status": "CONFIRMED",
        "clinician": "Dr. Arjun Prasad",
        "department": "General Medicine",
        "appointment_date": "2026-09-10",
        "appointment_time": "16:00",
        "location": "Sunrise Family Clinic, Indiranagar",
        "fee_due": 350.0,
    },
    "APT-1051": {
        "appointment_id": "APT-1051",
        "patient_id": "PAT-203",
        "patient_name": "Arjun Rao",
        "status": "SHIPPED",  # checked in; visit underway
        "clinician": "Dr. Meera Iyer",
        "department": "Cardiology",
        "appointment_date": "2026-09-07",
        "appointment_time": "09:15",
        "location": "Sunrise Family Clinic, Indiranagar",
        "fee_due": 0.0,
    },
    "APT-1052": {
        "appointment_id": "APT-1052",
        "patient_id": "PAT-204",
        "patient_name": "Priya Nair",
        "status": "DELIVERED",  # visit completed
        "clinician": "Dr. Kavya Reddy",
        "department": "Dermatology",
        "appointment_date": "2026-09-05",
        "appointment_time": "12:00",
        "location": "Sunrise Family Clinic, Indiranagar",
        "fee_due": 900.0,
    },
    "APT-1060": {
        "appointment_id": "APT-1060",
        "patient_id": "PAT-201",
        "patient_name": "Ravi Kumar",
        "status": "DELIVERED",  # last month's completed follow-up
        "clinician": "Dr. Meera Iyer",
        "department": "Cardiology",
        "appointment_date": "2026-08-09",
        "appointment_time": "10:30",
        "location": "Sunrise Family Clinic, Indiranagar",
        "fee_due": 0.0,
    },
    "APT-1061": {
        "appointment_id": "APT-1061",
        "patient_id": "PAT-205",
        "patient_name": "Farhan Ali",
        "status": "CANCELLED",
        "clinician": "Dr. Kavya Reddy",
        "department": "Dermatology",
        "appointment_date": "2026-09-11",
        "appointment_time": "11:00",
        "location": "Sunrise Family Clinic, Indiranagar",
        "fee_due": 400.0,
    },
    "APT-1062": {
        "appointment_id": "APT-1062",
        "patient_id": "PAT-206",
        "patient_name": "Sneha Iyer",
        "status": "CONFIRMED",
        "clinician": "Dr. Arjun Prasad",
        "department": "General Medicine",
        "appointment_date": "2026-09-12",
        "appointment_time": "18:30",
        "location": "Sunrise Family Clinic, Indiranagar",
        "fee_due": 350.0,
    },
    "APT-1070": {
        "appointment_id": "APT-1070",
        "patient_id": "PAT-207",
        "patient_name": "Vikram Bose",
        "status": "SHIPPED",  # in the waiting room now
        "clinician": "Dr. Arjun Prasad",
        "department": "General Medicine",
        "appointment_date": "2026-09-07",
        "appointment_time": "10:00",
        "location": "Sunrise Family Clinic, Indiranagar",
        "fee_due": 0.0,
    },
}

_SEED_PATIENTS: dict[str, dict] = {
    "PAT-201": {"phone": "+91-9840010203", "name": "Ravi Kumar",
                "appointments": ["APT-1042", "APT-1060"]},
    "PAT-202": {"phone": "+91-9840010204", "name": "Meena Sharma",
                "appointments": ["APT-1043"]},
    "PAT-203": {"phone": "98400-10205", "name": "Arjun Rao",
                "appointments": ["APT-1051"]},
    "PAT-204": {"phone": "+91-9840010206", "name": "Priya Nair",
                "appointments": ["APT-1052"]},
    "PAT-205": {"phone": "+91-9840010207", "name": "Farhan Ali",
                "appointments": ["APT-1061"]},
    "PAT-206": {"phone": "9840010208", "name": "Sneha Iyer",
                "appointments": ["APT-1062"]},
    "PAT-207": {"phone": "+91-98-400-10209", "name": "Vikram Bose",
                "appointments": ["APT-1070"]},
}


class ClinicBackend:
    """In-memory clinic PMS stand-in, structurally implementing the
    SupportBackend protocol (MockERP is the reference implementation shape:
    deepcopy discipline, one-shot failure injection, loose id/phone
    matching). Constructed once per deployment and passed to
    ToolGateway(erp=...) / build_orchestrator(erp=...) — the gateway,
    policy engine, and runner are byte-identical to the ecommerce demo's."""

    def __init__(self) -> None:
        self.appointments: dict[str, dict] = copy.deepcopy(_SEED_APPOINTMENTS)
        self.patients: dict[str, dict] = copy.deepcopy(_SEED_PATIENTS)
        self.adjustments: list[dict] = []   # initiated_refund -> billing adjustments
        self.pages: list[dict] = []         # record_handoff -> staff pages
        # Failure injection (MockERP parity): the NEXT operation raises like
        # a hung PMS so graceful timeout handling stays testable offline.
        self.fail_next = False

    def _check_live(self) -> None:
        if self.fail_next:
            self.fail_next = False  # one-shot: the NEXT operation fails
            raise TimeoutError("clinic pms backend timed out")

    # -- read paths ---------------------------------------------------------

    def get_order(self, order_id: str) -> dict | None:
        """Appointment lookup by id — callers and brains spell ids loosely
        ('apt1042' vs the stored 'APT-1042'), so the comparison runs on the
        alphanumeric-uppercase shape. Exact key always wins."""
        self._check_live()
        a = self.appointments.get(order_id)
        if a is not None:
            return copy.deepcopy(a)
        want = _apt_shape(order_id)
        for k, v in self.appointments.items():
            if _apt_shape(k) == want:
                return copy.deepcopy(v)
        return None

    def lookup_orders_by_phone(self, phone: str) -> list[dict]:
        """Fetch appointments for a caller-supplied phone number (the
        not-found ladder's alternate lookup — callers rarely know their
        appointment id). Suffix digit matching, MockERP semantics."""
        self._check_live()
        want = "".join(ch for ch in str(phone) if ch.isdigit())
        if not want:
            return []
        out = []
        for p in self.patients.values():
            have = "".join(ch for ch in str(p.get("phone", "")) if ch.isdigit())
            if have and (have == want or have.endswith(want)
                         or want.endswith(have)):
                for aid in p.get("appointments", []):
                    a = self.appointments.get(aid)
                    if a:
                        out.append(copy.deepcopy(a))
        return out

    def orders_for_customer(self, customer_id: str) -> list[str]:
        """Appointment ids booked under a patient record id."""
        self._check_live()
        p = self.patients.get(customer_id)
        return list(p["appointments"]) if p else []

    # -- clinic verbs on the generic surface ---------------------------------

    def cancel_order(self, order_id: str, reason: str) -> dict:
        """Cancel an appointment. The gateway's precondition (status not in
        SHIPPED/DELIVERED) already blocked completed visits before this runs;
        the store simply records the clinic's own outcome word CANCELLED."""
        self._check_live()
        a = self.appointments[order_id]
        a["status"] = "CANCELLED"
        a["cancel_reason"] = reason
        return copy.deepcopy(a)

    def reschedule_delivery(self, order_id: str, new_date: str) -> dict:
        """Reschedule an appointment — writes the clinic's own
        `appointment_date` field; record contents are backend data."""
        self._check_live()
        a = self.appointments[order_id]
        a["appointment_date"] = new_date
        return copy.deepcopy(a)

    def initiate_refund(self, order_id: str, amount: float, reason: str) -> dict:
        """Billing adjustment on a visit invoice (the clinic's honest
        equivalent of a refund request). Status moves to the generic
        REFUND_INITIATED word so downstream policy data reads it."""
        self._check_live()
        a = self.appointments[order_id]
        a["status"] = "REFUND_INITIATED"
        adjustment = {"appointment_id": order_id, "amount": amount,
                      "reason": reason,
                      "adjustment_id": f"ADJ-{len(self.adjustments) + 1:04d}"}
        self.adjustments.append(adjustment)
        return adjustment

    def mark_return(self, order_id: str, reason: str) -> dict:
        """Appointment pass-back / no-show reversal: the visit slot goes back
        to the scheduling queue (patient left before being seen, visit being
        re-booked by staff). Only reachable for SHIPPED/DELIVERED visits —
        the initiate_return precondition enforces that."""
        self._check_live()
        a = self.appointments[order_id]
        a["status"] = "RETURN_REQUESTED"
        a["return_reason"] = reason
        return copy.deepcopy(a)

    def record_handoff(self, reason: str) -> dict:
        """Page the on-call clinic staff (the clinic's human-handoff valve:
        front desk, duty doctor, or emergency response — the reason text
        routes the page)."""
        self._check_live()
        page = {"reason": reason,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.pages.append(page)
        return {"handed_off": True, "paged": "on-call clinic staff",
                "reason": reason}
