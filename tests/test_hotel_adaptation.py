# tests/test_hotel_adaptation.py — the adaptation front door, end to end.
"""The hotel tenant exists ONLY as bundle + fixture data. Guarded here:
the bundle validates; the approved proposals compile onto a
FixtureGenericBackend; a scripted conversation searches, books, and
cancels governed; the brain surface contains NO e-commerce legacy tools;
a proposed-status variant registers nothing. Zero domain code anywhere —
this file only wires what exists."""
import json
import subprocess
import sys
from pathlib import Path

from voiceagent.decisionlog import DecisionLog
from voiceagent.fixture_backend import FixtureGenericBackend
from voiceagent.runtime import build_orchestrator

from tests.test_orchestrator import ScriptedBrain, reply, tc

ROOT = Path(__file__).resolve().parents[1]
HOTEL = ROOT / "data" / "tenants" / "hotel-demo"
FIXTURE = ROOT / "data" / "fixtures" / "hotel.json"
VALIDATOR = ROOT / "scripts" / "validate_tenant.py"
FRONTIER_URL = {"VOICEAGENT_FRONTIER_URL": "https://fake/v1"}


def test_hotel_bundle_validates():
    r = subprocess.run([sys.executable, str(VALIDATOR), str(HOTEL)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[PASS]" in r.stdout


# --- the wired system ---------------------------------------------------------

def _orch(**kw):
    return build_orchestrator(dict(FRONTIER_URL), tenant="hotel-demo",
                              erp=FixtureGenericBackend(FIXTURE), **kw)


def test_surface_is_hotel_only_no_ecommerce_legacy():
    orch = _orch()
    surface = set(orch._deployment.gateway_tools)
    assert {"search_rooms", "get_room", "get_booking", "create_booking",
            "modify_booking", "cancel_booking",
            "escalate_to_human", "end_call"} <= surface
    assert not any(n.startswith(("fetch_order", "order_", "cancel_order",
                                 "initiate_refund", "reschedule_delivery"))
                   for n in surface)
    meta = orch._deployment.gateway_tools["create_booking"]
    assert meta["side_effects"] is True
    assert meta["parameters"]["properties"]["guest_name"] == {"type": "string"}


def test_search_is_a_governed_read():
    orch = _orch(decision_log=DecisionLog())
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t1", "search_rooms", check_in="2026-09-20",
                        check_out="2026-09-21")]),
        reply("We have the deluxe room at $55 and the standard at $35 "
              "per night for those dates."),
    ])
    res = orch.handle_turn("s-search", "any rooms for September 20th?",
                           authenticated=True)
    assert res.actions[0]["tool"] == "search_rooms"
    assert res.actions[0]["verdict"] == "ALLOW" and res.actions[0]["ok"]
    assert res.actions[0]["value"]["available"][0]["room_id"] == "RM-101"


def test_booking_requires_confirmation_and_mutates_fixture():
    log = DecisionLog()
    orch = _orch(decision_log=log)
    be = orch.runner.gateway.erp
    # Turn 1: the brain asks for confirmation (no tool call) — the
    # side-effects contract on the surface is what drives that behavior.
    orch.brain.client = ScriptedBrain([
        reply("Shall I book the deluxe room for Ravi Kumar, "
              "check-in September 20th?"),
    ])
    res1 = orch.handle_turn("s-book", "please book the deluxe room",
                            authenticated=True)
    assert not res1.actions  # nothing executed without confirmation
    assert be.get_resource("booking", "B-1001") is None  # fixture untouched
    # Turn 2: guest confirms — NOW the brain calls the governed tool.
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t2", "create_booking", guest_name="Ravi Kumar",
                        room_id="RM-101", check_in="2026-09-20")]),
        reply("You're all set — booking B-1001, deluxe room, "
              "check-in September 20th."),
    ])
    res2 = orch.handle_turn("s-book", "yes, please book it",
                            authenticated=True)
    act = res2.actions[0]
    assert act["tool"] == "create_booking" and act["verdict"] == "ALLOW"
    assert act["ok"] and act["value"]["booking_id"] == "B-1001"
    assert act["value"]["status"] == "BOOKED"
    # The fixture backend MUTATED — the demo data is real state.
    assert be.get_resource("booking", "B-1001")["guest_name"] == "Ravi Kumar"
    assert log.query(action="create_booking", verdict="ALLOW")


def test_cancel_is_high_risk_governed_with_precondition():
    log = DecisionLog()
    orch = _orch(decision_log=log)
    gw = orch.runner.gateway
    # Seed a booking directly through the same governed surface.
    gw.execute("create_booking", {"guest_name": "Nina Roy",
                                  "room_id": "RM-102",
                                  "check_in": "2026-09-20"})
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t1", "cancel_booking", booking_id="B-1001")]),
        reply("Your reservation B-1001 has been cancelled."),
    ])
    res = orch.handle_turn("s-cancel", "cancel my reservation",
                           authenticated=True)
    act = res.actions[0]
    assert act["tool"] == "cancel_booking" and act["verdict"] == "ALLOW"
    assert act["ok"] and act["value"]["status"] == "CANCELLED"
    # The precondition the OWNER added during review is enforced: a second
    # cancel of the same booking is refused before any execution.
    res2 = gw.execute("cancel_booking", {"booking_id": "B-1001"})
    assert not res2.ok and "precondition_failed" in res2.error
    assert log.query(action="cancel_booking")


def test_proposed_variant_registers_nothing(tmp_path):
    import shutil
    import yaml
    bundle = tmp_path / "hotel-proposed"
    shutil.copytree(HOTEL, bundle)
    pf = bundle / "proposals.yaml"
    doc = yaml.safe_load(pf.read_text(encoding="utf-8"))
    for entry in doc["proposals"]:
        entry["status"] = "proposed"
    pf.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    orch = build_orchestrator(dict(FRONTIER_URL), tenant=str(bundle),
                              erp=FixtureGenericBackend(FIXTURE))
    surface = set(orch._deployment.gateway_tools)
    assert not ({"search_rooms", "get_room", "get_booking",
                 "create_booking", "modify_booking",
                 "cancel_booking"} & surface)
    assert {"escalate_to_human", "end_call"} <= surface  # valves remain
