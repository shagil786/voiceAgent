# tests/test_proposals_deploy.py — deployable proposals (ADR-005 wiring).
"""The example-repairs bundle proves proposals are a DEPLOYMENT artifact,
not just a library: an approved proposal YAML compiles onto the gateway and
joins the brain's surface through build_orchestrator — while a `proposed`
entry records but never compiles. Guarded here:
1. The bundle validates (validate_tenant.py) including proposals.yaml.
2. build_orchestrator(tenant=example-repairs, erp=RepairsBackend) compiles
   the two APPROVED proposals into governed tools; the PROPOSED one is
   absent from gateway specs, bindings, AND the brain surface.
3. The brain surface (deployment.gateway_tools) contains the proposal tools
   + platform valves, and NO classic ecommerce tools.
4. Compiled tools execute governed end-to-end (read + cancel, precondition
   enforced) against the injected GenericBackend.
Fully offline: stub FrontierClient only.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from voiceagent.demo_repairs import RepairsBackend
from voiceagent.runtime import build_orchestrator
from voiceagent.proposals import load_proposals_yaml

ROOT = Path(__file__).resolve().parents[1]
REPAIRS = ROOT / "data" / "tenants" / "example-repairs"
VALIDATOR = ROOT / "scripts" / "validate_tenant.py"


def _orch():
    return build_orchestrator(
        {"VOICEAGENT_FRONTIER_URL": "https://fake/v1"},
        tenant="example-repairs",
        erp=RepairsBackend())


def test_bundle_validates_including_proposals():
    r = subprocess.run([sys.executable, str(VALIDATOR), str(REPAIRS)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[PASS]" in r.stdout


def test_approved_proposals_compile_proposed_does_not():
    orch = _orch()
    gw = orch.runner.gateway
    surface = set(gw.specs) & set(gw.bindings)
    assert "fetch_booking_status" in surface
    assert "cancel_booking" in surface
    assert "quote_repair" not in surface  # PROPOSED — never compiles


def test_brain_surface_has_proposals_valves_no_classic_tools():
    orch = _orch()
    dep_tools = set(orch._deployment.gateway_tools)
    assert {"fetch_booking_status", "cancel_booking",
            "escalate_to_human", "end_call"} <= dep_tools
    assert "quote_repair" not in dep_tools
    assert not any(n.startswith(("fetch_order", "order_", "cancel_order",
                                 "initiate_", "reschedule_delivery"))
                   for n in dep_tools)


def test_approved_tools_execute_governed():
    orch = _orch()
    gw = orch.runner.gateway
    res = gw.execute("fetch_booking_status", {"booking_id": "BK-3001"})
    assert res.ok and res.value["service"] == "AC repair"
    res2 = gw.execute("cancel_booking",
                      {"booking_id": "BK-3001", "reason": "changed plan"})
    assert res2.ok and res2.value["status"] == "CANCELLED"
    # precondition still enforced through the compiled resource fetch
    res3 = gw.execute("cancel_booking",
                      {"booking_id": "BK-3003", "reason": "x"})
    assert not res3.ok and "precondition_failed" in res3.error


def test_proposal_file_records_three_entries():
    props = load_proposals_yaml(REPAIRS / "proposals.yaml")
    by_status = {p.name: p.status for p in props}
    assert by_status["fetch_booking_status"] == "approved"
    assert by_status["cancel_booking"] == "approved"
    assert by_status["quote_repair"] == "proposed"


def test_invalid_proposals_yaml_fails_fast_at_orchestrator_build():
    """A bundle whose approved proposal is invalid must refuse to start —
    committed approval data can never fail silently."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tenant.json").write_text(
            '{"name": "broken", "persona": "x"}')
        (root / "proposals.yaml").write_text(
            "proposals:\n"
            "  - name: 'Bad Name!'\n"
            "    description: x\n"
            "    params: [a]\n"
            "    action: a\n"
            "    operation: a\n"
            "    status: approved\n")
        from voiceagent.runtime import _bundle_proposals
        with pytest.raises(Exception):
            # _bundle_proposals loads; build_orchestrator validates on use
            build_orchestrator(
                {"VOICEAGENT_FRONTIER_URL": "https://fake/v1"},
                tenant=str(root))
