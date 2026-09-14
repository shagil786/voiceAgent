# tests/test_onboard_cli.py — the adaptation front door CLI.
"""Running onboard on the committed hotel spec produces a fail-closed
scaffold: every draft proposed, high-risk policy ESCALATE, tenant.json
parseable — and the output passes through load_proposals_yaml unchanged."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data" / "fixtures" / "hotel-openapi.yaml"
CLI = ROOT / "scripts" / "onboard.py"


def _run(tmp_out):
    return subprocess.run(
        [sys.executable, str(CLI), "--spec", str(SPEC),
         "--out", str(tmp_out), "--name", "Grand Hotel"],
        capture_output=True, text=True, cwd=ROOT)


def test_onboard_prints_discovery_report_and_writes_bundle(tmp_path):
    out = tmp_path / "hotel-demo"
    r = _run(out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Discovered API: Grand Hotel PMS" in r.stdout
    assert "6 operations" in r.stdout
    assert "[HIGH RISK] cancel_booking" in r.stdout
    for f in ("proposals.yaml", "policies.yaml", "tools.yaml", "tenant.json",
              "README.md"):
        assert (out / f).exists(), f
    assert (out / "intents").is_dir() and (out / "knowledge").is_dir()
    tools = (out / "tools.yaml").read_text(encoding="utf-8")
    for valve in ("escalate_to_human:", "end_call:", "record_feedback:"):
        assert valve in tools, valve


def test_onboard_output_is_fail_closed_by_construction(tmp_path):
    out = tmp_path / "hotel-demo"
    _run(out)
    from voiceagent.proposals import load_proposals_yaml
    props = load_proposals_yaml(out / "proposals.yaml")
    assert len(props) == 6
    assert all(p.status == "proposed" for p in props)
    policies = (out / "policies.yaml").read_text(encoding="utf-8")
    assert "cancel_booking:\n  escalate: true" in policies
    tenant = json.loads((out / "tenant.json").read_text(encoding="utf-8"))
    assert tenant["name"] == "Grand Hotel"
    assert tenant["persona"]


def test_onboard_scaffold_loads_through_the_bundle_loader(tmp_path):
    out = tmp_path / "hotel-demo"
    _run(out)
    from voiceagent.tenant import Tenant
    t = Tenant.load(out)
    assert t.config.name == "Grand Hotel"
    # The scaffold's brain surface is valves-only: without an explicit
    # tools.yaml a fresh bundle would fall back to the e-commerce builtin
    # gateway tools (runtime._bundle_gateway_tools).
    from voiceagent.runtime import gateway_tools_from_yaml
    surface = set(gateway_tools_from_yaml(out / "tools.yaml"))
    assert surface == {"escalate_to_human", "end_call", "record_feedback"}


def test_onboard_without_name_slugifies_the_spec_title(tmp_path):
    out = tmp_path / "slug-demo"
    r = subprocess.run(
        [sys.executable, str(CLI), "--spec", str(SPEC), "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    tenant = json.loads((out / "tenant.json").read_text(encoding="utf-8"))
    assert tenant["name"] == "grand-hotel-pms"
