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
