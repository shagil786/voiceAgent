# tests/test_default_bundle.py — Task E: the default tenant is a real bundle.
"""The platform's built-in demo tenant used to be hardcoded Python content
(voiceagent.demo_data) imported by runtime.py / intent.py / tools.py. It now
lives in the COMMITTED bundle data/tenants/default/ (unlike gitignored
customer bundles — it IS the platform's built-in demo tenant) and loads
through the SAME Tenant machinery as named tenants. These tests pin the
bundle content (byte-identical to the historical demo data) and the
dependency-flow rule: no demo_data imports in runtime/intent/tools."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import yaml

from voiceagent.tenant import Tenant

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "tenants" / "default"

# The historical demo data, pinned here verbatim (moved out of
# voiceagent.demo_data — the strings must survive the move byte-identically).
IDENTITY = "You are Acme's voice support agent."
KNOWLEDGE = {
    "eta": "Deliveries occur between 9:00 and 19:00 local time.",
    "cancel_policy": "Orders that already shipped cannot be cancelled.",
}


# --- the bundle is committed and well-formed ----------------------------------

def test_bundle_exists_with_tenant_json():
    cfg = json.loads((BUNDLE / "tenant.json").read_text())
    assert cfg["name"] == "default"
    assert cfg["currency"] == "$"
    # flat persona string = the Acme identity, wrapped as Persona(role=...)
    assert cfg["persona"] == "Acme's voice support agent"


def test_bundle_knowledge_files_match_the_historical_demo_kb():
    d = BUNDLE / "knowledge"
    got = {f.stem: f.read_text(encoding="utf-8") for f in sorted(d.glob("*.md"))}
    assert got == KNOWLEDGE


def test_bundle_intents_use_the_tenant_yaml_schema():
    """File NAME = intent label, YAML list = exemplars — the same schema as
    data/tenants/example-acme/intents/, and the full historical exemplar
    corpus (including the multilingual M5a/M5b/M5c sets) survives the move."""
    d = BUNDLE / "intents"
    files = sorted(d.glob("*.yaml"))
    assert files, "default bundle must declare intents/*.yaml"
    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert isinstance(data, list) and data, f
        assert all(isinstance(x, str) and x.strip() for x in data)
    # every historical intent label is declared
    assert {f.stem for f in files} >= {
        "order_status", "refund", "cancel_order", "refund_info",
        "delivery_eta", "high_value_refund", "reschedule_delivery"}


def test_bundle_erp_fixture_matches_the_historical_demo_data():
    data = json.loads((BUNDLE / "erp_fixture.json").read_text())
    assert set(data) == {"orders", "customers"}
    assert set(data["orders"]) == {"ORD-4821", "ORD-7734", "ORD-9021",
                                   "ORD-9022"}
    assert set(data["customers"]) == {"CUST-001", "CUST-002"}
    assert data["orders"]["ORD-4821"]["amount"] == 1299.0
    assert data["customers"]["CUST-001"]["phone"] == "+91-9876543210"


# --- the loaders serve bundle data through the tenant machinery ---------------

def test_intent_exemplars_flow_from_the_bundle():
    from voiceagent.intent import INTENT_EXEMPLARS, load_default_exemplars
    assert INTENT_EXEMPLARS is not None and INTENT_EXEMPLARS
    assert INTENT_EXEMPLARS == load_default_exemplars()
    # the same surface a named tenant bundle yields
    assert INTENT_EXEMPLARS == Tenant.load(BUNDLE).intent_exemplars()


def test_mockerp_fixture_loads_from_the_bundle_file():
    from voiceagent.tools import MockERP
    erp = MockERP()
    assert erp.get_order("ORD-4821")["status"] == "CONFIRMED"
    assert erp.orders_for_customer("CUST-002") == ["ORD-9021", "ORD-9022"]


def test_make_deployment_serves_the_default_bundle():
    # Byte-identity with the historical built-in deployment is pinned in
    # test_runtime_tenant.py; here we pin that the DATA comes from the bundle
    # (identity compiled from the bundle persona, knowledge from knowledge/)
    # and that the platform block composes governance + the action examples
    # derived from the builtin composed surface.
    from voiceagent.runtime import BUILTIN_GATEWAY_TOOLS, make_deployment
    from voiceagent.runtime import platform_prompt
    from voiceagent.tenant import compile_persona_block
    dep = make_deployment()
    assert dep.system_prompt.startswith(IDENTITY)
    assert dep.knowledge == KNOWLEDGE
    tenant = Tenant.load(BUNDLE)
    assert dep.system_prompt == (compile_persona_block(tenant.config.persona)
                                 + " "
                                 + platform_prompt(BUILTIN_GATEWAY_TOOLS))


# --- dependency flow: core modules no longer import demo_data ------------------

def test_core_modules_import_no_demo_data():
    for mod in ("runtime", "intent", "tools"):
        import importlib
        src = inspect.getsource(importlib.import_module(f"voiceagent.{mod}"))
        assert "from voiceagent.demo_data" not in src, mod
        assert "import demo_data" not in src, mod
