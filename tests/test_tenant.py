# tests/test_tenant.py — M6a: tenant config seam (persona/currency as DATA).
import json
from pathlib import Path

from voiceagent.tenant import TenantConfig
from voiceagent.entities import extract_entities
from voiceagent.agent import build_agent
from tests.test_agent import FakeIndex, FakeLLM

def test_load_missing_file_returns_historical_defaults(tmp_path):
    t = TenantConfig.load(tmp_path / "missing.json")
    assert t.persona.role.startswith("customer support assistant")
    # Deliberate pin update: the platform default currency is "$" (USA/UK-first
    # target market); a tenant declares its own currency in tenant.json.
    assert t.currency == "$"

def test_load_tenant_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"name": "acme-bank",
                             "persona": "a phone-banking assistant for Acme Bank",
                             "currency": "$"}))
    t = TenantConfig.load(p)
    assert t.name == "acme-bank" and t.currency == "$"
    assert "Acme Bank" in t.persona.role  # legacy flat-string persona wrapped

def test_persona_reaches_the_system_prompt():
    from voiceagent.tenant import TenantConfig as T
    agent = build_agent(FakeIndex(), FakeLLM(),
                        tenant=T(persona="a voice agent for Acme Air"))
    assert "Acme Air" in agent._system_prompt

def test_default_persona_is_neutral():
    # Deliberate pin update: the default persona is now the neutral
    # "customer support assistant" (was "...for an Indian ecommerce
    # company"), and the neutral default must compile into the prompt.
    from voiceagent.tenant import DEFAULT_PERSONA_ROLE
    assert DEFAULT_PERSONA_ROLE == "customer support assistant"
    agent = build_agent(FakeIndex(), FakeLLM())
    assert agent._system_prompt.startswith(
        "You are a customer support assistant. ")
    assert "Indian ecommerce" not in agent._system_prompt

def test_currency_parameterized_entities():
    e = extract_entities("I want a refund of $250 for my order", currency="$")
    assert e.amount == 250.0
    # Deliberate pin update: the rupee word forms are currency-scoped now, so
    # the ₹ behaviour is pinned with an explicit currency, not the default.
    e2 = extract_entities("refund of ₹25,000 please", currency="₹")
    assert e2.amount == 25000.0


def test_platform_default_currency_is_dollar():
    from voiceagent.tenant import DEFAULT_CURRENCY
    assert DEFAULT_CURRENCY == "$"


# ---------------------------------------------------------------------------
# M6b: STRUCTURED persona — compiled into the prompt so compliance rules
# (may_promise / never_say) are data, reviewable and CI-assertable.
# ---------------------------------------------------------------------------

from voiceagent.tenant import Persona

def test_structured_persona_compiles_into_prompt():
    tenant = TenantConfig(persona=Persona(
        role="a phone-banking assistant for Acme Bank",
        tone="calm and factual",
        may_promise=["a callback within 24 hours", "a ticket reference"],
        never_say=["guaranteed refund", "legal advice"],
    ))
    agent = build_agent(FakeIndex(), FakeLLM(), tenant=tenant)
    sp = agent._system_prompt
    assert "phone-banking assistant for Acme Bank" in sp
    assert "Tone: calm and factual." in sp
    assert "a callback within 24 hours" in sp
    assert "Never say or imply: guaranteed refund; legal advice." in sp

def test_compiled_prompt_forbids_unlisted_promises():
    tenant = TenantConfig(persona=Persona(may_promise=["a ticket reference"]))
    sp = agent_prompt = build_agent(FakeIndex(), FakeLLM(),
                                    tenant=tenant)._system_prompt
    assert "Never promise anything else." in sp

def test_legacy_flat_persona_key_still_loads(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"persona": "a support bot for OldCo"}))
    t = TenantConfig.load(p)
    assert t.persona.role == "a support bot for OldCo"

def test_never_say_rules_are_machine_assertable():
    """The compliance story: CI can assert forbidden claims never enter the
    compiled prompt surface."""
    tenant = TenantConfig(persona=Persona(never_say=["guaranteed refund"]))
    sp = build_agent(FakeIndex(), FakeLLM(), tenant=tenant)._system_prompt
    assert "Never say or imply: guaranteed refund." in sp


# ---------------------------------------------------------------------------
# M6b: the TENANT BUNDLE — one namespace per customer, typed surfaces
# (tenant.json / intents/*.yaml / policies.yaml / knowledge/), every surface
# optional with built-in fallback. This is the Control Plane's per-customer
# artifact, validated by scripts/validate_tenant.py.
# ---------------------------------------------------------------------------

import yaml
from voiceagent.tenant import Tenant
from voiceagent.intent import IntentClassifier

def _make_bundle(root, persona_currency="$", exemplar=None, policy=True):
    root = Path(root)
    (root / "intents").mkdir(parents=True)
    (root / "tenant.json").write_text(json.dumps({
        "name": "acme", "currency": persona_currency,
        "persona": {"role": "an assistant for Acme",
                    "never_say": ["legal advice"]}}))
    if exemplar is not None:
        (root / "intents" / "track_order.yaml").write_text(
            yaml.safe_dump(exemplar))
    if policy:
        (root / "policies.yaml").write_text(
            "escalate:\n  - fraud\norder_status:\n  allow: true\n")
    return root

def test_missing_bundle_falls_back_to_all_builtins(tmp_path):
    t = Tenant.load(tmp_path / "nope")
    assert not t.exists
    assert t.intent_exemplars() is None      # caller falls back to built-ins
    assert t.policy_file() is None
    assert t.knowledge_dir() is None
    assert t.config.currency == "$"  # platform default (deliberate change)

def test_bundle_surfaces_resolve(tmp_path):
    root = _make_bundle(tmp_path / "acme", exemplar=["where is my order"])
    t = Tenant.load(root)
    assert t.exists and t.config.name == "acme"
    assert t.intent_exemplars() == {"track_order": ["where is my order"]}
    assert t.policy_file().endswith("policies.yaml")
    assert "never_say" in (root / "tenant.json").read_text()

def test_tenant_exemplars_drive_the_classifier(tmp_path):
    root = _make_bundle(tmp_path / "acme",
                        exemplar=["where is my acme package zyx",
                                  "status of my acme order"])
    t = Tenant.load(root)
    clf = IntentClassifier(exemplars=t.intent_exemplars())
    action, score = clf.classify("where is my acme package zyx")
    assert action == "track_order" and score > 0.5
    assert clf._intents == ["track_order"]  # tenant taxonomy, not built-ins

def test_bundle_knowledge_dir(tmp_path):
    root = _make_bundle(tmp_path / "acme")
    (root / "knowledge").mkdir()
    (root / "knowledge" / "faq.md").write_text("# FAQ\nReturn in 30 days.")
    t = Tenant.load(root)
    assert t.knowledge_dir() is not None
    assert "Return in 30 days" in (Path(t.knowledge_dir()) / "faq.md").read_text()


# ---------------------------------------------------------------------------
# Sprint A1: the tenant bundle DECLARES its action vocabulary, derived from
# the surfaces it already owns — intents/*.yaml file names (the classifier
# taxonomy IS the action vocabulary) + the governed `action` declarations in
# tools.yaml — with optional info-only extras in tenant.json `actions`.
# One source per concept; no second list to keep in sync.
# ---------------------------------------------------------------------------

def test_action_vocabulary_derived_from_intents_and_tools_yaml(tmp_path):
    root = _make_bundle(tmp_path / "acme", exemplar=["what do I owe"])
    (root / "intents" / "check_balance.yaml").write_text(
        yaml.safe_dump(["what is my balance"]))
    (root / "tools.yaml").write_text(
        "tools:\n  escalate_to_human:\n    action: escalate_to_human\n"
        "  fetch_order_status:\n    action: order_status\n")
    t = Tenant.load(root)
    assert t.action_vocabulary() == ["check_balance", "escalate_to_human",
                                     "order_status", "track_order"]


def test_action_vocabulary_tenant_json_info_only_extras(tmp_path):
    root = _make_bundle(tmp_path / "acme")
    (root / "tenant.json").write_text(json.dumps({
        "name": "acme", "actions": ["check_balance"]}))
    t = Tenant.load(root)
    assert t.action_vocabulary() == ["check_balance"]


def test_action_vocabulary_none_when_bundle_declares_nothing(tmp_path):
    root = _make_bundle(tmp_path / "acme")  # no intents files, no tools.yaml
    assert Tenant.load(root).action_vocabulary() is None
    assert Tenant.load(tmp_path / "nope").action_vocabulary() is None


def test_example_acme_declares_its_vocabulary():
    acme = Tenant.load(Path(__file__).resolve().parents[1]
                       / "data" / "tenants" / "example-acme")
    vocab = acme.action_vocabulary()
    assert "order_status" in vocab and "escalate_to_human" in vocab
    # Derived from bundle data, NOT the demo e-commerce list.
    assert "recharge" not in vocab and "roaming" not in vocab
