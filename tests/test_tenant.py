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
    assert t.currency == "₹"

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

def test_default_persona_keeps_benchmark_byte_identical():
    agent = build_agent(FakeIndex(), FakeLLM())
    assert "Indian ecommerce company" in agent._system_prompt

def test_currency_parameterized_entities():
    e = extract_entities("I want a refund of $250 for my order", currency="$")
    assert e.amount == 250.0
    e2 = extract_entities("refund of ₹25,000 please")
    assert e2.amount == 25000.0  # default unchanged


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
