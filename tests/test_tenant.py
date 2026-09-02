# tests/test_tenant.py — M6a: tenant config seam (persona/currency as DATA).
import json
from pathlib import Path

from voiceagent.tenant import TenantConfig
from voiceagent.entities import extract_entities
from voiceagent.agent import build_agent
from tests.test_agent import FakeIndex, FakeLLM

def test_load_missing_file_returns_historical_defaults(tmp_path):
    t = TenantConfig.load(tmp_path / "missing.json")
    assert t.persona.startswith("customer support assistant")
    assert t.currency == "₹"

def test_load_tenant_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"name": "acme-bank",
                             "persona": "a phone-banking assistant for Acme Bank",
                             "currency": "$"}))
    t = TenantConfig.load(p)
    assert t.name == "acme-bank" and t.currency == "$"
    assert "Acme Bank" in t.persona

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
