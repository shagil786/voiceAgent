# tests/test_runtime_tenant.py — the tenant bundle compiles into the governed
# Deployment. Guards two contracts: (1) with NO tenant configured the built-in
# Acme deployment is byte-identical to the pre-bundle behavior, (2) with a
# bundle, identity / tool surface / knowledge / policy are DATA — onboarding a
# customer must not touch code. Fully offline: stub FrontierClient only.
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from voiceagent.policy import PolicyContext
from voiceagent.runtime import (
    MAX_KNOWLEDGE_CHARS,
    PLATFORM_PROMPT_BASE,
    build_orchestrator,
    make_deployment,
)
from voiceagent.tenant import Persona, Tenant, compile_persona_block

ROOT = Path(__file__).resolve().parents[1]
ACME = ROOT / "data" / "tenants" / "example-acme"
VALIDATOR = ROOT / "scripts" / "validate_tenant.py"
FRONTIER_URL = {"VOICEAGENT_FRONTIER_URL": "https://fake/v1"}

# The pre-tenant built-in prompt, pinned byte-for-byte: the no-tenant path must
# keep serving exactly what shipped before bundles existed.
BUILTIN_ACME_PROMPT = (
    "You are Acme's voice support agent. Be concise and warm — your "
    "replies are spoken aloud. You may propose governed actions "
    "(fetch_order_status, reschedule_delivery, cancel_order, "
    "initiate_return, escalate_to_human) — the policy layer decides; "
    "if a verdict blocks you, explain it plainly to the customer. "
    "Authenticate context comes from the session; never invent order "
    "details — fetch them. Never invent URLs, tracking links, or "
    "reference numbers: if the customer asks for a tracking link, "
    "offer to send it over WhatsApp instead of reading one out. "
    "Only promise actions that exist in your tool surface (order "
    "status, reschedule, cancel, return, refund via human approval, "
    "human handoff) — never say you are doing something you have no "
    "tool for. If the customer is upset or asks for a human agent, "
    "propose escalate_to_human with a short reason."
)


class _FakeLLM:
    """Stub FrontierClient: first reply proposes one governed tool call, then a
    text reply closes the turn (same two-step pattern as tests/test_runtime.py)."""

    def __init__(self, tool_name="fetch_order_status", arguments=None):
        self.calls = []
        self._n = 0
        self._tool_name = tool_name
        self._arguments = arguments or {}

    def chat(self, messages, tools=None, tool_choice="auto",
             temperature=0.4, max_tokens=512):
        from voiceagent.swarm.frontier import FrontierReply, FrontierToolCall
        self.calls.append(messages)
        self._n += 1
        if tools and self._n == 1:
            return FrontierReply(
                content=None,
                tool_calls=[FrontierToolCall(
                    id="c1", name=self._tool_name,
                    arguments=dict(self._arguments))],
                model="fake", latency_s=0.001, raw={})
        return FrontierReply(content="Done.", tool_calls=[], model="fake",
                             latency_s=0.001, raw={})


# --- 1. no tenant configured => byte-identical -------------------------------

def test_no_tenant_deployment_is_byte_identical():
    dep = make_deployment()
    assert dep.name == "acme_support"
    assert dep.system_prompt == BUILTIN_ACME_PROMPT
    assert set(dep.gateway_tools) == {"fetch_order_status", "reschedule_delivery",
                                      "cancel_order", "escalate_to_human",
                                      "initiate_return"}
    assert set(dep.knowledge) == {"eta", "cancel_policy"}
    assert dep.metadata == {}


def test_platform_prompt_composes_builtin_identity():
    # The built-in prompt is the platform governance base + the Acme identity
    # sentence — composition, not a second prompt source.
    dep = make_deployment()
    assert dep.system_prompt == ("You are Acme's voice support agent. "
                                 + PLATFORM_PROMPT_BASE)


# --- 2. example-acme bundle drives the Deployment -----------------------------

def test_example_acme_bundle_drives_the_deployment():
    dep = make_deployment(tenant=Tenant.load(ACME))
    assert dep.name == "example-acme"
    assert dep.system_prompt.startswith(PLATFORM_PROMPT_BASE)
    assert ("a customer-support voice assistant for Acme (example tenant)"
            in dep.system_prompt)
    assert "Tone: warm and concise." in dep.system_prompt
    assert ("You may promise exactly: a ticket reference; a callback within "
            "24 hours. Never promise anything else.") in dep.system_prompt
    assert "Never say or imply: guaranteed refund; legal advice." in \
        dep.system_prompt
    assert set(dep.gateway_tools) == {"fetch_order_status", "escalate_to_human"}
    assert "support-basics" in dep.knowledge
    assert dep.metadata["languages"] == ["en", "es"]
    assert dep.metadata["tenant"] == "example-acme"


def test_example_acme_via_build_orchestrator_tenant_arg(monkeypatch):
    monkeypatch.chdir(ROOT)  # bare bundle names resolve under data/tenants/
    orch = build_orchestrator(env=dict(FRONTIER_URL), tenant="example-acme")
    orch.brain.client = _FakeLLM()
    dep = orch._deployment
    assert dep.name == "example-acme"
    assert "example tenant" in dep.system_prompt
    assert set(dep.gateway_tools) == {"fetch_order_status", "escalate_to_human"}
    res = orch.handle_turn("s1", "where is my order ORD-1", authenticated=True)
    assert res.reply and res.actions[0]["tool"] == "fetch_order_status"


# --- frontier persona compiler ------------------------------------------------

def test_compile_persona_block_frontier_shape():
    block = compile_persona_block(Persona(
        role="a voice agent for Acme", tone="warm",
        may_promise=["a ticket"], never_say=["legal advice"]))
    assert block == ("You are a voice agent for Acme. Tone: warm. "
                     "You may promise exactly: a ticket. "
                     "Never promise anything else. "
                     "Never say or imply: legal advice.")
    # The ACTION-line tail is the local-LLM platform instruction (agent.py);
    # the frontier prompt must never inherit it.
    assert "ACTION:" not in block


def test_compile_persona_block_minimal_persona():
    assert compile_persona_block(Persona(role="a bot")) == "You are a bot."


# --- 3/4. tools.yaml validation fails fast ------------------------------------

def _bundle_with_tools_yaml(root: Path, body: str) -> Path:
    root.mkdir(parents=True)
    (root / "tenant.json").write_text(json.dumps({"name": "bundle-b"}))
    (root / "tools.yaml").write_text(body)
    return root


def test_unknown_tool_name_fails_fast(tmp_path):
    root = _bundle_with_tools_yaml(
        tmp_path / "b1",
        "tools:\n  grant_refund:\n    action: refund\n")
    with pytest.raises(ValueError, match="unknown tool"):
        make_deployment(tenant=Tenant.load(root))


def test_missing_escalate_valve_fails_fast(tmp_path):
    root = _bundle_with_tools_yaml(
        tmp_path / "b2",
        "tools:\n  fetch_order_status:\n    action: order_status\n")
    with pytest.raises(ValueError, match="escalate_to_human"):
        make_deployment(tenant=Tenant.load(root))


# --- 5. knowledge cap ----------------------------------------------------------

def test_knowledge_cap_drops_beyond_cap_in_sorted_order(tmp_path):
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "a.md").write_text("a" * 4000)
    (tmp_path / "knowledge" / "b.md").write_text("b" * MAX_KNOWLEDGE_CHARS)
    dep = make_deployment(tenant=Tenant.load(tmp_path))
    # Sorted-id prefix fits under the cap; b.md is dropped whole.
    assert set(dep.knowledge) == {"a"}
    assert len(dep.knowledge["a"]) <= MAX_KNOWLEDGE_CHARS


# --- 6. policy comes from the bundle ------------------------------------------

def test_bundle_policy_drives_the_policy_engine():
    orch = build_orchestrator(env=dict(FRONTIER_URL), tenant="example-acme")
    pol = orch.runner.policy
    assert pol.policies["complaint"]["escalate_when"] == {"frustrated": True}
    assert pol.evaluate(
        "complaint", PolicyContext(signals={"frustrated": True})).verdict == \
        "ESCALATE"
    # Least privilege: example-acme declares only order_status + complaint.
    assert pol.evaluate("reschedule_delivery").verdict == "DENY"


def test_least_privilege_deny_fed_back_to_brain(tmp_path, monkeypatch):
    # Bundle with the example-acme policy surface (order_status + complaint
    # only) but NO tools.yaml, so the built-in tool surface still exposes
    # reschedule_delivery and the proposal reaches the governed runner.
    monkeypatch.chdir(ROOT)
    root = tmp_path / "least-priv"
    root.mkdir()
    (root / "tenant.json").write_text(json.dumps({"name": "least-priv"}))
    (root / "policies.yaml").write_text(
        "escalate:\n  - fraud\n\n"
        "order_status:\n  allow: true\n\n"
        "complaint:\n  allow: true\n  escalate_when:\n    frustrated: true\n")
    orch = build_orchestrator(
        env=dict(FRONTIER_URL), tenant=str(root),
        deployment=make_deployment())  # built-in surface: reschedule proposeable
    orch.brain.client = _FakeLLM(
        tool_name="reschedule_delivery",
        arguments={"order_id": "ORD-1", "new_date": "2026-09-10"})
    res = orch.handle_turn("s1", "move my delivery to friday",
                           authenticated=True)
    assert res.actions and res.actions[0]["tool"] == "reschedule_delivery"
    assert res.actions[0]["verdict"] == "DENY"
    assert any("least privilege" in r for r in res.actions[0]["reasons"])


# --- 7. VOICEAGENT_TENANT env seam ---------------------------------------------

def test_voiceagent_tenant_env_dict_seam():
    orch = build_orchestrator(env={"VOICEAGENT_FRONTIER_URL": "https://fake/v1",
                                   "VOICEAGENT_TENANT": "example-acme"})
    assert orch._deployment.name == "example-acme"
    assert orch._deployment.metadata["tenant"] == "example-acme"


def test_voiceagent_tenant_os_environ_seam(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("VOICEAGENT_FRONTIER_URL", "https://fake/v1")
    monkeypatch.setenv("VOICEAGENT_TENANT", "example-acme")
    orch = build_orchestrator()
    assert orch._deployment.name == "example-acme"


# --- 8. the CI gate ------------------------------------------------------------

def test_validate_tenant_gate_passes_example_acme():
    r = subprocess.run([sys.executable, str(VALIDATOR), str(ACME)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_validate_tenant_gate_rejects_unknown_tool(tmp_path):
    root = _bundle_with_tools_yaml(
        tmp_path / "bad",
        "tools:\n  not_a_real_tool:\n    action: order_status\n")
    r = subprocess.run([sys.executable, str(VALIDATOR), str(root)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1
    assert "unknown tool" in r.stdout
