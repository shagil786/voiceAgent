# src/voiceagent/runtime.py — the one place the governed Orchestrator is built.
"""The governed agent runtime: a single assembly seam for the Orchestrator so
every entry point (LiveKit worker, REPL, tests) drops in the SAME brain.

Contract (non-negotiable, mirrors the limb plan): LiveKit / HTTP are transport;
`Orchestrator.handle_turn` is the ONLY brain. This module wires the frontier
brain + governed tool runner + policy engine + a Deployment's tool surface into
one `Orchestrator`, so no entry point can accidentally serve the legacy Agent
or an ungoverned swarm path.

Importable with zero heavy deps (no livekit, no llama.cpp, no network): the
frontier client is only constructed when `config_from_env()` resolves a URL, and
model loads happen lazily inside the integration adapters, not here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from voiceagent.decisionlog import DecisionLog
from voiceagent.memory import InMemoryMemory
from voiceagent.orchestrator import Deployment, Orchestrator
from voiceagent.policy import PolicyEngine, load_policies
from voiceagent.swarm.frontier import (
    FrontierAgentBridge,
    FrontierClient,
    config_from_env,
)
from voiceagent.tools import (
    DEFAULT_TOOL_SPECS,
    GovernedToolRunner,
    MockERP,
    ToolGateway,
)
from voiceagent.tenant import DEFAULT_CURRENCY, Tenant, compile_persona_block

# Default policy file + deployment name. A real deployment overrides the
# Deployment (system prompt, gateway tools, knowledge) per business; the policy
# file lives in git as the company's support/compliance artifact.
DEFAULT_POLICY_PATH = "data/policies/policies.yaml"
DEFAULT_DEPLOYMENT_NAME = "acme_support"

# Platform-level governance boilerplate for the frontier system prompt: the
# spoken-aloud brevity rule, the propose-vs-policy contract, never-invent
# rules, promise-only-what-your-tools-do, and the escalation guidance. This is
# platform code-level policy — tenant bundles supply identity/persona AROUND
# it, never instead of it.
PLATFORM_PROMPT_BASE = (
    "Be concise and warm — your replies are spoken aloud. You may propose "
    "governed actions (fetch_order_status, reschedule_delivery, cancel_order, "
    "initiate_return, escalate_to_human) — the policy layer decides; if a "
    "verdict blocks you, explain it plainly to the customer. Authenticate "
    "context comes from the session; never invent order details — fetch "
    "them. Never invent URLs, tracking links, or reference numbers: if the "
    "customer asks for a tracking link, offer to send it over WhatsApp "
    "instead of reading one out. Only promise actions that exist in your "
    "tool surface (order status, reschedule, cancel, return, refund via "
    "human approval, human handoff) — never say you are doing something you "
    "have no tool for. If the customer is upset or asks for a human agent, "
    "propose escalate_to_human with a short reason.")
_BUILTIN_IDENTITY = "You are Acme's voice support agent."

# The built-in tool surface and knowledge: what a deployment gets when no
# tenant bundle overrides them. Keys are code bindings (DEFAULT_TOOL_SPECS);
# this dict only decides what the brain may PROPOSE.
BUILTIN_GATEWAY_TOOLS: dict[str, dict] = {
    "fetch_order_status": {"action": "order_status"},
    "reschedule_delivery": {"action": "reschedule_delivery"},
    "cancel_order": {"action": "cancel_order"},
    "escalate_to_human": {
        "action": "escalate_to_human",
        "side_effects": True,
        "description": "Page a human agent to take over this call. "
                       "Provide a short reason for the handoff.",
        "parameters": {"type": "object",
                       "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]},
    },
    "initiate_return": {
        "action": "initiate_return",
        "side_effects": True,
        "description": "Request a return for a shipped or delivered "
                       "order (params: order_id, reason).",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"},
                                      "reason": {"type": "string"}},
                       "required": ["order_id", "reason"]},
    },
}
BUILTIN_KNOWLEDGE: dict[str, str] = {
    "eta": "Deliveries occur between 9:00 and 19:00 local time.",
    "cancel_policy": "Orders that already shipped cannot be cancelled.",
}

# Total injected knowledge is capped because deploy() joins it into the
# system prompt — an unbounded KB would bloat every single turn.
MAX_KNOWLEDGE_CHARS = 6000


def gateway_tools_from_yaml(path: str | Path) -> dict[str, dict]:
    """Load a tenant bundle's tools.yaml into a Deployment.gateway_tools
    surface — the DEPLOYMENT surface (what the brain may propose), NOT the
    ToolGateway.from_yaml params/preconditions shape (how tools execute).
    Fail fast: tool names are code bindings (DEFAULT_TOOL_SPECS) so an
    unknown declaration can never silently no-op, and escalate_to_human must
    stay proposeable in any declared surface — the safety valve."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or not isinstance(raw.get("tools", {}),
                                                   dict):
        raise ValueError(
            "tools.yaml: expected a top-level 'tools' mapping of "
            "tool name -> spec")
    tools: dict = raw.get("tools") or {}
    unknown = sorted(set(tools) - set(DEFAULT_TOOL_SPECS))
    if unknown:
        raise ValueError(
            f"tools.yaml: unknown tool(s) {', '.join(unknown)} — bindings "
            "are code (DEFAULT_TOOL_SPECS), declarations are data")
    if "escalate_to_human" not in tools:
        raise ValueError(
            "tools.yaml: 'escalate_to_human' must be declared — the brain "
            "must always be able to propose a human handoff")
    surface: dict[str, dict] = {}
    for name, meta in tools.items():
        if not isinstance(meta, dict) or not meta.get("action"):
            raise ValueError(f"tools.yaml: '{name}' must declare an 'action'")
        surface[name] = dict(meta)
    return surface


def _cap_knowledge(knowledge: dict[str, str]) -> dict[str, str]:
    """Sorted-id prefix of the knowledge that fits under MAX_KNOWLEDGE_CHARS.
    Whole-file granularity: a file that does not fit is dropped together with
    everything after it — a truncated FAQ could assert the opposite of the
    text it cut off."""
    capped: dict[str, str] = {}
    total = 0
    for kid in sorted(knowledge):
        text = knowledge[kid]
        if total + len(text) > MAX_KNOWLEDGE_CHARS:
            break
        capped[kid] = text
        total += len(text)
    return capped


def _bundle_gateway_tools(tenant: Tenant) -> dict[str, dict]:
    p = tenant.root / "tools.yaml"
    if p.exists():
        return gateway_tools_from_yaml(p)
    return dict(BUILTIN_GATEWAY_TOOLS)


def _bundle_knowledge(tenant: Tenant) -> dict[str, str]:
    d = tenant.knowledge_dir()
    if d is None:
        return dict(BUILTIN_KNOWLEDGE)
    files = sorted(Path(d).glob("*.md"))
    knowledge = {f.stem: f.read_text(encoding="utf-8") for f in files}
    if not knowledge:
        return dict(BUILTIN_KNOWLEDGE)
    return _cap_knowledge(knowledge)


def _resolve_tenant(tenant: str | None,
                    env: dict[str, str] | None) -> Tenant | None:
    """Explicit `tenant` arg wins, then VOICEAGENT_TENANT (same env
    precedence as config_from_env: the passed env dict replaces os.environ).
    A value containing '/' is a bundle path; otherwise it is a bundle NAME
    under data/tenants/. No value -> None (built-in deployment)."""
    e = os.environ if env is None else env
    value = tenant or e.get("VOICEAGENT_TENANT") or None
    if not value:
        return None
    root = Path(value) if "/" in value else Path("data/tenants") / value
    bundle = Tenant.load(root)
    # Tenant.load falls back to platform defaults for a MISSING root, which
    # here would silently serve default-identity + platform-wide policy — a
    # typo'd VOICEAGENT_TENANT must never widen policy, so fail fast instead.
    if not bundle.exists:
        raise ValueError(
            f"tenant bundle not found: {root} — VOICEAGENT_TENANT must name "
            "a bundle under data/tenants/ or a bundle path")
    return bundle


def make_deployment(
    tenant: "Tenant | None" = None,
    policy_path: str = DEFAULT_POLICY_PATH,
) -> Deployment:
    """Build the governed Deployment: prompt + gateway tool surface + inline
    knowledge. With a tenant bundle, identity/persona, tool surface,
    knowledge and metadata all come from data/tenants/<name>/ — onboarding a
    customer is data, not code. tenant=None reproduces the built-in Acme
    deployment byte-identically (policy_path is accepted for API symmetry;
    the policy engine is wired in build_orchestrator)."""
    if tenant is None:
        return Deployment(
            name=DEFAULT_DEPLOYMENT_NAME,
            system_prompt=_BUILTIN_IDENTITY + " " + PLATFORM_PROMPT_BASE,
            gateway_tools=dict(BUILTIN_GATEWAY_TOOLS),
            knowledge=dict(BUILTIN_KNOWLEDGE),
        )
    return Deployment(
        name=tenant.config.name,
        system_prompt=PLATFORM_PROMPT_BASE + "\n\n"
                      + compile_persona_block(tenant.config.persona),
        gateway_tools=_bundle_gateway_tools(tenant),
        knowledge=_bundle_knowledge(tenant),
        metadata={"languages": tenant.language_set(),
                  "tenant": tenant.config.name},
    )


def build_orchestrator(
    env: dict[str, str] | None = None,
    policy_path: str = DEFAULT_POLICY_PATH,
    *,
    tenant: str | None = None,
    erp: Any | None = None,
    memory: Any | None = None,
    decision_log: Any | None = None,
    deployment: Deployment | None = None,
    max_tool_rounds: int = 3,
) -> Orchestrator | None:
    """Assemble the governed Orchestrator. Returns None when no frontier brain
    is configured (VOICEAGENT_FRONTIER_URL unset) so callers fail FAST with an
    explicit message instead of crashing later on `orchestrator.handle_turn`.

    `env` overrides os.environ for config resolution (lets tests pin a stub
    frontier). `tenant` selects a tenant bundle by NAME under data/tenants/ or
    by bundle PATH; when it resolves (explicit arg, else VOICEAGENT_TENANT),
    the bundle's policies.yaml and Deployment drive the brain. `erp`/`memory`/
    `decision_log` are injectable so a real backend or a test double can be
    substituted without touching the wiring.
    """
    cfg = config_from_env(env)
    if cfg is None:
        return None

    bundle = _resolve_tenant(tenant, env)
    log = decision_log or DecisionLog()
    # The bundle's policy file IS the least-privilege artifact: undeclared
    # actions get a DENY fed back to the brain. Only a bundle that declares no
    # policy file falls back to the platform policy_path.
    policy_src = (bundle.policy_file() or policy_path) if bundle else policy_path
    # Currency is tenant data: the bundle declares it, the platform default
    # covers a no-tenant deployment. Feeds the policy reason strings.
    currency = bundle.config.currency if bundle else DEFAULT_CURRENCY
    policy = PolicyEngine(load_policies(policy_src), currency=currency)
    runner = GovernedToolRunner(
        ToolGateway(erp=erp or MockERP()), policy, decision_log=log)
    brain = FrontierAgentBridge(FrontierClient(cfg))
    orch = Orchestrator(
        brain, runner=runner, memory=memory or InMemoryMemory(),
        decision_log=log, max_tool_rounds=max_tool_rounds)
    orch.deploy(deployment or make_deployment(tenant=bundle,
                                              policy_path=policy_path))
    return orch
