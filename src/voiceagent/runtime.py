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
from voiceagent.tools import GovernedToolRunner, MockERP, ToolGateway

# Default policy file + deployment name. A real deployment overrides the
# Deployment (system prompt, gateway tools, knowledge) per business; the policy
# file lives in git as the company's support/compliance artifact.
DEFAULT_POLICY_PATH = "data/policies/policies.yaml"
DEFAULT_DEPLOYMENT_NAME = "acme_support"


def make_deployment(
    name: str = DEFAULT_DEPLOYMENT_NAME,
    policy_path: str = DEFAULT_POLICY_PATH,
) -> Deployment:
    """Build the governed Deployment: prompt + gateway tool surface + inline
    knowledge. This is the per-business drop-in; everything here is data, not
    code, so onboarding a tenant = editing these values (or loading a bundle)."""
    return Deployment(
        name=name,
        system_prompt=(
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
            "propose escalate_to_human with a short reason."),
        gateway_tools={
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
        },
        knowledge={
            "eta": "Deliveries occur between 9:00 and 19:00 local time.",
            "cancel_policy": "Orders that already shipped cannot be cancelled.",
        },
    )


def build_orchestrator(
    env: dict[str, str] | None = None,
    policy_path: str = DEFAULT_POLICY_PATH,
    *,
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
    frontier). `erp`/`memory`/`decision_log` are injectable so a real backend
    or a test double can be substituted without touching the wiring.
    """
    cfg = config_from_env(env)
    if cfg is None:
        return None

    log = decision_log or DecisionLog()
    policy = PolicyEngine(load_policies(policy_path))
    runner = GovernedToolRunner(
        ToolGateway(erp=erp or MockERP()), policy, decision_log=log)
    brain = FrontierAgentBridge(FrontierClient(cfg))
    orch = Orchestrator(
        brain, runner=runner, memory=memory or InMemoryMemory(),
        decision_log=log, max_tool_rounds=max_tool_rounds)
    orch.deploy(deployment or make_deployment(policy_path=policy_path))
    return orch
