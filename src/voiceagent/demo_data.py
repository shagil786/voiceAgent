# src/voiceagent/demo_data.py — DEMO TENANT DATA, not core.
"""The built-in demo deployment's business data (the historical "Acme"
e-commerce demo). This module exists so orchestration core ships NO business
vocabulary: everything here is consumed ONLY as a fallback when no tenant
bundle / policy / deployment declares the corresponding data.

A real tenant never imports this module — their bundle (data/tenants/<name>/
with intents/*.yaml + tools.yaml + policies.yaml) declares its own action
vocabulary and tool contracts; see voiceagent.tenant.Tenant.action_vocabulary.
"""
from __future__ import annotations

from voiceagent.tools import ToolSpec

# The demo deployment's action vocabulary (formerly agent.DEFAULT_ACTIONS,
# where it was core code). Consumed only when neither the policy engine
# (PolicyEngine.known_actions) nor the assembly seam declares a vocabulary.
DEMO_TENANT_ACTIONS = [
    "order_status", "refund", "cancel_order", "address_change",
    "payment_declined", "recharge", "billing", "return", "replacement",
    "otp", "fraud", "account_closure", "delivery_delay", "product_info",
    "invoice", "plan_change", "roaming", "network_issue", "complaint",
    "high_value_refund", "refund_info", "delivery_eta",
]

# Demo tool contracts for intent-actions that have NO code-bound tool (they
# are policy/guard data, never executed through the ToolGateway — hence
# params=()). `facts` are the customer-visible guarantees the echo guardrail
# enforces when the customer states them; formerly agent.KEYWORD_FACTS, where
# they were core code. A tenant declares its own contracts via tools.yaml
# `facts:` on real tools; these demo entries are the no-bundle fallback.
DEMO_TENANT_CONTRACT_SPECS: dict[str, ToolSpec] = {
    "fraud": ToolSpec(params=(), facts=("block",)),
    "otp": ToolSpec(params=(), facts=("otp",)),
    "billing": ToolSpec(params=(), facts=("bill",)),
    "payment_declined": ToolSpec(params=(), facts=("declined",)),
    "recharge": ToolSpec(params=(), facts=("fail", "recharge")),
    "refund_info": ToolSpec(params=(), facts=("refund",)),
    "delivery_eta": ToolSpec(params=(), facts=("order", "delivery")),
}
