# src/voiceagent/swarm/arbiter.py
"""Consensus and Conflict Resolution Arbiter for multi-agent swarm.
Strictly evaluates priority: Regulatory Compliance > Risk > Pricing > Sales.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voiceagent.swarm.blackboard import (
    PRIORITY_COMPLIANCE,
    PRIORITY_PRICING,
    PRIORITY_RISK,
    PRIORITY_SALES,
    Proposal,
)


@dataclass
class ArbiterDecision:
    """The synthesized consensus decision executed by the Frontman and Gateway."""
    spoken_content: str
    action: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    mandatory_disclosures: list[str] = field(default_factory=list)
    vetoes_applied: list[str] = field(default_factory=list)
    sidecar_actions: list[dict[str, Any]] = field(default_factory=list)
    winning_agent: str = "frontman"


class ConsensusArbiter:
    """Deterministic arbiter resolving conflicting multi-agent proposals."""

    def arbitrate(self, proposals: list[Proposal], default_content: str = "") -> ArbiterDecision:
        if not proposals:
            return ArbiterDecision(spoken_content=default_content)

        # Separate proposals by class
        compliance_proposals = [p for p in proposals if p.priority >= PRIORITY_COMPLIANCE]
        risk_proposals = [p for p in proposals if PRIORITY_RISK <= p.priority < PRIORITY_COMPLIANCE]
        pricing_proposals = [p for p in proposals if PRIORITY_PRICING <= p.priority < PRIORITY_RISK]
        sales_proposals = [p for p in proposals if p.priority < PRIORITY_PRICING]

        vetoes: list[str] = []
        disclosures: list[str] = []
        sidecars: list[dict[str, Any]] = []

        # 1. Check Compliance Vetoes & Mandatory Disclosures
        for comp in compliance_proposals:
            if comp.veto:
                vetoes.append(f"COMPLIANCE_VETO: {comp.veto_reason or 'regulatory violation'}")
            if comp.content:
                disclosures.append(comp.content)
            if comp.metadata.get("sidecar"):
                sidecars.append(comp.metadata["sidecar"])

        # A compliance veto DISPOSES: nothing below it may execute. The
        # terminal is the real governed handoff (escalate_to_human), never a
        # synthetic action — disclosures and vetoes still reach the caller.
        compliance_vetoes = [v for v in vetoes if v.startswith("COMPLIANCE_VETO")]
        if compliance_vetoes:
            spoken = ("This needs a human colleague to proceed. "
                      + " ".join(disclosures)).strip()
            return ArbiterDecision(
                spoken_content=spoken,
                action="escalate_to_human",
                params={"reason": "; ".join(compliance_vetoes)},
                mandatory_disclosures=disclosures,
                vetoes_applied=vetoes,
                sidecar_actions=sidecars,
            )

        # 2. Check Risk/Credit Vetoes
        risk_blocked = False
        for r in risk_proposals:
            if r.veto:
                vetoes.append(f"RISK_VETO: {r.veto_reason or 'credit limit/risk breach'}")
                risk_blocked = True
            if r.metadata.get("sidecar"):
                sidecars.append(r.metadata["sidecar"])

        # If risk is blocked, we cannot execute sales offers. The terminal is
        # the real governed handoff — `risk_escalation` was a synthetic
        # action no gateway binds, i.e. an escalation dead-end.
        if risk_blocked:
            risk_lead = risk_proposals[0] if risk_proposals else None
            spoken = risk_lead.content if risk_lead else "Your request requires additional risk verification."
            blocked = next((p for p in sales_proposals if p.action), None)
            params: dict[str, Any] = {
                "reason": "; ".join(v for v in vetoes
                                    if v.startswith("RISK_VETO")),
            }
            if blocked is not None:
                params["blocked_action"] = blocked.action
                params.update(blocked.params)
            return ArbiterDecision(
                spoken_content=spoken,
                action="escalate_to_human",
                params=params,
                vetoes_applied=vetoes,
                mandatory_disclosures=disclosures,
            )

        # 3. Check Pricing Bounds
        price_adjusted = False
        active_pricing: Proposal | None = None
        pricing_blocked = False
        for pr in pricing_proposals:
            if pr.veto:
                vetoes.append(f"PRICING_VETO: {pr.veto_reason or 'below minimum floor'}")
                pricing_blocked = True
            active_pricing = pr
            if pr.metadata.get("sidecar"):
                sidecars.append(pr.metadata["sidecar"])

        # A pricing veto blocks the priced action: executing the sale below
        # the floor would violate the bound the veto exists to enforce.
        if pricing_blocked:
            return ArbiterDecision(
                spoken_content=("This needs a human colleague to proceed. "
                                + " ".join(disclosures)).strip(),
                action="escalate_to_human",
                params={"reason": "; ".join(
                    v for v in vetoes if v.startswith("PRICING_VETO"))},
                mandatory_disclosures=disclosures,
                vetoes_applied=vetoes,
                sidecar_actions=sidecars,
            )

        # 4. Resolve Primary Action and Content
        winning_proposal: Proposal | None = None

        # If a sales proposal exists, check if pricing modifies or bounds it
        if sales_proposals:
            primary_sales = sales_proposals[0]
            winning_proposal = primary_sales
            if primary_sales.metadata.get("sidecar"):
                sidecars.append(primary_sales.metadata["sidecar"])

            # If pricing agent intervened (concession curve floor enforced)
            if active_pricing and active_pricing.content:
                spoken = f"{primary_sales.content} {active_pricing.content}".strip()
                params = {**primary_sales.params, **active_pricing.params}
            else:
                spoken = primary_sales.content
                params = primary_sales.params
        elif active_pricing:
            winning_proposal = active_pricing
            spoken = active_pricing.content
            params = active_pricing.params
        elif risk_proposals:
            winning_proposal = risk_proposals[0]
            spoken = risk_proposals[0].content
            params = risk_proposals[0].params
        else:
            spoken = default_content
            params = {}

        # 5. Append Mandatory Statutory Disclosures (Regulatory Watchdog)
        if disclosures:
            spoken = f"{spoken} {' '.join(disclosures)}".strip()

        return ArbiterDecision(
            spoken_content=spoken,
            action=winning_proposal.action if winning_proposal else None,
            params=params,
            mandatory_disclosures=disclosures,
            vetoes_applied=vetoes,
            sidecar_actions=sidecars,
            winning_agent=winning_proposal.source_agent if winning_proposal else "frontman",
        )
