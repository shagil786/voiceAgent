# src/voiceagent/swarm/specialist.py
"""Domain-Agnostic Pluggable Specialist Agent Framework.
Allows the Autonomous Swarm to run ANY vertical or business domain:
Real Estate, B2B SaaS, Automotive, Insurance, Healthcare, or custom trained domains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from voiceagent.swarm.blackboard import (
    PRIORITY_SALES,
    BlackboardState,
    Proposal,
)


@dataclass
class SpecialistTool:
    """A tool declared for the specialist agent with standard JSON Schema."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any] | None = None


@dataclass
class SpecialistSpec:
    """Declarative specification for ANY business vertical or domain."""
    domain_id: str  # e.g., "automotive", "b2b_saas", "real_estate", "insurance"
    name: str
    role_description: str
    system_prompt: str
    catalog: list[dict[str, Any]] = field(default_factory=list)
    tools: list[SpecialistTool] = field(default_factory=list)
    statutory_disclosures: list[str] = field(default_factory=list)
    concession_curve: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class DomainSpecialist:
    """A fully configurable, domain-agnostic specialist agent.
    Can be trained, configured, or prompted for ANY commercial vertical.
    """

    def __init__(
        self,
        spec: SpecialistSpec,
        llm_client: Any | None = None,
        custom_handler: Callable[[BlackboardState, str], Coroutine[Any, Any, Proposal | None]] | None = None,
    ):
        self.spec = spec
        self.llm = llm_client
        self._custom_handler = custom_handler

    async def handle_turn(self, state: BlackboardState, user_text: str) -> Proposal | None:
        """Handle turn by evaluating domain catalog, tools, or custom handler."""
        if self._custom_handler:
            return await self._custom_handler(state, user_text)

        low = user_text.lower()

        # Match against domain catalog items (if provided)
        matched_item: dict[str, Any] | None = None
        for item in self.spec.catalog:
            keywords = item.get("keywords", [])
            name = item.get("name", "").lower()
            if name in low or any(k.lower() in low for k in keywords):
                matched_item = item
                break

        if matched_item:
            item_name = matched_item.get("name", "item")
            item_desc = matched_item.get("description", "")
            item_price = matched_item.get("price", "")
            sidecar = matched_item.get("sidecar")

            spoken = f"I recommend our {item_name}. {item_desc}"
            if item_price:
                spoken += f" It is priced at {item_price}."

            return Proposal(
                source_agent=self.spec.domain_id,
                priority=PRIORITY_SALES,
                action="recommend_domain_item",
                params={"item_id": matched_item.get("id"), "item": matched_item},
                content=spoken.strip(),
                metadata={"sidecar": sidecar} if sidecar else {},
            )

        return None


# ---------------------------------------------------------------------------
# Pre-built Domain Factory Templates for instant instantiation
# ---------------------------------------------------------------------------

def create_domain_specialist(domain: str, **custom_overrides) -> DomainSpecialist:
    """Factory helper to create specialized agents across multiple industries."""
    if domain == "luxury_automotive":
        spec = SpecialistSpec(
            domain_id="luxury_automotive",
            name="Automotive Concierge",
            role_description="Assists with test-drive bookings, EV range specs, and financing.",
            system_prompt="You are a senior product specialist for luxury electric vehicles.",
            catalog=[
                {
                    "id": "EV-SUV-01",
                    "name": "Apex e-SUV GT",
                    "keywords": ["suv", "electric", "apex", "test drive", "ev"],
                    "description": "Features dual-motor all-wheel drive, 620 km WLTP range, and 0-100 in 3.6s.",
                    "price": "₹78.5 Lakhs",
                    "sidecar": {"type": "whatsapp_doc", "title": "Apex e-SUV Brochure", "asset": "apex_gt_specsheet.pdf"},
                }
            ],
            statutory_disclosures=["FAME-II subsidy subject to state transport registration terms."],
        )
    elif domain == "b2b_saas":
        spec = SpecialistSpec(
            domain_id="b2b_saas",
            name="SaaS Enterprise Account Executive",
            role_description="Enterprise software qualification, security compliance, and seat licensing.",
            system_prompt="You are an enterprise account executive for cloud security software.",
            catalog=[
                {
                    "id": "PLAN-ENT",
                    "name": "Enterprise Shield Plan",
                    "keywords": ["enterprise", "security", "saml", "sso", "seats", "annual"],
                    "description": "Includes 24/7 dedicated SOC monitoring, custom SAML/SSO, and SOC2 compliance.",
                    "price": "₹12,000 per user/year",
                    "sidecar": {"type": "whatsapp_doc", "title": "SOC2 Security Whitepaper", "asset": "soc2_type2_report.pdf"},
                }
            ],
            statutory_disclosures=["Annual contracts billed annually upfront with 99.95% uptime SLA guarantee."],
        )
    elif domain == "insurance":
        spec = SpecialistSpec(
            domain_id="insurance",
            name="Health & Term Insurance Advisor",
            role_description="Underwrites policy limits, pre-existing condition checks, and premium quotes.",
            system_prompt="You are a licensed insurance advisory agent.",
            catalog=[
                {
                    "id": "HEALTH-1CR",
                    "name": "Comprehensive 1 Crore Health Cover",
                    "keywords": ["health", "medical", "insurance", "hospital", "claim", "cover"],
                    "description": "Covers zero room rent capping, global emergency coverage, and restore benefits.",
                    "price": "₹1,450/month",
                    "sidecar": {"type": "whatsapp_doc", "title": "Policy Benefit Breakdown", "asset": "health_1cr_benefits.pdf"},
                }
            ],
            statutory_disclosures=["Insurance is the subject matter of solicitation. IRDAI Reg No. 129."],
        )
    else:
        # Generic pluggable domain
        spec = SpecialistSpec(
            domain_id=domain,
            name=custom_overrides.get("name", f"{domain.capitalize()} Specialist"),
            role_description=custom_overrides.get("role", "Domain specialist agent"),
            system_prompt=custom_overrides.get("prompt", "You are a professional specialist."),
            catalog=custom_overrides.get("catalog", []),
            statutory_disclosures=custom_overrides.get("disclosures", []),
        )

    return DomainSpecialist(spec=spec)
