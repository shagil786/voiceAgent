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
    """Factory helper to create specialized agents across multiple industries.

    Specs are loaded from the vertical packs on disk via
    :func:`voiceagent.packs.load_vertical` (imported lazily to avoid a
    circular import — packs builds the SpecialistSpec objects defined here).

    Packed verticals ignore ``custom_overrides``; overrides apply only to
    the generic fallback spec built for unknown domain ids.
    """
    from voiceagent.packs import load_vertical

    try:
        return DomainSpecialist(spec=load_vertical(domain))
    except FileNotFoundError:
        pass

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
