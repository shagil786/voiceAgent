# src/voiceagent/swarm/agents/negotiator.py
"""Bounded Mathematical Concession Engine.
Replaces unconstrained LLM discounting with a strict concession curve:
List Price -> Value Resistance -> Soft Concession -> Hard Floor (requires token).
"""
from __future__ import annotations

from typing import Any

from voiceagent.swarm.blackboard import (
    PRIORITY_PRICING,
    BlackboardState,
    Proposal,
)


class BoundedNegotiator:
    """Mathematical concession agent preventing margin give-away."""

    def __init__(
        self,
        list_price: float = 15000000.0,  # 1.50 Cr
        soft_floor: float = 14800000.0,  # 1.48 Cr (free covered parking)
        hard_floor: float = 14600000.0,  # 1.46 Cr (absolute floor, requires instant token)
    ):
        self.list_price = list_price
        self.soft_floor = soft_floor
        self.hard_floor = hard_floor

    async def handle_turn(self, state: BlackboardState, user_text: str) -> Proposal | None:
        low = user_text.lower()
        keywords = ["discount", "kam", "reduce", "budget", "expensive", "concession", "price", "cr", "lakh", "cheap", "cheaper", "deal", "offer"]
        if not any(k in low for k in keywords):
            return None

        # Check negotiation history turns
        rounds = sum(1 for t in state.history if any(k in t.get("text", "").lower() for k in keywords))

        if rounds == 0:
            # Stage 1: Resist concession, anchor on list price and scarcity
            return Proposal(
                source_agent="negotiator",
                priority=PRIORITY_PRICING,
                action="anchor_list_price",
                params={"offered_price": self.list_price},
                content=(
                    f"The price is anchored at ₹{self.list_price/1e7:.2f} Cr given the prime high-floor corner position. "
                    "However, if we finalize today, I can request the management to waive the clubhouse fee."
                ),
            )
        elif rounds == 1:
            # Stage 2: Soft concession (Free parking value-add)
            return Proposal(
                source_agent="negotiator",
                priority=PRIORITY_PRICING,
                action="soft_concession",
                params={"offered_price": self.soft_floor, "perk": "free_covered_parking"},
                content=(
                    f"I can step down to ₹{self.soft_floor/1e7:.2f} Cr and bundle a reserved covered parking bay "
                    "valued at ₹5 Lakhs at zero cost."
                ),
                metadata={"sidecar": {"type": "whatsapp_doc", "asset": "parking_allocation_letter"}},
            )
        else:
            # Stage 3: Hard Floor — requires immediate token deposit
            return Proposal(
                source_agent="negotiator",
                priority=PRIORITY_PRICING,
                action="hard_floor_close",
                params={"offered_price": self.hard_floor, "requires_token": True, "token_amount": 100000.0},
                content=(
                    f"The absolute floor approved by the developer is ₹{self.hard_floor/1e7:.2f} Cr, strictly conditional "
                    "on locking in your ₹1 Lakh token deposit right now on this call."
                ),
                metadata={"sidecar": {"type": "payment_link", "amount": 100000.0, "purpose": "token_deposit"}},
            )
