# src/voiceagent/swarm/agents/closer.py
"""Commercial Sales Closer Agent for Real Estate and Financial Origination.
Handles inventory lookup, rewards math, and appointment scheduling.
"""
from __future__ import annotations

from typing import Any

from voiceagent.swarm.blackboard import (
    PRIORITY_SALES,
    BlackboardState,
    Proposal,
)


class CommercialSalesCloser:
    """Specialized closer agent for high-ticket commercial transactions."""

    async def handle_turn(self, state: BlackboardState, user_text: str) -> Proposal | None:
        low = user_text.lower()

        # Real Estate inventory lookup
        if any(w in low for w in ["3bhk", "2bhk", "apartment", "bandra", "flat"]):
            return Proposal(
                source_agent="sales_closer",
                priority=PRIORITY_SALES,
                action="recommend_property",
                params={"unit": "3BHK-1402", "carpet_area": "1250 sqft", "location": "Bandra West"},
                content=(
                    "I have an exclusive high-floor 3BHK unit in Bandra West with 1,250 sq.ft carpet area, "
                    "panoramic sea views, and possession scheduled for December 2026."
                ),
                metadata={
                    "sidecar": {
                        "type": "whatsapp_doc",
                        "title": "3BHK Luxury Floor Plan",
                        "asset": "bandra_west_3bhk_floorplan.pdf",
                    }
                },
            )

        # Site visit booking
        if any(w in low for w in ["visit", "see", "dekhna", "saturday", "sunday", "tomorrow"]):
            return Proposal(
                source_agent="sales_closer",
                priority=PRIORITY_SALES,
                action="schedule_site_visit",
                params={"location": "Bandra Experience Centre", "slot": "Saturday 11:00 AM"},
                content="I can reserve a priority VIP site visit for you this Saturday at 11:00 AM with our senior project architect.",
                metadata={
                    "sidecar": {
                        "type": "calendar_invite",
                        "event": "VIP Site Visit: Bandra West Tower",
                        "time": "Saturday 11:00 AM",
                    }
                },
            )

        # Fintech card calculation
        if any(w in low for w in ["card", "credit", "cashback", "spends", "amazon"]):
            return Proposal(
                source_agent="sales_closer",
                priority=PRIORITY_SALES,
                action="recommend_credit_card",
                params={"card": "HDFC Millennia", "annual_cashback": 24000.0},
                content=(
                    "Based on your monthly spends, the Millennia card delivers 5% direct cashback on Amazon and Swiggy, "
                    "yielding approximately ₹24,000 in direct yearly savings."
                ),
                metadata={
                    "sidecar": {
                        "type": "whatsapp_doc",
                        "title": "Rewards & Cashback Comparison Table",
                        "asset": "millennia_rewards_matrix.pdf",
                    }
                },
            )

        return None
