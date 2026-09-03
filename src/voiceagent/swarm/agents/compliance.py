# src/voiceagent/swarm/agents/compliance.py
"""Regulatory Compliance & Statutory Consent Watchdog.
Guarantees mandatory legal disclosures on banking and real estate transactions.
"""
from __future__ import annotations

from typing import Any

from voiceagent.swarm.blackboard import (
    PRIORITY_COMPLIANCE,
    BlackboardState,
    Proposal,
)


class ComplianceWatchdog:
    """Monitors live calls and strictly enforces statutory disclosures."""

    def __init__(
        self,
        rera_number: str = "P51800031245",
        banking_apr_disclosure: str = "Standard interest rate is 14.5% APR with zero annual fee for year one.",
    ):
        self.rera_number = rera_number
        self.banking_apr = banking_apr_disclosure

    async def handle_turn(self, state: BlackboardState, user_text: str) -> Proposal | None:
        low = user_text.lower()
        disclosed = state.metadata.get("disclosures_given", set())

        # If call mentions booking or closing or token payment, check compliance
        if any(k in low for k in ["token", "book", "deposit", "buy", "deal", "card", "apply"]):
            # Real Estate disclosure
            if "rera" not in disclosed and any(k in low for k in ["flat", "bhk", "property", "token", "book"]):
                disclosed.add("rera")
                state.metadata["disclosures_given"] = disclosed
                return Proposal(
                    source_agent="compliance",
                    priority=PRIORITY_COMPLIANCE,
                    action="statutory_disclosure",
                    content=f"Please note this project is registered under MahaRERA No. {self.rera_number}.",
                    metadata={"sidecar": {"type": "whatsapp_doc", "asset": "rera_certificate"}},
                )

            # Banking disclosure
            if "apr" not in disclosed and any(k in low for k in ["card", "loan", "apply", "interest"]):
                disclosed.add("apr")
                state.metadata["disclosures_given"] = disclosed
                return Proposal(
                    source_agent="compliance",
                    priority=PRIORITY_COMPLIANCE,
                    action="statutory_disclosure",
                    content=f"Statutory disclosure: {self.banking_apr}",
                )

        return None
