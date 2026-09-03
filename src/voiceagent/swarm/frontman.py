# src/voiceagent/swarm/frontman.py
"""The Fast Voice Frontman (<200ms).
Maintains conversational rhythm, handles acoustic fillers,
and interfaces asynchronously with the Deep Cognitive Swarm.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from voiceagent.swarm.arbiter import ArbiterDecision, ConsensusArbiter
from voiceagent.swarm.blackboard import Blackboard, Proposal


# Natural conversational fillers by intent category (English + Hinglish)
FILLERS: dict[str, list[str]] = {
    "inventory": [
        "Sure, let me pull up our live property inventory for you right now...",
        "Ek second, main live inventory check kar raha hu...",
    ],
    "credit": [
        "Let me calculate the optimal card rewards and APR for your spending profile...",
        "Zaroor, main aapke liye best credit limit aur rewards calculate kar raha hu...",
    ],
    "pricing": [
        "Let me check the developer's maximum approved concession on this unit...",
        "Main dekh leta hu is unit par kitna best discount mil sakta hai...",
    ],
    "general": [
        "Just a moment, let me verify those details for you...",
        "Bas ek second, main details check kar raha hu...",
    ],
}


@dataclass
class FrontmanResponse:
    spoken_text: str
    filler_text: str | None = None
    action: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    sidecars_dispatched: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


class VoiceFrontman:
    """The low-latency Voice Frontman coordinating real-time conversation
    with the Deep Cognitive Swarm."""

    def __init__(
        self,
        blackboard: Blackboard,
        arbiter: ConsensusArbiter | None = None,
        llm_client: Any | None = None,
        on_sidecar_dispatch: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.blackboard = blackboard
        self.arbiter = arbiter or ConsensusArbiter()
        self.llm = llm_client
        self.on_sidecar = on_sidecar_dispatch

    def get_filler(self, user_text: str) -> str:
        """Select a natural acoustic filler in < 5ms to maintain voice pacing."""
        low = user_text.lower()
        if any(w in low for w in ["flat", "bhk", "property", "apartment", "location", "square feet"]):
            return FILLERS["inventory"][0]
        if any(w in low for w in ["card", "credit", "cibil", "loan", "apr", "interest", "limit"]):
            return FILLERS["credit"][0]
        if any(w in low for w in ["price", "discount", "offer", "concession", "budget", "kam karo"]):
            return FILLERS["pricing"][0]
        return FILLERS["general"][0]

    async def handle_turn(
        self,
        user_text: str,
        authenticated: bool = False,
        need_filler: bool = True,
    ) -> FrontmanResponse:
        t0 = time.time()
        filler: str | None = None
        if need_filler:
            filler = self.get_filler(user_text)

        # Asynchronously dispatch to the Deep Cognitive Swarm
        proposals = await self.blackboard.dispatch(user_text, timeout_s=1.5)

        # Arbitrate between sub-agent proposals
        decision = self.arbiter.arbitrate(
            proposals,
            default_content="I understand. How else can I assist you today?",
        )

        # Trigger any omnichannel sidecars (WhatsApp, payment link)
        for sidecar in decision.sidecar_actions:
            if self.on_sidecar:
                if asyncio.iscoroutinefunction(self.on_sidecar):
                    await self.on_sidecar(sidecar)
                else:
                    self.on_sidecar(sidecar)

        latency_ms = (time.time() - t0) * 1000.0
        return FrontmanResponse(
            spoken_text=decision.spoken_content,
            filler_text=filler,
            action=decision.action,
            params=decision.params,
            sidecars_dispatched=decision.sidecar_actions,
            latency_ms=latency_ms,
        )
