# src/voiceagent/swarm/blackboard.py
"""In-memory Blackboard State and Asynchronous Event Bus for concurrent
multi-agent swarm collaboration at voice speed (<200ms latency).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class CallerProfile:
    customer_id: str = "CUST-001"
    phone: str = ""
    name: str = "Valued Customer"
    authenticated: bool = False
    risk_tier: str = "standard"  # low, standard, high
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """A proposal emitted concurrently by a specialized worker agent."""
    source_agent: str
    priority: int  # Higher number = higher priority in arbitration
    action: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    veto: bool = False
    veto_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class BlackboardState:
    """Shared blackboard visible to all sub-agents and the Frontman."""
    session_id: str
    profile: CallerProfile = field(default_factory=CallerProfile)
    history: list[dict[str, str]] = field(default_factory=list)
    open_goals: list[str] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append_turn(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text, "ts": str(time.time())})


# Priority hierarchy for arbitration
PRIORITY_COMPLIANCE = 100
PRIORITY_RISK = 80
PRIORITY_PRICING = 60
PRIORITY_SALES = 40
PRIORITY_SENTIMENT = 20


class Blackboard:
    """Asynchronous blackboard event bus coordinating specialized sub-agents
    in parallel, avoiding turn-based sequential message passing."""

    def __init__(self, session_id: str, profile: CallerProfile | None = None):
        self.state = BlackboardState(
            session_id=session_id,
            profile=profile or CallerProfile(),
        )
        self._agents: dict[
            str, Callable[[BlackboardState, str], Coroutine[Any, Any, Proposal | None]]
        ] = {}

    def register_agent(
        self,
        name: str,
        handler: Callable[[BlackboardState, str], Coroutine[Any, Any, Proposal | None]],
    ) -> None:
        """Register a sub-agent handler to run on dispatched goals."""
        self._agents[name] = handler

    async def dispatch(
        self, user_text: str, timeout_s: float = 1.0
    ) -> list[Proposal]:
        """Dispatch customer input to all registered sub-agents in parallel.
        Collects all proposals within timeout_s without blocking the voice loop."""
        self.state.append_turn("customer", user_text)
        self.state.proposals.clear()

        tasks = [
            asyncio.create_task(self._safe_invoke(name, handler, user_text))
            for name, handler in self._agents.items()
        ]
        if not tasks:
            return []

        done, pending = await asyncio.wait(
            tasks, timeout=timeout_s, return_when=asyncio.ALL_COMPLETED
        )

        for p in pending:
            p.cancel()

        proposals: list[Proposal] = []
        for t in done:
            try:
                res = t.result()
                if res is not None:
                    proposals.append(res)
            except Exception:
                pass

        # Sort proposals by priority descending
        proposals.sort(key=lambda x: x.priority, reverse=True)
        self.state.proposals = proposals
        return proposals

    async def _safe_invoke(
        self,
        name: str,
        handler: Callable[[BlackboardState, str], Coroutine[Any, Any, Proposal | None]],
        user_text: str,
    ) -> Proposal | None:
        try:
            return await handler(self.state, user_text)
        except Exception:
            return None
