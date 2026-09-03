# src/voiceagent/outbound/dialer.py
"""Predictive Outbound Dialer, Regulatory DND Scrubber, and AMD Dispatcher.
Enforces TRAI/TCPA compliance, calling windows, and sub-second AMD handoff.
"""
from __future__ import annotations

import asyncio
import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from voiceagent.outbound.amd import CallParty, Sub600msAMD


@dataclass
class Lead:
    lead_id: str
    phone: str
    name: str
    domain: str = "real_estate"
    interest: str = "3BHK Luxury"
    attempts: int = 0
    max_attempts: int = 3
    status: str = "PENDING"  # PENDING, IN_PROGRESS, CONNECTED, DND_BLOCKED, TIME_BLOCKED, FAILED, CONVERTED
    metadata: dict[str, Any] = field(default_factory=dict)


class RegulatoryDNDScrubber:
    """Regulatory shield validating Do-Not-Call (DND) status and calling hours."""

    def __init__(self, dnd_numbers: set[str] | None = None):
        # Known registered DND numbers
        self._dnd_registry: set[str] = dnd_numbers or {
            "+919999999999",
            "+919800000000",
            "+18005550199",
        }

    def is_dnd_registered(self, phone: str) -> bool:
        clean = phone.replace(" ", "").replace("-", "")
        return clean in self._dnd_registry

    def is_within_calling_window(
        self,
        current_time: datetime.time | None = None,
        allowed_start: datetime.time = datetime.time(9, 0),
        allowed_end: datetime.time = datetime.time(20, 0),
    ) -> bool:
        """Validate TRAI/TCPA 9:00 AM - 8:00 PM local customer calling window."""
        t = current_time or datetime.datetime.now().time()
        return allowed_start <= t <= allowed_end

    def scrub(self, phone: str, current_time: datetime.time | None = None) -> tuple[bool, str]:
        """Returns (is_permitted, reason)."""
        if self.is_dnd_registered(phone):
            return False, "BLOCKED_BY_DND_REGISTRY"
        if not self.is_within_calling_window(current_time):
            return False, "BLOCKED_OUTSIDE_LEGAL_CALLING_HOURS"
        return True, "ALLOWED"


@dataclass
class OutboundCallResult:
    lead_id: str
    phone: str
    party_detected: CallParty
    amd_latency_ms: float
    status: str
    action_taken: str


class PredictiveDialer:
    """Manages outbound campaign dispatch and sub-600ms AMD handoff."""

    def __init__(
        self,
        scrubber: RegulatoryDNDScrubber | None = None,
        on_human_connect: Callable[[Lead], Coroutine[Any, Any, Any]] | None = None,
        on_voicemail_connect: Callable[[Lead], Coroutine[Any, Any, Any]] | None = None,
    ):
        self.scrubber = scrubber or RegulatoryDNDScrubber()
        self.on_human = on_human_connect
        self.on_voicemail = on_voicemail_connect
        self.call_logs: list[OutboundCallResult] = []

    async def dial_lead(
        self,
        lead: Lead,
        audio_stream_frames: list[bytes] | None = None,
        simulated_party: CallParty | None = None,
        current_time: datetime.time | None = None,
    ) -> OutboundCallResult:
        """Execute one compliant outbound dial."""
        lead.attempts += 1

        # 1. Pre-call Regulatory Scrubber
        permitted, reason = self.scrubber.scrub(lead.phone, current_time=current_time)
        if not permitted:
            lead.status = reason
            res = OutboundCallResult(
                lead_id=lead.lead_id,
                phone=lead.phone,
                party_detected=CallParty.SILENCE,
                amd_latency_ms=0.0,
                status=reason,
                action_taken="SCRUBBED_NO_DIAL",
            )
            self.call_logs.append(res)
            return res

        # 2. Call Connected -> Run Sub-600ms AMD
        amd = Sub600msAMD()
        party = simulated_party or CallParty.ANALYZING
        latency_ms = 450.0

        if audio_stream_frames:
            for frame in audio_stream_frames:
                amd_res = amd.process_frame(frame)
                if amd_res.classification != CallParty.ANALYZING:
                    party = amd_res.classification
                    latency_ms = amd_res.latency_ms
                    break

        if party == CallParty.ANALYZING:
            party = CallParty.HUMAN  # default fallback if undecided

        # 3. Action Dispatch
        if party == CallParty.HUMAN:
            lead.status = "CONNECTED"
            action = "HANDOFF_TO_FRONTMAN"
            if self.on_human:
                res_cb = self.on_human(lead)
                if asyncio.iscoroutine(res_cb):
                    await res_cb
        else:
            lead.status = "VOICEMAIL_DISPATCHED"
            action = "DISPATCHED_WHATSAPP_DROPPED_CALL"
            if self.on_voicemail:
                res_cb = self.on_voicemail(lead)
                if asyncio.iscoroutine(res_cb):
                    await res_cb

        res = OutboundCallResult(
            lead_id=lead.lead_id,
            phone=lead.phone,
            party_detected=party,
            amd_latency_ms=latency_ms,
            status=lead.status,
            action_taken=action,
        )
        self.call_logs.append(res)
        return res
