# src/voiceagent/outbound/dialer.py
"""Predictive Outbound Dialer, Regulatory DND Scrubber, and AMD Dispatcher.
Enforces TRAI/TCPA compliance, calling windows, and sub-second AMD handoff.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from voiceagent.outbound.amd import CallParty, Sub600msAMD

logger = logging.getLogger(__name__)

# Repo root anchor (src/voiceagent/outbound/ -> parents[3]) so the
# jurisdiction data file resolves regardless of process cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_JURISDICTIONS_FILE = _REPO_ROOT / "data" / "jurisdictions.yaml"

_FALLBACK_WINDOW = (datetime.time(9, 0), datetime.time(20, 0))


def _parse_hhmm(raw: object) -> datetime.time | None:
    """'HH:MM' -> time, None when malformed (never raise on operator data)."""
    try:
        hour, _, minute = str(raw).partition(":")
        return datetime.time(int(hour), int(minute))
    except (ValueError, TypeError):
        return None


def _jurisdiction_file_data() -> tuple[dict, dict]:
    """(windows, seeds) from data/jurisdictions.yaml — ({}, {}) on any
    failure (missing file, bad YAML, bad rows): operator data must never
    take the dialer down; the verified code table below stays the floor."""
    try:
        import yaml
        raw = yaml.safe_load(_JURISDICTIONS_FILE.read_text(
            encoding="utf-8")) or {}
    except Exception:
        logger.warning("jurisdictions.yaml unreadable; using built-in "
                       "verified table only", exc_info=True)
        return {}, {}
    if not isinstance(raw, dict):
        return {}, {}
    windows, seeds = {}, {}
    for code, entry in (raw.get("jurisdictions") or {}).items():
        if not isinstance(entry, dict):
            continue
        start = _parse_hhmm(entry.get("window_start"))
        end = _parse_hhmm(entry.get("window_end"))
        if start is None or end is None:
            logger.warning("jurisdictions.yaml: %r has a malformed window; "
                           "skipped", code)
            continue
        windows[str(code)] = (start, end)
        seed = entry.get("seed_dnd") or []
        seeds[str(code)] = {str(n) for n in seed if n}
    return windows, seeds


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
    """Regulatory shield validating Do-Not-Call (DND) status and calling hours.

    Jurisdiction model: calling windows and seed DND lists are per-jurisdiction
    CONFIGURATION, not code. CALLING_WINDOWS carries the windows this platform
    has actually operated under (IN TRAI / US TCPA — both 09:00-20:00 local),
    EXTENDED by data/jurisdictions.yaml (file wins): operators add a
    jurisdiction with a YAML edit after verifying local law — never code.
    Any other jurisdiction MUST be configured explicitly by the operator;
    the platform deliberately does NOT invent windows for jurisdictions it
    has not verified. Default behavior (no country_code) is byte-identical
    to the pre-configuration behavior."""

    # Jurisdiction code -> (window_start, window_end), local customer time.
    # ONLY verified jurisdictions belong in this table. TRAI (India) and
    # TCPA (US) both use 09:00-20:00; do not add entries without a source.
    CALLING_WINDOWS: dict[str, tuple[datetime.time, datetime.time]] = {
        "IN": (datetime.time(9, 0), datetime.time(20, 0)),   # TRAI
        "US": (datetime.time(9, 0), datetime.time(20, 0)),   # TCPA
    }

    # Seed DND entries by jurisdiction (demo/known-registered numbers the
    # platform has actually seen). Operators supply the real registry per
    # deployment via dnd_numbers=.
    SEED_DND: dict[str, set[str]] = {
        "IN": {"+919999999999", "+919800000000"},
        "US": {"+18005550199"},
    }

    @classmethod
    def _windows(cls) -> dict[str, tuple[datetime.time, datetime.time]]:
        """Verified code table, extended by data/jurisdictions.yaml (file
        wins on conflict — the operator verified local law after the code
        shipped). Operators add jurisdictions with a YAML edit, never code."""
        file_windows, _ = _jurisdiction_file_data()
        return {**cls.CALLING_WINDOWS, **file_windows}

    @classmethod
    def _seeds(cls) -> dict[str, set[str]]:
        """Seed DND merged the same way (file extends the code seeds)."""
        _, file_seeds = _jurisdiction_file_data()
        merged = {k: set(v) for k, v in cls.SEED_DND.items()}
        for code, seed in file_seeds.items():
            merged.setdefault(code, set()).update(seed)
        return merged

    def __init__(
        self,
        dnd_numbers: set[str] | None = None,
        country_code: str | None = None,
        allowed_start: datetime.time | None = None,
        allowed_end: datetime.time | None = None,
    ):
        """dnd_numbers replaces the seed registry entirely when given.
        country_code selects seed DND + calling window from the verified
        tables (unknown code = empty seed + 09:00-20:00 fallback + a warning
        that the jurisdiction is unverified). Explicit allowed_start/end
        override the table window (operator-verified custom windows)."""
        if dnd_numbers is not None:
            self._dnd_registry: set[str] = set(dnd_numbers)
        elif country_code:
            seeds = self._seeds().get(country_code, set())
            self._dnd_registry = set(seeds)
            if not seeds:
                logger.warning(
                    "unverified jurisdiction %r: no seed DND registry; "
                    "supply dnd_numbers= for production use", country_code)
        else:
            # No jurisdiction given: historical default = union of all seeds
            # (byte-identical to the pre-configuration hardcoded registry).
            self._dnd_registry = set().union(*self.SEED_DND.values())
        self.country_code = country_code
        if allowed_start is not None or allowed_end is not None:
            # Partial override: fill the missing bound from the jurisdiction
            # table (or the historical default), so callers can shift just one
            # bound without silently dropping the other.
            base = self._windows().get(country_code or "", _FALLBACK_WINDOW)
            self._window = (
                allowed_start if allowed_start is not None else base[0],
                allowed_end if allowed_end is not None else base[1],
            )
        elif country_code in self._windows():
            self._window = self._windows()[country_code]
        else:
            # Historical default (IN/US share it): 09:00-20:00 local.
            self._window = (datetime.time(9, 0), datetime.time(20, 0))

    def is_dnd_registered(self, phone: str) -> bool:
        clean = phone.replace(" ", "").replace("-", "")
        return clean in self._dnd_registry

    def is_within_calling_window(
        self,
        current_time: datetime.time | None = None,
        allowed_start: datetime.time | None = None,
        allowed_end: datetime.time | None = None,
    ) -> bool:
        """Validate the configured legal calling window in the customer's
        local time. Per-call bounds (when given) override the instance
        window; the default remains 09:00-20:00 local (TRAI/TCPA)."""
        start = self._window[0] if allowed_start is None else allowed_start
        end = self._window[1] if allowed_end is None else allowed_end
        t = current_time or datetime.datetime.now().time()
        return start <= t <= end

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
