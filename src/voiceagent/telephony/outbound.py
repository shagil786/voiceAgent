"""Outbound SIP dial: room -> SIP participant -> AMD handoff seam.

LiveKit is transport only. `dial_out` places the call; `classify_early_audio`
is the offline-testable seam that runs the existing `Sub600msAMD` over the
first 600ms of room audio after connect (full loopback drill: Task 5).

Regulatory note: callers (see `scripts/livekit_dial.py`) MUST run
`RegulatoryDNDScrubber.scrub` before dialing — the library never dials an
unscrubbed number on its own.

All `livekit` imports are function-level (lazy) so unit tests never require
the dep on the cold path.
"""
from __future__ import annotations

import asyncio
import math
import sys
import time
from typing import Any, Callable, Iterable

from voiceagent.outbound.amd import CallParty, Sub600msAMD

CONNECTED = "connected"
FAILED = "failed"
TIMEOUT = "timeout"

_POLL_ACTIVE = "active"
_POLL_FAILED = "failed"
_POLL_RINGING = "ringing"

_CADENCE_S = 1.0

_DECIDED = {
    CallParty.HUMAN: "human",
    CallParty.MACHINE: "machine",
    CallParty.BEEP: "beep",
}


def _default_create(
    *,
    api: Any,
    room_name: str,
    to_number: str,
    trunk_id: str,
    from_number: str | None = None,
) -> Any:
    """Place the SIP call via the installed `livekit-api` SDK.

    Verified against installed `livekit-api` source (NOT the brief sketch):
    `SipService.create_sip_participant` takes a single
    `CreateSIPParticipantRequest` protobuf — there is no
    `create_sip_participant(room=, to=, trunk=)` kwargs form. Field names
    (`room_name`, `sip_call_to`, `sip_trunk_id`, `sip_number`) verified
    against `livekit.protocol.sip`. The SDK method is async; this sync
    wrapper owns the one `asyncio.run` so the library stays sync.
    """
    from livekit.protocol import sip as sip_proto

    req = sip_proto.CreateSIPParticipantRequest(
        room_name=room_name,
        sip_call_to=to_number,
        sip_trunk_id=trunk_id,
    )
    if from_number:
        req.sip_number = from_number
    return asyncio.run(api.sip.create_sip_participant(req))


def _default_poll(api: Any, room_name: str, seen: list[bool] | None = None) -> str:
    """Map room participants to `active|failed|ringing`.

    Verified: `SIPParticipantInfo` carries NO call-status field, so status
    comes from `RoomService.list_participants` instead: any JOINED/ACTIVE
    participant -> `active`; any participant but none live -> `failed` only
    when a DISCONNECTED participant is present, else `ringing`. Empty room
    -> `ringing` on a fresh dial, but `failed` once `seen` records that
    participants were present (fast hangup, not another 30s of ringing).
    `seen` is owned per-`dial_out` call (fresh list each dial); omitting it
    makes a stateless single-shot. Transient errors print a stderr warning
    and count as `ringing` for that iteration only.
    """
    if seen is None:
        seen = []
    try:
        from livekit.protocol import models, room as room_proto

        resp = asyncio.run(
            api.room.list_participants(room_proto.ListParticipantsRequest(room=room_name))
        )
        participants = list(resp.participants)
        if not participants:
            return _POLL_FAILED if seen else _POLL_RINGING
        seen.append(True)
        if any(
            p.state in (models.ParticipantInfo.State.ACTIVE, models.ParticipantInfo.State.JOINED)
            for p in participants
        ):
            return _POLL_ACTIVE
        if any(p.state == models.ParticipantInfo.State.DISCONNECTED for p in participants):
            return _POLL_FAILED
        return _POLL_RINGING
    except Exception as exc:
        print(f"livekit poll warning: {exc}", file=sys.stderr)
        return _POLL_RINGING


def dial_out(
    api: Any,
    room_name: str,
    to_number: str,
    trunk_id: str,
    timeout_s: float = 30,
    poll: Callable[[str], str] | None = None,
    create: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] | None = None,
    from_number: str | None = None,
) -> str:
    """Dial `to_number` into `room_name` via `trunk_id`.

    Returns `"connected" | "failed" | "timeout"`. `create`/`poll`/`sleep`
    are injectable (defaults hit the live SDK / `time.sleep`); the poll
    loop runs at 1s cadence until `active`/`failed` or `timeout_s`. Poll
    exceptions warn on stderr and count as `ringing` for that iteration.
    The loop is also iteration-capped at `timeout_s` cadences, so a no-op
    `sleep` fake finishes instantly with the same outcome as wall-clock.
    """
    _create = create if create is not None else _default_create
    _create(api=api, room_name=room_name, to_number=to_number, trunk_id=trunk_id,
            from_number=from_number)
    seen: list[bool] = []
    _poll = poll if poll is not None else (lambda r: _default_poll(api, r, seen))
    _sleep = sleep if sleep is not None else time.sleep

    deadline = time.monotonic() + timeout_s
    max_polls = max(1, math.ceil(timeout_s / _CADENCE_S))
    for _ in range(max_polls):
        try:
            status = _poll(room_name)
        except Exception as exc:
            print(f"livekit poll warning: {exc}", file=sys.stderr)
            status = _POLL_RINGING
        if status == _POLL_ACTIVE:
            return CONNECTED
        if status == _POLL_FAILED:
            return FAILED
        if time.monotonic() >= deadline:
            return TIMEOUT
        _sleep(min(_CADENCE_S, deadline - time.monotonic()))
    return TIMEOUT


def classify_early_audio(frames_iter: Iterable[bytes]) -> str:
    """Run `Sub600msAMD` over 20ms frames; first decisive label wins.

    Reuses `Sub600msAMD().process_frame` (default 16k/20ms/600ms ctor —
    verified against `outbound/amd.py`); returns `human|machine|beep`, or
    `timeout` when the iterator exhausts (or AMD decides SILENCE) first.
    """
    amd = Sub600msAMD()
    for frame in frames_iter:
        result = amd.process_frame(bytes(frame))
        if result.classification in _DECIDED:
            return _DECIDED[result.classification]
        if result.classification == CallParty.SILENCE:
            return TIMEOUT
    return TIMEOUT
