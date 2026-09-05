#!/usr/bin/env python3
"""LiveKit outbound dial: scrub -> create room -> dial -> disposition.

Usage:
    .venv/bin/python scripts/livekit_dial.py --to +15551234567 --room call-demo [--trunk TRUNK_ID]

Regulatory gate: `RegulatoryDNDScrubber` runs BEFORE any dial; blocked
numbers exit non-zero without touching the SIP trunk. Secrets and trunk
defaults come from `.env` via `RuntimeConfig` (`LIVEKIT_*`); `--trunk`
overrides `LIVEKIT_TRUNK_ID`, the trunk ID is the caller-ID source and
`LIVEKIT_NUMBER` is passed as the SIP caller number when set.

After `connected`, the first 600ms of room audio goes to
`classify_early_audio` (AMD handoff seam); the full worker-join + loopback
drill lands in Task 5 — until then, point a worker at the room with
`scripts/livekit_worker.py` (it joins `call-*` rooms on webhook events).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.config import load_config
from voiceagent.outbound.dialer import RegulatoryDNDScrubber
from voiceagent.telephony.outbound import dial_out

logger = logging.getLogger("livekit_dial")


def _api(config):
    """Zero-arg FACTORY for the LiveKit client: dial_out constructs it inside
    each call's event loop, because a LiveKitAPI's aiohttp session is bound
    to the loop it was built under and cannot cross asyncio.run boundaries."""
    from livekit.api import LiveKitAPI

    return lambda: LiveKitAPI(
        url=config.livekit_url, api_key=config.livekit_key, api_secret=config.livekit_secret
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="E.164 destination number")
    parser.add_argument("--room", required=True, help="Room name for the call")
    parser.add_argument("--trunk", default=None, help="Outbound trunk ID (default: LIVEKIT_TRUNK_ID)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config()

    trunk_id = args.trunk or config.livekit_trunk_id
    if not trunk_id:
        print("missing trunk: pass --trunk or set LIVEKIT_TRUNK_ID")
        return 2

    permitted, reason = RegulatoryDNDScrubber().scrub(args.to)
    if not permitted:
        print(f"blocked: {reason} (no dial placed)")
        return 2

    async def _create_room() -> None:
        from livekit.protocol import room as room_proto

        api = _api(config)()
        try:
            await api.room.create_room(room_proto.CreateRoomRequest(name=args.room))
        finally:
            await api.aclose()

    asyncio.run(_create_room())
    logger.info("room %s ready; start a worker (scripts/livekit_worker.py) to join it", args.room)

    disposition = dial_out(
        _api(config),
        args.room,
        args.to,
        trunk_id,
        from_number=config.livekit_number,
    )
    print(f"disposition: {disposition}")
    if disposition == "connected":
        logger.info(
            "connected — run the first 600ms of room audio through "
            "classify_early_audio for the AMD handoff (Task 5 loopback)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
