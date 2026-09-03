# tests/test_outbound.py
"""Tests for Sub-600ms AMD, Regulatory DND Scrubber, and Outbound Predictive Dialer."""
import asyncio
import datetime
import math
import pytest

from voiceagent.outbound.amd import CallParty, Sub600msAMD
from voiceagent.outbound.dialer import Lead, OutboundCallResult, PredictiveDialer, RegulatoryDNDScrubber


def _make_speech_frame(freq=300, sr=16000, dur_s=0.02):
    n = int(sr * dur_s)
    frames = bytearray()
    for i in range(n):
        v = int(32767 * 0.7 * math.sin(2 * math.pi * freq * i / sr))
        frames += v.to_bytes(2, "little", signed=True)
    return bytes(frames)


def _make_silence_frame(sr=16000, dur_s=0.02):
    return b"\x00\x00" * int(sr * dur_s)


def _make_beep_frame(freq=1000, sr=16000, dur_s=0.02):
    n = int(sr * dur_s)
    frames = bytearray()
    for i in range(n):
        v = int(32767 * 0.8 * math.sin(2 * math.pi * freq * i / sr))
        frames += v.to_bytes(2, "little", signed=True)
    return bytes(frames)


def test_sub_600ms_amd_human_greeting():
    """Human greeting: 300ms speech burst followed by 160ms pause -> Classified as HUMAN < 500ms."""
    amd = Sub600msAMD()
    speech = _make_speech_frame()
    silence = _make_silence_frame()

    # 15 frames of speech = 300ms ("Hello?")
    for _ in range(15):
        res = amd.process_frame(speech)
        assert res.classification == CallParty.ANALYZING

    # 8 frames of silence = 160ms (pause waiting for response)
    human_res = None
    for _ in range(8):
        res = amd.process_frame(silence)
        if res.classification == CallParty.HUMAN:
            human_res = res
            break

    assert human_res is not None
    assert human_res.classification == CallParty.HUMAN
    assert human_res.latency_ms <= 500.0  # Decided under 500ms!


def test_sub_600ms_amd_machine_continuous_speech():
    """Voicemail: continuous speech exceeding 550ms -> Classified as MACHINE."""
    amd = Sub600msAMD()
    speech = _make_speech_frame()

    machine_res = None
    # 30 frames of continuous speech = 600ms
    for _ in range(30):
        res = amd.process_frame(speech)
        if res.classification == CallParty.MACHINE:
            machine_res = res
            break

    assert machine_res is not None
    assert machine_res.classification == CallParty.MACHINE
    assert machine_res.latency_ms <= 600.0


def test_sub_600ms_amd_carrier_beep():
    """Answering machine prompt beep at 1000Hz -> Immediate BEEP classification."""
    amd = Sub600msAMD()
    beep = _make_beep_frame(freq=1000)

    res = amd.process_frame(beep)
    assert res.classification == CallParty.BEEP


def test_regulatory_dnd_scrubber():
    """Verify DND blacklist blocking and 9 AM - 8 PM legal calling window."""
    scrubber = RegulatoryDNDScrubber(dnd_numbers={"+919999999999"})

    # DND registered number is blocked
    ok, reason = scrubber.scrub("+919999999999")
    assert not ok
    assert reason == "BLOCKED_BY_DND_REGISTRY"

    # Permitted number during legal hours (e.g. 2:00 PM)
    ok2, reason2 = scrubber.scrub("+919876543210", current_time=datetime.time(14, 0))
    assert ok2
    assert reason2 == "ALLOWED"

    # Permitted number outside legal hours (e.g. 11:00 PM)
    ok3, reason3 = scrubber.scrub("+919876543210", current_time=datetime.time(23, 0))
    assert not ok3
    assert reason3 == "BLOCKED_OUTSIDE_LEGAL_CALLING_HOURS"


def test_predictive_dialer_handoff_lifecycle():
    """Verify dialer executes DND scrub, AMD analysis, and proper action dispatch."""
    async def _run():
        humans_handled = []
        voicemails_handled = []

        dialer = PredictiveDialer(
            on_human_connect=lambda lead: humans_handled.append(lead.lead_id),
            on_voicemail_connect=lambda lead: voicemails_handled.append(lead.lead_id),
        )

        daytime = datetime.time(14, 0)

        # 1. Lead 1: Legitimate lead, Human answers
        lead_human = Lead(lead_id="LEAD-01", phone="+919876543210", name="Rohan Mehta")
        res1 = await dialer.dial_lead(lead_human, simulated_party=CallParty.HUMAN, current_time=daytime)
        assert res1.action_taken == "HANDOFF_TO_FRONTMAN"
        assert res1.status == "CONNECTED"
        assert len(humans_handled) == 1

        # 2. Lead 2: Legitimate lead, Machine/Voicemail answers
        lead_vm = Lead(lead_id="LEAD-02", phone="+919876543211", name="Ananya Sen")
        res2 = await dialer.dial_lead(lead_vm, simulated_party=CallParty.MACHINE, current_time=daytime)
        assert res2.action_taken == "DISPATCHED_WHATSAPP_DROPPED_CALL"
        assert res2.status == "VOICEMAIL_DISPATCHED"
        assert len(voicemails_handled) == 1

        # 3. Lead 3: DND registered number -> Blocked prior to dial
        lead_dnd = Lead(lead_id="LEAD-03", phone="+919999999999", name="DND Contact")
        res3 = await dialer.dial_lead(lead_dnd, current_time=daytime)
        assert res3.action_taken == "SCRUBBED_NO_DIAL"
        assert res3.status == "BLOCKED_BY_DND_REGISTRY"

    asyncio.run(_run())
