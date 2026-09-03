# tests/test_stream.py
import pytest
import math
from voiceagent.telephony.stream import BargeInController, StreamingVAD, BargeInEvent


def _make_pcm_frame(freq=440, sr=16000, dur_s=0.02, silent=False):
    n = int(sr * dur_s)
    if silent:
        return b"\x00\x00" * n
    frames = bytearray()
    for i in range(n):
        v = int(32767 * 0.6 * math.sin(2 * math.pi * freq * i / sr))
        frames += v.to_bytes(2, "little", signed=True)
    return bytes(frames)


def test_barge_in_controller_lifecycle():
    barge_in_events = []

    def on_barge(evt: BargeInEvent):
        barge_in_events.append(evt)

    controller = BargeInController(on_barge_in=on_barge)
    assert not controller.is_speaking

    controller.start_speaking("turn-101")
    assert controller.is_speaking
    assert controller.active_turn_id == "turn-101"

    evt = controller.trigger_barge_in()
    assert evt is not None
    assert evt.interrupted_turn_id == "turn-101"
    assert not controller.is_speaking
    assert len(barge_in_events) == 1

    # Second trigger on already interrupted turn returns None
    assert controller.trigger_barge_in() is None


def test_streaming_vad_speech_start_and_end():
    vad = StreamingVAD(speech_pad_ms=40, silence_timeout_ms=60)
    speech_frame = _make_pcm_frame(silent=False)
    silence_frame = _make_pcm_frame(silent=True)

    # Initially not in speech
    assert not vad.in_speech

    # Frame 1: speech, but pad not yet satisfied
    e1 = vad.process_frame(speech_frame)
    assert not e1["speech_started"]

    # Frame 2: speech, pad (40ms = 2 frames) met -> speech started
    e2 = vad.process_frame(speech_frame)
    assert e2["speech_started"]
    assert vad.in_speech

    # Send 3 silence frames (60ms timeout = 3 frames) -> speech ended
    vad.process_frame(silence_frame)
    vad.process_frame(silence_frame)
    e5 = vad.process_frame(silence_frame)
    assert e5["speech_ended"]
    assert e5["complete_audio"] is not None
    assert not vad.in_speech


def test_streaming_vad_triggers_barge_in_when_speaking():
    interrupted = []
    controller = BargeInController(on_barge_in=lambda evt: interrupted.append(evt))
    controller.start_speaking("agent-turn-42")

    vad = StreamingVAD(barge_in_controller=controller, speech_pad_ms=20)
    speech_frame = _make_pcm_frame(silent=False)

    # Customer speaks while agent is speaking
    events = vad.process_frame(speech_frame)
    assert events["speech_started"]
    assert events["barge_in"] is not None
    assert events["barge_in"].interrupted_turn_id == "agent-turn-42"
    assert not controller.is_speaking
    assert len(interrupted) == 1
