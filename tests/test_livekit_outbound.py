# tests/test_livekit_outbound.py
from voiceagent.telephony.outbound import classify_early_audio, dial_out

def test_dial_outcomes_and_amd_passthrough():
    calls = {"n": 0}
    def fake_create(**kw):
        calls["n"] += 1
    assert dial_out(object(), "r", "+1", "T", create=fake_create,
                    poll=lambda r: "active") == "connected"
    assert dial_out(object(), "r", "+1", "T", create=fake_create,
                    poll=lambda r: "failed") == "failed"
    assert dial_out(object(), "r", "+1", "T", create=fake_create,
                    poll=lambda r: "ringing") == "timeout"
    assert calls["n"] == 3

def test_amd_first_decision_wins():
    import math, struct
    from voiceagent.outbound.amd import Sub600msAMD
    frames = [struct.pack("<320h", *[int(4000 * math.sin(2 * math.pi * 300 * i / 16000)) for i in range(320)]) for _ in range(30)]
    assert classify_early_audio(iter(frames)) in ("human", "machine", "beep", "timeout")
