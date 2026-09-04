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
                    poll=lambda r: "ringing", sleep=lambda s: None) == "timeout"
    assert calls["n"] == 3

def test_amd_first_decision_wins():
    import math, struct
    from voiceagent.outbound.amd import Sub600msAMD
    frames = [struct.pack("<320h", *[int(4000 * math.sin(2 * math.pi * 300 * i / 16000)) for i in range(320)]) for _ in range(30)]
    assert classify_early_audio(iter(frames)) in ("human", "machine", "beep", "timeout")

def test_poll_errors_tolerated_then_connected(capsys):
    attempts = {"n": 0}
    def flaky(r):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise RuntimeError("boom")
        return "active"
    assert dial_out(object(), "r", "+1", "T", create=lambda **kw: None,
                    poll=flaky, sleep=lambda s: None) == "connected"
    assert attempts["n"] == 3
    assert "livekit poll warning: boom" in capsys.readouterr().err

def test_default_poll_empty_after_seen_fails():
    from types import SimpleNamespace
    from voiceagent.telephony import outbound as ob
    # ParticipantInfo.State.JOINED == 1 (verified against livekit.protocol.models).
    script = [[], [SimpleNamespace(state=1)], []]
    calls = {"i": 0}
    class FakeRoom:
        async def list_participants(self, req):
            i = calls["i"]
            calls["i"] += 1
            return SimpleNamespace(participants=script[min(i, 2)])
    api = SimpleNamespace(room=FakeRoom())
    seen: list = []
    assert ob._default_poll(api, "r", seen) == "ringing"
    assert ob._default_poll(api, "r", seen) == "active"
    assert ob._default_poll(api, "r", seen) == "failed"
