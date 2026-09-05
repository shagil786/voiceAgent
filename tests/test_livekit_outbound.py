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


def test_factory_api_constructed_inside_loop_and_closed():
    # Regression (first live PSTN drill): a real LiveKitAPI binds its aiohttp
    # session to the constructing loop, so dial_out MUST build it inside each
    # asyncio.run. A zero-arg factory => the default create constructs, uses,
    # and closes the client inside its own loop; an eager object (legacy test
    # fakes) is used as-is and never touches a loop.
    from types import SimpleNamespace

    events: list = []

    class FakeClient:
        def __init__(self):
            events.append("constructed")

        @property
        def sip(self):
            async def create(req):
                events.append("created")
                return SimpleNamespace(participant_identity="p")
            return SimpleNamespace(create_sip_participant=create)

        async def aclose(self):
            events.append("closed")

    # Factory path: the DEFAULT create runs (no create injection).
    outcome = dial_out(lambda: FakeClient(), "r", "+91", "T",
                       poll=lambda r: "active")
    assert outcome == "connected"
    assert events == ["constructed", "created", "closed"]

    # Eager-object path (legacy fakes) still works untouched via injection.
    created = []
    outcome = dial_out(object(), "r", "+91", "T",
                       create=lambda **kw: created.append(kw) or object(),
                       poll=lambda r: "active")
    assert outcome == "connected"
    assert created and events == ["constructed", "created", "closed"]
