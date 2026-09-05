# tests/test_livekit_inbound.py
import asyncio
import time
import pytest
from types import SimpleNamespace
from voiceagent.telephony.inbound import (
    _wait_for_sip_track_async, ensure_room_sample_rate, make_turn_fn,
    wait_for_sip_track, webhook_handler,
)

class FakeOrch:
    def __init__(self): self.turns = []
    def handle_turn(self, session_id, user_text, **kw):
        self.turns.append(user_text)
        from types import SimpleNamespace
        return SimpleNamespace(reply="Hi there", actions=[])

def test_greeting_turn_and_utterance_wiring(tmp_path):
    fn = make_turn_fn(FakeOrch(), "s1", asr=lambda pcm: "hello",
                      tts=lambda text: (text, b"\x00\x00" * 8000))
    reply, wav = fn(b"\x01\x02" * 8000)
    assert reply == "Hi there" and isinstance(wav, bytes) and len(wav) > 0

def test_webhook_filters_by_prefix():
    joined = []
    h = webhook_handler(config=None, join_room=joined.append,
                        validate=lambda body, sig: {"event": "room_started", "room": {"name": "call-abc"}})
    assert h("{}", "sig") is True and joined == ["call-abc"]
    h2 = webhook_handler(config=None, join_room=joined.append,
                         validate=lambda body, sig: {"event": "room_started", "room": {"name": "other"}})
    assert h2("{}", "sig") is False and len(joined) == 1

def test_wait_for_sip_track_eventual_then_found():
    calls = {"n": 0}
    def get_tracks():
        calls["n"] += 1
        return None if calls["n"] < 3 else "track"
    assert wait_for_sip_track(get_tracks, timeout_s=5, sleep=lambda s: None) == "track"

def test_wait_for_sip_track_timeout_returns_none():
    assert wait_for_sip_track(lambda: None, timeout_s=0.05, sleep=lambda s: None) is None

def test_default_validate_missing_creds_fail_closed():
    config = SimpleNamespace(livekit_key=None, livekit_secret=None, livekit_room_prefix="call-")
    h = webhook_handler(config, join_room=lambda r: None)
    assert h("{}", "sig") is False

def test_room_sample_rate_guard():
    ensure_room_sample_rate(48000)
    with pytest.raises(ValueError):
        ensure_room_sample_rate(16000)

def test_async_waiter_track_immediately():
    assert asyncio.run(_wait_for_sip_track_async(lambda: "track", 15)) == "track"

def test_async_waiter_timeout_returns_none():
    t0 = time.monotonic()
    assert asyncio.run(_wait_for_sip_track_async(lambda: None, 0.02)) is None
    assert time.monotonic() - t0 < 5


# --- capture_frame regression: unawaited coroutine = silent calls -----------

def _fake_source():
    class FakeSource:
        def __init__(self):
            self.frames = []
            self.awaited = 0

        async def capture_frame(self, frame):
            self.awaited += 1
            self.frames.append(frame)

    return FakeSource()


def test_publish_pcm16_awaits_capture_frame():
    """_publish_pcm16 must await capture_frame — a bare call publishes nothing."""
    import struct

    from voiceagent.telephony.inbound import _publish_pcm16

    source = _fake_source()
    # 100ms of tone @16k int16 mono
    wav = struct.pack("<1600h", *([1000] * 1600))
    asyncio.run(_publish_pcm16(source, wav))
    assert source.awaited > 0
    assert all(
        f.samples_per_channel == 480 and f.sample_rate == 48000 and f.num_channels == 1
        for f in source.frames
    )  # 10ms @48k mono


def test_playback_pump_polls_session_and_stops():
    """Pump drains the session queue (not a copy), so barge-in clears silence it."""
    import asyncio as aio

    from voiceagent.telephony.inbound import _playback_pump

    source = _fake_source()
    chunks = [b"\x00" * 960]

    class FakeSession:
        def take_playback(self):
            return chunks.pop(0) if chunks else None

    stop = aio.Event()

    async def stop_after_first(frame):
        source.awaited += 1
        stop.set()

    source.capture_frame = stop_after_first

    async def run():
        await aio.wait_for(_playback_pump(source, FakeSession(), stop), timeout=2.0)

    asyncio.run(run())
    assert source.awaited == 1  # one chunk played, then stop -> no hang, no drain-after-barge-in
