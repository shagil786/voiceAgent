# tests/test_livekit_inbound.py
import pytest
from types import SimpleNamespace
from voiceagent.telephony.inbound import (
    ensure_room_sample_rate, make_turn_fn, wait_for_sip_track, webhook_handler,
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
