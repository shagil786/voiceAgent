# tests/test_voice.py
from voiceagent.voice import tts_latency

def test_tts_latency_returns_seconds():
    s = tts_latency("hello")
    # may be 0.0 if piper is unavailable in CI; must not be negative
    assert s >= 0.0
