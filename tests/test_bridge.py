# tests/test_bridge.py
from voiceagent.telephony.livekit_bridge import BridgeSession

def _wav16_of_speech_like():
    import math, struct
    n = 16000  # 1s of tone-as-speech (energy VAD fires on RMS)
    return struct.pack(f"<{n}h", *[int(8000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n)])

def test_utterance_roundtrip_and_bargein_cancel():
    calls = []
    def on_utterance(pcm):
        calls.append(pcm)
        import struct
        wav = struct.pack(f"<{len(pcm)//2}h", *[1000] * (len(pcm) // 2))
        return ("hello", wav)
    s = BridgeSession(on_utterance=on_utterance)
    tone = _wav16_of_speech_like()
    for i in range(0, len(tone), 640):  # 20ms 16k frames
        s.feed_pcm16(tone[i:i+640])
    for _ in range(25):  # trailing silence to endpoint the turn
        s.feed_pcm16(bytes(640))
    assert len(calls) == 1
    chunks = []
    while (c := s.take_playback()) is not None:
        chunks.append(c)
    assert chunks and len(chunks[0]) == 960  # 10ms @48k mono int16
    # barge-in cancels pending playback
    s2 = BridgeSession(on_utterance=on_utterance)
    for i in range(0, len(tone), 640):
        s2.feed_pcm16(tone[i:i+640])
    for _ in range(25):
        s2.feed_pcm16(bytes(640))
    assert s2.take_playback() is not None
    s2.feed_pcm16(tone[:640])  # speech while speaking → barge-in path arms
    s2.barge_in()  # explicit external trigger also clears
    assert s2.take_playback() is None
    s2.stop(); s.stop()

def test_drain_clears_speaking_no_spurious_bargein():
    import struct
    barges = []
    def on_utterance(pcm):
        return ("hello", struct.pack(f"<{len(pcm)//2}h", *[1000] * (len(pcm) // 2)))
    s = BridgeSession(on_utterance=on_utterance, on_barge_in=barges.append)
    tone = _wav16_of_speech_like()
    for i in range(0, len(tone), 640):
        s.feed_pcm16(tone[i:i+640])
    for _ in range(25):
        s.feed_pcm16(bytes(640))
    assert s._vad.barge_in.is_speaking is True
    while s.take_playback() is not None:
        pass
    assert s._vad.barge_in.is_speaking is False
    for _ in range(5):  # fresh speech onset must not barge in
        s.feed_pcm16(tone[:640])
    assert barges == []
    s.stop()

def test_feed_after_stop_is_noop():
    calls = []
    def on_utterance(pcm):
        calls.append(pcm)
        return ("hi", b"\x00\x00")
    s = BridgeSession(on_utterance=on_utterance)
    s.stop()
    s.feed_pcm16(_wav16_of_speech_like()[:640])
    assert calls == [] and s.take_playback() is None

def test_odd_reply_wav_raises():
    import pytest
    s = BridgeSession(on_utterance=lambda p: ("hi", b"\x00\x01\x02"))
    tone = _wav16_of_speech_like()
    with pytest.raises(ValueError, match="int16-aligned"):
        for i in range(0, len(tone), 640):
            s.feed_pcm16(tone[i:i+640])
        for _ in range(25):
            s.feed_pcm16(bytes(640))
