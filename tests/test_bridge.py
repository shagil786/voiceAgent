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
