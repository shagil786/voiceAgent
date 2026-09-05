# tests/test_loopback.py
# Loopback drill mechanics: caller WAV through BridgeSession -> reply WAV.
# No models, no network — the stub turn fn stands in for ASR/Orchestrator/TTS.
import wave


def test_caller_wav_roundtrip():
    import math, struct
    from voiceagent.telephony.livekit_bridge import BridgeSession
    n = 16000
    wav = struct.pack(f"<{n}h", *[int(6000 * math.sin(2 * math.pi * 330 * i / 16000)) for i in range(n)])
    out = []
    s = BridgeSession(on_utterance=lambda pcm: ("ok", struct.pack(f"<{len(pcm)//2}h", *[500] * (len(pcm) // 2))))
    for i in range(0, len(wav), 640):
        s.feed_pcm16(wav[i:i+640])
    for _ in range(25):
        s.feed_pcm16(bytes(640))
    while (c := s.take_playback()) is not None:
        out.append(c)
    assert b"".join(out) and s.take_playback() is None
    s.stop()


def test_wav_file_loopback_writes_48k_reply(tmp_path):
    """Real file path: write 16k WAV -> 20ms chunk feed -> collect reply ->
    write 48k WAV. Exercises `resample_16k_to_48k` + chunking end to end
    without loading any ASR/TTS model."""
    import math, struct

    from voiceagent.telephony.livekit_bridge import BridgeSession

    caller_path = tmp_path / "caller.wav"
    reply_path = tmp_path / "reply.wav"

    n = 16000  # 1s caller tone (enough energy for the VAD to endpoint)
    pcm16 = struct.pack(
        f"<{n}h", *[int(6000 * math.sin(2 * math.pi * 330 * i / 16000)) for i in range(n)]
    )
    with wave.open(str(caller_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16)

    turns = []

    def stub_turn(pcm: bytes) -> tuple[str, bytes]:
        turns.append(len(pcm))
        # 400ms fixed 16k reply (stand-in for the governed turn's TTS output)
        m = 6400
        return "ok", struct.pack(
            f"<{m}h", *[int(4000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(m)]
        )

    session = BridgeSession(on_utterance=stub_turn)
    with wave.open(str(caller_path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())

    # 20ms chunks (640 bytes); wave may return a short trailing frame, and
    # feed_pcm16 fail-fasts on non-640-byte input, so drop the partial.
    reply_chunks = []
    for i in range(0, len(raw) - 639, 640):
        session.feed_pcm16(raw[i:i + 640])
    for _ in range(25):  # trailing silence to endpoint the turn
        session.feed_pcm16(bytes(640))
    while (c := session.take_playback()) is not None:
        reply_chunks.append(c)
    session.stop()

    # One endpointed utterance; the VAD pads with pre/post-roll silence, so
    # only assert it carries the full caller PCM (plus padding), not exact size.
    assert len(turns) == 1 and turns[0] >= n * 2
    assert reply_chunks

    with wave.open(str(reply_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"".join(reply_chunks))

    with wave.open(str(reply_path), "rb") as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1
        assert w.getnframes() > 0
        # 400ms reply upsampled 16k->48k stays 400ms (3:1 ratio)
        assert abs(w.getnframes() - 48000 * 0.4) <= 480  # within 10ms
