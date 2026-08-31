# tests/test_asr.py
import wave
import tempfile
from pathlib import Path
from voiceagent.asr import transcribe_wav

def _write_wav(path, sample_rate=16000):
    import math
    n = int(sample_rate * 0.5)
    frames = bytearray()
    for i in range(n):
        v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / sample_rate))
        frames += v.to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))

def test_transcribe_wav_returns_string():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.wav"
        _write_wav(p)
        text = transcribe_wav(str(p))
        assert isinstance(text, str)
