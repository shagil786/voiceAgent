# tests/test_voice_agent.py
import wave
import tempfile
from pathlib import Path
from voiceagent.voice_agent import voice_turn

class FakeAgent:
    def handle(self, text, authenticated=False, amount=None, conv_id=""):
        return type("R", (), {
            "text": "Your order ORD-1 is on the way.",
            "action": "order_status",
            "decision": type("D", (), {"verdict": "ALLOW", "reasons": ["ok"]})(),
        })()

def _tone_wav(path, sr=16000, dur_s=0.3):
    import math
    n = int(sr * dur_s)
    frames = bytearray()
    for i in range(n):
        v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / sr))
        frames += v.to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(bytes(frames))

def test_voice_turn_returns_transcript_reply_and_tts():
    with tempfile.TemporaryDirectory() as d:
        audio = str(Path(d) / "in.wav")
        out = str(Path(d) / "out.wav")
        _tone_wav(audio)
        res = voice_turn(FakeAgent(), audio, out_audio=out)
        assert isinstance(res.transcript, str)
        assert res.reply == "Your order ORD-1 is on the way."
        assert res.action == "order_status"
        assert res.decision == "ALLOW"
        assert res.latency_s >= 0
        assert Path(out).exists()
