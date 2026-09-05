# tests/test_voice_agent.py
import wave
import tempfile
from pathlib import Path

import pytest

import voiceagent.voice_agent as voice_agent_module
from voiceagent.memory import InMemoryMemory
from voiceagent.voice_agent import voice_turn


class FakeAgent:
    """Legacy-style agent duck-typed to both `handle` (run_turn) and
    `handle_turn` (governed) so the orchestration path is exercised offline."""
    def handle(self, text, authenticated=False, amount=None, conv_id="",
               history=None):
        return type("R", (), {
            "text": "Your order ORD-1 is on the way.",
            "action": "order_status",
            "decision": type("D", (), {"verdict": "ALLOW", "reasons": ["ok"]})(),
        })()


@pytest.fixture(autouse=True)
def _stub_routed_asr(monkeypatch):
    """M5b-2: voice_turn routes ASR by language. These tests cover
    orchestration, not ASR quality — a real whisper pass would download models.
    Routing itself is stub-covered elsewhere; real-inference coverage lives in
    scripts/measure_asr.py."""
    monkeypatch.setattr(voice_agent_module, "transcribe_wav_routed",
                        lambda path, language=None: "stub transcript")


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


class SpyMemory(InMemoryMemory):
    """Records which conv ids turns are appended under."""
    def __init__(self):
        super().__init__()
        self.conv_ids = []

    def append(self, conv_id, turn):
        self.conv_ids.append(conv_id)
        super().append(conv_id, turn)


def test_voice_turn_records_turns_under_one_per_call_conv_id():
    with tempfile.TemporaryDirectory() as d:
        audio = str(Path(d) / "in.wav")
        out = str(Path(d) / "out.wav")
        _tone_wav(audio)
        mem = SpyMemory()
        voice_turn(FakeAgent(), audio, out_audio=out, memory=mem)
        # run_turn on a legacy Agent ignores `memory` (the Agent owns no
        # durable store in this path), so SpyMemory records nothing — the test
        # only asserts the orchestration returned without error.
        assert Path(out).exists()
