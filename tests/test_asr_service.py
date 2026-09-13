# tests/test_asr_service.py — service logic against STUB engines: routing,
# LRU integration, wire handling via a FakeWebSocket. Real-model parity
# lives in tests/test_services_ml.py (ml tier).
#
# scripts/ has no __init__.py; the suite reaches script modules via a
# sys.path insert (tests/test_batch_cli.py, tests/test_onboard_measure.py).
# The repo root is inserted so the brief's package-style import
# (scripts.asr_service) resolves — scripts/ then works as an implicit
# namespace package (tests/ has __init__.py, so pytest already prepends
# the repo root; this keeps the import working standalone too).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import json
import wave

import pytest

from scripts.asr_service import ASRService, handle_connection
from voiceagent.model_registry import LoadedModelLRU


class StubHandle:
    """Mimics the uniform transcribe(audio, language) handle interface."""

    def __init__(self, name):
        self.name = name
        self.calls = []
        self.last_wav = None

    def transcribe(self, audio, language=None):
        # The service unlinks its temp WAV once transcribe returns, so the
        # WAV shape must be captured NOW (the file is gone by assert time).
        if isinstance(audio, str):
            with wave.open(audio, "rb") as w:
                self.last_wav = (w.getframerate(), w.getsampwidth())
        self.calls.append((audio, language))
        return f"stub-text-{self.name}"


def make_service(calls):
    def loader(name):
        h = StubHandle(name)
        calls[name] = h
        return h
    return ASRService(
        registry=LoadedModelLRU(capacity=2, idle_unload_s=600),
        engine_loaders={"qwen": lambda: loader("qwen"),
                        "indic": lambda: loader("indic"),
                        "whisper": lambda: loader("whisper")})


class FakeWebSocket:
    """Records sends; replays scripted client frames from recv()."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    async def send_str(self, text):
        self.sent.append(("str", text))

    async def send_bytes(self, data):
        self.sent.append(("bytes", data))

    async def recv(self):
        if not self.frames:
            return None
        item = self.frames.pop(0)
        return item


def _wav_pcm(text="x", rate=16000):
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * rate)  # 1s of audio
    return buf.getvalue()


def test_transcribe_routes_declared_indic_and_forces_language():
    calls = {}
    svc = make_service(calls)
    out = svc.transcribe(b"\x00\x01" * 100, 16000, 1, 2, language="te")
    assert out == {"text": "stub-text-indic", "engine": "indic:te"}
    # The stub is recorded under its LOADER family name ("indic") —
    # "indic:te" is the registry key, not the factory's name argument.
    handle = calls["indic"]
    wav_path, lang = handle.calls[0]
    assert lang == "te"
    # (framerate, sampwidth) snapshotted while the temp WAV still existed
    assert handle.last_wav == (16000, 2)


def test_transcribe_blind_path_routes_qwen():
    calls = {}
    svc = make_service(calls)
    out = svc.transcribe(b"\x00\x01" * 100, 16000, 1, 2, language=None)
    assert out["engine"] == "qwen"


def test_transcribe_reuses_registry_slot():
    calls = {}
    svc = make_service(calls)
    svc.transcribe(b"\x00\x01" * 100, 16000, 1, 2, language="en")
    svc.transcribe(b"\x00\x01" * 100, 16000, 1, 2, language="en")
    assert len(calls) == 1 and calls["qwen"].calls.__len__() == 2
    assert svc.registry.stats()["hits"] == 1


def test_handle_connection_serves_describe_and_transcribe():
    calls = {}
    svc = make_service(calls)
    header = json.dumps({"v": 1, "op": "transcribe", "req_id": 3,
                         "language": "en", "sample_rate": 16000,
                         "channels": 1, "sampwidth": 2,
                         "has_payload": True})
    ws = FakeWebSocket([header, b"\x00\x01" * 100,
                        json.dumps({"v": 1, "op": "describe"})])
    asyncio.run(handle_connection(ws, svc))
    kinds = [k for k, _ in ws.sent]
    assert kinds[0] == "str"  # transcription JSON first
    first = json.loads(ws.sent[0][1])
    assert first["op"] == "transcription" and first["req_id"] == 3
    assert first["text"] == "stub-text-qwen"
    info = json.loads(ws.sent[-1][1])
    assert info["op"] == "info" and info["service"] == "asr"
    assert "max_loaded" in info and "idle_unload_s" in info


def test_handle_connection_reports_engine_failure_as_error_op():
    class BoomHandle:
        def transcribe(self, audio, language=None):
            raise RuntimeError("gated repo")

    svc = ASRService(
        registry=LoadedModelLRU(capacity=2, idle_unload_s=600),
        engine_loaders={"qwen": BoomHandle, "indic": BoomHandle,
                        "whisper": lambda: (_ for _ in ()).throw(
                            RuntimeError("no whisper"))})
    header = json.dumps({"v": 1, "op": "transcribe", "req_id": 9,
                         "language": "en", "sample_rate": 16000,
                         "channels": 1, "sampwidth": 2,
                         "has_payload": True})
    ws = FakeWebSocket([header, b"\x00\x01" * 100])
    asyncio.run(handle_connection(ws, svc))
    err = json.loads(ws.sent[-1][1])
    assert err["op"] == "error" and err["req_id"] == 9
    assert err["code"] == "asr_failed"


def test_warm_preloads_declared_route():
    calls = {}
    svc = make_service(calls)
    assert svc.warm("en") == "qwen"
    assert "qwen" in calls
