# tests/test_services_ml.py — REAL-model service processes on ephemeral
# ports: wire parity with the in-process path, warmup parity, and a Qwen
# smoke gated on the model already being cached (ml tier only).
import asyncio
import wave
from pathlib import Path

import pytest
from aiohttp import web

pytestmark = pytest.mark.ml

from scripts.asr_service import ASRService, build_app as asr_app
from scripts.tts_service import TTSService, build_app as tts_app
from voiceagent import asr_client, tts_client


QWEN_CACHE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf"
WHISPER_CACHE = Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small"


def _boot(build):
    """Boot a service app on an ephemeral port; return (url, stop).

    The AppRunner must live on a background thread's event loop that keeps
    running: booting via asyncio.run closes that loop, which dead-serves
    the socket (clients hang on the ws handshake forever) — same lesson as
    tests/test_asr_client.py::_serve.
    """
    import threading
    runner = web.AppRunner(build())
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(
            web.TCPSite(runner, "127.0.0.1", 0).start())
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    if not started.wait(timeout=10):
        raise RuntimeError("service failed to boot")
    url = f"ws://127.0.0.1:{runner.addresses[0][1]}/ws"

    def _stop():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.run_until_complete(runner.cleanup())
        loop.close()

    return url, _stop


def _tone_wav(tmp_path, seconds=1.0, rate=16000):
    import math
    p = tmp_path / "tone.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / rate))
            .to_bytes(2, "little", signed=True)
            for i in range(int(rate * seconds))))
    return str(p)


def test_asr_service_whisper_smoke(tmp_path, monkeypatch):
    """Real whisper-small loads inside the service process and answers the
    wire — the full remote path with a real engine."""
    if not WHISPER_CACHE.exists():
        pytest.skip("whisper small not cached")
    url, stop = _boot(lambda: asr_app(ASRService()))
    monkeypatch.setenv("VOICEAGENT_ASR_URL", url)
    try:
        text = asr_client.transcribe_wav_routed(_tone_wav(tmp_path),
                                                language="en")
        assert isinstance(text, str)  # tone -> empty-ish transcript is FINE
    finally:
        stop()


def test_asr_wire_parity_stub_engines(tmp_path, monkeypatch):
    """Same wav, same stub engine: in-process router and wire path must
    return identical text (drift tripwire for the wire codec)."""
    from voiceagent.model_registry import LoadedModelLRU

    class EchoHandle:
        def transcribe(self, audio, language=None):
            return "parity-text"

    loaders = {"qwen": EchoHandle, "indic": EchoHandle, "whisper": EchoHandle}
    url, stop = _boot(lambda: asr_app(ASRService(
        registry=LoadedModelLRU(capacity=2, idle_unload_s=600),
        engine_loaders=loaders)))
    monkeypatch.setenv("VOICEAGENT_ASR_URL", url)
    try:
        wire = asr_client.transcribe_wav_routed(_tone_wav(tmp_path),
                                                language="en")
    finally:
        stop()
    monkeypatch.delenv("VOICEAGENT_ASR_URL")
    # In-process comparison against the SAME stub (monkeypatch the router).
    import voiceagent.asr as asr_mod
    monkeypatch.setattr(asr_mod, "_get_qwen_asr", lambda: EchoHandle())
    monkeypatch.setattr(asr_mod, "_get_whisper_small", lambda: EchoHandle())
    local = asr_mod.transcribe_wav_routed(_tone_wav(tmp_path), language="en")
    assert wire == local == "parity-text"


def test_tts_service_real_voice_smoke(tmp_path, monkeypatch):
    if not Path("data/models/en_US-amy-medium.onnx").exists():
        pytest.skip("en voice not in data/models")
    url, stop = _boot(lambda: tts_app(TTSService()))
    monkeypatch.setenv("VOICEAGENT_TTS_URL", url)
    try:
        out = str(tmp_path / "reply.wav")
        tts_client.speak("Your order ORD-9021 ships today.", "en",
                         out_path=out)
        with wave.open(out, "rb") as w:
            assert w.getnframes() > 0
    finally:
        stop()


def test_asr_service_qwen_smoke(tmp_path, monkeypatch):
    if not QWEN_CACHE.exists():
        pytest.skip("Qwen3-ASR not cached")
    url, stop = _boot(lambda: asr_app(ASRService()))
    monkeypatch.setenv("VOICEAGENT_ASR_URL", url)
    try:
        text = asr_client.transcribe_wav_routed(_tone_wav(tmp_path),
                                                language="en")
        assert isinstance(text, str)
    finally:
        stop()
