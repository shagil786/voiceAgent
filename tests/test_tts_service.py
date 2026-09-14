# tests/test_tts_service.py — TTS service with a stub voice loader + the
# brain dispatcher against a real loopback server (no piper downloads).
#
# scripts/ has no __init__.py; the suite reaches script modules via a
# sys.path insert (tests/test_asr_client.py precedent) so the brief's
# package-style import (scripts.tts_service) resolves.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import io
import threading
import wave

from aiohttp import web

from scripts.tts_service import TTSService, build_app
from voiceagent import tts_client
from voiceagent.model_registry import LoadedModelLRU
from voiceagent.tts import VOICE_REGISTRY


class StubVoice:
    """Duck-typed PiperVoice stand-in: writes a real, parseable WAV through
    the Wave_write the service opened (real PiperVoice sets the format
    itself; replicate that contract or Wave_write.close() raises)."""

    def synthesize_wav(self, text, w, syn_config=None):
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 100)


class BigWavVoice:
    """Synthesizes a >4 MiB WAV — aiohttp 3.9's client-side default
    max_msg_size — to prove the dispatcher's raised inbound cap."""

    NFRAMES = 2_646_000  # 120s of 22050 Hz/16-bit mono ≈ 5.29 MB

    def synthesize_wav(self, text, w, syn_config=None):
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * self.NFRAMES)


def make_service():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    svc = TTSService(registry=reg,
                     voice_loader=lambda name, d: StubVoice(),
                     registry_map=dict(VOICE_REGISTRY))
    return svc


def _serve(monkeypatch, service=None):
    """Boot a real stub service on an ephemeral port; return (url, stop).

    Brief-note adaptation (verified empirically in Task 4): booting via
    asyncio.run closes that loop, which dead-serves the socket — so the
    AppRunner lives on a background thread's loop that keeps running.
    Same surface: AppRunner + TCPSite on port 0 + runner.addresses.
    """
    runner = web.AppRunner(build_app(service or make_service()))
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
        raise RuntimeError("stub tts service failed to boot")
    url = f"ws://127.0.0.1:{runner.addresses[0][1]}/ws"

    def _stop():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.run_until_complete(runner.cleanup())
        loop.close()

    return url, _stop


def test_synthesize_returns_wav_bytes_and_voice():
    svc = make_service()
    out = svc.synthesize("hello there", language="en")
    assert out["voice"].startswith("en_US")
    buf = io.BytesIO(out["wav"])
    with wave.open(buf, "rb") as w:
        assert w.getnchannels() == 1


def test_client_speak_writes_wav_file(tmp_path, monkeypatch):
    url, stop = _serve(monkeypatch)
    monkeypatch.setenv("VOICEAGENT_TTS_URL", url)
    try:
        out = str(tmp_path / "reply.wav")
        path = tts_client.speak("hello there", out_path=out)
        assert path == out
        with wave.open(out, "rb") as w:
            assert w.getnframes() > 0
    finally:
        stop()


def test_client_speak_roundtrips_wav_over_4mib(tmp_path, monkeypatch):
    """The synthesize reply carries the whole WAV in one binary frame;
    aiohttp 3.9's client default max_msg_size is 4 MiB (~95s of 22050 Hz
    16-bit mono), so the dispatcher raises it to match the service's
    inbound cap (64 MiB) or long replies die as MessageTooBig."""
    svc = TTSService(registry=LoadedModelLRU(capacity=2, idle_unload_s=600),
                     voice_loader=lambda name, d: BigWavVoice(),
                     registry_map=dict(VOICE_REGISTRY))
    url, stop = _serve(monkeypatch, service=svc)
    monkeypatch.setenv("VOICEAGENT_TTS_URL", url)
    try:
        out = str(tmp_path / "big.wav")
        assert tts_client.speak("hello there", out_path=out) == out
        with wave.open(out, "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 22050
            assert w.getnframes() == BigWavVoice.NFRAMES
    finally:
        stop()


def test_client_env_unset_uses_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv("VOICEAGENT_TTS_URL", raising=False)
    import voiceagent.tts as tts_mod
    monkeypatch.setattr(tts_mod, "speak",
                        lambda text, language=None, out_path=None: "legacy")
    assert tts_client.speak("x") == "legacy"
