# tests/test_asr_client.py — dispatcher: remote (env set, real aiohttp
# loopback server with STUB engines), legacy (env unset), fail-closed
# (server error -> GenericBackendError, never a local fallback).
#
# scripts/ has no __init__.py; the suite reaches script modules via a
# sys.path insert (tests/test_asr_service.py precedent) so the brief's
# package-style import (scripts.asr_service) resolves.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import json
import threading

import pytest
from aiohttp import web

from scripts.asr_service import ASRService, build_app
from voiceagent import asr_client
from voiceagent.generic_backend import GenericBackendError
from voiceagent.model_registry import LoadedModelLRU


class StubHandle:
    def transcribe(self, audio, language=None):
        return "wire-text"


def stub_service():
    return ASRService(
        registry=LoadedModelLRU(capacity=2, idle_unload_s=600),
        engine_loaders={"qwen": StubHandle, "indic": StubHandle,
                        "whisper": StubHandle})


def _serve(monkeypatch):
    """Boot a real stub service on an ephemeral port; return (url, stop).

    Brief-note adaptation: booting via asyncio.run closes that loop, which
    dead-serves the socket (verified empirically — clients hang forever),
    so the AppRunner lives on a background thread's loop that keeps
    running. Same surface: AppRunner + TCPSite on port 0 + runner.addresses.
    """
    runner = web.AppRunner(build_app(stub_service()))
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
        raise RuntimeError("stub asr service failed to boot")
    url = f"ws://127.0.0.1:{runner.addresses[0][1]}/ws"

    def _stop():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.run_until_complete(runner.cleanup())
        loop.close()

    return url, _stop


def _write_wav(tmp_path):
    import wave
    p = tmp_path / "utt.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 1600)
    return str(p)


def test_remote_roundtrip(tmp_path, monkeypatch):
    url, stop = _serve(monkeypatch)
    monkeypatch.setenv("VOICEAGENT_ASR_URL", url)
    try:
        text = asr_client.transcribe_wav_routed(_write_wav(tmp_path),
                                                language="en")
        assert text == "wire-text"
    finally:
        stop()


def test_env_unset_dispatches_to_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv("VOICEAGENT_ASR_URL", raising=False)
    # Legacy function is lazily imported and actually used.
    called = {}
    import voiceagent.asr as asr_mod

    def fake_legacy(path, language=None):
        called["args"] = (path, language)
        return "legacy-text"

    monkeypatch.setattr(asr_mod, "transcribe_wav_routed", fake_legacy)
    assert asr_client.transcribe_wav_routed("x.wav", language="en") \
        == "legacy-text"
    assert called["args"] == ("x.wav", "en")


def test_remote_failure_raises_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEAGENT_ASR_URL", "ws://127.0.0.1:9/ws")  # nothing there
    monkeypatch.setenv("VOICEAGENT_SERVICE_TIMEOUT_S", "2")
    import voiceagent.asr as asr_mod
    monkeypatch.setattr(asr_mod, "transcribe_wav_routed",
                        lambda *a, **k: pytest.fail("fallback happened"))
    with pytest.raises(GenericBackendError):
        asr_client.transcribe_wav_routed(_write_wav(tmp_path), language="en")


def test_dispatcher_works_inside_a_running_loop(tmp_path, monkeypatch):
    """The telephony worker calls this from async context — the dispatcher
    must bridge to a worker thread instead of asyncio.run-on-running-loop."""
    url, stop = _serve(monkeypatch)
    monkeypatch.setenv("VOICEAGENT_ASR_URL", url)

    async def driver():
        return asr_client.transcribe_wav_routed(_write_wav(tmp_path),
                                                language="en")

    try:
        assert asyncio.run(driver()) == "wire-text"
    finally:
        stop()


def test_warm_dispatch(tmp_path, monkeypatch):
    url, stop = _serve(monkeypatch)
    monkeypatch.setenv("VOICEAGENT_ASR_URL", url)
    try:
        assert asr_client.warmup_asr_for_language("en") == "en"
        asr_client.warmup_asr()  # warm op, returns None
    finally:
        stop()
