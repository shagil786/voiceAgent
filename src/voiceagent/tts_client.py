"""Brain-side TTS dispatcher — speak/synthesize_to_wav/warm_tts over the
TTS service when VOICEAGENT_TTS_URL is set, legacy in-process otherwise.
speech_text normalization happens HERE (brain-owned, deterministic
mechanics); the service receives already-speakable text. Fail-closed: a
declared service that is down raises GenericBackendError — no fallback.

The remote call is a blocking round-trip; callers inside a running event
loop (telephony worker) are bridged through a worker thread because
asyncio.run cannot nest (same shape as asr_client).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time

from voiceagent.generic_backend import GenericBackendError
from voiceagent.service_protocol import FrameReader, json_text
from voiceagent.tts import speech_text

logger = logging.getLogger(__name__)


def _url() -> str | None:
    return os.environ.get("VOICEAGENT_TTS_URL") or None


def _timeout() -> float:
    return float(os.environ.get("VOICEAGENT_SERVICE_TIMEOUT_S", 300))


def _run_remote(coro):
    """Bridge a blocking remote call into whatever context we're in."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result(timeout=_timeout() + 30)


async def _remote_speak(url: str, text: str, language: str | None,
                        out_path: str) -> tuple[str, float]:
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=_timeout())
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(url) as ws:
            await ws.send_str(json_text({
                "op": "synthesize", "req_id": 1, "language": language,
                "text": text}))
            reader = FrameReader()
            while True:
                # aiohttp 3.9 removed the recv() alias: receive() and treat
                # any non-data frame (CLOSE/CLOSED/ERROR) as mid-request
                # death — same contract the brief wrote against recv().
                frame = await ws.receive()
                if frame.type not in (aiohttp.WSMsgType.TEXT,
                                      aiohttp.WSMsgType.BINARY):
                    raise GenericBackendError("tts service closed the "
                                              "connection mid-request")
                msg = reader.feed(frame.data)
                if msg is None:
                    continue
                if msg.op == "synthesis_done":
                    if msg.payload is None:
                        raise GenericBackendError(
                            "tts response missing WAV payload")
                    # The payload IS the complete WAV file the service
                    # synthesized — write it verbatim, no re-containering.
                    with open(out_path, "wb") as f:
                        f.write(msg.payload)
                    return out_path, float(msg.fields.get("seconds", 0.0))
                if msg.op == "error":
                    raise GenericBackendError(
                        f"tts service error: {msg.fields.get('code')}: "
                        f"{msg.fields.get('message')}")
    raise GenericBackendError("tts service never completed the request")


def speak(text: str, language: str | None = None,
          out_path: str | None = None) -> str:
    url = _url()
    if url is None:
        from voiceagent.tts import speak as _legacy
        return _legacy(text, language=language, out_path=out_path)
    import tempfile
    if out_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
    speakable = speech_text(text)
    try:
        _run_remote(_remote_speak(url, speakable, language, out_path))
    except GenericBackendError:
        raise
    except Exception as e:
        raise GenericBackendError(
            f"tts service unreachable ({type(e).__name__}: {e})") from e
    return out_path


def synthesize_to_wav(text: str, out_path: str) -> float:
    url = _url()
    if url is None:
        from voiceagent.tts import synthesize_to_wav as _legacy
        return _legacy(text, out_path)
    t0 = time.time()
    speak(text, out_path=out_path)
    return time.time() - t0


def warm_tts(language: str | None = None) -> str | None:
    url = _url()
    if url is None:
        from voiceagent.tts import get_tts_handle
        return get_tts_handle().warm(language)

    async def _go():
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=_timeout())
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(url) as ws:
                await ws.send_str(json_text(
                    {"op": "warm", "req_id": 1, "language": language}))
                while True:
                    # aiohttp 3.9 removed recv(): receive() and treat any
                    # non-data frame as a closed connection.
                    frame = await ws.receive()
                    if frame.type not in (aiohttp.WSMsgType.TEXT,
                                          aiohttp.WSMsgType.BINARY):
                        raise GenericBackendError("warmup: connection closed")
                    msg = FrameReader().feed(frame.data)
                    if msg is None:
                        continue
                    if msg.op == "ok":
                        return msg.fields.get("warmed")
                    if msg.op == "error":
                        raise GenericBackendError(
                            f"warmup failed: {msg.fields.get('message')}")

    try:
        return _run_remote(_go())
    except GenericBackendError:
        raise
    except Exception as e:
        raise GenericBackendError(
            f"tts service unreachable ({type(e).__name__}: {e})") from e
