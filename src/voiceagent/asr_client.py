"""Brain-side ASR dispatcher — the exact signatures of voiceagent.asr's
voice-loop entries, backed by the ASR service when VOICEAGENT_ASR_URL is
set, by the legacy in-process handles when it is not (spec 2026-09-14).

Fail-closed: with a URL declared, connect/send/timeout/engine errors raise
GenericBackendError (the governed timeout vocabulary) — NEVER a silent
fallback to in-process. The service-side whisper fallback (router
semantics) still applies inside the service; client-side failures are
incidents.

The remote call is a blocking round-trip; callers inside a running event
loop (telephony worker) are bridged through a worker thread because
asyncio.run cannot nest.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import wave

from voiceagent.generic_backend import GenericBackendError
from voiceagent.service_protocol import FrameReader, json_text

logger = logging.getLogger(__name__)


def _url() -> str | None:
    return os.environ.get("VOICEAGENT_ASR_URL") or None


def _timeout() -> float:
    return float(os.environ.get("VOICEAGENT_SERVICE_TIMEOUT_S", 300))


def _read_wav(path: str) -> tuple[bytes, int, int, int]:
    with wave.open(str(path), "rb") as w:
        return (w.readframes(w.getnframes()), w.getframerate(),
                w.getnchannels(), w.getsampwidth())


async def _remote_transcribe(url: str, path: str,
                             language: str | None) -> str:
    import aiohttp
    pcm, rate, ch, sw = _read_wav(path)
    timeout = aiohttp.ClientTimeout(total=_timeout())
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(url) as ws:
            await ws.send_str(json_text({
                "op": "transcribe", "req_id": 1, "language": language,
                "sample_rate": rate, "channels": ch, "sampwidth": sw,
                "has_payload": True}))
            await ws.send_bytes(pcm)
            reader = FrameReader()
            while True:
                # aiohttp 3.9 removed the recv() alias: receive() and treat
                # any non-data frame (CLOSE/CLOSED/ERROR) as mid-request
                # death — same contract the brief wrote against recv().
                frame = await ws.receive()
                if frame.type not in (aiohttp.WSMsgType.TEXT,
                                      aiohttp.WSMsgType.BINARY):
                    raise GenericBackendError("asr service closed the "
                                              "connection mid-request")
                msg = reader.feed(frame.data)
                if msg is None:
                    continue
                if msg.op == "transcription":
                    return msg.fields["text"]
                if msg.op == "error":
                    raise GenericBackendError(
                        f"asr service error: {msg.fields.get('code')}: "
                        f"{msg.fields.get('message')}")


def _run_remote(coro):
    """Bridge a blocking remote call into whatever context we're in."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result(timeout=_timeout() + 30)


def transcribe_wav_routed(path: str, language: str | None = None) -> str:
    url = _url()
    if url is None:
        from voiceagent.asr import transcribe_wav_routed as _legacy
        return _legacy(path, language=language)
    try:
        return _run_remote(_remote_transcribe(url, path, language))
    except GenericBackendError:
        raise
    except Exception as e:
        raise GenericBackendError(
            f"asr service unreachable ({type(e).__name__}: {e})") from e


def warmup_asr() -> None:
    url = _url()
    if url is None:
        from voiceagent.asr import warmup_asr as _legacy
        return _legacy()
    _warm_op(url, None)


def warmup_asr_for_language(lang: str | None) -> str | None:
    url = _url()
    if url is None:
        from voiceagent.asr import warmup_asr_for_language as _legacy
        return _legacy(lang)
    return _warm_op(url, lang)


def _warm_op(url: str, language: str | None) -> str | None:
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
            f"asr service unreachable ({type(e).__name__}: {e})") from e
