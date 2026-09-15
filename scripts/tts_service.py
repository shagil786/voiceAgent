"""TTS capability service — wraps TTSHandle's voice resolution + synthesis;
speech_text stays brain-side (the client normalizes before sending; this
service calls synthesize_speakable semantics — voice resolution + synthesis,
never re-normalization). Voices live behind LoadedModelLRU slots so the
resident set stays capped and idle voices unload (spec 2026-09-14).

Wire: see voiceagent/service_protocol.py. A synthesize answer is one JSON
header ("synthesis_done", has_payload: true) followed by one binary frame
carrying the complete WAV file. Inference is serialized through a
single-worker executor (matches the in-process behavior; revisited in
Phase 3 per-session split).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import WSMsgType, web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent import tts as tts_mod  # noqa: E402
from voiceagent.model_registry import LoadedModelLRU  # noqa: E402
from voiceagent.service_protocol import (  # noqa: E402
    FrameReader, ProtocolError, error_body, json_text, require)

logger = logging.getLogger("tts_service")
EXECUTOR = ThreadPoolExecutor(max_workers=1)


class TTSService:
    def __init__(self, registry=None, voice_loader=None, registry_map=None,
                 model_dir=None, mms_loader=None):
        self._model_dir = model_dir or "data/models"
        self._registry_map = registry_map if registry_map is not None \
            else dict(tts_mod.VOICE_REGISTRY)
        self._registry_map.update(tts_mod.voice_overrides_from_env())
        self._voice_loader = voice_loader or tts_mod._real_voice_loader
        # mms: voices (piper-gap languages) dispatch to the MMS backend —
        # same split as TTSHandle._get_voice; injectable for tests.
        self._mms_loader = mms_loader or tts_mod._load_mms_voice
        self.registry = registry or LoadedModelLRU(
            capacity=int(os.environ.get("VOICEAGENT_SERVICE_MAX_LOADED", 2)),
            idle_unload_s=float(
                os.environ.get("VOICEAGENT_SERVICE_IDLE_UNLOAD_S", 600)))
        self._resolver = tts_mod.TTSHandle(
            registry=self._registry_map)  # voice_for/warm only; never synth

    def synthesize(self, text: str, language: str | None) -> dict:
        """ALREADY-SPEAKABLE text -> {"wav": bytes, "voice": str,
        "seconds": float}. speech_text is NEVER applied here — the brain
        owns normalization; spelled IDs must survive verbatim."""
        t0 = time.time()
        _, voice_name = self._resolver.voice_for(language, text)
        loader = (self._mms_loader if tts_mod.is_mms_voice(voice_name)
                  else self._voice_loader)
        voice = self.registry.get(
            f"voice:{voice_name}", lambda: loader(voice_name, self._model_dir))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            length_scale = self._resolver._length_scale
            if length_scale != tts_mod.DEFAULT_LENGTH_SCALE:
                from piper.config import SynthesisConfig
                voice.synthesize_wav(text, w, syn_config=SynthesisConfig(
                    length_scale=length_scale))
            else:
                voice.synthesize_wav(text, w)  # type: ignore[attr-defined]
        return {"wav": buf.getvalue(), "voice": voice_name,
                "seconds": round(time.time() - t0, 3)}

    def warm(self, language: str | None) -> str | None:
        """Preload the language's voice slot and return the voice name
        (mirrors TTSHandle.warm's contract)."""
        base = (str(language or "en").strip().lower().replace("_", "-")
                .split("-")[0] or "en")
        _, voice_name = self._resolver.voice_for(base, "")
        loader = (self._mms_loader if tts_mod.is_mms_voice(voice_name)
                  else self._voice_loader)
        self.registry.get(
            f"voice:{voice_name}", lambda: loader(voice_name, self._model_dir))
        return voice_name

    def describe(self) -> dict:
        stats = self.registry.stats()
        return {"op": "info", "service": "tts",
                "voices": sorted(set(self._registry_map.values())),
                "max_loaded": stats["capacity"],
                "idle_unload_s": float(
                    os.environ.get("VOICEAGENT_SERVICE_IDLE_UNLOAD_S", 600)),
                "loaded_keys": stats["loaded_keys"]}


async def handle_connection(ws, service: TTSService) -> None:
    """Serve one WebSocket connection until the peer closes."""
    reader = FrameReader()
    loop = asyncio.get_running_loop()
    while True:
        # aiohttp 3.9 removed the recv() alias (caught by Task 4's
        # real-socket tests): receive() and map any non-data frame
        # (CLOSE/CLOSING/CLOSED/ERROR) to the old recv() close contract.
        frame = await ws.receive()
        if frame.type not in (WSMsgType.TEXT, WSMsgType.BINARY):
            return
        raw = frame.data
        if raw is None:
            return
        try:
            msg = reader.feed(raw)
        except ProtocolError as e:
            await ws.send_str(json_text(error_body(None, "protocol", str(e))))
            return
        if msg is None:
            continue
        req_id = msg.req_id
        try:
            if msg.op == "synthesize":
                require(msg.fields, "text")
                result = await loop.run_in_executor(
                    EXECUTOR,
                    lambda m=msg: service.synthesize(
                        m.fields["text"], m.fields.get("language")))
                wav = result.pop("wav")
                # Wire contract: header declares the payload, WAV bytes
                # follow — the client's FrameReader completes on the bytes.
                await ws.send_str(json_text({
                    "op": "synthesis_done", "req_id": req_id,
                    "has_payload": True, **result}))
                await ws.send_bytes(wav)
            elif msg.op == "describe":
                await ws.send_str(json_text(service.describe()))
            elif msg.op == "warm":
                warmed = await loop.run_in_executor(
                    EXECUTOR,
                    lambda: service.warm(msg.fields.get("language")))
                await ws.send_str(json_text(
                    {"op": "ok", "req_id": req_id, "warmed": warmed}))
            else:
                await ws.send_str(json_text(error_body(
                    req_id, "protocol", f"unknown op '{msg.op}'")))
        except Exception as e:  # load/synthesis failures — one answer per req
            logger.exception("request failed")
            await ws.send_str(json_text(
                error_body(req_id, "tts_failed", f"{type(e).__name__}: {e}")))


def _check_token(request) -> bool:
    token = os.environ.get("VOICEAGENT_SERVICE_TOKEN")
    return token is None or request.headers.get("X-VoiceAgent-Token") == token


def build_app(service: TTSService) -> web.Application:
    app = web.Application()

    async def ws_handler(request):
        if not _check_token(request):
            return web.Response(status=401, text="bad service token")
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024)
        await ws.prepare(request)
        try:
            await handle_connection(ws, service)
        finally:
            await ws.close()
        return ws

    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/health", lambda r: web.json_response({"ok": True}))
    return app


async def _periodic_evict(service: TTSService) -> None:
    while True:
        await asyncio.sleep(60)
        service.registry.evict_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8711)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    service = TTSService()
    app = build_app(service)

    async def _start_evictor(app_):
        # aiohttp deprecated app.loop; create the task from the running
        # loop instead (plan-note ruling, task 3 brief).
        app_["evictor"] = asyncio.create_task(_periodic_evict(service))

    app.on_startup.append(_start_evictor)
    logger.info("tts service listening on %s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
