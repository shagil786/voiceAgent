"""ASR capability service (spec 2026-09-14-service-split-design.md).

Wraps the EXISTING language-routed handles unchanged (QwenASRHandle /
IndicASRHandle / WhisperASRHandle) behind LoadedModelLRU slots keyed by the
routed engine ("qwen" / "indic:<base>" / "whisper"). Routing + the
engine-failure fallback to whisper are REUSED from voiceagent.asr by
injecting registry-backed engine getters — service semantics are
in-process semantics by construction, not a reimplementation.

Wire: see voiceagent/service_protocol.py. Inference is serialized through
a single-worker executor (handles are not thread-safe; matches the
in-process behavior; revisited in Phase 3 per-session split).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent import asr as asr_mod  # noqa: E402
from voiceagent.model_registry import LoadedModelLRU  # noqa: E402
from voiceagent.service_protocol import (  # noqa: E402
    FrameReader, ProtocolError, error_body, json_text, require)

logger = logging.getLogger("asr_service")

EXECUTOR = ThreadPoolExecutor(max_workers=1)


class ASRService:
    """Router + registry-backed engines + whisper fallback, sync API."""

    def __init__(self, registry=None, supported=None, routes=None,
                 engine_loaders=None, warn=None):
        self.registry = registry or LoadedModelLRU(
            capacity=int(os.environ.get("VOICEAGENT_SERVICE_MAX_LOADED", 2)),
            idle_unload_s=float(
                os.environ.get("VOICEAGENT_SERVICE_IDLE_UNLOAD_S", 600)))
        self._supported = (supported if supported is not None
                           else asr_mod.INDIC_CONFORMER_LANGUAGES)
        self._routes = routes if routes is not None else asr_mod._ASR_ROUTES
        self._loaders = engine_loaders or {
            "qwen": lambda: asr_mod.QwenASRHandle(),
            "indic": lambda: asr_mod.IndicASRHandle(),
            "whisper": lambda: asr_mod.WhisperASRHandle(model="small"),
        }
        self._warn = warn or (lambda m: logger.warning(m))

    def _loader_for(self, key):
        """Factory for a routed registry key: "indic:te" -> the "indic"
        engine loader (registry keys carry the language, loader keys do
        not)."""
        return self._loaders[key.split(":", 1)[0]]

    def _route(self, language):
        base = asr_mod._normalize_lang(language)
        if self._routes.get(base or "") == "indic":
            if base in self._supported:
                return f"indic:{base}", base
            self._warn(f"language '{base}' is not supported by the Indic "
                       "conformer; falling back to the Qwen core engine")
        return "qwen", base

    def transcribe(self, pcm: bytes, sample_rate: int, channels: int,
                   sampwidth: int, language: str | None) -> dict:
        key, base = self._route(language)
        handle = self.registry.get(key, self._loader_for(key))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(int(channels))
                w.setsampwidth(int(sampwidth))
                w.setframerate(int(sample_rate))
                w.writeframes(pcm)
            try:
                text = handle.transcribe(path, language=base)
            except Exception:
                fb = self.registry.get("whisper", self._loaders["whisper"])
                self._warn(f"engine '{key}' failed; falling back to whisper "
                           "small (in-process router semantics)")
                text = fb.transcribe(path, language=base)
            return {"text": str(text).strip(), "engine": key}
        finally:
            os.unlink(path)

    def warm(self, language: str | None) -> str | None:
        base = asr_mod._normalize_lang(language)
        if base is None:
            key = "qwen"
        elif self._routes.get(base) == "indic" and base in self._supported:
            key = f"indic:{base}"
        else:
            key = "qwen"
        self.registry.get(key, self._loader_for(key))
        self.registry.get("whisper", self._loaders["whisper"])
        return key

    def describe(self) -> dict:
        stats = self.registry.stats()
        return {"op": "info", "service": "asr",
                "engines": ["qwen", "indic", "whisper"],
                "max_loaded": stats["capacity"],
                "idle_unload_s": float(
                    os.environ.get("VOICEAGENT_SERVICE_IDLE_UNLOAD_S", 600)),
                "loaded_keys": stats["loaded_keys"],
                "stats": {k: v for k, v in stats.items()
                          if k != "loaded_keys"}}


async def handle_connection(ws, service: ASRService) -> None:
    """Serve one WebSocket connection until the peer closes."""
    reader = FrameReader()
    loop = asyncio.get_running_loop()
    while True:
        raw = await ws.recv()
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
            if msg.op == "transcribe":
                require(msg.fields, "sample_rate")
                if msg.payload is None:
                    raise ProtocolError("transcribe without payload")
                result = await loop.run_in_executor(
                    EXECUTOR,
                    lambda m=msg: service.transcribe(
                        m.payload, m.fields["sample_rate"],
                        m.fields.get("channels", 1),
                        m.fields.get("sampwidth", 2),
                        m.fields.get("language")))
                await ws.send_str(json_text({
                    "op": "transcription", "req_id": req_id, **result}))
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
        except Exception as e:  # engine/load failures — one answer per req
            logger.exception("request failed")
            await ws.send_str(json_text(
                error_body(req_id, "asr_failed", f"{type(e).__name__}: {e}")))


async def _periodic_evict(service: ASRService) -> None:
    while True:
        await asyncio.sleep(60)
        service.registry.evict_idle()


def _check_token(request) -> bool:
    token = os.environ.get("VOICEAGENT_SERVICE_TOKEN")
    return token is None or request.headers.get("X-VoiceAgent-Token") == token


def build_app(service: ASRService) -> web.Application:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8710)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--idle-unload", type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.capacity is not None:
        os.environ["VOICEAGENT_SERVICE_MAX_LOADED"] = str(args.capacity)
    if args.idle_unload is not None:
        os.environ["VOICEAGENT_SERVICE_IDLE_UNLOAD_S"] = str(args.idle_unload)
    service = ASRService()
    app = build_app(service)

    async def _start_evictor(app_):
        # aiohttp deprecated app.loop; create the task from the running
        # loop instead (plan-note ruling, task 3 brief).
        app_["evictor"] = asyncio.create_task(_periodic_evict(service))

    app.on_startup.append(_start_evictor)
    logger.info("asr service listening on %s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
