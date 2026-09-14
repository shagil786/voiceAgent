# Service Split Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move ASR and TTS out of the brain process into lazy-loaded, LRU-managed WebSocket services so the brain imports zero ML libraries and the 16 GB / 13:48 test run becomes structurally impossible.

**Architecture:** Wyoming-style narrow waist — JSON control frame + optional binary frame over WebSocket (aiohttp). Two new service scripts wrap the *existing* ASR handles and `TTSHandle` unchanged; brain-side dispatcher modules keep the exact legacy function signatures and switch on `VOICEAGENT_ASR_URL`/`VOICEAGENT_TTS_URL` (set → remote fail-closed; unset → byte-identical in-process). An `ml` pytest marker deselects model-weight tests by default.

**Tech Stack:** Python 3.12, aiohttp 3.9.5 (already installed via livekit, must be declared), pytest 8.3.4 (no pytest-asyncio — async tests are sync functions driving `asyncio.run`), SQLite-free (no new storage).

**Spec:** `docs/superpowers/specs/2026-09-14-service-split-design.md` — read it first; this plan implements its Goals 1–7 and Migration steps 1–6.

## Global Constraints

- Python: always `.venv/bin/python` (repo convention; voiceagent is not pip-installed).
- **No pytest-asyncio in this repo** — every async test is a sync `def` using `asyncio.run(...)` (a bare `async def` test is silently SKIPPED).
- **macOS torch-before-faiss:** knowledge.py must import sentence-transformers (torch) before `import faiss` — never reorder; conftest's torch import is deleted in Task 7 only because knowledge.py's internal order already guarantees it.
- **No silent fallback:** when `VOICEAGENT_ASR_URL`/`VOICEAGENT_TTS_URL` is set, a service failure raises — never fall back to in-process. Use `GenericBackendError` (`src/voiceagent/generic_backend.py:55`, a `TimeoutError` subclass — the governed timeout vocabulary).
- Services bind `127.0.0.1` by default; `X-VoiceAgent-Token` checked only when `VOICEAGENT_SERVICE_TOKEN` is set.
- Wire audio is declared, never assumed: transcribe requests carry `sample_rate`/`channels`/`sampwidth` in the JSON header; synthesize responses carry complete WAV file bytes at the voice's native rate.
- Inference is serialized per service (single-worker executor) — handles are not thread-safe; this matches today's behavior and is revisited in Phase 3.
- Tests must not download models in the default (`not ml`) tier.
- Existing behavior with env unset must stay byte-identical — pinned by existing tests, which must pass unmodified except for `ml` marker lines.

## File Structure

```
Create:
  src/voiceagent/service_protocol.py   # wire format codec (pure stdlib)
  src/voiceagent/model_registry.py     # LoadedModelLRU (pure stdlib)
  src/voiceagent/asr_client.py         # brain-side ASR dispatcher + remote client
  src/voiceagent/tts_client.py         # brain-side TTS dispatcher + remote client
  scripts/asr_service.py               # ASR service process (wraps existing handles)
  scripts/tts_service.py               # TTS service process (wraps existing TTSHandle)
  tests/test_service_protocol.py
  tests/test_model_registry.py
  tests/test_asr_service.py
  tests/test_tts_service.py
  tests/test_asr_client.py
  tests/test_tts_client.py
  tests/test_import_contract.py
  tests/test_services_ml.py
Modify:
  requirements.txt                     # declare aiohttp
  src/voiceagent/tts.py                # split speak() -> speech_text + synthesize_speakable()
  src/voiceagent/config.py             # DEFAULT_VOICES via module __getattr__ (lazy)
  src/voiceagent/voice_agent.py        # swap ASR/TTS imports to dispatchers
  src/voiceagent/voice.py              # swap ASR import to dispatcher
  src/voiceagent/telephony/inbound.py  # swap ASR import to dispatcher
  src/voiceagent/intent.py             # SentenceTransformer import -> inside IntentClassifier.__init__
  tests/conftest.py                    # delete torch import
  scripts/livekit_worker.py            # warmup block -> dispatcher warmup
  scripts/local_call.py                # warmup import -> dispatcher
  pyproject.toml                       # ml marker + addopts -m "not ml"
  tests/test_train_adapter.py, tests/test_asr.py, tests/test_tts.py,
  tests/test_knowledge_rag.py          # @pytest.mark.ml lines
  docs/telephony-runbook.md            # service boot section
```

---

### Task 1: Wire protocol codec

**Files:**
- Create: `src/voiceagent/service_protocol.py`
- Create: `tests/test_service_protocol.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces: `PROTOCOL_VERSION = 1`; `ProtocolError(ValueError)`; `Message` dataclass (`op: str`, `req_id: int | None`, `fields: dict`, `payload: bytes | None`); `FrameReader().feed(msg: str | bytes) -> Message | None` (returns a Message when a message completes, else `None`); `json_text(fields: dict) -> str`; `require(fields: dict, key: str)` (raises `ProtocolError` when missing); `error_body(req_id, code, message) -> dict`.

- [ ] **Step 1: Declare aiohttp in requirements**

Add to `requirements.txt` (after the livekit block, same comment style):

```
# Capability services (2026-09-14 service split): aiohttp already arrives
# transitively via livekit; declaring it pins the WS server/client stack the
# ASR/TTS services and brain clients are built on.
aiohttp==3.9.5
```

- [ ] **Step 2: Write the failing tests**

`tests/test_service_protocol.py`:

```python
# tests/test_service_protocol.py — wire codec: JSON header + one optional
# binary frame per message, version-checked, req_id-correlated.
import json

import pytest

from voiceagent.service_protocol import (
    PROTOCOL_VERSION, FrameReader, ProtocolError, error_body, json_text,
    require)


def test_text_message_is_self_complete():
    r = FrameReader()
    msg = r.feed(json_text({"v": 1, "op": "describe"}))
    assert msg.op == "describe" and msg.req_id is None and msg.payload is None


def test_declared_payload_completes_on_binary_frame():
    r = FrameReader()
    assert r.feed(json_text({"v": 1, "op": "transcribe", "req_id": 7,
                             "language": "en", "sample_rate": 16000,
                             "channels": 1, "sampwidth": 2,
                             "has_payload": True})) is None
    msg = r.feed(b"\x01\x02\x03\x04")
    assert msg.op == "transcribe" and msg.req_id == 7
    assert msg.payload == b"\x01\x02\x03\x04"
    assert msg.fields["sample_rate"] == 16000


def test_binary_after_self_complete_header_is_error():
    r = FrameReader()
    r.feed(json_text({"v": 1, "op": "describe"}))
    with pytest.raises(ProtocolError):
        r.feed(b"a")


def test_binary_before_header_rejected():
    with pytest.raises(ProtocolError):
        FrameReader().feed(b"a")


def test_wrong_version_rejected():
    with pytest.raises(ProtocolError):
        FrameReader().feed(json.dumps({"v": 99, "op": "describe"}))


def test_missing_op_rejected():
    with pytest.raises(ProtocolError):
        FrameReader().feed(json.dumps({"v": PROTOCOL_VERSION}))


def test_require_and_error_body():
    with pytest.raises(ProtocolError, match="language"):
        require({"op": "transcribe"}, "language")
    err = error_body(5, "model_load_failed", "boom")
    assert err == {"v": PROTOCOL_VERSION, "op": "error", "req_id": 5,
                   "code": "model_load_failed", "message": "boom"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_service_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voiceagent.service_protocol'`

- [ ] **Step 4: Implement the codec**

`src/voiceagent/service_protocol.py`:

```python
"""Wyoming-style narrow-waist wire protocol for VoiceAgent capability
services (spec 2026-09-14-service-split-design.md).

v1 framing: every message is one JSON text frame; a message MAY carry
exactly one binary frame, announced by `"has_payload": true` in its header
(raw PCM for transcribe requests, WAV bytes for synthesize responses).
Payload-less messages are self-complete the moment their header arrives —
no close-time flush races, and both directions agree via the explicit
flag. Audio shape is declared in the JSON header
(sample_rate/channels/sampwidth) — never assumed. Unknown version or op is
a protocol error: close the connection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """A frame violated the v1 wire contract."""


@dataclass
class Message:
    op: str
    req_id: int | None
    fields: dict
    payload: bytes | None


def json_text(fields: dict) -> str:
    body = {"v": PROTOCOL_VERSION, **fields}
    return json.dumps(body)


def require(fields: dict, key: str) -> None:
    if key not in fields:
        raise ProtocolError(f"missing required field '{key}'")


def error_body(req_id: int | None, code: str, message: str) -> dict:
    return {"v": PROTOCOL_VERSION, "op": "error", "req_id": req_id,
            "code": code, "message": message}


class FrameReader:
    """Assembles (JSON header, optional single binary payload) messages.

    A header WITHOUT has_payload is self-complete: feed() returns its
    Message immediately. A header WITH has_payload stashes until the next
    binary frame completes it. A binary frame with no stashed header, or a
    new header while a payload is still pending, is a ProtocolError."""

    def __init__(self):
        self._header: dict | None = None

    def feed(self, msg: str | bytes) -> Message | None:
        if isinstance(msg, (bytes, bytearray, memoryview)):
            if self._header is None:
                raise ProtocolError("binary frame before JSON header")
            header, self._header = self._header, None
            return Message(op=header["op"], req_id=header.get("req_id"),
                           fields=header, payload=bytes(msg))
        try:
            header = json.loads(msg)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"invalid JSON header: {e}") from e
        if not isinstance(header, dict) or header.get("v") != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version in {header!r}")
        if "op" not in header:
            raise ProtocolError("header missing 'op'")
        if header.get("has_payload"):
            self._header = header
            return None
        if self._header is not None:
            raise ProtocolError("new header while a binary payload is "
                                "still pending")
        return Message(op=header["op"], req_id=header.get("req_id"),
                       fields=header, payload=None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_service_protocol.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add src/voiceagent/service_protocol.py tests/test_service_protocol.py requirements.txt
git commit -m "feat(services): v1 wire protocol codec — JSON header + one binary frame, Wyoming-style"
```

---

### Task 2: LoadedModelLRU registry

**Files:**
- Create: `src/voiceagent/model_registry.py`
- Create: `tests/test_model_registry.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces: `LoadedModelLRU(capacity: int, idle_unload_s: float, clock: Callable[[], float] = time.monotonic)` with methods `get(key: str, loader: Callable[[], object]) -> object` (hit touches; miss loads; per-key load serialized; capacity evicts oldest), `evict_idle() -> int` (drops entries idle > idle_unload_s, returns count), `stats() -> dict` (`{"hits","misses","evictions","capacity","loaded_keys"}`), `remove(key: str) -> bool`. Eviction drops the reference (models free via GC; HF cache makes re-load cheap) — no closer hooks in v1.

- [ ] **Step 1: Write the failing tests**

`tests/test_model_registry.py`:

```python
# tests/test_model_registry.py — capacity + idle eviction + per-key load
# serialization, all with fake models and a fake clock (no real weights).
import threading
import time

from voiceagent.model_registry import LoadedModelLRU


class FakeModel:
    loads = 0

    def __init__(self, key):
        self.key = key
        FakeModel.loads += 1


def test_first_get_loads_second_get_hits():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    a = reg.get("qwen", lambda: FakeModel("qwen"))
    b = reg.get("qwen", lambda: FakeModel("qwen"))
    assert a is b and FakeModel.loads == 1
    assert reg.stats()["hits"] == 1 and reg.stats()["misses"] == 1


def test_capacity_evicts_oldest():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    reg.get("a", lambda: FakeModel("a"))
    reg.get("b", lambda: FakeModel("b"))
    reg.get("a", lambda: FakeModel("a"))   # touch a -> b is now oldest
    reg.get("c", lambda: FakeModel("c"))   # evicts b
    assert reg.stats()["evictions"] == 1
    assert reg.stats()["loaded_keys"] == ["a", "c"]


def test_evict_idle_drops_only_stale_entries():
    now = [1000.0]
    reg = LoadedModelLRU(capacity=2, idle_unload_s=10, clock=lambda: now[0])
    reg.get("a", lambda: FakeModel("a"))
    now[0] = 1005.0
    reg.get("b", lambda: FakeModel("b"))
    now[0] = 1012.0                        # a idle 12s > 10, b idle 7s
    assert reg.evict_idle() == 1
    assert reg.stats()["loaded_keys"] == ["b"]


def test_per_key_load_is_serialized_under_concurrency():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    gate = threading.Event()
    loads = []

    def slow_loader():
        loads.append(threading.current_thread().name)
        gate.wait(timeout=5)
        return FakeModel("qwen")

    results = []

    def worker():
        results.append(reg.get("qwen", slow_loader))

    t1 = threading.Thread(target=worker)
    t1.start()
    time.sleep(0.05)                       # let t1 enter the loader
    t2 = threading.Thread(target=worker)
    t2.start()
    time.sleep(0.05)
    gate.set()
    t1.join(); t2.join()
    assert results[0] is results[1]
    assert len(loads) == 1                 # second caller reused the load


def test_remove():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    reg.get("a", lambda: FakeModel("a"))
    assert reg.remove("a") is True
    assert reg.remove("a") is False
    assert reg.get("a", lambda: FakeModel("a")) is not None
    assert FakeModel.loads == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_model_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/voiceagent/model_registry.py`:

```python
"""Capacity-capped, idle-unloading registry of loaded models — the lazy
home for capability services (spec 2026-09-14-service-split-design.md).

Eviction drops the handle reference; weights free via GC and re-load hits
the local HF cache (no re-download). Thread-safe: a short lock guards the
map, per-key locks serialize concurrent first-loads so a cold key never
loads twice and unrelated keys never block each other.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable


class LoadedModelLRU:
    def __init__(self, capacity: int, idle_unload_s: float,
                 clock: Callable[[], float] = time.monotonic):
        self._capacity = max(1, int(capacity))
        self._idle_s = float(idle_unload_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._loading: dict[str, threading.Lock] = {}
        self._entries: OrderedDict[str, tuple[object, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, loader: Callable[[], object]) -> object:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                obj, _ = entry
                self._entries.move_to_end(key)
                self._entries[key] = (obj, self._clock())
                self._hits += 1
                return obj
            self._misses += 1
            load_lock = self._loading.setdefault(key, threading.Lock())
        with load_lock:
            with self._lock:  # re-check: another thread may have loaded
                entry = self._entries.get(key)
                if entry is not None:
                    obj, _ = entry
                    self._entries.move_to_end(key)
                    self._entries[key] = (obj, self._clock())
                    return obj
                obj = loader()
                self._evict_for_capacity(protected=key)
                self._entries[key] = (obj, self._clock())
                return obj

    def _evict_for_capacity(self, protected: str) -> None:
        while len(self._entries) >= self._capacity:
            oldest = next(iter(self._entries))
            if oldest == protected and len(self._entries) == 1:
                break
            del self._entries[oldest]
            self._evictions += 1

    def evict_idle(self) -> int:
        with self._lock:
            now = self._clock()
            stale = [k for k, (_, last) in self._entries.items()
                     if now - last > self._idle_s]
            for k in stale:
                del self._entries[k]
            return len(stale)

    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses,
                    "evictions": self._evictions,
                    "capacity": self._capacity,
                    "loaded_keys": list(self._entries)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_model_registry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/model_registry.py tests/test_model_registry.py
git commit -m "feat(services): LoadedModelLRU — capacity-capped, idle-unloading model registry"
```

---

### Task 3: ASR service

**Files:**
- Create: `scripts/asr_service.py`
- Create: `tests/test_asr_service.py`

**Interfaces:**
- Consumes: `voiceagent.asr` (`get_asr_for_language(lang, engines, supported, warn)`, `QwenASRHandle()`, `IndicASRHandle()`, `WhisperASRHandle(model="small")`, `_normalize_lang`, `_ASR_ROUTES`, `INDIC_CONFORMER_LANGUAGES`, `_get_whisper_small`), `voiceagent.service_protocol` (Task 1), `voiceagent.model_registry.LoadedModelLRU` (Task 2).
- Produces (for Task 4's remote client and Task 8's ml tests): `ASRService(registry=None, supported=None, routes=None, engine_loaders=None, warn=None)` with sync methods `transcribe(pcm: bytes, sample_rate: int, channels: int, sampwidth: int, language: str | None) -> dict` (`{"text", "engine"}`, engine key `"qwen" | "indic:<base>" | "whisper"`), `warm(language: str | None) -> str | None`, `describe() -> dict`; `handle_connection(ws: AnyReader, service: ASRService) -> None` (async; `AnyReader` = anything with async `send_str/send_bytes/recv()` returning `str | bytes | None`); `build_app(service) -> aiohttp.web.Application`; `main()` (argparse `--port` default 8710, `--host` default 127.0.0.1, `--capacity` default `VOICEAGENT_SERVICE_MAX_LOADED` or 2, `--idle-unload` default `VOICEAGENT_SERVICE_IDLE_UNLOAD_S` or 600).

- [ ] **Step 1: Write the failing tests**

`tests/test_asr_service.py` — the engine is a stub recording its inputs; a FakeWebSocket drives `handle_connection` without a real socket (no pytest-asyncio):

```python
# tests/test_asr_service.py — service logic against STUB engines: routing,
# LRU integration, wire handling via a FakeWebSocket. Real-model parity
# lives in tests/test_services_ml.py (ml tier).
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

    def transcribe(self, audio, language=None):
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
    handle = calls["indic:te"]
    wav_path, lang = handle.calls[0]
    assert lang == "te"
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2


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
```

Note: `scripts/` has no `__init__.py` — add the import path the way the test suite already does for scripts (check how existing script-importing tests do it, e.g. `tests/test_onboard_cli.py`); if they use `sys.path.insert`, mirror that at the top of the test file with a comment.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_asr_service.py -v`
Expected: FAIL — import error (`scripts.asr_service` missing)

- [ ] **Step 3: Implement the service**

`scripts/asr_service.py`:

```python
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
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

from voiceagent import asr as asr_mod
from voiceagent.model_registry import LoadedModelLRU
from voiceagent.service_protocol import (
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

    def _registry_engine(self, key):
        def _get():
            return self.registry.get(key, self._loaders[key])
        return _get

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
        handle = self.registry.get(key, self._loaders[key])
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
        self.registry.get(key, self._loaders[key])
        self.registry.get("whisper", self._loaders["whisper"])
        return base

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
    app.on_startup.append(
        lambda app_: app_.loop.create_task(_periodic_evict(service)))
    logger.info("asr service listening on %s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_asr_service.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/asr_service.py tests/test_asr_service.py
git commit -m "feat(services): ASR service — registry-backed existing handles over the v1 wire"
```

---

### Task 4: Brain-side ASR client + dispatcher

**Files:**
- Create: `src/voiceagent/asr_client.py`
- Create: `tests/test_asr_client.py`

**Interfaces:**
- Consumes: `scripts.asr_service.build_app` + `ASRService` with stub engines (Task 3), `voiceagent.service_protocol` (Task 1), `voiceagent.generic_backend.GenericBackendError`.
- Produces (the swap surface Tasks 6 uses): `transcribe_wav_routed(path: str, language: str | None = None) -> str`; `warmup_asr() -> None`; `warmup_asr_for_language(lang: str | None) -> str | None` — all three dispatch on `VOICEAGENT_ASR_URL`: set → remote (fail-closed via `GenericBackendError`), unset → legacy `voiceagent.asr` functions (imported lazily inside the call). Remote timeout = `VOICEAGENT_SERVICE_TIMEOUT_S` (default 300 — cold model load takes ~100 s).

- [ ] **Step 1: Write the failing tests**

`tests/test_asr_client.py` — real aiohttp server on an ephemeral port with STUB engines (fast, no models); plus env-unset legacy dispatch and no-fallback semantics:

```python
# tests/test_asr_client.py — dispatcher: remote (env set, real aiohttp
# loopback server with STUB engines), legacy (env unset), fail-closed
# (server error -> GenericBackendError, never a local fallback).
import asyncio
import json

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
    """Boot a real stub service on an ephemeral port; return (url, stop)."""
    runner = web.AppRunner(build_app(stub_service()))

    async def _boot():
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return f"ws://127.0.0.1:{runner.addresses[0][1]}/ws"

    url = asyncio.run(_boot())

    def _stop():
        asyncio.run(runner.cleanup())

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_asr_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voiceagent.asr_client'`

- [ ] **Step 3: Implement the dispatcher**

`src/voiceagent/asr_client.py`:

```python
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
                raw = await ws.recv()
                if raw is None:
                    raise GenericBackendError("asr service closed the "
                                              "connection mid-request")
                msg = reader.feed(raw)
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
                    raw = await ws.recv()
                    if raw is None:
                        raise GenericBackendError("warmup: connection closed")
                    msg = FrameReader().feed(raw)
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
```

One test caveat to resolve while implementing: `TestServer` + `runner.setup()` — if the ephemeral-port boot proves fiddly, boot `web.AppRunner` on port 0 and read `runner.addresses` for the resolved port; the test helper `_serve` above may be adjusted for that as long as the tests' assertions stand.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_asr_client.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/asr_client.py tests/test_asr_client.py
git commit -m "feat(services): brain-side ASR dispatcher — remote fail-closed, legacy env-unset, loop-safe"
```

---

### Task 5: TTS service + client (+ synthesize_speakable split)

**Files:**
- Modify: `src/voiceagent/tts.py`
- Create: `scripts/tts_service.py`
- Create: `src/voiceagent/tts_client.py`
- Create: `tests/test_tts_service.py`
- Test: `tests/test_tts.py` (existing tests must stay green)

**Interfaces:**
- Consumes: `voiceagent.tts` (`TTSHandle`, `VOICE_REGISTRY`, `speech_text`, `resolve_voice_lang`), Tasks 1–2 modules.
- Produces: `TTSHandle.synthesize_speakable(text: str, language: str | None, out_path: str) -> str` (voice resolution + synthesis WITHOUT speech_text; `speak()` becomes `speech_text(text)` + this); `scripts/tts_service.TTSService(transcribe-like sync API): synthesize(text: str, language: str | None) -> dict` (`{"wav": bytes, "voice": str, "seconds": float}`), `warm(language) -> str | None`, `describe() -> dict`; `handle_connection(ws, service)`; `build_app(service)`. Brain side: `tts_client.speak(text, language=None, out_path=None) -> str`, `tts_client.synthesize_to_wav(text, out_path) -> float`, `tts_client.warm_tts(language=None) -> str | None` — dispatching on `VOICEAGENT_TTS_URL` exactly like Task 4.

- [ ] **Step 1: Split speak() in tts.py (failing test first)**

Add to `tests/test_tts.py` (stub-loader pattern already used in that file):

```python
def test_synthesize_speakable_skips_normalization():
    """synthesize_speakable is the wire contract: the client (brain) already
    applied speech_text, so the handle must NOT re-normalize (spelled IDs
    must survive verbatim)."""
    from voiceagent.tts import TTSHandle
    captured = {}

    class StubVoice:
        def synthesize_wav(self, text, w, syn_config=None):
            captured["text"] = text

    handle = TTSHandle(registry={"en": "en_US-amy-medium"},
                       voice_loader=lambda name, d: StubVoice())
    handle.synthesize_speakable("O R D, 9 0 2 1", "en", out_path="/tmp/x.wav")
    assert captured["text"] == "O R D, 9 0 2 1"  # NOT re-folded/spelled


def test_speak_still_normalizes():
    from voiceagent.tts import TTSHandle
    captured = {}

    class StubVoice:
        def synthesize_wav(self, text, w, syn_config=None):
            captured["text"] = text

    handle = TTSHandle(registry={"en": "en_US-amy-medium"},
                       voice_loader=lambda name, d: StubVoice())
    handle.speak("**ORD-9021**", "en", out_path="/tmp/x.wav")
    assert captured["text"] == "O R D, 9 0 2 1"
```

Run: `.venv/bin/python -m pytest tests/test_tts.py::test_synthesize_speakable_skips_normalization -v` → FAIL (no attribute).

Refactor in `src/voiceagent/tts.py`: extract the synthesis block of `speak()` (voice_for → _get_voice → wave write, keeping the `length_scale` SynthesisConfig branch verbatim) into:

```python
    def synthesize_speakable(self, text: str, language: str | None,
                             out_path: str) -> str:
        """Synthesize ALREADY-SPEAKABLE text (speech_text applied upstream —
        the wire contract for the TTS service). speak() = speech_text + this."""
        _, voice_name = self.voice_for(language, text)
        voice = self._get_voice(voice_name)
        with wave.open(out_path, "wb") as w:
            if self._length_scale != DEFAULT_LENGTH_SCALE:
                from piper.config import SynthesisConfig
                voice.synthesize_wav(  # type: ignore[attr-defined]
                    text, w, syn_config=SynthesisConfig(
                        length_scale=self._length_scale))
            else:
                voice.synthesize_wav(text, w)  # type: ignore[attr-defined]
        return out_path
```

and `speak()` becomes: `text = speech_text(text)` + resolve out_path (unchanged temp-file block) + `return self.synthesize_speakable(text, language, out_path)`.

Run: `.venv/bin/python -m pytest tests/test_tts.py -v` → PASS (all, including the two new ones).

- [ ] **Step 2: Write the failing service/client tests**

`tests/test_tts_service.py`:

```python
# tests/test_tts_service.py — TTS service with a stub voice loader + the
# brain dispatcher against a real loopback server (no piper downloads).
import asyncio
import io
import json
import wave

import pytest
from aiohttp import web

from scripts.tts_service import TTSService, build_app, handle_connection
from voiceagent import tts_client
from voiceagent.model_registry import LoadedModelLRU
from voiceagent.tts import VOICE_REGISTRY


class StubVoice:
    def synthesize_wav(self, text, w, syn_config=None):
        with wave.open(w) as _:  # w is a wave.Wave_write; write minimal pcm
            pass
        # write real frames so the WAV is parseable
        w._file.write(b"\x00\x00" * 100)


def make_service():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    svc = TTSService(registry=reg,
                     voice_loader=lambda name, d: StubVoice(),
                     registry_map=dict(VOICE_REGISTRY))
    return svc


def test_synthesize_returns_wav_bytes_and_voice():
    svc = make_service()
    out = svc.synthesize("hello there", language="en")
    assert out["voice"].startswith("en_US")
    buf = io.BytesIO(out["wav"])
    with wave.open(buf, "rb") as w:
        assert w.getnchannels() == 1


def test_client_speak_writes_wav_file(tmp_path, monkeypatch):
    svc = make_service()
    runner = web.AppRunner(build_app(svc))

    async def boot():
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return f"ws://127.0.0.1:{runner.addresses[0][1]}/ws"

    url = asyncio.run(boot())
    monkeypatch.setenv("VOICEAGENT_TTS_URL", url)
    try:
        out = str(tmp_path / "reply.wav")
        path = tts_client.speak("hello there", out_path=out)
        assert path == out
        with wave.open(out, "rb") as w:
            assert w.getnframes() > 0
    finally:
        asyncio.run(runner.cleanup())


def test_client_env_unset_uses_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv("VOICEAGENT_TTS_URL", raising=False)
    import voiceagent.tts as tts_mod
    monkeypatch.setattr(tts_mod, "speak",
                        lambda text, language=None, out_path=None: "legacy")
    assert tts_client.speak("x") == "legacy"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tts_service.py -v`
Expected: FAIL — `ModuleNotFoundError` (tts_service missing)

- [ ] **Step 4: Implement the service + client**

`scripts/tts_service.py` (mirror of Task 3's shape; only the service class + synthesize op differ — the connection handler, token check, build_app, main are the same pattern as `asr_service.py` with `service="tts"`, default port 8711, and the synthesize op returning WAV bytes before a done frame):

```python
"""TTS capability service — wraps TTSHandle's voice resolution + synthesis;
speech_text stays brain-side (the client normalizes before sending; this
service calls synthesize_speakable, which never re-normalizes)."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

from voiceagent import tts as tts_mod
from voiceagent.model_registry import LoadedModelLRU
from voiceagent.service_protocol import (
    FrameReader, ProtocolError, error_body, json_text, require)

logger = logging.getLogger("tts_service")
EXECUTOR = ThreadPoolExecutor(max_workers=1)


class TTSService:
    def __init__(self, registry=None, voice_loader=None, registry_map=None,
                 model_dir=None):
        self._model_dir = model_dir or "data/models"
        self._registry_map = registry_map if registry_map is not None \
            else dict(tts_mod.VOICE_REGISTRY)
        self._registry_map.update(tts_mod.voice_overrides_from_env())
        self._voice_loader = voice_loader or tts_mod._real_voice_loader
        self.registry = registry or LoadedModelLRU(
            capacity=int(os.environ.get("VOICEAGENT_SERVICE_MAX_LOADED", 2)),
            idle_unload_s=float(
                os.environ.get("VOICEAGENT_SERVICE_IDLE_UNLOAD_S", 600)))
        self._resolver = tts_mod.TTSHandle(
            registry=self._registry_map)  # voice_for/warm only; never synth

    def synthesize(self, text: str, language: str | None) -> dict:
        t0 = time.time()
        _, voice_name = self._resolver.voice_for(language, text)
        voice = self.registry.get(
            f"voice:{voice_name}",
            lambda: self._voice_loader(voice_name, self._model_dir))
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
        base = (str(language or "en").strip().lower().replace("_", "-")
                .split("-")[0] or "en")
        _, voice_name = self._resolver.voice_for(base, "")
        self.registry.get(
            f"voice:{voice_name}",
            lambda: self._voice_loader(voice_name, self._model_dir))
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
        except Exception as e:
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
    app.on_startup.append(
        lambda app_: app_.loop.create_task(_periodic_evict(service)))
    logger.info("tts service listening on %s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

`src/voiceagent/tts_client.py`:

```python
"""Brain-side TTS dispatcher — speak/synthesize_to_wav/warm_tts over the
TTS service when VOICEAGENT_TTS_URL is set, legacy in-process otherwise.
speech_text normalization happens HERE (brain-owned, deterministic
mechanics); the service receives already-speakable text. Fail-closed: a
declared service that is down raises GenericBackendError — no fallback."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
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
                raw = await ws.recv()
                if raw is None:
                    raise GenericBackendError("tts service closed the "
                                              "connection mid-request")
                msg = reader.feed(raw)
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
                    raw = await ws.recv()
                    if raw is None:
                        raise GenericBackendError("warmup: connection closed")
                    msg = FrameReader().feed(raw)
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
```

Implementation notes: (a) the WAV-bytes delivery uses `FrameReader`'s one-binary-frame rule — the client writes `msg.payload` to `out_path` when it arrives; drop the dead `"binary"`-op branch if the reader makes it unreachable; (b) the StubVoice in the test writes via `w._file` — if that private attr is brittle, change the stub to build a real in-memory WAV and copy frames; the assertions are what matter.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tts_service.py tests/test_tts.py -v`
Expected: PASS (3 new + all existing tts tests)

- [ ] **Step 6: Commit**

```bash
git add src/voiceagent/tts.py src/voiceagent/tts_client.py scripts/tts_service.py tests/test_tts_service.py tests/test_tts.py
git commit -m "feat(services): TTS service + brain dispatcher — speech_text stays brain-side, voices behind LRU"
```

---

### Task 6: Call-site swaps + config lazy import

**Files:**
- Modify: `src/voiceagent/voice_agent.py:10,13`
- Modify: `src/voiceagent/voice.py:84`
- Modify: `src/voiceagent/telephony/inbound.py:36`
- Modify: `scripts/livekit_worker.py:66,86,93` (warmup block)
- Modify: `scripts/local_call.py:120`
- Modify: `src/voiceagent/config.py:42` (lazy DEFAULT_VOICES)
- Test: `tests/test_service_swap.py` (new wiring test)

**Interfaces:**
- Consumes: `voiceagent.asr_client` (`transcribe_wav_routed`, `warmup_asr`, `warmup_asr_for_language`), `voiceagent.tts_client` (`speak`, `synthesize_to_wav`, `warm_tts`) — identical signatures to the legacy modules (Tasks 4–5).
- Produces: no new surface; behavior routing only.

- [ ] **Step 1: Write the failing wiring test**

`tests/test_service_swap.py`:

```python
# tests/test_service_swap.py — the voice path enters through the
# dispatchers, so env-selected services are reachable from every entry
# (loopback, HTTP chat, telephony) without per-entry wiring.
import subprocess
import sys


SWAP_SUBPROCESS = """
import sys, json
sys.path.insert(0, "src")
import voiceagent.voice_agent as va
import voiceagent.telephony.inbound as inbound
mods = set(sys.modules)
print(json.dumps({
    "voice_agent_uses_dispatcher": "voiceagent.asr_client" in mods,
    "inbound_lazy_ok": True,  # inbound imports inside the function
}))
"""

CONFIG_LAZY_SUBPROCESS = """
import sys, json
sys.path.insert(0, "src")
import voiceagent.config as config
print(json.dumps({"tts_loaded": "voiceagent.tts" in sys.modules}))
config.DEFAULT_VOICES  # touch it
print(json.dumps({"tts_loaded_after_touch": "voiceagent.tts" in sys.modules}))
"""


def test_voice_agent_imports_dispatchers():
    r = subprocess.run([sys.executable, "-c", SWAP_SUBPROCESS],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert '"voice_agent_uses_dispatcher": true' in r.stdout


def test_config_default_voices_is_lazy():
    lines = [ln for ln in subprocess.run(
        [sys.executable, "-c", CONFIG_LAZY_SUBPROCESS],
        capture_output=True, text=True).stdout.splitlines() if ln.startswith("{")]
    assert '"tts_loaded": false' in lines[0]
    assert '"tts_loaded_after_touch": true' in lines[1]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_service_swap.py -v`
Expected: FAIL — voice_agent still imports `voiceagent.asr` directly; config still imports tts eagerly.

- [ ] **Step 3: Make the swaps**

1. `src/voiceagent/voice_agent.py`: `from voiceagent.asr import transcribe_wav_routed` → `from voiceagent.asr_client import transcribe_wav_routed`; `from voiceagent.tts import speak` → `from voiceagent.tts_client import speak`. No other change (signatures identical).
2. `src/voiceagent/voice.py:84`: `from voiceagent.asr import transcribe_wav_routed` → `from voiceagent.asr_client import transcribe_wav_routed`.
3. `src/voiceagent/telephony/inbound.py:36`: same swap.
4. `scripts/local_call.py:120`: `from voiceagent.asr import warmup_asr` → `from voiceagent.asr_client import warmup_asr`.
5. `scripts/livekit_worker.py` warmup block: `from voiceagent.asr import warmup_asr` → `from voiceagent.asr_client import warmup_asr`; `from voiceagent.asr import warmup_asr_for_language` → `from voiceagent.asr_client import warmup_asr_for_language`; `from voiceagent.tts import get_tts_handle` + `voice = get_tts_handle().warm(language)` → `from voiceagent.tts_client import warm_tts` + `voice = warm_tts(language)`. Update the block's comments to note the dispatcher decides in-process vs service.
6. `src/voiceagent/config.py`: replace the module-level import with a module `__getattr__` (PEP 562) preserving the single-source comment:

```python
# SINGLE SOURCE: tts.VOICE_REGISTRY (a verbatim copy here drifted stale —
# it kept the old male Hindi voice after the registry moved to priyamvada).
# Lazy via module __getattr__: config consumers must not import the tts
# module (and its voice/download plumbing) merely for the default names.
def __getattr__(name):
    if name == "DEFAULT_VOICES":
        from voiceagent.tts import VOICE_REGISTRY
        return VOICE_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 4: Run the wiring test + the affected suites**

Run: `.venv/bin/python -m pytest tests/test_service_swap.py tests/test_voice_agent.py tests/test_config.py tests/test_livekit_inbound.py -v`
Expected: PASS. If a test pins the direct `voiceagent.asr` import, update the pin to `asr_client` (behavior contract unchanged) and note it in the commit body.

- [ ] **Step 5: Full-suite gate — DEFERRED to Task 8 (ruling 2026-09-14)**

Do NOT run the full suite in this task: until Task 7's markers exist, a full run loads every model and spikes ~16-20 GB RAM on the 16 GB Mac mini (violates the no-heavy-runs-without-warning constraint; it actually happened during execution and was killed). Task 6's changes are import swaps — targeted verification is sufficient: run the wiring test + the stub-based suites touching the swapped call sites (test_service_swap, test_voice_agent, test_config, test_livekit_inbound, test_voice, test_tts_service, test_asr_client — all fast). The full-suite before/after measurement happens in Task 8 where the fast tier exists.

- [ ] **Step 6: Commit**

```bash
git add src/voiceagent/voice_agent.py src/voiceagent/voice.py src/voiceagent/telephony/inbound.py src/voiceagent/config.py scripts/livekit_worker.py scripts/local_call.py tests/test_service_swap.py
git commit -m "feat(services): voice path enters through ASR/TTS dispatchers; config DEFAULT_VOICES lazy"
```

---

### Task 7: Import surgery + ml test tier + import contract

**Files:**
- Modify: `tests/conftest.py` (delete torch import)
- Modify: `src/voiceagent/intent.py:30` (lazy ST import)
- Modify: `pyproject.toml` (ml marker + addopts)
- Modify: `tests/test_train_adapter.py`, `tests/test_asr.py`, `tests/test_tts.py`, `tests/test_knowledge_rag.py` (marker lines)
- Create: `tests/test_import_contract.py`

**Interfaces:**
- Consumes: Tasks 1–6 (dispatchers exist).
- Produces: default `pytest` run = fast tier (no model weights); `pytest -m ml` = full heavy tier; CI-assertable import boundary.

- [ ] **Step 1: Make intent.py's ST import lazy**

In `src/voiceagent/intent.py`: delete line 30 (`from sentence_transformers import SentenceTransformer`). In `IntentClassifier.__init__`, immediately before the two `SentenceTransformer(...)` constructions (lines ~86–87):

```python
        # Lazy: importing this module must stay ML-free (import-contract
        # test) — the embedders load only when a classifier is BUILT.
        from sentence_transformers import SentenceTransformer
```

Verify: `.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import voiceagent.intent; assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules; print('intent import clean')"` → prints `intent import clean`.

- [ ] **Step 2: Delete the conftest torch import**

Replace `tests/conftest.py` contents with:

```python
"""Test-suite-wide configuration.

Historical note (2026-09-14 service split): this module used to import
torch at session start because macOS segfaults when faiss loads before
torch's OpenMP runtime. The guard now lives at the ONLY faiss import site
(voiceagent/knowledge.py, sentence-transformers before `import faiss`),
and the default test tier must not load ML libraries at all — see
tests/test_import_contract.py.
"""
```

Then run `pytest tests/test_knowledge.py tests/test_knowledge_rag.py tests/test_intent.py -q` (the faiss-adjacent suites) to confirm no segfault and no new failures.

- [ ] **Step 3: Register the ml marker + default deselect in pyproject**

In `[tool.pytest.ini_options]` add:

```toml
markers = [
    "ml: loads real model weights or voices (deselected by default; run with -m ml)",
]
addopts = "-m 'not ml'"
```

- [ ] **Step 4: Mark the heavy tests**

Apply `@pytest.mark.ml` (add `import pytest` where missing) to:
- `tests/test_train_adapter.py::test_train_loop_end_to_end` (torch/peft training loop; `test_empty_data_refuses` stays fast),
- `tests/test_asr.py` — file-level `pytestmark = pytest.mark.ml` (real whisper tiny),
- `tests/test_tts.py::test_synthesize_to_wav_creates_file` (real piper synth),
- `tests/test_knowledge_rag.py::test_real_encoder_retrieval_ranks_matching_section_first` (real LaBSE encode).

Sweep for stragglers (tests that load weights but got missed):

```bash
.venv/bin/python -m pytest tests -q --durations=15 -p no:cacheprovider
```

Any fast-tier test showing >10 s gets inspected: if it loads a real model, mark it `ml` too (record each such add in the commit body). Repeat until the fast tier's slowest test is <10 s.

- [ ] **Step 5: Write the import-contract test**

`tests/test_import_contract.py`:

```python
# tests/test_import_contract.py — the CI-assertable architecture boundary:
# with service URLs declared, importing the brain + dispatchers loads ZERO
# ML libraries. Runs each probe in a fresh subprocess (the pytest session
# itself may have torch resident from other tiers).
import json
import subprocess
import sys

ML_MODULES = ["torch", "faiss", "sentence_transformers", "transformers",
              "faster_whisper", "piper", "librosa", "peft"]

REMOTE_PROBE = """
import sys, json
sys.path.insert(0, "src")
import voiceagent.runtime as rt          # the brain assembly seam
import voiceagent.asr_client, voiceagent.tts_client
import voiceagent.voice_agent             # the voice-loop entry
bad = [m for m in {ml} if m in sys.modules]
print(json.dumps({{"bad": bad}}))
""".format(json.dumps(ML_MODULES))

LEGACY_PROBE = """
import sys, json
sys.path.insert(0, "src")
import voiceagent.asr_client, voiceagent.tts_client   # dispatchers only
print(json.dumps({{"ml_at_import": [m for m in {ml} if m in sys.modules]}}))
""".format(json.dumps(ML_MODULES))


def _run(probe: str, env_extra: dict) -> dict:
    import os
    env = dict(os.environ)
    env.pop("VOICEAGENT_ASR_URL", None)
    env.pop("VOICEAGENT_TTS_URL", None)
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_brain_imports_zero_ml_with_services_declared():
    out = _run(REMOTE_PROBE, {"VOICEAGENT_ASR_URL": "ws://127.0.0.1:8710/ws",
                              "VOICEAGENT_TTS_URL": "ws://127.0.0.1:8711/ws"})
    assert out["bad"] == [], "brain loaded ML modules: %s" % out["bad"]


def test_dispatchers_import_clean_without_urls():
    out = _run(LEGACY_PROBE, {})
    assert out["ml_at_import"] == [], out["ml_at_import"]
```

- [ ] **Step 6: Run the contract + measure the fast tier**

Run:
```bash
.venv/bin/python -m pytest tests/test_import_contract.py -v
time .venv/bin/python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -3
/usr/bin/time -l .venv/bin/python -m pytest tests -q -p no:cacheprovider 2>&1 | grep -i "maximum resident"
```
Expected: contract tests PASS; fast tier wall time far below the 828 s baseline and peak RSS far below ~16 GB. Record BOTH numbers (plus the baseline 828 s / ~16 GB) in the commit body — they are the sprint's headline result.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py src/voiceagent/intent.py pyproject.toml tests/test_train_adapter.py tests/test_asr.py tests/test_tts.py tests/test_knowledge_rag.py tests/test_import_contract.py
git commit -m "feat(services): ml test tier + import contract — the brain cannot load ML libraries"
```

---

### Task 8: ml-tier service tests + e2e proof + runbook

**Files:**
- Create: `tests/test_services_ml.py`
- Modify: `docs/telephony-runbook.md` (service boot section)

**Interfaces:**
- Consumes: Tasks 3–7 complete (services, clients, markers).
- Produces: wire-vs-in-process parity proof (ml tier), e2e proof at the standing bar, operational docs.

- [ ] **Step 1: Write the ml-tier tests**

`tests/test_services_ml.py`:

```python
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
    runner = web.AppRunner(build())

    async def _go():
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return f"ws://127.0.0.1:{runner.addresses[0][1]}/ws"

    url = asyncio.run(_go())
    return url, lambda: asyncio.run(runner.cleanup())


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
```

- [ ] **Step 2: Run the ml tier**

Run: `.venv/bin/python -m pytest tests -m ml -q -p no:cacheprovider`
Expected: PASS (including the pre-existing marked tests from Task 7). Cold whisper/Qwen loads make this the minutes-long tier — that is its job; it is never part of the default run.

- [ ] **Step 3: E2E proof against live services**

Boot both services and the e2e check exactly as the runbook will document:

```bash
set -a; source .env; set +a
.venv/bin/python scripts/asr_service.py --port 8710 > /tmp/asr_service.log 2>&1 &
.venv/bin/python scripts/tts_service.py --port 8711 > /tmp/tts_service.log 2>&1 &
export VOICEAGENT_ASR_URL=ws://127.0.0.1:8710/ws VOICEAGENT_TTS_URL=ws://127.0.0.1:8711/ws
.venv/bin/python scripts/voice_e2e_check.py
```

Expected: the standing bar — 4/4 PASS + the 2 honest XFAILs (hinglish ASR garble → Kaggle fine-tune; blind native callers by design), with asr/tts turn timings comparable to the in-process baseline. Then kill the services and re-run once WITHOUT the env vars to prove the legacy in-process path still passes identically.

- [ ] **Step 4: Document in the runbook**

Add a "Capability services (ASR/TTS)" section to `docs/telephony-runbook.md`: boot order (services before the worker), the two URL env vars, ports 8710/8711, `VOICEAGENT_SERVICE_MAX_LOADED`/`VOICEAGENT_SERVICE_IDLE_UNLOAD_S`/`VOICEAGENT_SERVICE_TIMEOUT_S`/`VOICEAGENT_SERVICE_TOKEN`, the health endpoint (`GET /health`), and the invariant: **URLs unset = exact legacy in-process behavior; URLs set = fail-closed remote (no fallback)**. Update the README architecture row if one exists.

- [ ] **Step 5: Final full-suite verification + commit**

```bash
.venv/bin/python -m pytest tests -q -p no:cacheprovider   # fast tier
.venv/bin/python -m pytest tests -m ml -q -p no:cacheprovider  # heavy tier
```

Both green → commit:

```bash
git add tests/test_services_ml.py docs/telephony-runbook.md
git commit -m "test(services): real-model wire parity + e2e proof; runbook service section"
```

---

## Completion checklist (maps to spec Goals 1–7)

- [x] `service_protocol.py` v1 codec + tests (G1)
- [x] `LoadedModelLRU` + tests (G2/G3 lazy-load core)
- [x] ASR service wrapping existing handles + tests (G2)
- [x] TTS service + `synthesize_speakable` split + tests (G3)
- [x] Brain dispatchers, fail-closed, loop-safe + tests (G4)
- [x] Import surgery: conftest clean, intent.py lazy, config lazy (G5)
- [x] Import-contract test green (G5, CI-assertable boundary)
- [x] `ml` marker registered; default tier < 10 s/test; before/after wall + peak-RSS recorded (G6)
- [x] voice_e2e_check 4/4 + 2 XFAIL against services; legacy path re-verified (G7)
- [x] Runbook section committed

## Sprint headline (measured 2026-09-14, single-process fast tier)

| Metric | Baseline (pre-split) | After (phase 1 done) |
|---|---|---|
| Fast-tier wall time | 828 s | **99.6 s** (8.3x) |
| Peak RSS | ~16 GB | **1.05 GB** (15x) |
| Result | 1089 passed / 3 skipped / 1 xfailed, no aborts | same |

Note: the T7-era 209 s figure was the 6-chunk sweep (per-process startup
overhead); the single-process number above is the honest tier measurement.
The full-suite abort root cause is documented in the 4b5b804 commit body
(duplicate libomp; ST-before-faiss restored at session start).
