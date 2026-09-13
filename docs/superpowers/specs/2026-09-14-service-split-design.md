# Service Split — Phase 1: ASR + TTS Out of the Brain

**Date:** 2026-09-14
**Status:** Approved design (user approved 2026-09-14), pending implementation plan
**Sprint:** "Real Architecture" Phase 1 — capability services, lazy-loaded, never all at once

## Summary

Every production voice agent studied on 2026-09-14 (LiveKit Agents, Pipecat,
TEN Framework, Vapi/Retell, Wyoming) converges on the same DNA: the session
brain carries **zero model weights**, every capability is a network service,
and heavy compute is isolated in its own process. Our repo is the outlier —
one Python process imports torch, sentence-transformers, faiss, whisper and
piper together (evidence: `tests/conftest.py:8` torch import; eager
sentence_transformers/faiss imports in `intent.py:30`, `knowledge.py:10,15`;
the test suite peaks at ~16 GB RAM and 13:48 wall for 1068 tests because the
whole ML stack is resident in one pytest process).

Phase 1 splits the two heaviest, most self-contained capabilities — ASR and
TTS — into standalone service processes speaking a Wyoming-style narrow-waist
protocol (JSON control + raw PCM). The brain gets thin clients implementing
the *existing* handle surfaces, so callers don't change. Lazy loading moves
to its correct home: **inside** each model service (registry + LRU + idle
unload). The deterministic BASE tier becomes physically enforced — the brain
process cannot load torch even by accident, and a CI import-contract test
asserts it.

Topology becomes deployment config, not code: Phase 1 runs everything on
localhost ports; the same wire protocol later spans machines (Phase 3 VPS)
without code changes.

## Goals

1. `service_protocol.py` — versioned JSON-header + PCM framing over
   WebSocket, with a `describe` capability handshake (Wyoming-inspired,
   aiohttp — already installed via livekit).
2. `scripts/asr_service.py` — wraps the existing language-routed handles
   (QwenASRHandle / IndicASRHandle / WhisperASRHandle) **unchanged**, adds
   the LRU model registry: load `(engine, language)` on first request,
   unload idle + capacity-capped.
3. `scripts/tts_service.py` — wraps `TTSHandle` unchanged; voice lazy-load
   via the same LRU; `speech_text()` normalization stays brain-side
   (deterministic mechanics are brain-owned; the wire carries final
   speakable text and returns audio).
4. Brain clients (`asr_client.py`, `tts_client.py`) implementing the
   existing surfaces (`transcribe_wav_routed`, `speak`,
   `synthesize_to_wav`). Selection by env: `VOICEAGENT_ASR_URL` /
   `VOICEAGENT_TTS_URL` set → remote; unset → in-process legacy. **No
   silent fallback** — service failure raises the same backend-error class
   the escalation path already handles.
5. Import surgery + CI contract: conftest's torch import deleted; the
   torch-before-faiss guard moves to its only true home (`knowledge.py`,
   immediately before `import faiss`); `intent.py` SentenceTransformer load
   goes lazy; a test asserts `import voiceagent.runtime` (with service URLs
   set) leaves none of
   `{torch, faiss, sentence_transformers, transformers, faster_whisper,
   piper, librosa, peft}` in `sys.modules`.
6. Test tiers: `ml` marker for model-weight tests (deselected by default);
   fast tier runs the brain + stubbed services — sub-3-minute, no torch
   resident. `pytest -m ml` restores the full heavy run.
7. Proof: `scripts/voice_e2e_check.py` against localhost services passes at
   the standing bar (4/4 PASS + 2 honest XFAILs), and `livekit_worker.py`
   boots with remote ASR/TTS (warmup becomes service warmup calls — the
   worker never loads models itself).

## Non-goals (this sprint)

- RAG/embedding service extraction (Phase 2 — ST + faiss stay in-process
  behind lazy imports until then).
- Per-session worker processes (Phase 3, LiveKit job-subprocess style —
  needed only at real call concurrency).
- Streaming ASR/TTS over the wire (Phase 1 is utterance-level, matching the
  current chunked pipeline; the wire format reserves binary framing so
  Sprint B full-duplex streams without a protocol break).
- GPU box, VPS deploy, compose supervision (Phase 3; Phase 1 services are
  plain localhost processes with a documented boot order).
- LLM adapters (already remote services), training loop (stays script-side
  batch, never in the serving path).

## Architecture

```
Telnyx SIP ──► LiveKit SFU ──► livekit_worker (the BRAIN — stays LIGHT)
                                dialogue, policy, GovernedToolRunner,
                                tenant bundle, LLM adapters (remote)
                                     │  VOICEAGENT_ASR_URL / VOICEAGENT_TTS_URL
              ┌──────────────────────┴───────────────────┐
              ▼ WS JSON+PCM                              ▼ WS JSON+PCM
     ┌──────────────────┐                       ┌──────────────────┐
     │ asr_service.py   │                       │ tts_service.py   │
     │ LRU registry     │                       │ LRU registry     │
     │  ├ Qwen handle   │                       │  ├ piper voices  │
     │  ├ Indic handle  │                       │  (lazy per voice)│
     │  └ whisper (fb)  │                       └──────────────────┘
     │  (lazy per lang) │
     └──────────────────┘
     localhost:8710 (default)                   localhost:8711 (default)

Env unset (or URLs unset) ⇒ exact legacy in-process behavior, byte-for-byte.
```

The handles themselves are untouched platform code. The sprint adds the wire,
the registry, the clients, and the import discipline.

## Component design

### 1. `src/voiceagent/service_protocol.py` (new, stdlib + aiohttp)

Wire format — every request is one JSON text frame, optionally followed by
binary frame(s):

- `{"v":1, "op":"describe"}` →
  `{"v":1, "op":"info", "service":"asr", "engines":[{name, languages}...],
   "max_loaded":2, "idle_unload_s":600}` (tts: `voices:[...]`).
- transcribe: text `{"v":1,"op":"transcribe","req_id":N,"language":"te",
  "declared":true}` + one binary frame (16 kHz mono s16le PCM, the
  canonical wire rate) → `{"v":1,"op":"transcription","req_id":N,
  "text":..., "engine":"indic-ctc", "detected":"te"}`.
- synthesize: text `{"v":1,"op":"synthesize","req_id":N,"language":"hi-IN",
  "text":"..."}` → binary frame(s) (PCM, same canonical rate) → final
  `{"v":1,"op":"synthesis_done","req_id":N,"seconds":1.84}`.
- warmup: `{"v":1,"op":"warm","req_id":N,"language":...}` (tts: voice) →
  `{"v":1,"op":"ok","req_id":N}`.
- `{"v":1,"op":"error","req_id":N,"code":"model_load_failed",
  "message":...}` — unknown `v` or `op` is a protocol error, connection
  closed with `code:"protocol"`.

Auth: optional `X-VoiceAgent-Token` header checked against
`VOICEAGENT_SERVICE_TOKEN`; services bind `127.0.0.1` by default. No token
required on loopback (Wyoming's trust model, documented).

### 2. `src/voiceagent/model_registry.py` (new, generic, no ML imports)

`LoadedModelLRU`: `get(key, loader)`, `touch(key)`, `evict_idle(now)`,
`capacity` from `VOICEAGENT_SERVICE_MAX_LOADED` (default 2), idle from
`VOICEAGENT_SERVICE_IDLE_UNLOAD_S` (default 600). Thread-locked; per-key
inference serialized (handles are not thread-safe — same behavior as today,
documented); hit/miss/evict counters exposed on `describe`. Stub-tested
(fast tier) with fake model classes.

### 3. `scripts/asr_service.py` (new)

aiohttp WS server. On boot: loads the routing data exactly as
`asr.transcribe_wav_routed` does (same `data/lang/*.yaml` sources), wraps
`get_asr_for_language` behind the LRU — the registry key is the routed
engine slot (`qwen` / `indic:<lang>` / `whisper`). `warm` op mirrors
`warmup_asr_for_language`. PCM→WAV wrapping reuses the existing chunk
plumbing; WAV path behavior of `transcribe_wav_routed` is preserved for
in-process mode only.

### 4. `scripts/tts_service.py` (new)

aiohttp WS server wrapping `TTSHandle`; registry key = resolved voice name
(`resolve_voice_lang` logic, same env overrides). Brain sends
`speech_text()`-normalized text (normalization stays a pure brain function
with its existing tests).

### 5. Brain clients: `src/voiceagent/asr_client.py`, `tts_client.py` (new)

- `RemoteASR` implements `transcribe_wav_routed(path_or_pcm, language=…)`;
  `RemoteTTS` implements `speak(text, language=…)` and
  `synthesize_to_wav(text, out_path)` — the exact surfaces callers use
  today (`voice_agent.py:59`, `voice.py:84-87`, `livekit_worker.py`
  warmup block, `scripts/local_call.py`).
- Selection helper `client_for(service)` in `config.py`: URL env set →
  remote; unset → in-process. Call sites change only in *where* they get
  the handle, not how they call it.
- Failure semantics: connect/send/timeout raise the existing
  backend-error type (same class as `backend_timeout` in the tool
  gateway) so DialogueTracker/escalation handle it with zero new paths.
  **No retry-then-local-fallback** — a declared service that is down is an
  incident, not a silent quality change.

### 6. Import surgery (existing files, minimal diffs)

- `tests/conftest.py`: the module-level `import torch` is deleted; its
  docstring comment moves to `knowledge.py`.
- `src/voiceagent/knowledge.py`: `import torch` immediately before
  `import faiss` (the OpenMP segfault guard, at the only place faiss is
  imported).
- `src/voiceagent/intent.py:30`: `from sentence_transformers import
  SentenceTransformer` moves inside the classifier factory (first-use load).
- `src/voiceagent/config.py:42`: `VOICE_REGISTRY` import from `tts` becomes
  a lazy attribute (voice names are data; no reason to import piper plumbing
  for a default list).

### 7. Test tiers (pyproject + marks)

- `markers = ["ml: loads real model weights or voices (deselected by
  default)"]`, `addopts = "-m 'not ml'"`.
- Marked `ml` (initial sweep, ~40 tests): `test_train_adapter.py` (torch/
  peft loop), the real-ST test in `test_knowledge_rag.py:548`, model tests
  in `test_asr.py`, real-handle tests in `test_tts.py`, `test_benchmark.py`,
  e2e voice tests.
- Service tests are tier-split: protocol/LRU/clients tested against a stub
  engine (fast); `pytest -m ml` spins real services on ephemeral ports.
- New import-contract test (fast): with both URLs set, import
  `voiceagent.runtime` + both clients, assert the ML module set is absent
  from `sys.modules`, and assert `transcribe`/`speak` dispatch to the
  remote path.

## Migration steps (each independently shippable)

1. `service_protocol.py` + `model_registry.py` with fast tests (pure
   logic, no models).
2. `asr_service.py` + `asr_client.py` + call-site swap + contract tests.
3. `tts_service.py` + `tts_client.py` + call-site swap.
4. Import surgery + marker sweep + addopts; record suite wall-time and
   peak-RSS before/after in the PR description (the 16 GB/13:48 numbers
   are the baseline being beaten).
5. E2E proof: `voice_e2e_check.py` with services up (standing 4/4 + 2
   XFAIL bar), `livekit_worker.py` boot against services, legacy in-process
   path re-verified (env unset).
6. RUNBOOK section: boot order, env vars, ports, token, health/describe.

## Testing strategy

Fast tier (default): protocol framing round-trips, LRU evict/idle/capacity
with fakes, client selection + no-fallback failure semantics, import
contract, marker hygiene (`pytest -m ml` includes; default excludes).
ML tier: real service processes with real models on ephemeral ports,
`transcribe_wav_routed` parity (same text in-process vs over the wire for
canned samples), warmup parity, voice_e2e against services.

## Risks

- **Latency**: one localhost WS hop per utterance (~sub-ms wire + existing
  inference). E2E script already reports asr/tts turn timings — compared
  before/after in step 5.
- **Concurrency**: handles serialize per engine key (current behavior);
  multi-call scaling is a Phase 3 concern (per-session workers), not
  Phase 1.
- **Double implementation drift**: in-process legacy path stays as the
  reference; the ml-tier parity test pins remote == local on canned
  samples, so drift fails CI.
- **Lazy ST in `intent.py`**: the sidecar classifier is warmed at worker
  boot today (the 17.5s first-turn mute fix); in remote mode the warmup
  call goes to the service, in legacy mode the existing factory warmup
  still runs — both covered in step 5.
