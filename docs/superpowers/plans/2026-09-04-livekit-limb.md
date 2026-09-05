# LiveKit Telephony Limb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LiveKit Cloud carries real PSTN audio both ways (+1 240 231 5037 inbound; outbound dial-out) while the governed Orchestrator stays the only brain — LiveKit is transport, never the agent.

**Architecture:** A worker process bridges LiveKit rooms and the existing turn loop: inbound webhook (`room_started` for `call-*`) → mint token → `rtc.Room` join → subscribe the SIP participant's 48kHz audio → numpy-resample to 16kHz → existing `StreamingVAD` turn detection → `Orchestrator.handle_turn` (ASR text in) → chunked Piper reply → upsample → publish via `AudioSource`, with `BargeInController` cancelling playback. Outbound reverses it: our dialer creates the room, the worker joins, `api.sip.create_sip_participant` dials out, existing `Sub600msAMD` classifies the early audio, and `campaign_turn` runs only on human answer. No Agents-framework pipeline, no second brain, no policy bypass: every spoken turn is one governed `handle_turn`.

**Tech Stack:** Python 3.12, `livekit` (realtime SDK, provides `livekit.rtc`) + `livekit-api` (server SDK) (new approved deps — first non-stdlib addition since foundation; pin exact versions in `requirements.txt` with a comment recording the resolution date), `numpy` (already a dep — resampling), existing VAD/ASR/TTS/Orchestrator/dialer/AMD untouched.

**Spec:** Design spec import path is `Orchestrator.handle_turn` / `campaign_turn` (NOT legacy `voice_agent.voice_turn`, which drives the old `Agent` path and stays for CLI demos). LiveKit primitives per current docs: inbound trunk + dispatch rule (already configured: `voice-agent`, Individual, `call-`, +12402315037) → SIP participant auto-joins room; outbound via `CreateSIPParticipant` (stored trunk or inline); `wait_until_answered` semantics; voicemail answers SIP 200 OK so AMD stays mandatory.

## Global Constraints

- The brain is always `Orchestrator.handle_turn` / `campaign_turn` with the deployment's policy + gateway + decision log; the limb proposes nothing and executes nothing by itself.
- Audio contract: room side 48kHz mono int16 (assert `frame.sample_rate`, resample if different); pipeline side 16kHz mono int16 (VAD/ASR/TTS native). Resample with numpy linear interpolation only.
- Secrets (`LIVEKIT_URL/KEY/SECRET`) come from `.env` (already present, gitignored) via `RuntimeConfig` extension; never logged, never in code.
- All tests run offline: fake rooms/tracks/SIP API; zero PSTN cost in CI. The single live PSTN drill is a documented manual runbook, not a test.
- Existing pytest suite stays green. New deps pinned in `requirements.txt`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/voiceagent/config.py` (extend) | `livekit_url/key/secret/number/trunk_id/room_prefix` fields + `LIVEKIT_*` env reads |
| `src/voiceagent/telephony/audio.py` | numpy 48k↔16k resample + 10ms/20ms chunking helpers (pure functions) |
| `src/voiceagent/telephony/livekit_bridge.py` | `BridgeSession`: subscribe→VAD→`on_utterance`→publish loop with barge-in cancellation; `on_utterance` callback `(pcm16: bytes) -> (reply_text, reply_wav16)` |
| `src/voiceagent/telephony/inbound.py` | Webhook receiver (stdlib `http.server`, livekit `WebhookReceiver` validation) → token mint (`AccessToken`) → join → greet via `handle_turn("(Inbound call connected — greet the caller.)")` → bridge loop → leave on disconnect |
| `src/voiceagent/telephony/outbound.py` | `dial_out(api, room_name, to_number, trunk_id, timeout_s) -> str` (`connected|failed|timeout`) via `create_sip_participant` + participant-join wait; AMD handoff point documented for `campaign_turn` |
| `scripts/livekit_worker.py` | Worker entrypoint: serves webhook, spawns per-room sessions, graceful shutdown |
| `scripts/livekit_dial.py` | CLI: create room + dial + report disposition (manual PSTN drill vehicle) |
| `docs/telephony-runbook.md` | Dispatch rule, numbers, trunk, test-call procedure, per-minute cost notes, recording-consent reminder |
| `tests/test_livekit_audio.py`, `tests/test_bridge.py`, `tests/test_livekit_inbound.py`, `tests/test_livekit_outbound.py` | Offline tests with fakes |

---

### Task 1: Config extension + deps + audio resample

**Files:**
- Modify: `src/voiceagent/config.py`, `requirements.txt`
- Create: `src/voiceagent/telephony/audio.py`
- Test: `tests/test_livekit_audio.py`

**Interfaces:**
- Consumes: `RuntimeConfig/load_config` conventions (Task: read `config.py` first — same dataclass + env pattern).
- Produces: `RuntimeConfig` gains `livekit_url/key/secret/number: str|None = None`, `livekit_trunk_id: str|None = None`, `livekit_room_prefix: str = "call-"` reading `LIVEKIT_URL/KEY/SECRET/NUMBER/TRUNK_ID/ROOM_PREFIX`; `resample_48k_to_16k(pcm: bytes) -> bytes`, `resample_16k_to_48k(pcm: bytes) -> bytes`, `chunk_frames(pcm: bytes, frame_ms: int, sample_rate: int) -> list[bytes]` in `telephony/audio.py`.

- [x] **Step 1: Write the failing test**

```python
# tests/test_livekit_audio.py
import math, struct
from voiceagent.telephony.audio import (
    chunk_frames, resample_16k_to_48k, resample_48k_to_16k)

def _sine_16k(ms=100, hz=440):
    n = 16 * ms
    return struct.pack(f"<{n}h", *[int(3000 * math.sin(2 * math.pi * hz * i / 16000)) for i in range(n)])

def test_roundtrip_preserves_tone():
    tone = _sine_16k()
    up = resample_16k_to_48k(tone)
    assert len(up) == len(tone) * 3
    back = resample_48k_to_16k(up)
    assert len(back) == len(tone)
    import struct as st
    amps = st.unpack(f"<{len(back)//2}h", back)
    assert max(abs(a) for a in amps) > 1000  # tone survived

def test_chunking_exact():
    assert len(chunk_frames(_sine_16k(ms=100), 20, 16000)) == 5

def test_config_livekit_fields():
    from voiceagent.config import load_config
    c = load_config(env={"LIVEKIT_URL": "wss://x", "LIVEKIT_ROOM_PREFIX": "call-"})
    assert c.livekit_url == "wss://x" and c.livekit_number is None
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_livekit_audio.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.telephony.audio'`

- [x] **Step 3: Write minimal implementation**

`audio.py`: numpy linear interp — `np.frombuffer(pcm, dtype=np.int16).astype(np.float32)`, `np.interp(new_idx, old_idx, x)`, clip to int16 range, `.tobytes()`. Ratios exactly 3:1 / 1:3. `chunk_frames`: split into `sample_rate*frame_ms//1000` samples per chunk (drop trailing partial). Config: 5 new fields + env reads in `load_config` (same style as frontier keys); do NOT change existing fields. `requirements.txt`: `livekit==1.1.17` + `livekit-api==1.2.1` (resolved 2026-09-04; NOTE: there is no `livekit-rtc` package on PyPI — the realtime SDK is `livekit`, imported as `from livekit import rtc`); then `.venv/bin/pip install` them for later tasks (network needed once).

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_livekit_audio.py tests/test_config.py -q`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/voiceagent/telephony/audio.py src/voiceagent/config.py requirements.txt tests/test_livekit_audio.py
git commit -m "feat: livekit config, pinned deps, PCM resample helpers"
```

---

### Task 2: Bridge session (room audio ↔ turn loop, offline-testable)

**Files:**
- Create: `src/voiceagent/telephony/livekit_bridge.py`
- Test: `tests/test_bridge.py`

**Interfaces:**
- Consumes: `StreamingVAD/BargeInController` (`telephony/stream.py:85-156`); defines its own `on_utterance` callback contract (see Produces).
- Produces: `BridgeSession(on_utterance, on_barge_in=None)` with `feed_pcm16(frame_20ms: bytes) -> None` (raises `ValueError` on non-640-byte frames), `take_playback() -> bytes | None` (960-byte 48k chunks), `barge_in() -> None`, `stop() -> None`; `on_utterance: Callable[[bytes], tuple[str, bytes]]` (16k PCM in → `(reply_text, reply_wav_16k)` out).

Contract (exact, keeps the room out of this file for offline tests): the session owns VAD + playback queue. `feed_pcm16` accumulates; when VAD emits `complete_audio`, session calls `asr_stub(audio)` — NO: ASR belongs to wiring (Task 4), not the session. Instead session emits completed utterances via callback: constructor takes `on_utterance: Callable[[bytes], tuple[str, bytes]]` receiving 16k PCM, returning `(reply_text, reply_wav_16k)`. Session upsamples WAV to 48k, splits into 10ms chunks, queues; `take_playback` pops one chunk (returns None when idle/after `stop`). Barge-in: VAD's controller is constructed with `on_barge_in` that clears the playback queue (`stop audition, keep session`). STT/TTS/Orchestrator wiring lives in Task 4's `scripts/livekit_worker.py` + `telephony/inbound.py`, NOT here — this task is pure pump mechanics.

- [x] **Step 1: Write the failing test**

```python
# tests/test_bridge.py
from voiceagent.telephony.livekit_bridge import BridgeSession

def _wav16_of_speech_like():
    import math, struct
    n = 16000  # 1s of tone-as-speech (energy VAD fires on RMS)
    return struct.pack(f"<{n}h", *[int(8000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n)])

def test_utterance_roundtrip_and_bargein_cancel():
    calls = []
    def on_utterance(pcm):
        calls.append(pcm)
        import struct
        wav = struct.pack(f"<{len(pcm)//2}h", *[1000] * (len(pcm) // 2))
        return ("hello", wav)
    s = BridgeSession(on_utterance=on_utterance)
    tone = _wav16_of_speech_like()
    for i in range(0, len(tone), 640):  # 20ms 16k frames
        s.feed_pcm16(tone[i:i+640])
    for _ in range(25):  # trailing silence to endpoint the turn
        s.feed_pcm16(bytes(640))
    assert len(calls) == 1
    chunks = []
    while (c := s.take_playback()) is not None:
        chunks.append(c)
    assert chunks and len(chunks[0]) == 960  # 10ms @48k mono int16
    # barge-in cancels pending playback
    s2 = BridgeSession(on_utterance=on_utterance)
    for i in range(0, len(tone), 640):
        s2.feed_pcm16(tone[i:i+640])
    for _ in range(25):
        s2.feed_pcm16(bytes(640))
    assert s2.take_playback() is not None
    s2.feed_pcm16(tone[:640])  # speech while speaking → barge-in path arms
    s2.barge_in()  # explicit external trigger also clears
    assert s2.take_playback() is None
    s2.stop(); s.stop()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bridge.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write minimal implementation**

`BridgeSession`: owns `StreamingVAD()` (defaults: 16k, 20ms frames, own `BargeInController` whose `on_barge_in` clears `self._play` deque); `start_speaking(turn_id)` called when queuing reply chunks (so VAD can fire barge-in); public `barge_in()` triggers controller (room layer may also call on loud uplink). `feed_pcm16` must receive exactly 320-sample (640-byte) frames — raise `ValueError` otherwise (fail fast on clock mismatch). `take_playback` returns 960-byte chunks or None. `stop()` clears queue + marks stopped (further `take_playback` → None).

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bridge.py tests/test_livekit_audio.py -q`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/voiceagent/telephony/livekit_bridge.py tests/test_bridge.py
git commit -m "feat: offline-testable room-to-turn bridge session"
```

---

### Task 3: Inbound worker (webhook → join → greet → loop → hangup)

**Files:**
- Create: `src/voiceagent/telephony/inbound.py`, `scripts/livekit_worker.py`
- Test: `tests/test_livekit_inbound.py` (fakes only — fake room/track/webhook payload, no network)

**Interfaces:**
- Consumes: `BridgeSession` (Task 2); `RuntimeConfig` (Task 1); `Orchestrator/Deployment/handle_turn` + `transcribe_wav_routed` + `speak`-to-bytes (read exact WAV-bytes TTS entry: `tts.py` `speak()` writes files — for bytes, synthesize to temp WAV then read; reuse that pattern, do not add TTS API).
- Produces: `make_turn_fn(orchestrator, session_id, language=None)` (ASR bytes→text via temp WAV + `transcribe_wav_routed`, `handle_turn`, reply→temp-WAV bytes via `speak`, returns `(reply, wav_bytes)`); `webhook_handler(config, join_room)` (validates via livekit `WebhookReceiver`, filters `room_started` with `room.name.startswith(room_prefix)`, calls `join_room(room_name)`); `run_room_session(room_name, config, deps)` (mint `AccessToken`, `rtc.Room` connect, subscribe first SIP audio track, greeting turn, pump loop on `AudioStream` frames with sample-rate assert, leave on participant-disconnect).

- [x] **Step 1: Write the failing test**

```python
# tests/test_livekit_inbound.py
from voiceagent.telephony.inbound import make_turn_fn, webhook_handler

class FakeOrch:
    def __init__(self): self.turns = []
    def handle_turn(self, session_id, user_text, **kw):
        self.turns.append(user_text)
        from types import SimpleNamespace
        return SimpleNamespace(reply="Hi there", actions=[])

def test_greeting_turn_and_utterance_wiring(tmp_path):
    fn = make_turn_fn(FakeOrch(), "s1", asr=lambda pcm: "hello",
                      tts=lambda text: (text, b"\x00\x00" * 8000))
    reply, wav = fn(b"\x01\x02" * 8000)
    assert reply == "Hi there" and isinstance(wav, bytes) and len(wav) > 0

def test_webhook_filters_by_prefix():
    joined = []
    h = webhook_handler(config=None, join_room=joined.append,
                        validate=lambda body, sig: {"event": "room_started", "room": {"name": "call-abc"}})
    assert h("{}", "sig") is True and joined == ["call-abc"]
    h2 = webhook_handler(config=None, join_room=joined.append,
                         validate=lambda body, sig: {"event": "room_started", "room": {"name": "other"}})
    assert h2("{}", "sig") is False and len(joined) == 1
```

Note: `make_turn_fn(orch, session_id, asr=None, tts=None, language=None)` — injectable asr/tts for tests (defaults: temp-WAV `transcribe_wav_routed` + `speak`), because no model loads in CI.

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_livekit_inbound.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write minimal implementation**

Per interfaces. `webhook_handler` signature `(config, join_room, validate=None)` — default validate uses livekit `WebhookReceiver(TokenVerifier(config.key, config.secret))` (verified against livekit-api==1.2.1 — NOT `WebhookReceiver(key, secret)`); injectable for tests. Prefix: `(config.livekit_room_prefix if config is not None else "call-")` (the test passes `config=None`). `run_room_session` imports `from livekit import rtc` lazily (function-level, so unit tests never need the dep installed... they will be installed post-Task-1; still lazy-import to keep cold paths light). Greeting: `turn_fn` equivalent with transcript `"(Inbound call connected — greet the caller.)"` spoken once after subscribe. Disconnect: leave room + return (worker loop respawns per webhook; no shared state).

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_livekit_inbound.py tests/test_bridge.py -q`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/voiceagent/telephony/inbound.py scripts/livekit_worker.py tests/test_livekit_inbound.py
git commit -m "feat: inbound webhook worker with governed greeting"
```

---

### Task 4: Outbound dial (room → SIP participant → AMD → campaign)

**Files:**
- Create: `src/voiceagent/telephony/outbound.py`, `scripts/livekit_dial.py`
- Test: `tests/test_livekit_outbound.py` (fakes only)

**Interfaces:**
- Consumes: `RuntimeConfig` (trunk_id, number as caller ID); existing `dialer.RegulatoryDNDScrubber` + `amd.Sub600msAMD` (read exact interfaces first — reuse, do not reimplement); `campaign_turn` (orchestrator).
- Produces: `dial_out(api, room_name, to_number, trunk_id, timeout_s=30, poll=None) -> str` returning `"connected" | "failed" | "timeout"`; `poll` injectable `(room_name) -> str` mapping to `active|failed|ringing` (default polls LiveKitAPI participant `sip.callStatus`); `scripts/livekit_dial.py` CLI (`--to --room [--trunk]`: create room (API), worker-join note, dial, print disposition; AMD + campaign wiring documented as Task-5-integrated? No — wire here: after `connected`, run existing AMD over the first 600ms of room audio? Offline-testable seam: `classify_early_audio(frames_iter) -> str` calling `Sub600msAMD().process_frame` per 20ms frame, returning first decisive `human|machine|beep|timeout`).

- [x] **Step 1: Write the failing test**

```python
# tests/test_livekit_outbound.py
from voiceagent.telephony.outbound import classify_early_audio, dial_out

def test_dial_outcomes_and_amd_passthrough():
    calls = {"n": 0}
    def fake_create(**kw):
        calls["n"] += 1
    assert dial_out(object(), "r", "+1", "T", create=fake_create,
                    poll=lambda r: "active") == "connected"
    assert dial_out(object(), "r", "+1", "T", create=fake_create,
                    poll=lambda r: "failed") == "failed"
    assert dial_out(object(), "r", "+1", "T", create=fake_create,
                    poll=lambda r: "ringing", sleep=lambda s: None) == "timeout"
    assert calls["n"] == 3

def test_amd_first_decision_wins():
    import math, struct
    from voiceagent.outbound.amd import Sub600msAMD
    frames = [struct.pack("<320h", *[int(4000 * math.sin(2 * math.pi * 300 * i / 16000)) for i in range(320)]) for _ in range(30)]
    assert classify_early_audio(iter(frames)) in ("human", "machine", "beep", "timeout")
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_livekit_outbound.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [x] **Step 3: Write minimal implementation**

`dial_out(api, room_name, to_number, trunk_id, timeout_s=30, poll=None, create=None, sleep=None)`: default `create` calls `api.sip.create_sip_participant(CreateSIPParticipantRequest(room_name=..., sip_call_to=to_number, sip_trunk_id=trunk_id))` (verified against livekit-api==1.2.1 — request-object form, NOT kwargs); default `poll` maps `list_participants` → `ParticipantInfo.State` (`SIPParticipantInfo` carries no call status); default `sleep` is `time.sleep` (injectable → timeout test runs instantly). Poll loop: 1s cadence until `active`/`failed` or `timeout_s` → `"timeout"`. `classify_early_audio`: feed frames to `Sub600msAMD` (verify ctor from `outbound/amd.py` first), return on first decisive label else `"timeout"` after iterator exhausts. Regulatory note in docstring: CLI must run `RegulatoryDNDScrubber` before dialing (wire the check into `livekit_dial.py` main, not the library).

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_livekit_outbound.py tests/test_livekit_inbound.py -q`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/voiceagent/telephony/outbound.py scripts/livekit_dial.py tests/test_livekit_outbound.py
git commit -m "feat: outbound SIP dial with AMD handoff seam"
```

---

### Task 5: Runbook + README + loopback drill

**Files:**
- Create: `docs/telephony-runbook.md`, `scripts/livekit_loopback.py`
- Modify: `README.md` (status table: limb row)
- Test: `tests/test_loopback.py` (two fake rtc peers through BridgeSession — no network)

**Interfaces:**
- Consumes: Tasks 1–4 surfaces.
- Produces: runbook (dispatch rule recap, numbers/trunk inventory, webhook URL setup, test-call procedure with expected logs, per-minute cost table with TODAY's LiveKit Cloud + trunk rates marked verify-at-billing, recording-consent reminder for property/sales use); `livekit_loopback.py` (caller WAV → 16k frames → session A → WAV reply → assert non-empty + VAD endpointed ≥1 turn); README limb row → `⏳ limb on branch, first PSTN drill pending`.

- [x] **Step 1: Write the failing test**

```python
# tests/test_loopback.py
def test_caller_wav_roundtrip():
    import math, struct
    from voiceagent.telephony.livekit_bridge import BridgeSession
    n = 16000
    wav = struct.pack(f"<{n}h", *[int(6000 * math.sin(2 * math.pi * 330 * i / 16000)) for i in range(n)])
    out = []
    s = BridgeSession(on_utterance=lambda pcm: ("ok", struct.pack(f"<{len(pcm)//2}h", *[500] * (len(pcm) // 2))))
    for i in range(0, len(wav), 640):
        s.feed_pcm16(wav[i:i+640])
    for _ in range(25):
        s.feed_pcm16(bytes(640))
    while (c := s.take_playback()) is not None:
        out.append(c)
    assert b"".join(out) and s.take_playback() is None
    s.stop()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loopback.py -q`
Expected: PASS already (proves bridge reuse) — honest split: this task's new content is the script + docs; note it in the report.

- [x] **Step 3: Write script + docs + README row**

`livekit_loopback.py`: plays a caller WAV (arg) through a BridgeSession with a stub turn fn, writes reply WAV out, prints turn count + timings. Runbook per interfaces. README: limb row + number (mask? No — DID is public by design when dialed; still, list as configured, not shouted).

- [x] **Step 4: Run tests to verify**

Run: `.venv/bin/python -m pytest tests/test_loopback.py tests/test_bridge.py -q && .venv/bin/python scripts/livekit_loopback.py --help`
Expected: PASS + usage printed

- [x] **Step 5: Commit**

```bash
git add scripts/livekit_loopback.py docs/telephony-runbook.md tests/test_loopback.py README.md
git commit -m "feat: loopback drill, telephony runbook, README limb status"
```

---

## Out of scope (explicitly post-limb)

- LiveKit Agents-framework pipeline (we stay transport-only by design).
- UK/India numbers + outbound trunk provisioning (console work, same pattern as +1).
- Call recording storage/retention policy beyond the consent reminder.
- WebRTC browser/mobile clients (rooms support them free, but no client work).
