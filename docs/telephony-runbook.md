# Telephony Runbook — first live PSTN call (LiveKit limb)

Operational runbook for the first real inbound/outbound PSTN call through the
LiveKit limb. Branch: `feat/livekit-limb`. The offline loopback drill
(`scripts/livekit_loopback.py`) must pass before any step here that costs money.

Architecture invariant: LiveKit is transport only. Every spoken turn — including
the greeting — is one governed `Orchestrator.handle_turn` / `campaign_turn`
through the same brain as chat and voice CLI. No LiveKit Agents framework, no
second brain.

## Prerequisites

- **LiveKit Cloud project** (console.livekit.cloud) with a US SIP trunk region.
- **`.env` (gitignored)** containing:
  - `LIVEKIT_URL`, `LIVEKIT_KEY`, `LIVEKIT_SECRET` — from the project's Settings → Keys.
  - `VOICEAGENT_FRONTIER_URL`, `VOICEAGENT_FRONTIER_MODEL`, `VOICEAGENT_FRONTIER_KEY` —
    the governed brain. **Required**: without them `scripts/livekit_worker.py`
    refuses to start (exit 2) rather than serve calls with no orchestrator.
  - `LIVEKIT_NUMBER` (+1 240 231 5037) and `LIVEKIT_TRUNK_ID` (outbound trunk).
- **`VOICEAGENT_TENANT=<name>`** — loads the tenant bundle from
  `data/tenants/<name>/`. Unset = the built-in demo deployment (Acme) serves
  calls with a stderr warning; fine for the first mechanical drill, never for
  a real deployment.
- **`VOICEAGENT_DEFAULT_LANG=<code>`** — the trunk's known query language
  (e.g. `te`, `ta`, `hi`). Unset = blind whisper ASR that never auto-routes to
  the Indic engine; Telugu/Tamil callers get whisper instead of the routed path.
- Python env with `livekit` + `livekit-api` installed (pinned in `requirements.txt`).

## Trunk + dispatch rule inventory

Fill trunk IDs from your LiveKit console — **this doc does not invent them.**

| Resource | Where | Value / rule |
|---|---|---|
| Inbound SIP trunk | Console → SIP → Trunks | name `voice-agent`, type **Individual**, numbers: +1 240 231 5037 |
| Dispatch rule | Console → SIP → Dispatch | match rooms with prefix **`call-`**, create SIP participant (room per call) |
| Inbound DID | Console → SIP | +1 240 231 5037 (configurable via `LIVEKIT_NUMBER`) |
| Outbound trunk | Console → Telephony → SIP trunks (outbound) | external provider (Telnyx/Twilio); stored trunk ID in `LIVEKIT_TRUNK_ID` — native LiveKit numbers CANNOT dial out |

Worker + dispatch contract: the worker only joins rooms whose name starts with
`config.livekit_room_prefix` (default `call-`, override `LIVEKIT_ROOM_PREFIX`),
and only on `room_started` events — dispatch-rule rooms and worker filter must
agree on the prefix.

## Webhook setup

1. Start the worker on a publicly reachable URL:
   ```bash
   .venv/bin/python scripts/livekit_worker.py --port 8080
   ```
   (Behind a tunnel or load balancer for local dev; the URL must be reachable
   from LiveKit Cloud.)
2. Register the webhook URL in the LiveKit console (Settings → Webhooks),
   events: `room_started` at minimum.
3. Signature validation is fail-closed: the worker checks the `Authorization`
   header with livekit's `WebhookReceiver(TokenVerifier(key, secret))`. Any
   missing/invalid signature → HTTP 404 and **no** session spawn. A wrong
   key/secret pair in `.env` shows up here as every webhook 404ing.

## Inbound test-call procedure (first live call)

1. Start the worker (above). Confirm the startup line:
   `webhook listening on :8080 (prefix 'call-')`.
2. Dial **+1 240 231 5037** from any phone.
3. Expected worker log sequence:
   - `POST / HTTP/1.1" 200` — webhook accepted (404 here = signature or prefix problem).
   - `spawned session thread for room call-<id>` — one daemon thread per room.
   - Note: `run_room_session` itself is log-quiet; the next observable signal
     is audible — the caller hears the governed greeting within a few seconds
     of the SIP track appearing. If no SIP track shows within 15s the session
     exits without greeting (no dead-air calls).
4. Speak; each VAD-endpointed utterance runs one governed turn and the reply
   plays back. Speech while the agent is speaking barges in (playback is
   cancelled at session level).
5. Hang up; the worker's session thread exits on the disconnect event
   (worker logs the room's `participant_disconnected` via the livekit room
   callback; the thread then returns).

## Outbound test-call procedure

**IMPORTANT (verified 2026-09-05): LiveKit-native Phone Numbers (bought in the
LiveKit console, e.g. +1 240 231 5037) are INBOUND ONLY.** LiveKit's docs:
"LiveKit Phone Numbers currently only supports inbound calling. Support for
outbound calls is coming soon." A native number cannot dial out, and
`CreateSIPParticipant` returns `permission_denied` ("project is not authorised
to initiate outbound calls"). Outbound dialing REQUIRES an outbound SIP trunk
from an external provider (Telnyx, Twilio, Plivo — LiveKit's tested list):

1. Sign up at a provider (Telnyx gives $5 trial credit, no card; Twilio gives
   $15 trial credit) and buy/enable a number there.
2. Enable the destination country in the provider portal (Telnyx: Outbound
   Voice Profile → Allowed Destinations → India; Twilio: Voice Geographic
   Permissions). Trial accounts can typically only call numbers you verify.
3. LiveKit Console → Telephony → SIP trunks → New **outbound** trunk with the
   provider's credentials/number; note the trunk ID.
4. `export LIVEKIT_TRUNK_ID=<that id>` and run the procedure below.

```bash
.venv/bin/python scripts/livekit_dial.py --to +15551234567 --room call-demo-1
```

1. **DND scrub gate first**: `RegulatoryDNDScrubber().scrub(--to)` runs BEFORE
   any dial — blocked numbers print `blocked: <reason>` and exit 2 without
   touching the SIP trunk.
2. Room `call-demo-1` is created, then `dial_out` places the SIP call
   (`LIVEKIT_TRUNK_ID` trunk, `LIVEKIT_NUMBER` as caller ID when set) and
   polls at 1s cadence.
3. Output: `disposition: connected | failed | timeout`.
4. On `connected`, the first 600ms of room audio goes through
   `classify_early_audio` (`Sub600msAMD`): voicemail answers SIP 200 OK, so
   AMD stays mandatory — only a `human` classification should lead into
   `campaign_turn`.
5. A worker must be running to join the room and speak (same webhook flow —
   `call-*` prefix); the dial CLI places the call but never talks.

## Loopback drill (no PSTN cost)

Run before any paid step, and after any bridge/audio change:

```bash
.venv/bin/python scripts/livekit_loopback.py            # synthetic 330Hz tone
.venv/bin/python scripts/livekit_loopback.py --wav caller.wav --out data/out/loopback-reply.wav
```

Feeds a 16k mono WAV through `BridgeSession` in 20ms frames (the exact shape
the room pump feeds) with a stub turn fn, and writes the 48k reply WAV.
Success: `loopback OK`, ≥1 endpointed utterance, non-empty reply file, and the
reply duration matches the stub (0.40s) — proving VAD endpointing, 16k↔48k
resample, chunking, and the playback queue without a model or network.

## Costs

**VERIFY AT BILLING — rates change; do not trust this table at go-live.**
Check console.livekit.cloud → Billing and your SIP provider's current rate
card before the first paid drill, and fill the actuals in.

| Line item | Unit | Rate | Where to verify |
|---|---|---|---|
| LiveKit Cloud media (audio) | per participant-minute | VERIFY AT BILLING | Console → Billing → Media |
| LiveKit Cloud SIP | per minute / per leg | VERIFY AT BILLING | Console → Billing → SIP |
| Inbound DID rental | monthly | VERIFY AT BILLING | Console → SIP → Trunks |
| Outbound PSTN termination (US) | per minute | VERIFY AT BILLING | SIP trunk provider rate card |
| Frontier brain (e.g. Groq) | per 1M tokens | VERIFY AT BILLING | provider pricing page |

Budget rule for drills: one short inbound call + one short outbound call is a
few participant-minutes of media plus a few PSTN minutes — keep first drills
under a minute each.

## Recording consent

The platform does **not** record by default (no recording is started anywhere
in the limb). If you enable recording for outbound/property/sales calls, you
are responsible for jurisdiction-appropriate consent — several US states are
**two-party consent** (all parties must agree); announce and capture consent
before recording, and store/retain recordings under an owner-approved policy.

## Known limits

- **Blind-ASR path**: no per-call language identification. Without
  `VOICEAGENT_DEFAULT_LANG`, Telugu/Tamil callers get blind whisper (never the
  Indic-routed engine). Set the trunk's language per deployment.
- **Barge-in is session-level, not semantic**: uplink speech clears the
  playback queue; it does not understand *what* was said mid-reply.
- **Greeting is one governed turn**: `handle_turn("(Inbound call connected —
  greet the caller.)")` spoken once after the SIP track appears — not a canned
  file, and not multi-turn negotiation.
- **One room = one session thread**: rooms are joined on `room_started` only;
  a session that misses the event (worker restart mid-call) is not recovered.
- `run_room_session` is log-quiet by design; success is the audible greeting,
  failure modes surface as thread exit without greeting (no SIP track in 15s)
  or webhook 404s (signature/prefix mismatch).
