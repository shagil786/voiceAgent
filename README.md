# VoiceAgent — Global Adaptive Agent

> **The AI agent that drops into any business with some context — then learns.**
> Owner pastes a website + describes the business in chat → the agent self-builds
> its bundle (knowledge, invented tools, policies, evals), answers from approved
> sources immediately, connects external tools only after explicit owner approval,
> and gets better every turn: instant owner corrections go live same-turn, and
> every contact gets a scoped memory so a corrected mistake is never repeated
> with the same person.

**Contract (non-negotiable): the model proposes; the deterministic spine disposes.**
Every tool call is policy-gated, precondition-checked, and decision-logged.
Nothing external executes without approval. Customer text can never rewrite
global behavior — only owner-approved versions go live.

## How it works

```
Owner chat + approved sources (site, docs, allowlisted web)
        ↓
BUNDLE COMPILER (writes only, never executes)
  knowledge + SpecialistSpec + tools.json (PROPOSED) + policies.yaml + evals
        ↓
APPROVAL GATE — knowledge goes live on approval; tools move
  PROPOSED → APPROVED → CONNECTED (auth probe + benign dry-run, recorded)
        ↓
LIVE RUNTIME — Orchestrator + frontier cloud brain (any OpenAI-compat endpoint)
  governed tools → policy → gateway → ERP → audit log → reply
        ↓
LEARNING LOOPS
  instant: owner correction → vN+1 bundle → self-checks → live same-turn
  per-person: contact profiles (prefs, corrections, open items) consulted each turn
  batch (next): nightly outcomes → tuning proposals, owner-approved
```

## Status — what exists today

| Layer | State |
|---|---|
| **Runtime** (`orchestrator.py`, `swarm/frontier.py`) — bounded turn loop, native function-calling, governed tools, escalation as auditable action | ✅ live (text), 0.3–1.5s/turn on Groq (dev runs) |
| **Deploy foundation** (`deploy/`: bundle schema v1, scoped crawl, compiler, gate + dry-run, runnable self-checks, mechanical go-live) | ✅ on `main` |
| **Learn loops** (`learn/`: correction classifier, instant patches, E.164+alias profiles with TTL + delete/export, orchestrator seam, anonymous-contact isolation) | ✅ on `main` |
| **Governance** (policy DSL + validator 7/7, decision log, injection guard, frustration detector) | ✅ enforced |
| **Voice I/O** (Qwen3-ASR primary + whisper fallback + Indic routing, Piper TTS, barge-in/VAD) | ✅ built, needs fresh e2e latency run |
| **Outbound** (dialer with DND scrub + window, sub-600ms AMD logic, swarm organs) | ✅ built, synthetic-only validation |
| **Tests** | ✅ 409 collected, 406 passed / 2 skipped / 1 xfailed |
| **Batch-learn job, operator packs, adversarial harness** | ⏳ Plans 3–4 |
| **Telephony limb** (SIP/WebRTC + real number), **real ERP adapter**, **packaging** (Docker/VPS) | ❌ the remaining organs — MockERP, stub sidecars only |

Design: `docs/superpowers/specs/2026-09-03-global-adaptive-agent-design.md`.
Plans: `docs/superpowers/plans/2026-09-04-instant-learn-profiles.md` (done),
`2026-09-03-global-adaptive-agent-foundation.md` (done).

## Quickstart

```bash
# chat demo (text, local brain options)
PYTHONPATH=src .venv/bin/python scripts/chat.py
python scripts/chat_server.py  # → http://127.0.0.1:8000

# live frontier brain (needs .env — see .env.example)
PYTHONPATH=src .venv/bin/python scripts/live_conversation.py --scripted

# tests (fast subset; full suite ~100s)
.venv/bin/python -m pytest tests/test_bundle.py tests/test_gate.py \
  tests/test_profiles.py tests/test_instant.py tests/test_learn_loop.py -q
```

`.env` (gitignored): `VOICEAGENT_FRONTIER_URL`, `VOICEAGENT_FRONTIER_MODEL`,
`VOICEAGENT_FRONTIER_KEY`.

## Key invariants (for contributors)

- `bundle.schema_version = 1` frozen; loader rejects unknown versions.
- Crawl never leaves the owner domain + ≤3-URL allowlist (50 pages / depth 3).
- Only `CONNECTED` tools execute; scope widening resets to `APPROVED`.
- Owner channel (`instant_correct`) is the sole global writer; customer channel
  writes profile candidates only. Anonymous contacts share nothing.
- Every behavior change ships with tests; reviews gate every task.
