# voiceAgent — runbook

Self-hosted, governed, audited multilingual voice agent. The control-plane
console (voiceAgentDashboard) talks to this repo **only** through its control
API — never its files.

## Prerequisites

- `.venv` (Python 3.12), deps installed: `.venv/bin/python -m pytest -q`
- `.env` (gitignored) with the frontier + Bedrock config (see below)

## 1. Control API server (the dashboard's data source)

```bash
VOICEAGENT_CONTROL_TOKEN=<token> \
VOICEAGENT_AUDIT_DB=data/out/audit.sqlite \
.venv/bin/python scripts/control_server.py 8081 127.0.0.1
```

Endpoints: `GET /api/control/{status,calls,ratings,summary,scores}` and
`POST /api/control/onboard/{preview,deploy}`. Binds 127.0.0.1 only; requires
the bearer token on every request (CORS enabled for the console).

## 2. Run a governed conversation (frontier brain)

```bash
PYTHONPATH=src VOICEAGENT_AUDIT_DB=data/out/audit.sqlite \
.venv/bin/python scripts/live_conversation.py --scripted   # 4-turn demo
# ...or interactive REPL (omit --scripted), 'exit' to quit
```

Every turn prints brain latency, tool calls, policy verdicts, and the reply.
Decisions append to the audit DB, which the dashboard streams.

## 3. Provider fallback chain

The brain tries providers in order and fails over on 429/5xx/unreachable:

| Order | Provider | Env vars |
|---|---|---|
| 1 (primary) | Groq (OpenAI-compat) | `VOICEAGENT_FRONTIER_URL`, `_MODEL`, `_KEY` |
| 2 (optional) | any OpenAI-compat | `VOICEAGENT_FRONTIER_FALLBACK_URL`, `_MODEL`, `_KEY` |
| 3 (optional) | Amazon Bedrock Converse (GLM/DeepSeek) | `VOICEAGENT_BEDROCK_API_KEY` (+ `_MODEL_ID`, `_REGION`) |

Bedrock auth: `VOICEAGENT_BEDROCK_API_KEY` (bearer, recommended) or
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (SigV4). See `.env.example`.

## 4. Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## 5. Telephony (LiveKit)

- Offline bridge drill (no PSTN/cost): `.venv/bin/python scripts/livekit_loopback.py`
- Live PSTN: see `docs/telephony-runbook.md` (needs `LIVEKIT_URL/API_KEY/API_SECRET/NUMBER/TRUNK_ID`).
