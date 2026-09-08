# DEPLOY.md — container/VPS runbook

One image (`voiceagent:latest`), two long-running services. Config comes from
the environment only — `.env` is gitignored and never enters the image.

## Prerequisites

- Docker Engine with the compose plugin (**v2.24+** — the compose file uses
  optional `env_file`). No other host dependencies.
- Frontier brain env (`VOICEAGENT_FRONTIER_URL` at minimum) — both services
  **fail fast (exit 2)** without it and, under `restart: unless-stopped`,
  restart-loop by design until configured.
- For the `worker`: a LiveKit Cloud project + a publicly reachable webhook
  URL — see `docs/telephony-runbook.md` before enabling telephony.

## Build

```bash
docker compose build          # or: docker build -t voiceagent:latest .
```

The build compiles `llama-cpp-python` from source in a throwaway builder
stage (no prebuilt aarch64 wheel exists); the runtime image contains no
compilers.

## Run

```bash
cp .env.example .env          # fill in values; never committed
docker compose up -d chat     # governed demo HTTP agent on :8000
docker compose up -d worker   # LiveKit telephony webhook on :8080
```

Precedence: host shell environment (e.g. `VOICEAGENT_TENANT=example-clinic
docker compose up chat`) overrides `.env`. An empty passthrough string is
treated as unset by the code.

## Environment variables

Required:

| Variable | Services | Notes |
|---|---|---|
| `VOICEAGENT_FRONTIER_URL` | chat, worker | Hard gate — unset ⇒ exit 2 |
| `LIVEKIT_URL` | worker | LiveKit Cloud project URL |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | worker | `LIVEKIT_KEY`/`LIVEKIT_SECRET` accepted as legacy aliases; wrong pair ⇒ every webhook 404s (fail-closed signature check) |

Optional / defaulted:

| Variable | Default | Notes |
|---|---|---|
| `VOICEAGENT_FRONTIER_MODEL` | `gpt-4o-mini` | Set to your frontier model |
| `VOICEAGENT_FRONTIER_KEY` | none | Bearer key for the frontier |
| `VOICEAGENT_TENANT` | unset | Tenant bundle name under `data/tenants/`; unset serves the built-in demo deployment (stderr warning) |
| `VOICEAGENT_DEFAULT_LANG` | unset | Worker: trunk's known query language (e.g. `te`, `ta`); unset = blind whisper ASR |
| `VOICEAGENT_AUDIT_DB` | in-memory log | Set `/app/data/out/audit.db` to persist the audit trail |
| `VOICEAGENT_MEMORY_DB` | unset (inert) | Set `/app/data/out/intent_memory.db` to enable learned intent memory |
| `VOICEAGENT_MODELS_DIR` | `data/models` | NOT baked into the image — see "Local models" below |
| `VOICEAGENT_EMBEDDING_SPACE` | `latin` | Retrieval embedding space |
| `VOICEAGENT_CANDIDATE_MODELS` | registry names | Comma-separated stems, smallest-first |
| `VOICEAGENT_VOICES` | built-in registry | Comma `lang:path` pairs (piper ONNX) |
| `VOICEAGENT_HTTP_RATE_LIMIT` | unset (off) | chat: max API requests/min per client IP (`/api/turn`, `/api/history`); over budget ⇒ `429` + `Retry-After`. Static pages are never limited |
| `VOICEAGENT_HTTP_RATE_WINDOW_S` | `60` | chat: rate-limit window seconds |
| `VOICEAGENT_TRUST_PROXY` | `false` | Set `true` ONLY behind a trusted reverse proxy: keys the rate limiter on the client-supplied `X-Forwarded-For` (leftmost hop). Default `false` = socket peer — trusting XFF when directly exposed lets clients rotate the header and bypass the limit |
| `VOICEAGENT_HF_TOKEN` | unset | HF token for gated downloads |
| `LIVEKIT_NUMBER` / `LIVEKIT_TRUNK_ID` | unset | Outbound dialing (DID + trunk) |
| `LIVEKIT_ROOM_PREFIX` | `call-` | Must match the LiveKit dispatch rule |

## Data persistence

| Path | Volume | Contents |
|---|---|---|
| `/app/data/out` | `voiceagent_state` | `memory.db` (chat demo conversation memory, hardcoded `data/out/memory.db`), plus any SQLite paths you point `VOICEAGENT_AUDIT_DB` / `VOICEAGENT_MEMORY_DB` at |
| `/app/data/index` | `voiceagent_index` | `chunks.pkl` (RAG chunk-embedding cache; rebuilt automatically if absent) |
| `/app/data/tenants`, `/app/data/policies`, `/app/data/knowledge` | baked into image | Read-only — update by editing the repo and rebuilding |

Notes:

- SQLite on a shared named volume is single-host state, not a network DB. If
  chat and worker both write the audit trail, give each service its own
  `VOICEAGENT_AUDIT_DB` filename (SQLite single-writer locking).
- **Local models**: the GB-scale `data/models` cache is deliberately not
  baked in. The model registry falls back to a built-in list and the frontier
  brain needs no local models; the local GGUF/ONNX fallback path is only
  available if you mount the cache read-only:
  `-v ./data/models:/app/data/models:ro`.

## Health checks

```bash
# chat — UI page (200, HTML):
curl -fsS http://127.0.0.1:8000/
# chat — one governed turn (JSON out):
curl -fsS -X POST http://127.0.0.1:8000/api/turn \
  -H 'Content-Type: application/json' -d '{"text": "hi"}'

# worker — webhook is POST-only; a 404 proves the listener is up (signature
# validation fails closed — only LiveKit's signed events get 200):
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8080/
```

Worker startup log line: `webhook listening on :8080 (prefix 'call-')`.

## Telephony warning

The `worker`'s webhook **must be reachable from LiveKit Cloud over the public
internet** (tunnel, load balancer, or public VPS port). Register the URL in
the LiveKit console (Settings → Webhooks, `room_started` at minimum) and keep
`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` exactly matching the console — the
full first-call procedure is `docs/telephony-runbook.md`.

## Container notes

- Runs as non-root `appuser` (uid 1000); `/app/data` is owned by it.
- `PYTHONPATH=/app/src`; the package runs from source, never pip-installed.
