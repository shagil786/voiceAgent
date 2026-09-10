# Data flows & residency inventory

Where caller data lives, what leaves the machine, and how to erase it.
Review this file during onboarding — data residency is a **deployment
property**: this inventory tells you every store and egress so you can
place them in the right jurisdiction. The platform defaults to local
files; only the rows marked CLOUD leave the host.

## At rest (all local SQLite files by default)

| Store | Path (env override) | Contents | Erasure |
|---|---|---|---|
| Audit trail | `VOICEAGENT_AUDIT_DB` (unset = in-memory only) | turn decisions: ts, conv_id, action, verdict, reasons. **No phone numbers, no transcripts, no audio.** | `forget_caller.py --erase-session` / `--purge` |
| Intent memory | `VOICEAGENT_MEMORY_DB` (unset = fully inert) | caller utterance fragments + ratings, keyed by tenant + session_id. **Raw caller text lives here** — the highest-sensitivity store. | same CLI; episodes also TTL-cull on consolidate |
| Chat transcripts | `VOICEAGENT_CHAT_MEMORY_DB` (default `data/out/memory.db`) | **full turn text** for demo chat-server conversations | same CLI (`chat_turns`) |
| ERP dev service | `data/erp/erp.sqlite` (gitignored) | demo orders/customers incl. phone numbers. Local dev only — production points `VOICEAGENT_ERP_URL` at the org's own system (the system of record; erasure there is the org's procedure). | delete the file (reseeds from fixtures) |

No audio or transcripts are persisted anywhere by the platform. The phone
slot (`inbound.py`) is per-call memory only.

## In flight (CLOUD egress)

| Call | Destination | Payload | Control |
|---|---|---|---|
| Brain (required) | `VOICEAGENT_FRONTIER_URL` (+ optional fallback URL) | system prompt + turn transcript + tool results | operator-chosen endpoint; self-hosted OpenAI-compatible servers keep this in-jurisdiction |
| Brain fallback | `VOICEAGENT_BEDROCK_*` (optional) | same | `VOICEAGENT_BEDROCK_REGION` pins the AWS region |
| Telephony transport | `LIVEKIT_URL` | realtime audio | operator's LiveKit project/region |

ASR (whisper/Qwen/conformer), TTS (piper), embeddings, RAG and policy all
run **on-host** — audio and text never leave for these stages.

## Retention & erasure

- Nothing deletes by default. Set `VOICEAGENT_DATA_RETENTION_DAYS=N` and
  run `scripts/forget_caller.py --purge` on a schedule (cron/systemd).
- One conversation: `scripts/forget_caller.py --erase-session <conv_id>`
  (audit entries + memory episodes + ratings; prints per-store counts).
- Memory episodes additionally TTL-cull during consolidation
  (`ttl_days`, default `EPISODE_TTL_DAYS`).
