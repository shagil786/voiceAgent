# Data surfaces — the files that teach the agent a language or a business.
No code changes are ever needed for either; add or edit files, commit,
deploy. Code holds MECHANISM only (parsers, routers, guardrails); every
table below is DATA.

## Platform language data: `data/lang/<code>.yaml`

One file per language. Add-a-language = add-a-file (proven in
tests/test_langdata.py: a synthetic `xx.yaml` parses and the engine
serves it with zero code changes).

| key | meaning | used by |
|---|---|---|
| `code` | language code (matches langid output) | everything |
| `tts_voice` | piper voice id (`en_US-lessac-medium`); absent = fallback voice + warning | tts `VOICE_REGISTRY` |
| `asr_engine` | `indic` routes to the IndicConformer; absent = Qwen core | asr router |
| `alias_of` | resolve to another code for engines/voices (`hinglish: hi`); detection keeps the original | asr hints/forcing, tts, inbound containment |
| `scripts` | script names claimed for detection (see `scripts.yaml`) | langid |
| `detect_as` | detected code for a shared block (`mr` claims Devanagari, detects as `hi`) | langid |
| `detect_tokens` + `detect_stage` | Latin lexicon + check stage (`global` before `hinglish`) | langid |
| `numbers.words` / `.scales` / `.hundred` | flat value tables; tens+units accumulate, irregular systems enumerate (Hindi 0-99, French 80) | entities |
| `digits` | 10-char native-script digit row, merged into one map | entities |
| `garbles` | observed ASR mishearing → canonical word | entities |
| `sentiment` | frustration phrases (English always scanned too) | sentiment |
| `companions` | sibling scripts scanned together (`hi` ↔ `hinglish`) | sentiment |
| `currency_words` | `{SYM: [regex forms]}` — each form mints amounts ONLY for its own currency | entities |

## Platform script/model data

- `data/scripts.yaml` — unicode block facts (`Devanagari: [[0x0900, 0x097F]]`,
  incl. unclaimed Arabic for tokenization). Languages claim scripts; the
  table itself knows no language.
- `data/asr_engines.yaml` — third-party support lists straight from model
  cards (`indic:` 22 codes). Routing decisions live in lang files; this
  file only answers "does the engine support it, else warn + fall back".

## Tenant business data: `data/tenants/<name>/`

| file | meaning |
|---|---|
| `tenant.json` | identity: persona, languages served, currency |
| `intents/<action>.yaml` | caller exemplars per action (the classifier taxonomy) |
| `tools.yaml` | governed tool surface (`action:` renames to policy actions) |
| `proposals.yaml` | human-approved tool proposals (committing = approval) |
| `entities.yaml` | `record_ids:` — reference-number shapes (`code`, `digit_pattern` with group(1), optional `prefix_pattern`, `bare_digits`, `min/max_digits`); validated by `Tenant.record_id_shapes()` and `validate_tenant.py` |
| `policies.yaml` | per-action verdicts, thresholds, `escalate_when` |
| `knowledge/` | source documents the agent answers from |

## Onboarding generation (website + doc → bundle)

`POST /api/control/onboard/preview {source: {url|text}, interview}` returns
the deterministic compiler baseline PLUS brain-drafted surfaces when a
frontier is configured (`VOICEAGENT_LLM_BASE_URL` + `VOICEAGENT_LLM_MODEL`;
silent fallback otherwise):

`{deploy_id, spec, knowledge, tools[], policies, evals[], intents{},
entities|null, questions[], drafted, note}`

- Drafted tools are `PROPOSED` (uppercase deploy state); intents key to
  drafted actions only; entities shapes are regex-validated (invalid ones
  are dropped and become questions).
- `questions[]` is the dashboard contract: `{id, prompt, why,
  kind: text|choice|multi, options[], answer_key}`. `answer_key` names the
  interview key the answer fills — the dashboard renders each question,
  merges answers into `interview`, and re-previews. Gap detectors cover:
  offering, top asks, handoff triggers, greeting, served languages
  (auto-hinting what the content detects), ERP hookup, risky promises
  found in the text, ID examples, and any invalid draft.
- Approval stays human: `POST /api/control/onboard/deploy` runs gate +
  self-checks; only then does anything go live.
