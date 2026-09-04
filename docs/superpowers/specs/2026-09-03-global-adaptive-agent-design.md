# Global Adaptive Agent — Design (v1)

**Date:** 2026-09-03
**Status:** Approved design, awaiting spec review before implementation plan
**Replaces framing:** India-commerce support bot → global-first, cloud-brain, any-vertical adaptive agent

## 1. Goal & non-goals

**Goal (v1):** Non-technical owner pastes a website URL + describes the business in chat → agent self-builds a runnable bundle (knowledge + invented tools as gated proposals + policies + evals), answers from approved sources immediately, invents connections but connects/executes externally only after explicit owner approval, learns instantly from owner corrections and nightly from outcome labels, remembers per-person context so it never repeats a corrected mistake with the same person.

**Non-goals (v1):**
- No open-internet auto-truth. Web = candidates, owner-approved sources = truth.
- No runtime tool-mining from customer traffic (v2). v1 mines tools only during onboarding.
- No autonomous deal/money closing. Property/offer/money actions draft + human signs. Full auto-close needs per-tool opt-in + transaction rails (v2 gate).
- No weight-level continual fine-tune. Learning = session memory → per-person profile → versioned bundle patches (+ batch proposals). Cloud fine-tune pipeline is v2+.

## 2. Locked decisions

| Decision | Choice | Source |
|---|---|---|
| Onboarding input | (b) URL + plain-words chat → self-built bundle, no dev YAML | owner 2026-09-03 |
| Tool invention | (c) agent invents + drafts connections, owner approval required before connect/execute | owner 2026-09-03 |
| Learning signals | (c) both: instant owner corrections (live same-turn) + batch outcomes (nightly proposals) | owner 2026-09-03 |
| Runtime mining | v2 (proposal queue only when built, no auto-connect) | owner 2026-09-03 |
| Vertical scope | Any-place via operator patterns; support/sales are examples | owner 2026-09-03 |
| Per-person memory | Yes: scoped profile per contact, global learns only anonymized approved patterns | owner 2026-09-03 |
| Architecture | Hybrid #3: compiler core (deterministic versioned bundle) + chat front; `Orchestrator` + `GovernedToolRunner` + decision log stay the enforcer, unchanged | recommended + approved |

## 3. Architecture

```
Owner chat + approved sources (site, docs, whitelisted web)
        ↓
  BUNDLE COMPILER (new; writes only, never executes)
   knowledge draft + SpecialistSpec + tools.json (PROPOSED)
   + policies.yaml draft + evals.json (10 self-checks)
        ↓
  APPROVAL GATE (owner: approve knowledge → live; approve tool → APPROVED → CONNECTED w/ dry-run)
        ↓
  LIVE RUNTIME (existing, unchanged semantics)
   Orchestrator.handle_turn ↔ frontier cloud brain
    → GovernedToolRunner (policy → gateway → decision log)
    → session memory + per-person profile + bundle knowledge
        ↓
  LEARNING LOOPS
   instant: owner correction → vN+1 live same-turn (self-checks green)
   batch: nightly outcomes → proposal queue → owner bulk approve
```

Existing modules reused with semantics preserved (one additive check only): `orchestrator.py` (`Deployment`, `handle_turn`, `campaign_turn`), `swarm/frontier.py` (cloud function-calling), `swarm/specialist.py` (`SpecialistSpec`), `tools.py` (+ additive tool-state check: PROPOSED/APPROVED never execute), `policy.py` (DSL), `memory.py` (+ profile store), `decisionlog.py` (every transition).

## 4. Components

### 4.1 Onboarding flow
Crawl owner URL (same-origin, robots-respecting, text-only v1; no login/JS-app scraping) with hard caps: max 50 pages, max depth 3 from seed URL, 15s per-page timeout. Scoping rule: the crawler never leaves the owner domain plus an explicit owner-approved allowlist (max ~3 URLs) — no discovery crawling, no following the web outward; it fetches only what the bundle needs. Staleness: each chunk stores `crawled_at`; re-crawl policy is manual-refresh in v1 (owner taps "refresh site" → new version diff; no background crawler). SPA/login/JS-heavy pages fail silently per §6 — they produce a listed gap, never a half-parse. v1 fallback (required, not v2): owner interview always offers "paste your docs manually" — pasted text / uploaded docs become first-class approved sources with `source: owner_paste` + timestamp, ranked above crawled chunks. Flow: crawl (or paste) → extract facts with source URLs → interview owner for 4 slots (offering, top-5 asks, never-promise/do, human-takeover triggers) → detect operator patterns (answer / resolve / qualify / follow_up / draft_action) → compile bundle v1 + 10 self-check chats. Knowledge approvable instantly; tools stay `PROPOSED`.

### 4.2 Bundle schema (per deployment, versioned)
`bundle.schema_version: 1` frozen from day one — every bundle file carries it; compiler rejects unknown versions instead of guessing, and version bumps require a migration note + golden-file tests (a checked-in `v1_example/` bundle that must load byte-identical). Layout `data/deployments/<id>/v<N>/`: `spec.json` (role, tone, patterns, disclosures), `tools.json` (name, description, params JSON-Schema, `state`, connection ref, policy action), `policies.yaml` (existing DSL; external default `require_approval:true`), `knowledge/` (chunk + source + crawl time), `evals.json` (self-checks). Compiler writes `vN+1` as a structural diff against the frozen schema (riskiest code in v1 — hence schema + goldens). `evals.json` self-checks are runnable by the orchestrator itself: each entry is `{turns: [...], assert: {contains | action | verdict}}` executed as real `handle_turn` calls; "go live" = 10/10 green mechanically, never aspirational. Live pointer flips only on approval + green self-checks; rollback = repoint to `vN`.

### 4.3 Approval gate (tool lifecycle)
`PROPOSED` (log-only, chat-visible as "once connected") → `APPROVED` (scope confirmed, no creds) → `CONNECTED` (creds + dry-run passed → callable under policy). Enforced by a state check in `GovernedToolRunner`; unknown/unconnected calls surface, never execute.

Dry-run (definition): because an invented tool has no real backend until the owner provides one, dry-run = (1) connector-level auth probe (credentials valid, scopes match the approved list — recorded, no side effects), plus (2) exactly one benign read-only call agreed with the owner in chat (e.g. `GET /listings?limit=1`, never a write), with full request/response stored owner-visible in the decision log. A tool becomes `CONNECTED` only after both probes pass and the owner confirms the recorded response looks right. Any scope widening later resets the tool to `APPROVED` until a fresh dry-run passes.

### 4.4 Instant-learn
Owner correction → patch-type classify (tone/fact/policy/exemplar) → `vN+1` → self-checks → live + changelog. Customer correction → session memory + profile candidate only; global needs owner nod (injection guard). Touching `CONNECTED` scopes or failing checks → stays proposed with reason.

### 4.5 Per-person memory
Contact key (decided): phone in E.164 as primary key + owner-aliased override (owner may rename/merge, e.g. "Sharma-family"), TTL + delete-on-request; affects `memory.py` schema, GDPR-style delete path, and dialer lookup. Profile per key: preferences, corrections, open items, consent. Read at turn start; written on correction/resolution. Never crosses contacts; owner-visible + deletable. Only anonymized ≥3-occurrence patterns proposed globally.

### 4.5a Eval-tagging contract (for Plan 3 batch-learn)
Approved customer-derived evals MUST carry `source_contact_hash` (SHA-256 of the contact key, never the key); Plan 3 MUST implement `purge_contact(hash)` to remove them. Owner-authored instant evals carry `source: owner`. Bundle loader ignores unknown eval fields (forward-compatible).

### 4.6 Batch-learn
Nightly job over labeled turns (resolved/escalated/thumbs/human-takeover) → proposals (exemplars, wording, threshold, knowledge gaps). Owner per-item/bulk approve → new versions. Eval set grows from approved outcomes only.

### 4.6a Batch job scope (v1)
Mine/approve is an on-demand CLI (`scripts/batch_learn.py`); scheduling is ops (cron/systemd), not code. Candidate enumeration uses explicit `--keys` (store protocol has no key listing by design). Proposal output capped at 50, deterministic order. Eval-source hashes enable `purge_contact`; bundle knowledge from batch carries `source: batch:<date>`. Purge covers bundle evals only; proposals/*.json history is retained in the owner-only deploy dir and scrubbed separately. Re-mining the same day overwrites today's proposals file — approve before re-mining. Purge --hash expects the SHA-256 of the stored (E.164-normalized) contact key.

## 5. Data flow (turn)
`contact key → profile + session → bundle knowledge → brain proposes → policy/tool-state check → gateway or escalate → reply → append session + profile deltas → (owner correction? instant patch) → (nightly? batch proposals)`.

## 6. Error handling
Crawl blocked → proceed on owner text + flag gap. Dry-run/self-check fail → stays proposed with reason. Connector down → `escalate_to_human` with context, no retry storm. Conflicting corrections → newest owner directive wins, logged. Unknown tool attempts → surfaced in reply + logged, never executed.

## 7. Testing
- 10 versioned self-checks per bundle (must-pass to go live).
- Existing pytest suite stays green; new tests for compiler diffs, gate states, instant-patch rollback, profile isolation, batch-proposal approval.
- Red-team set: injection → escalate, invented-tool pressure → stays proposed, cross-profile leak probe → isolated.
- Eval growth only from approved outcomes.

### 7.1 v1 success gate (measurable — this is what "works" means)
1. One real business onboarded in ≤30 min of owner time (paste URL + chat + approvals).
2. 10/10 bundle self-checks green before go-live, every version.
3. 50-turn adversarial eval: zero unapproved external calls, zero cross-profile leaks.
4. Instant-learn: owner correction live in ≤2 min (new version + green self-checks + changelog).
5. Full pytest suite green. All five must hold to call v1 done.

## 8. Risks
Regulated/high-value verticals (card/money, property claims) hallucinate without approved-source discipline → mitigated by candidates-only web + `never_promise`/disclosure defaults. Customer-driven global retraining → blocked by profile/global split. Tool hallucination → blocked by PROPOSED-default + dry-run.

## 9. Build order (for plan phase, not this spec)
1. Bundle schema + compiler (crawl → spec/tools/policies/knowledge/evals). 2. Approval gate + viewer/chat approvals. 3. Instant-learn + profiles. 4. Batch-learn job. 5. Operator-pattern packs (support + sales first, generic after). Runtime mining + auto-close + fine-tune stay v2.
