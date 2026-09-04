# Batch-Learn + Purge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nightly-or-on-demand job turns labeled outcomes + profile candidates into an owner-approved proposal queue (exemplars, wording, thresholds, knowledge gaps), approvals become versioned bundles, and any contact's derived evals can be purged by hash.

**Architecture:** New `voiceagent/learn/batch.py` (miner + applier, deterministic heuristics, no LLM) over three inputs — an outcome-label store (`learn/outcomes.py`), profile `pending_global` candidates (Plan 2), and decision-log verdicts. Anonymity rule: a customer pattern becomes a *global* proposal only with ≥3 occurrences across ≥3 distinct contact hashes (SHA-256, never raw keys). Owner approves per-item or bulk via `scripts/batch_learn.py`; each approval set becomes one bundle version gated by the existing stub-tier self-checks + `go_live`. `purge_contact` removes evals by `source_contact_hash` per spec §4.5a.

**Tech Stack:** Python 3.12, stdlib only (`hashlib`, `json`, `re`, `collections`), existing `pytest`, no new third-party deps.

**Spec:** `docs/superpowers/specs/2026-09-03-global-adaptive-agent-design.md` — this plan implements §4.6 (batch-learn) and the §4.5a `purge_contact` MUST. Builds on foundation (`deploy/bundle.py`, `deploy/selfcheck.py`) and Plan 2 (`learn/profiles.py` candidates, `learn/instant.py` versioning pattern, `EvalCheck`). Follow-up (separate plan): operator-pattern packs + adversarial harness + onboarding measurement (§7.1-1,2,3).

## Global Constraints

- Customer text NEVER writes the global bundle directly; only owner-approved proposals become versions. Anonymized global proposals require ≥3 occurrences across ≥3 distinct contact hashes.
- Contact hashes are SHA-256 hex of the contact key; raw keys never enter proposals, evals, or logs.
- Every approval set is a whole-bundle version; live pointer flips only on ≥10 all-pass self-checks; rollback = repoint.
- `purge_contact(contact_hash)` removes ALL evals carrying that hash and reports the count; unknown hash → count 0, no error.
- The nightly job is an on-demand CLI in v1 (cron/systemd wiring is ops docs, not code).
- No new third-party deps. Python 3.12. Existing pytest suite stays green.

---

## File structure

| File | Responsibility |
|---|---|
| `src/voiceagent/learn/outcomes.py` | `OutcomeLabel` (session_id, label, ts, note, contact_hash optional) + `InMemoryOutcomes` + `JsonlOutcomes(path)` (append JSONL, read all) with `record/query` |
| `src/voiceagent/learn/batch.py` | `hash_contact(key)`, `mine_proposals(...)`, `apply_approved(bundle, approvals)`, `purge_contact(bundle, contact_hash)`, proposal dict schema + queue read/write helpers |
| `scripts/batch_learn.py` | CLI: `--deploy DIR mine` (write proposal files) / `--approve IDS` / `--approve-all` / `--purge HASH` |
| `tests/test_outcomes.py` | Store tests |
| `tests/test_batch.py` | Miner/anonymity/apply/purge tests |

Proposal dict schema (exact): `{id, kind: exemplar|wording|threshold|knowledge_gap, title, detail, evidence: {count, distinct_hashes, hashes: sorted-distinct-hashes-capped-at-25, sample_quotes: [≤3, truncated 120 chars, hashes only — no raw keys]}, patch: {...kind-specific...}, status: proposed|approved|rejected}`. IDs: `f"{kind}-{n:03d}"` in mine order (deterministic: sort by (kind, title)). Titles: mined groups → `title = longest quote truncated 60 chars`, `detail = f"{count} reports from {distinct} contacts; majority {patch_type}"`; outcomes-keyword wording → `title = f"Repeated {word} complaints"`, `detail = f"{n} negative outcomes mention '{word}'"`.

---

### Task 1: Outcome-label store

**Files:**
- Create: `src/voiceagent/learn/outcomes.py`
- Test: `tests/test_outcomes.py`

**Interfaces:**
- Consumes: `now_ts` (`memory.py`).
- Produces: `OutcomeLabel` dataclass (`session_id: str, label: str` one of `resolved|escalated|thumbs_up|thumbs_down`, `ts: str, note: str = ""`, `contact_hash: str | None = None`); `InMemoryOutcomes` (`record(label)`, `query(label=None, session_id=None) -> list`); `JsonlOutcomes(path)` (same API; appends one JSON object per line on `record`, reads all on `query`; creates parent dirs).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_outcomes.py
from voiceagent.learn.outcomes import InMemoryOutcomes, JsonlOutcomes, OutcomeLabel

def test_record_query_and_labels():
    s = InMemoryOutcomes()
    s.record(OutcomeLabel(session_id="s1", label="resolved", ts="2026-09-04T00:00:00"))
    s.record(OutcomeLabel(session_id="s2", label="escalated", ts="2026-09-04T00:01:00"))
    assert [o.session_id for o in s.query(label="resolved")] == ["s1"]
    assert len(s.query()) == 2
    assert s.query(session_id="s9") == []

def test_jsonl_roundtrip(tmp_path):
    p = str(tmp_path / "sub" / "outcomes.jsonl")
    s = JsonlOutcomes(p)
    s.record(OutcomeLabel(session_id="s1", label="thumbs_down", ts="t", note="wrong fee"))
    assert JsonlOutcomes(p).query(label="thumbs_down")[0].note == "wrong fee"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_outcomes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.learn.outcomes'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/voiceagent/learn/outcomes.py
"""Outcome labels: owner/reported verdicts per session (batch-learn input)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

LABELS = ("resolved", "escalated", "thumbs_up", "thumbs_down")

@dataclass
class OutcomeLabel:
    session_id: str
    label: str  # one of LABELS
    ts: str
    note: str = ""
    contact_hash: str | None = None

class InMemoryOutcomes:
    def __init__(self):
        self._rows: list[OutcomeLabel] = []
    def record(self, label: OutcomeLabel) -> None:
        if label.label not in LABELS:
            raise ValueError(f"bad label {label.label!r}")
        self._rows.append(label)
    def query(self, label: str | None = None,
              session_id: str | None = None) -> list[OutcomeLabel]:
        out = self._rows
        if label is not None:
            out = [o for o in out if o.label == label]
        if session_id is not None:
            out = [o for o in out if o.session_id == session_id]
        return list(out)

class JsonlOutcomes(InMemoryOutcomes):
    def __init__(self, path: str):
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._rows.append(OutcomeLabel(**d))
    def record(self, label: OutcomeLabel) -> None:
        super().record(label)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(label)) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_outcomes.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/learn/outcomes.py tests/test_outcomes.py
git commit -m "feat: outcome-label store with JSONL backend"
```

---

### Task 2: Miner (clustering + anonymity + proposal types)

**Files:**
- Create: `src/voiceagent/learn/batch.py` (miner half: hashing, normalize, mine)
- Test: `tests/test_batch.py` (miner tests; applier tests land in Task 3 same file)

**Interfaces:**
- Consumes: candidate dicts `{quote, patch_type, session_id, ts}` + contact keys (caller hashes them); `OutcomeLabel`; `EvalCheck`/`Bundle` shapes (read-only here).
- Produces: `hash_contact(key: str) -> str` (SHA-256 hex); `normalize_quote(text: str) -> str` (lower, strip punctuation/extra spaces, truncate 200); `mine_proposals(candidates: list[dict], outcomes: list[OutcomeLabel], bundle: Bundle | None = None) -> list[dict]` (proposal schema above; deterministic order).

Mining rules (exact): group candidates by `normalize_quote(quote)`; a group becomes proposals only if `len(group) >= 3` AND `distinct contact hashes >= 3` (caller passes `contact_hash` per candidate in `candidate["contact_hash"]`; candidates without a hash count toward neither threshold). Every emitted proposal carries `evidence = {count, distinct_hashes, hashes: sorted(distinct)[:25], sample_quotes: ≤3 truncated 120 chars}`. Per qualifying group emit, by `patch_type` majority: `fact` → `knowledge_gap` (`patch: {text, source: "batch:<date>"}` with text = longest quote, source date = today UTC `YYYY-MM-DD`); `policy` → `threshold` (`patch: {never_promise_add: longest quote, needs_dsl_review: True}`); `tone` → `wording` (`patch: {tone_notes_add: longest quote}`); `exemplar` → `exemplar` (`patch: {user: longest quote, assert_contains: first 12 chars}`); ties → first in (`fact`, `policy`, `tone`, `exemplar`) order. Additionally, if `outcomes` labeled `thumbs_down`/`escalated` ≥3 share a `session`-level keyword (top non-stopword ≥4 chars appearing in ≥3 distinct sessions' notes — notes come from `OutcomeLabel.note`), emit one `wording` proposal titled `Repeated <word> complaints` with `patch: {tone_notes_add: ...}` left empty-string (owner words it). Cap output at 50 proposals (newest groups first by max ts? No — deterministic: sort groups by (-count, template); emit in that order; IDs assigned after final sort by (kind, title)). Sample quotes: ≤3 per proposal, each truncated to 120 chars; store ONLY quotes + count + distinct-hash count — never raw keys.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch.py
from voiceagent.learn.batch import hash_contact, mine_proposals, normalize_quote

def _cand(quote, ptype="fact", h="h1", ts="2026-09-01T00:00:00"):
    return {"quote": quote, "patch_type": ptype, "session_id": "s",
            "ts": ts, "contact_hash": h}

def test_hash_and_normalize():
    assert len(hash_contact("+911")) == 64 and hash_contact("+911") != hash_contact("+912")
    assert normalize_quote("  NO,  the Fee is 499!! ") == "no the fee is 499"

def test_anonymity_gate_needs_3x3():
    same = [_cand("No, fee is 499", h="h1"), _cand("No! fee is 499?", h="h1"),
            _cand("no fee is 499.", h="h1")]
    assert mine_proposals(same, []) == []  # 1 hash → nothing
    trio = [_cand("No, fee is 499", h="h1"), _cand("NO fee is 499!", h="h2"),
            _cand("no, fee is 499.", h="h3")]
    props = mine_proposals(trio, [])
    assert len(props) == 1 and props[0]["kind"] == "knowledge_gap"
    assert props[0]["evidence"]["distinct_hashes"] == 3
    assert props[0]["id"] == "knowledge_gap-000"

def test_policy_majority_and_cap():
    cands = [_cand(f"No, never promise X{i%2}", ptype="policy", h=f"h{i}") for i in range(6)]
    props = mine_proposals(cands, [])
    assert props and all(p["kind"] == "threshold" for p in props)
    assert all(p["patch"]["needs_dsl_review"] is True for p in props)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_batch.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.learn.batch'`

- [ ] **Step 3: Write minimal implementation**

Per rules above. `hash_contact`: `hashlib.sha256(key.encode()).hexdigest()`. `normalize_quote`: `re.sub(r"[^a-z0-9 ]", "", text.lower())`, collapse whitespace, `[:200]`. Stopwords for the outcomes-keyword rule: `{"the", "and", "for", "with", "this", "that", "have", "from", "they", "them", "then", "your", "about", "into"}`; keywords = words ≥4 chars not in stopwords. Proposal IDs after final `(kind, title)` sort: per-kind counters `f"{kind}-{n:03d}"`. Today UTC: `datetime.now(timezone.utc).date().isoformat()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_batch.py tests/test_outcomes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/learn/batch.py tests/test_batch.py
git commit -m "feat: batch miner with 3x3 anonymity gate"
```

---

### Task 3: Apply approvals + purge_contact

**Files:**
- Modify: `src/voiceagent/learn/batch.py` (append applier half)
- Test: extend `tests/test_batch.py` (append, do not rewrite)

**Interfaces:**
- Consumes: proposal dicts (Task 2); `Bundle`, `save_bundle`, `EvalCheck` (deploy/bundle.py).
- Produces: `apply_approved(bundle: Bundle, approvals: list[dict]) -> tuple[Bundle, dict]` (changelog `{applied: [ids], skipped: [{id, reason}]}`); `purge_contact(bundle: Bundle, contact_hash: str) -> tuple[Bundle, int]` (count removed); `read_proposals(path) / write_proposals(path, props)` (JSON list helpers).

Apply mapping (exact, copy-on-write via `copy.deepcopy`): `exemplar` → `evals.append(EvalCheck(name=f"batch-{id}", turns=[{"user": patch["user"]}], assert_={"contains": patch["assert_contains"]}))` + `spec.setdefault("eval_sources", {})[eval_name] = "|".join(evidence["all_hashes"])` (full uncapped list — `hashes` is display-only capped at 25; purge completeness depends on this). (Rationale, state in a comment: EvalCheck is frozen by the golden schema; hashes live in `spec.eval_sources` keyed by eval name — loader ignores nothing since evals.json shape is unchanged.) `wording` → `spec.setdefault("tone_notes", []).append(patch["tone_notes_add"])` (skip if empty-string with `skipped: "needs owner wording"`). `threshold` → `spec.setdefault("never_promise", []).append(patch["never_promise_add"])` + changelog `needs_dsl_review: True` (no DSL auto-edits, same rule as instant). `knowledge_gap` → `knowledge.append({"text": patch["text"], "source": patch["source"], "crawled_at": now-UTC-iso})`. Unknown kind → `skipped: "unknown kind"`. Non-`approved` status → `skipped: "not approved"` (caller filters, but defense in depth: only `status == "approved"` applies). `purge_contact`: drop evals whose `spec.eval_sources.get(name, "").split("|")` contains the hash + delete those keys; return `(new_bundle, count)`; unknown hash → `(bundle, 0)` (return input bundle object unchanged when count is 0).

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_apply_and_purge_roundtrip():
    from voiceagent.deploy.bundle import load_bundle
    from voiceagent.learn.batch import apply_approved, purge_contact
    b = load_bundle("data/deployments/_example/v1")
    approvals = [
        {"id": "exemplar-000", "kind": "exemplar", "title": "t", "detail": "d",
         "evidence": {"count": 3, "distinct_hashes": 3, "hashes": ["abc"],
                      "sample_quotes": []},
         "patch": {"user": "2BHK price?", "assert_contains": "2BHK price?"},
         "status": "approved"},
        {"id": "wording-000", "kind": "wording", "title": "t", "detail": "d",
         "evidence": {"count": 1, "distinct_hashes": 1, "hashes": [],
                      "sample_quotes": []},
         "patch": {"tone_notes_add": ""}, "status": "approved"},
    ]
    new, log = apply_approved(b, approvals)
    assert [a for a in log["applied"]] == ["exemplar-000"]
    assert log["skipped"][0]["id"] == "wording-000"
    assert new.spec["eval_sources"]["batch-exemplar-000"] == "abc"
    assert len(b.evals) == len(load_bundle("data/deployments/_example/v1").evals)
    pruned, n = purge_contact(new, "abc")
    assert n == 1 and not any("batch-exemplar-000" in e.name for e in pruned.evals)
    same, n0 = purge_contact(new, "nope")
    assert n0 == 0 and same is new
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_batch.py -q`
Expected: FAIL with `AttributeError`/`ImportError` (no `apply_approved`)

- [ ] **Step 3: Write minimal implementation**

Per mapping above. `read_proposals`/`write_proposals`: JSON list of proposal dicts (indent 2). `crawled_at`: `datetime.now(timezone.utc).isoformat()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_batch.py tests/test_outcomes.py tests/test_bundle.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/learn/batch.py tests/test_batch.py
git commit -m "feat: approval applier with eval-source sidecar and purge"
```

---

### Task 4: Nightly CLI (mine → approve → version)

**Files:**
- Create: `scripts/batch_learn.py`
- Test: `tests/test_batch_cli.py` (drives the script functions in-process, no subprocess)

**Interfaces:**
- Consumes: `mine_proposals`, `apply_approved`, `read/write_proposals` (Tasks 2–3); `JsonlOutcomes` (Task 1); `InMemoryProfiles`/`SQLiteProfiles` (Plan 2, read `pending_global` + contact keys for hashing); `load/save_bundle`, `run_self_checks`, `go_live`, `read_live`, `next_version`-equivalent (deploy/bundle.py + deploy/selfcheck.py + learn/instant.py `next_version` — import and reuse, do not reimplement).
- Produces: script with `mine(deploy_dir, outcomes_path, profiles_db, keys) -> proposals_path` and `approve(deploy_dir, ids | all, make_brain=None) -> dict` importable functions + `__main__` argparse (`mine --deploy --outcomes [--profiles-db] --keys k1,k2`, `approve --deploy (--ids a,b | --all)`, `purge --deploy --hash`).

Behavior (exact): `mine`: load outcomes (`JsonlOutcomes`), load profiles (`SQLiteProfiles(profiles_db)` if given else `InMemoryProfiles()` — note: InMemory is empty by construction; document that real mining reads the SQLite profiles DB), gather candidates = for each profile key: `hash_contact(key)` + each `pending_global` entry + `contact_hash`; call `mine_proposals`; `write_proposals(<deploy>/proposals/<YYYY-MM-DD>.json)`; return path. Profiles store needs key enumeration — `ProfileStore` protocol has no `keys()` method: read profiles via a new `all_keys()` helper ONLY IF missing... Plan 2 stores have no enumeration. Decision (no guessing): implement enumeration inside the CLI by accepting an explicit `--keys key1,key2` argument (owner's contact list or recent-call export) instead of expanding the store protocol. Document this. `approve`: read latest proposals file (lexicographically max in `proposals/`), filter ids (or all with status proposed), mark approved, `load_bundle(live-or-latest version dir)`, `apply_approved`, `save_bundle` to `next_version`, `run_self_checks` + `go_live`, `write_proposals` back with updated statuses, return `{version, applied, skipped, live}`. Fail-closed like instant (proposed saved, pointer untouched). `purge`: load live-or-latest, `purge_contact`, save as next version (no self-checks needed when only removals? Still run checks — cheap stub tier — then go_live; removal can only break eval-count floor: if evals drop below 10, go_live fails closed correctly).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch_cli.py
import sys
sys.path.insert(0, "scripts")
from batch_learn import approve, mine
from voiceagent.learn.outcomes import JsonlOutcomes, OutcomeLabel
from voiceagent.learn.profiles import Profile, SQLiteProfiles

def test_mine_approve_end_to_end(tmp_path):
    import shutil
    from voiceagent.deploy.bundle import load_bundle
    from voiceagent.learn.batch import EvalCheck
    shutil.copytree("data/deployments/_example/v1", tmp_path / "v1")
    b = load_bundle(tmp_path / "v1")
    b.evals = [EvalCheck(name=f"e{i:02d}", turns=[{"user": "Hi"}],
                         assert_={"contains": "Hi"}) for i in range(10)]
    from voiceagent.deploy.bundle import save_bundle
    save_bundle(b, tmp_path / "v1")
    db = SQLiteProfiles(str(tmp_path / "p.db"))
    from voiceagent.learn.batch import hash_contact
    for i in range(3):
        key = f"+910000000{i}"
        db.put(Profile(key=key, alias="", prefs=[], corrections=[], open_items=[],
                       pending_global=[{"quote": "No, fee is 499", "patch_type": "fact",
                                        "session_id": "s", "ts": "t"}],
                       consent={}, updated_at="2026-09-04T00:00:00"))
    keys = [f"+910000000{i}" for i in range(3)]
    path = mine(str(tmp_path), "nope.jsonl", str(tmp_path / "p.db"), keys=keys)
    import json
    props = json.loads(__import__("pathlib").Path(path).read_text())
    assert len(props) == 1 and props[0]["kind"] == "knowledge_gap"
    out = approve(str(tmp_path), ids=[props[0]["id"]])
    assert out["live"] is True and out["applied"] == [props[0]["id"]]
```

Note: `mine` signature `(deploy_dir, outcomes_path, profiles_db, keys)` — outcomes file may not exist ("nope.jsonl" → treat as empty, still mine candidates). Profiles DB missing → empty store, proposals only from nothing → []. Keep both tolerant (mine never crashes on missing inputs; returns []).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_batch_cli.py -q`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (no `batch_learn` module)

- [ ] **Step 3: Write minimal implementation**

Per behavior above. `sys.path` import in tests is the established pattern for scripts (see plan note). Argparse with three subcommands; functions importable. `next_version` imported from `learn.instant` (reuse). Base version: `read_live` else highest `v*`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_batch_cli.py tests/test_batch.py tests/test_outcomes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/batch_learn.py tests/test_batch_cli.py
git commit -m "feat: batch-learn CLI with mine/approve/purge"
```

---

### Task 5: Anonymity edge proof, queue bounds, docs

**Files:**
- Test: extend `tests/test_batch.py` (append)
- Docs: `docs/superpowers/specs/2026-09-03-global-adaptive-agent-design.md` — append `§4.6a` note (≤6 lines)

**Interfaces:**
- Consumes: Task 2–4 surfaces.
- Produces: proof that (a) single-hash flooding (100 candidates, 1 hash) yields zero proposals; (b) proposal output capped at 50 with deterministic order; (c) raw keys never appear in any proposal JSON (scan `json.dumps(props)` for each input key); (d) spec documents on-demand CLI + `--keys` enumeration decision.

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_flood_single_hash_yields_nothing_and_no_raw_keys():
    import json
    from voiceagent.learn.batch import mine_proposals
    cands = [{"quote": f"No, item {i} broke", "patch_type": "fact",
              "session_id": "s", "ts": "t", "contact_hash": "solo"} for i in range(100)]
    assert mine_proposals(cands, []) == []
    from voiceagent.learn.batch import hash_contact
    trio = [{"quote": "No, fee is 499", "patch_type": "fact", "session_id": "s",
             "ts": "t", "contact_hash": h} for h in ("a", "b", "c")]
    props = mine_proposals(trio, [])
    blob = json.dumps(props)
    assert "+911" not in blob and len(props) == 1

def test_proposal_cap_and_deterministic_order():
    from voiceagent.learn.batch import mine_proposals
    cands = []
    for g in range(60):
        for h in ("h1", "h2", "h3"):
            cands.append({"quote": f"No, thing{g} failed", "patch_type": "fact",
                          "session_id": "s", "ts": "t", "contact_hash": f"{h}-{g}"})
    props = mine_proposals(cands, [])
    assert len(props) == 50
    assert [p["id"] for p in props] == sorted(p["id"] for p in props)
```

Note: `contact_hash: f"{h}-{g}"` gives 3 DISTINCT hashes per group (180 distinct) — each group qualifies; cap cuts to 50. IDs sorted check works because single kind in this fixture.

- [ ] **Step 2: Run tests to verify**

Run: `.venv/bin/python -m pytest tests/test_batch.py -q`
Expected: FAIL (no cap → 60 proposals; flood already [] so that half passes — honest split: cap test fails, flood passes)

- [ ] **Step 3: Implement cap + spec note**

Cap in `mine_proposals`: after group ordering, `props = props[:50]` BEFORE the final `(kind, title)` sort (frequency wins over alphabetical; IDs assigned post-sort stay deterministic). Spec append (§4.6a, ≤6 lines):

```markdown
### 4.6a Batch job scope (v1)
Mine/approve is an on-demand CLI (`scripts/batch_learn.py`); scheduling is ops (cron/systemd), not code. Candidate enumeration uses explicit `--keys` (store protocol has no key listing by design). Proposal output capped at 50, deterministic order. Eval-source hashes enable `purge_contact`; bundle knowledge from batch carries `source: batch:<date>`.
```

- [ ] **Step 4: Run the wider net**

Run: `.venv/bin/python -m pytest tests/test_batch.py tests/test_batch_cli.py tests/test_outcomes.py tests/test_bundle.py tests/test_selfcheck.py tests/test_instant.py tests/test_profiles.py -q`
Expected: PASS, all green

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/learn/batch.py tests/test_batch.py docs/superpowers/specs/2026-09-03-global-adaptive-agent-design.md
git commit -m "feat: proposal cap, anonymity proof, batch scope note"
```

---

## Out of scope (explicitly Plan 4)

- Operator-pattern packs, 50-turn adversarial harness, onboarding measurement (§7.1-1,2,3).
- Auto-apply of proposals (all applications require owner approval — no decay timers, no auto-merge).
- Runtime tool-mining, auto-close, fine-tune — v2.
