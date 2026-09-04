# Instant-Learn + Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Owner corrections go live same-turn as versioned bundle patches; every contact gets a scoped profile (preferences, corrections, open items) keyed by phone + alias with TTL and delete/export, consulted each turn and never leaking across contacts.

**Architecture:** New `voiceagent/learn/` package (writes only versioned bundles, never executes external calls): deterministic correction classifier (lexicon patterns, owner/customer split) → instant patch applier (per-type bundle edits + stub-tier self-checks + `go_live`) → per-person profile store (in-memory + SQLite, E.164 keys, TTL, cascade delete). `Orchestrator` gains an additive `profiles` seam: contact-memory block merged into the system prefix at turn start, customer corrections captured to the profile after each turn. Owner corrections enter only via `instant_correct()` (never via customer `handle_turn` — injection separation).

**Tech Stack:** Python 3.12, stdlib only for new code (`re`, `sqlite3`, `threading`, `time`), existing `pytest`, no new third-party deps.

**Spec:** `docs/superpowers/specs/2026-09-03-global-adaptive-agent-design.md` — this plan implements §4.4 (instant-learn) and §4.5 (per-person memory) plus §7.1 gate 4 (correction live ≤2 min). Builds on the foundation plan (`docs/superpowers/plans/2026-09-03-global-adaptive-agent-foundation.md`): `Bundle`, `save_bundle`/`load_bundle`, `run_self_checks`, `go_live`, `read_live`. Follow-ups (separate plans): batch-learn job (§4.6), operator-pattern packs, 50-turn adversarial harness + onboarding measurement (§7.1-1,3).

## Global Constraints

- Customer text NEVER writes the global bundle and NEVER crosses contacts; only anonymized ≥3-occurrence patterns may be proposed globally, and that proposal queue is Plan 3 scope — this plan stores customer corrections as profile candidates only.
- Owner corrections enter only via `instant_correct()` (owner channel), never via customer `handle_turn`.
- Every applied owner patch is a whole-bundle version (`vN+1` via `save_bundle`); live pointer flips only on approval-equivalent (owner issued it) + green self-checks (≥10 all-pass); rollback = repoint.
- Contact key: phone normalized toward E.164 + owner-aliased override + TTL; affects `memory.py`-style schema, delete path, dialer lookup compatibility (key is a plain string).
- `PROFILE_TTL_DAYS = 365`; expired profiles prune on read + explicit prune job hook.
- No new third-party deps. Python 3.12. Existing pytest suite stays green.

---

## File structure

| File | Responsibility |
|---|---|
| `src/voiceagent/learn/__init__.py` | Package exports (`contact_key`, `classify_correction`, `instant_correct`, profile stores) |
| `src/voiceagent/learn/profiles.py` | Contact key resolution, `Profile` dataclass, `ProfileStore` protocol, `InMemoryProfiles`, `SQLiteProfiles` (WAL + lock, mirrors `memory.py`), alias map, session links, TTL prune, delete/export |
| `src/voiceagent/learn/corrections.py` | Deterministic `classify_correction(user_text, last_agent_text, is_owner) -> Correction` (lexicon patterns mirroring `sentiment.py` style; owner/global vs customer/candidate split) |
| `src/voiceagent/learn/instant.py` | `apply_owner_correction(bundle, correction) -> (Bundle, changelog_entry)`; `next_version(deploy_dir) -> str`; `instant_correct(deploy_dir, quote, context, make_brain) -> dict` (patch + stub-tier checks + go_live + timing) |
| `src/voiceagent/orchestrator.py` (modify) | Additive `profiles` ctor param; contact-memory block into system prefix; customer-correction capture; session linking; `delete_contact` wrapper |
| `src/voiceagent/memory.py` (no change) | Session histories cleared via existing `clear(conv_id)` during cascade |
| `tests/test_profiles.py` | Key/alias/TTL/store-parity/delete/export tests |
| `tests/test_corrections.py` | Classifier pattern/patch-type/scope tests |
| `tests/test_instant.py` | Patch mapping/versioning/go-live/timing tests |

---

### Task 1: Profile store (keys, TTL, both backends, delete/export)

**Files:**
- Create: `src/voiceagent/learn/__init__.py`
- Create: `src/voiceagent/learn/profiles.py`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: `CallerProfile` (`blackboard.py:13-20`: `customer_id`, `phone`, `name`, `authenticated`, `risk_tier`, `metadata`); `now_ts` (`memory.py:23-25`).
- Produces: `PROFILE_TTL_DAYS: int = 365`; `contact_key(profile: CallerProfile) -> str`; `normalize_phone(raw: str) -> str`; `Profile` dataclass (`key, alias, prefs: list[str], corrections: list[dict], open_items: list[str], pending_global: list[dict], consent: dict, updated_at: str`); `ProfileStore` protocol (`get, put, set_alias, resolve, link_session, sessions_for, delete_contact, export_contact, prune_expired`); `InMemoryProfiles`; `SQLiteProfiles(db_path)`.

Key rules (exact): `normalize_phone` strips spaces/dashes/parens/dots, keeps leading `+` then digits; if the stripped digits are non-empty, key = `+<digits>` when the raw started with `+` else `+<digits>` anyway (v1: assume already-E.164-or-local-digits; do NOT validate country codes); empty phone → key = `f"cid:{customer_id or 'unknown'}"`. `set_alias(alias, key)` maps owner names ("Sharma-family") → key; `resolve(alias_or_key)` returns the key (alias hit, else input if a profile exists, else input unchanged). `link_session(key, session_id)` records contact→sessions index (set semantics, no duplicates). `delete_contact(key)` removes profile + alias refs + session links, returns `{sessions: [...]}` for the caller to clear histories. `export_contact(key)` returns full profile dict or raises `KeyError`. TTL: `updated_at` bumped on every write; `get` returns `None` (and deletes) when older than `PROFILE_TTL_DAYS` (injectable `now` only in `prune_expired(now=None) -> int` for testability; `get` uses real time).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profiles.py
import pytest
from voiceagent.learn.profiles import (
    PROFILE_TTL_DAYS, InMemoryProfiles, Profile, SQLiteProfiles,
    contact_key, normalize_phone,
)
from voiceagent.swarm.blackboard import CallerProfile

def test_e164_key_and_fallback():
    assert PROFILE_TTL_DAYS == 365
    assert normalize_phone("+91 98765 43210") == "+919876543210"
    assert normalize_phone("(020) 7946-0018") == "+02079460018"
    p = CallerProfile(customer_id="C-9", phone="+91 98765 43210")
    assert contact_key(p) == "+919876543210"
    assert contact_key(CallerProfile(customer_id="C-9")) == "cid:C-9"

def test_alias_resolve_and_session_links():
    s = InMemoryProfiles()
    s.put(Profile(key="+911", alias="", prefs=[], corrections=[],
              open_items=[], pending_global=[], consent={},
              updated_at="2026-01-01T00:00:00"))
    s.set_alias("Sharma-family", "+911")
    assert s.resolve("Sharma-family") == "+911"
    s.link_session("+911", "sess-1"); s.link_session("+911", "sess-1")
    assert s.sessions_for("+911") == ["sess-1"]

def test_delete_returns_sessions_and_export_roundtrip(tmp_path):
    for cls, arg in ((InMemoryProfiles, ()), (SQLiteProfiles, (str(tmp_path / "p.db"),))):
        s = cls(*arg) if arg else cls()
        s.put(Profile(key="k1", alias="Fam", prefs=["3BHK only"], corrections=[],
                      open_items=["callback"], pending_global=[], consent={"recording": True},
                      updated_at="2026-09-01T00:00:00"))
        s.link_session("k1", "s1")
        assert s.export_contact("k1")["prefs"] == ["3BHK only"]
        assert s.delete_contact("k1") == {"sessions": ["s1"]}
        with pytest.raises(KeyError):
            s.export_contact("k1")

def test_ttl_prune(tmp_path):
    s = InMemoryProfiles()
    s.put(Profile(key="old", alias="", prefs=[], corrections=[], open_items=[],
                  pending_global=[], consent={}, updated_at="2020-01-01T00:00:00"))
    assert s.prune_expired(now="2026-09-04T00:00:00") == 1
    assert s.get("old") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profiles.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.learn'`

- [ ] **Step 3: Write minimal implementation**

`profiles.py`: `Profile` dataclass (all fields above, `prefs/corrections/open_items/pending_global` as lists, `consent` dict). `InMemoryProfiles`: dict + alias dict + links dict; `put` preserves a caller-provided `updated_at`, setting `now_ts()` only when empty (lets tests pin timestamps; live callers writing back a mutated profile must set `updated_at = now_ts()` first — Task 4 does); expired-via-`get` routes through the full `delete_contact` drop (profile + alias refs + session links); `get`/`put` deepcopy across the boundary (no live refs). `get` checks TTL via real current date (parse `updated_at[:10]`, compare day count — implement `_expired(updated_at)` helper using `datetime.date.fromisoformat`); expired → delete + return None. `SQLiteProfiles`: mirror `memory.py:77-85` (lock, WAL, NORMAL); tables `profiles(key TEXT PRIMARY KEY, alias TEXT, prefs_json, corrections_json, open_items_json, pending_json, consent_json, updated_at)` + `aliases(alias TEXT PRIMARY KEY, key TEXT)` + `links(key TEXT, session_id TEXT, PRIMARY KEY(key, session_id))`; JSON-serialize lists/dicts; same TTL semantics. `contact_key`/`normalize_phone` exactly per rules above.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_profiles.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/learn/__init__.py src/voiceagent/learn/profiles.py tests/test_profiles.py
git commit -m "feat: per-person profile store with E.164 keys, TTL, delete/export"
```

---

### Task 2: Deterministic correction classifier

**Files:**
- Create: `src/voiceagent/learn/corrections.py`
- Test: `tests/test_corrections.py`

**Interfaces:**
- Consumes: nothing (pure function; style mirrors `sentiment.py` lexicons).
- Produces: `Correction` dataclass (`is_correction: bool, patch_type: str` one of `tone|fact|policy|exemplar|none`, `quote: str` (≤280 chars), `scope: str` (`global` iff `is_owner and is_correction` else `candidate`/`none`)); `classify_correction(user_text: str, last_agent_text: str = "", is_owner: bool = False) -> Correction`.

Rules (exact, tested verbatim below): lowercased match. Correction leads: `("no,", "no ", "don't", "do not", "never", "wrong", "incorrect", "actually", "i told you", "you said", "stop saying", "not ", "but ")`. If no lead → `is_correction=False, patch_type="none", scope="none"`. Patch type precedence: `policy` if any of `("promise", "never ", "always", "escalat", "refund", "approv", "discount", "price")`; elif `exemplar` if any of `("say ", "respond with", "like this", "for example")`; elif `tone` if any of `("tone", "rude", "polite", "shorter", "short ", "hindi", "english", "hinglish", "language")`; else `fact`. `quote = user_text.strip()[:280]`. Scope: `global` iff `is_owner and is_correction`, `candidate` iff correction but not owner, else `none`. Customer corrections NEVER yield `global` (injection separation — hard rule).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corrections.py
from voiceagent.learn.corrections import classify_correction

def test_owner_fact_correction_is_global():
    c = classify_correction("No, that fee is 499 not 299", "Fee is 299", is_owner=True)
    assert (c.is_correction, c.patch_type, c.scope) == (True, "fact", "global")
    assert "499" in c.quote and len(c.quote) <= 280

def test_owner_policy_and_tone_types():
    assert classify_correction("Never promise loan approval", is_owner=True).patch_type == "policy"
    assert classify_correction("No, keep it shorter and polite", is_owner=True).patch_type == "tone"
    assert classify_correction("No, say 'site visits 10-6' like this", is_owner=True).patch_type == "exemplar"
    # precision: type keywords alone never trigger (R2 — correction needs a lead)
    assert classify_correction("What's the price?", is_owner=True).is_correction is False

def test_customer_correction_never_global():
    c = classify_correction("No, my flat is 3BHK not 2BHK", "Noted 2BHK", is_owner=False)
    assert c.is_correction is True and c.scope == "candidate"

def test_plain_chat_is_not_correction():
    c = classify_correction("What are the timings?", "10am to 6pm")
    assert (c.is_correction, c.patch_type, c.scope) == (False, "none", "none")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_corrections.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/voiceagent/learn/corrections.py
"""Deterministic correction classifier (no LLM): lexicon leads + type
precedence, mirroring sentiment.py. Owner/customer scoping is structural:
customer text can only ever be a candidate, never global."""
from __future__ import annotations

from dataclasses import dataclass

_LEADS = ("no,", "no ", "don't", "do not", "never", "wrong", "incorrect",
          "actually", "i told you", "you said", "stop saying", "not ", "but ")
_POLICY = ("promise", "never ", "always", "escalat", "refund", "approv",
           "discount", "price")
_EXEMPLAR = ("say ", "respond with", "like this", "for example")
_TONE = ("tone", "rude", "polite", "shorter", "short ", "hindi", "english",
         "hinglish", "language")

@dataclass
class Correction:
    is_correction: bool
    patch_type: str  # tone | fact | policy | exemplar | none
    quote: str
    scope: str  # global | candidate | none

def classify_correction(user_text: str, last_agent_text: str = "",
                        is_owner: bool = False) -> Correction:
    low = user_text.lower()
    quote = user_text.strip()[:280]
    if not any(l in low for l in _LEADS):
        return Correction(False, "none", quote, "none")
    if any(k in low for k in _POLICY):
        ptype = "policy"
    elif any(k in low for k in _EXEMPLAR):
        ptype = "exemplar"
    elif any(k in low for k in _TONE):
        ptype = "tone"
    else:
        ptype = "fact"
    scope = "global" if is_owner else "candidate"
    return Correction(True, ptype, quote, scope)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_corrections.py tests/test_profiles.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/learn/corrections.py tests/test_corrections.py
git commit -m "feat: deterministic correction classifier with owner/candidate split"
```

---

### Task 3: Instant patch applier (owner correction → live version)

**Files:**
- Create: `src/voiceagent/learn/instant.py`
- Test: `tests/test_instant.py`

**Interfaces:**
- Consumes: `Bundle`, `save_bundle`, `load_bundle` (deploy/bundle.py); `run_self_checks`, `go_live` (deploy/selfcheck.py); `Correction` (Task 2); `now_ts` (memory.py).
- Produces: `apply_owner_correction(bundle: Bundle, correction: Correction, context: str = "") -> tuple[Bundle, dict]`; `next_version(deploy_dir: str | Path) -> str`; `instant_correct(deploy_dir, quote: str, context: str = "", make_brain=None, actor: str = "owner") -> dict` returning `{version, passed: bool, checks: list, changelog: dict, live: bool}`.

Patch mapping (exact): copy-on-write via `copy.deepcopy`. `tone` → `spec.setdefault("tone_notes", []).append(quote)`. `fact` → `knowledge.append({"text": quote, "source": f"owner_correction:{ts}", "crawled_at": ts})`. `policy` → `spec.setdefault("never_promise", []).append(quote)` + changelog flag `needs_dsl_review=True` iff any of `("threshold", "amount", "₹", "rs.", "percent", "%", "above", "under", "over ")` in quote.lower() (free-text DSL edits stay manual via bundle viewer). `exemplar` → `knowledge.append({"text": "Exemplar guidance: " + quote, "source": f"owner_exemplar:{ts}", "crawled_at": ts})` (true eval-mining is Plan 3 scope). Non-owner (`scope != "global"`) → raise `ValueError("instant patch requires owner scope")`. Changelog: `{ts, actor, quote, patch_type, context}` + after checks `{version, passed, live, needs_dsl_review}`. `next_version`: scan `<deploy_dir>/v*` integer suffixes, return `f"v{max+1}"` (no versions → `"v1"`; ignore non-`v<int>` entries). `instant_correct`: `load_bundle(<deploy_dir>/<live or latest>)` → classify (is_owner=True) → apply → `run_self_checks(new, make_brain)` → `save_bundle(new, <deploy_dir>/<next>)` → `go_live(<deploy_dir>, next, results)` → return dict (live=True iff go_live True; on checks-fail the version is still saved as proposed, pointer untouched). Base version: `read_live(deploy_dir)` if set else highest `v*`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_instant.py
import pytest
from voiceagent.deploy.bundle import load_bundle
from voiceagent.learn.instant import (
    apply_owner_correction, instant_correct, next_version)
from voiceagent.learn.corrections import classify_correction

GOLDEN = "data/deployments/_example/v1"

def test_fact_patch_appends_knowledge_with_source():
    b = load_bundle(GOLDEN)
    c = classify_correction("No, the fee is 499", is_owner=True)
    new, log = apply_owner_correction(b, c)
    assert any(ch["source"].startswith("owner_correction:") for ch in new.knowledge)
    assert len(b.knowledge) + 1 == len(new.knowledge)  # copy-on-write
    assert log["patch_type"] == "fact"

def test_policy_patch_flags_dsl_review_on_amounts():
    b = load_bundle(GOLDEN)
    new, log = apply_owner_correction(
        b, classify_correction("Never promise refunds above 5000", is_owner=True))
    assert "Never promise refunds above 5000" in new.spec.get("never_promise", [])
    assert log["needs_dsl_review"] is True

def test_customer_scope_rejected_and_versions_increment(tmp_path):
    b = load_bundle(GOLDEN)
    cust = classify_correction("No, mine is 3BHK", is_owner=False)
    with pytest.raises(ValueError, match="owner scope"):
        apply_owner_correction(b, cust)
    assert next_version(tmp_path) == "v1"
    (tmp_path / "v3").mkdir()
    assert next_version(tmp_path) == "v4"

def test_instant_correct_goes_live_and_fast(tmp_path):
    import shutil, time
    from voiceagent.deploy.bundle import load_bundle, save_bundle
    shutil.copytree(GOLDEN, tmp_path / "v1")
    b = load_bundle(tmp_path / "v1")
    # golden ships 2 evals incl. an action-assert that fail-closes without a
    # wired runner; go_live needs ≥10 all-pass — so stage 10 contains-only
    # evals in-fixture (R3). This exercises the real live path (patch →
    # checks → pointer flip); golden fail-closed semantics stay covered in
    # selfcheck tests.
    from voiceagent.deploy.bundle import EvalCheck
    b.evals = [EvalCheck(name=f"live-{i:02d}", turns=[{"user": "Hello"}],
                         assert_={"contains": "Hello"}) for i in range(10)]
    save_bundle(b, tmp_path / "v1")
    t0 = time.monotonic()
    out = instant_correct(str(tmp_path), "No, visits run 10am to 6pm")
    dt = time.monotonic() - t0
    assert out["live"] is True and out["passed"] is True
    assert dt < 60  # §7.1-4 headroom: stub tier runs in ms; 60s guards CI flakes
    from voiceagent.deploy.bundle import read_live
    assert read_live(str(tmp_path)) == out["version"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_instant.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.learn.instant'`

- [ ] **Step 3: Write minimal implementation**

Per mapping above. `instant_correct` signature: `(deploy_dir, quote, context="", make_brain=None, actor="owner")`. Imports: `load_bundle/save_bundle/read_live` from `deploy.bundle`, `run_self_checks/go_live` from `deploy.selfcheck`, `classify_correction` from `learn.corrections`, `now_ts` from `memory`. Base version: `read_live(deploy_dir)` if set else highest `v<int>` dir. Safety note (why spec §4.4's "touching CONNECTED scopes stays proposed" needs no extra code): instant patches only ever edit `spec`/`knowledge` — they never modify `tools.json` or policy DSL entries, so CONNECTED tool scopes are unreachable by construction; state this invariant in a code comment above the patch mapping. Self-checks run the stub tier by default (`make_brain=None` → selfcheck default); document that live-brain variance is Plan 4 scope. On checks-fail: still `save_bundle` (proposed, auditable) but skip `go_live`, return `live=False` + `reason`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_instant.py tests/test_corrections.py tests/test_profiles.py tests/test_bundle.py tests/test_selfcheck.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/learn/instant.py tests/test_instant.py
git commit -m "feat: instant owner-correction patches with go-live and timing proof"
```

---

### Task 4: Orchestrator wiring (contact memory in, candidates out, session links)

**Files:**
- Modify: `src/voiceagent/orchestrator.py`
- Test: `tests/test_learn_loop.py` (new; uses `ScriptedBrain`/`reply`/`make_orchestrator`-style helpers from `tests/test_orchestrator.py` — import them, do not copy)

**Interfaces:**
- Consumes: `ProfileStore`, `contact_key` (Task 1); `classify_correction` (Task 2); existing `Orchestrator(brain, runner, memory, decision_log, max_tool_rounds)`, `handle_turn(session_id, user_text, *, profile, authenticated, _system_prefix)`, `_session_state`, `BlackboardState`, `CallerProfile`.
- Produces: `Orchestrator(..., profiles: ProfileStore | None = None)` (default None = pre-learn behavior, byte-identical replies); `handle_turn(..., contact_alias: str | None = None)`; `delete_contact(contact_or_alias: str) -> dict`; `export_contact(contact_or_alias: str) -> dict`.

Behavior (exact): in `__init__`, store `self.profiles` plus `self._profile_links: dict[str, str]` (session_id → contact key, process-local index only). In `handle_turn`, after `_session_state`: if `self.profiles is not None`: resolve key = `self.profiles.resolve(contact_alias)` if alias given else `contact_key(state.profile)`; `prof = self.profiles.get(key)`; if prof is not None: build block `"## Contact memory\n" + "\n".join(f"- {p}" for p in prefs) + corrections lines f"- Correction (use instead): {c['quote']}" + open items f"- Open: {o}"` (omit empty sections; cap block at 1500 chars); merge as `extra = (("## Outbound campaign call..." ))` — implement by prepending to the existing `_system_prefix` value (`prefix = (block + "\n\n" + (_system_prefix or "")) or None`) and pass the merged value down (do NOT change `_system_prefix` semantics for campaign_turn). Then `self.profiles.link_session(key, session_id)`. After the turn is recorded (after `self.memory.append` lines): `corr = classify_correction(user_text, final_text, is_owner=False)`; if `corr.is_correction`: `prof = self.profiles.get(key) or new Profile(key=key, ...)`; append `{"quote": corr.quote, "patch_type": corr.patch_type, "session_id": session_id, "ts": now_ts()}` to `prof.pending_global`; `self.profiles.put(prof)` (put bumps `updated_at`). `delete_contact`: resolve alias → key; `out = self.profiles.delete_contact(key)`; for each session in `out["sessions"]`: `self.memory.clear(sid)` + `self._sessions.pop(sid, None)`; return `out`. `export_contact`: resolve then delegate (KeyError propagates).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_learn_loop.py
from voiceagent.memory import InMemoryMemory
from voiceagent.orchestrator import Deployment, Orchestrator
from voiceagent.swarm.blackboard import CallerProfile
from voiceagent.swarm.frontier import FrontierAgentBridge
from voiceagent.learn.profiles import InMemoryProfiles, Profile
from tests.test_orchestrator import ScriptedBrain, reply

def _dep():
    return Deployment(name="acme", system_prompt="You are Acme.",
                      knowledge={"hours": "10am to 6pm"})

def _orch(**kw):
    kw.setdefault("memory", InMemoryMemory())
    orch = Orchestrator(brain=FrontierAgentBridge(ScriptedBrain([reply("Visits 10-6.")])), **kw)
    orch.deploy(_dep())
    return orch

def test_contact_memory_reaches_brain_and_candidates_captured():
    profs = InMemoryProfiles()
    profs.put(Profile(key="+911", alias="", prefs=["3BHK only"], corrections=[],
                      open_items=["callback Tue"], pending_global=[], consent={},
                      updated_at="2026-09-04T00:00:00"))
    orch = _orch(profiles=profs)
    orch.handle_turn("s1", "Any 2BHK?", profile=CallerProfile(phone="+91 911"))
    sent_first = orch.brain.client.calls[-1]["messages"][0]["content"]
    assert "3BHK only" in sent_first and "callback Tue" in sent_first
    assert profs.sessions_for("+911") == ["s1"]
    # candidate path: customer correction lands in pending_global, never bundle
    orch.handle_turn("s1", "No, mine is 3BHK not 2BHK",
                     profile=CallerProfile(phone="+911"))
    assert any("3BHK" in c["quote"] for c in profs.get("+911").pending_global)

def test_profiles_none_is_legacy_behavior():
    orch = _orch()
    r = orch.handle_turn("s9", "Hi", profile=CallerProfile(phone="+911"))
    assert r.reply == "Visits 10-6."

def test_delete_contact_cascades_sessions():
    profs = InMemoryProfiles()
    orch = _orch(profiles=profs)
    orch.handle_turn("sD", "Hi", profile=CallerProfile(phone="+911"))
    out = orch.delete_contact("+911")
    assert out == {"sessions": ["sD"]}
    assert orch.memory.history("sD") == []
```

Note: `brain.calls` access — verify against `ScriptedBrain.calls` (`tests/test_orchestrator.py:38`); `FrontierAgentBridge` wraps client (see `make_orchestrator` at `tests/test_orchestrator.py:107-110` for the exact wrap). If the bridge hides `.calls`, assert via `orch.brain` internals per that file — read it first, adjust the assertion to whatever exposes sent messages, never the reply text (already covered).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_learn_loop.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'profiles'`

- [ ] **Step 3: Write minimal implementation**

Per behavior above. Contact-memory block cap: build lines, `"\n".join`, truncate to 1500 chars. Merge order: contact block FIRST, then existing `_system_prefix`. New `Profile` for unknown keys: `Profile(key=key, alias="", prefs=[], corrections=[], open_items=[], pending_global=[], consent={}, updated_at=now_ts())` (put() bumps timestamp anyway).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_learn_loop.py tests/test_orchestrator.py tests/test_profiles.py tests/test_corrections.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/orchestrator.py tests/test_learn_loop.py
git commit -m "feat: orchestrator contact-memory seam with candidate capture"
```

---

### Task 5: Cascade proof, TTL on-read, timing gate, docs

**Files:**
- Modify: `src/voiceagent/learn/profiles.py` only if gaps found; otherwise tests + docs only
- Test: extend `tests/test_profiles.py` (append, do not rewrite)
- Docs: `docs/superpowers/specs/2026-09-03-global-adaptive-agent-design.md` — append `§4.5a` note (5 lines max, see step 3)

**Interfaces:**
- Consumes: Tasks 1–4 surfaces.
- Produces: proof that (a) `delete_contact` clears profile + linked session histories end-to-end via `Orchestrator.delete_contact` (already Task 4-tested; here add the SQLite-backed variant + export-after-delete KeyError), (b) expired profile prunes on plain `get` (SQLite variant), (c) §7.1-4 timing holds on a realistic bundle (golden + 5 knowledge chunks), (d) spec documents the Plan 3 eval-tagging contract.

- [ ] **Step 1: Write the failing tests (append to tests/test_profiles.py)**

```python
def test_sqlite_cascade_and_expiry(tmp_path):
    from voiceagent.learn.profiles import SQLiteProfiles
    s = SQLiteProfiles(str(tmp_path / "c.db"))
    s.put(Profile(key="k9", alias="Fam9", prefs=["x"], corrections=[],
                  open_items=[], pending_global=[], consent={},
                  updated_at="2020-01-01T00:00:00"))
    s.link_session("k9", "sess-9")
    assert s.get("k9") is None  # TTL prunes on read (2020 vs 365-day TTL)
    assert s.sessions_for("k9") == []  # links die with the profile
    with pytest.raises(KeyError):
        s.export_contact("k9")

def test_instant_correct_timing_on_realistic_bundle(tmp_path):
    import shutil, time
    from voiceagent.learn.instant import instant_correct
    shutil.copytree("data/deployments/_example/v1", tmp_path / "v1")
    t0 = time.monotonic()
    out = instant_correct(str(tmp_path), "No, support hours are 9am to 7pm")
    assert out["live"] is True and (time.monotonic() - t0) < 60
```

- [ ] **Step 2: Run tests to verify they fail-or-pass honestly**

Run: `.venv/bin/python -m pytest tests/test_profiles.py -q`
Expected: PASS if Task 1 covered SQLite TTL expiry on `get` (it tests `prune_expired` + `get`-returns-None for InMemory only) — if the SQLite `get` path does NOT prune, this FAILS and you must add expiry to `SQLiteProfiles.get` (same `_expired` helper, delete + return None). Either way, report which happened.

- [ ] **Step 3: Document the Plan 3 eval-tagging contract (spec append)**

Append to the spec file, new subsection at end of §4.5 (keep under 6 lines):

```markdown
### 4.5a Eval-tagging contract (for Plan 3 batch-learn)
Approved customer-derived evals MUST carry `source_contact_hash` (SHA-256 of the contact key, never the key); `purge_contact(hash)` removes them. Owner-authored instant evals carry `source: owner`. Bundle loader ignores unknown eval fields (forward-compatible).
```

Then make it true in code: `EvalCheck` gains no new field (forward-compat holds by ignoring); add one test in `tests/test_bundle.py` that an eval dict with an extra `source_contact_hash` key loads without error. If `load_bundle`/`EvalCheck` construction is strict (`EvalCheck(**e)` with exact kwargs would TypeError on extras), loosen it to pick known fields explicitly.

- [ ] **Step 4: Run the wider net and verify**

Run: `.venv/bin/python -m pytest tests/test_profiles.py tests/test_corrections.py tests/test_instant.py tests/test_learn_loop.py tests/test_bundle.py tests/test_selfcheck.py tests/test_gate.py tests/test_tools.py tests/test_orchestrator.py -q`
Expected: PASS, all green

- [ ] **Step 5: Commit**

```bash
git add tests/test_profiles.py tests/test_bundle.py src/voiceagent/learn/profiles.py docs/superpowers/specs/2026-09-03-global-adaptive-agent-design.md
git commit -m "feat: cascade proof, TTL on-read, eval-tag contract"
```

---

## Out of scope (explicitly later plans)

- Batch-learn nightly job + proposal queue + `purge_contact` implementation (§4.6) — Plan 3 (this plan only defines the tagging contract).
- Operator-pattern packs, 50-turn adversarial harness, onboarding measurement (§7.1-1,2,3) — Plan 4.
- Runtime tool-mining, auto-close, fine-tune — v2, not planned.
