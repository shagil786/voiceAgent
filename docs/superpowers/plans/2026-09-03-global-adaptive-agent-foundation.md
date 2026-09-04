# Global Adaptive Agent — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the versioned bundle foundation: frozen schema v1, scoped site-fetch + owner-paste ingestion, compiler that emits bundle diffs, approval gate with tool lifecycle + dry-run, and runnable self-checks that mechanically gate go-live.

**Architecture:** New `voiceagent/deploy/` package (writes only, never executes external calls) produces versioned bundles under `data/deployments/<id>/v<N>/`; live pointer flips only on approval + green self-checks. One additive state check in `GovernedToolRunner` enforces PROPOSED/APPROVED-never-execute. Self-checks run real `Orchestrator.handle_turn` against a deterministic stub brain (fast tier) with one live spot-check flag.

**Tech Stack:** Python 3.12, stdlib only for new code (`urllib`, `html.parser`, `json`, `sqlite3` untouched), existing `pytest` suite, no new third-party deps.

**Spec:** `docs/superpowers/specs/2026-09-03-global-adaptive-agent-design.md` — this plan implements §4.1 (crawl/paste → knowledge), §4.2 (schema + goldens + runnable evals), §4.3 (gate + dry-run), and the §7.1 gates 2–3 harness. Follow-ups (separate plans): instant-learn + per-person profiles (§3/§4.4–4.5), batch-learn job (§4.6), operator-pattern packs (§9-5).

## Global Constraints

- `bundle.schema_version = 1` frozen; loader rejects unknown versions, never guesses.
- Crawl caps: max 50 pages, max depth 3, 15s per-page timeout; never leaves owner domain + explicit allowlist (≤3 URLs); robots-respecting; text-only.
- Pasted/uploaded owner docs are first-class sources (`source: owner_paste`), ranked above crawled chunks.
- Tool states `PROPOSED → APPROVED → CONNECTED`; only `CONNECTED` (creds + dry-run passed + owner-confirmed response) is callable; scope widening resets to `APPROVED`.
- Dry-run = auth probe (no side effects) + exactly one agreed benign read-only call, full request/response stored owner-visible with secrets redacted.
- Contact key decision (E.164 + alias + TTL) is Plan 2 scope — this plan stores no per-person profiles.
- Go-live = owner approval + 10/10 self-checks green, mechanically executed; rollback = repoint live pointer.
- No new third-party deps. Python 3.12. Existing pytest suite stays green.

---

## File structure

| File | Responsibility |
|---|---|
| `src/voiceagent/deploy/__init__.py` | Package exports (`SCHEMA_VERSION`, `load_bundle`, `save_bundle`) |
| `src/voiceagent/deploy/bundle.py` | Frozen schema: dataclasses (`Bundle`, `ToolEntry`, `EvalCheck`), `load_bundle` (reject unknown `schema_version`), `save_bundle`, `diff_bundles` (structural diff vN → vN+1), live-pointer helpers (`read_live`, `write_live`) |
| `src/voiceagent/deploy/ingest.py` | Scoped fetch (caps, allowlist, robots, text extraction via stdlib `html.parser`) + owner-paste ingestion → `knowledge/` chunks with `{text, source, crawled_at}` |
| `src/voiceagent/deploy/compiler.py` | `compile_bundle(sources, interview) → Bundle`: builds `spec.json`, `tools.json` (all `PROPOSED`), `policies.yaml` draft (external default `require_approval:true`), `evals.json` (10 self-checks) |
| `src/voiceagent/deploy/gate.py` | Approval transitions (`approve_knowledge`, `approve_tool`, `record_dry_run`, `connect_tool`), dry-run record schema + secret redaction, reset-to-`APPROVED` on scope change |
| `src/voiceagent/deploy/selfcheck.py` | `run_self_checks(bundle, make_brain) → list[CheckResult]`: executes each eval as real `Orchestrator.handle_turn` calls against assertions |
| `src/voiceagent/tools.py` (modify) | Additive tool-state check in `GovernedToolRunner.run`: `PROPOSED`/`APPROVED`/unknown → blocked + decision-logged, never executed |
| `data/deployments/_example/v1/` (golden) | Checked-in golden bundle: `bundle.json`, `tools.json`, `policies.yaml`, `knowledge/`, `evals.json` — must load byte-identical |
| `tests/test_bundle.py` | Schema/golden/diff/live-pointer tests |
| `tests/test_ingest.py` | Caps, allowlist, robots, paste-ranking tests (stub HTTP, no network) |
| `tests/test_compiler.py` | Bundle-shape, PROPOSED-default, policy-default, 10-evals tests |
| `tests/test_gate.py` | Lifecycle, dry-run, redaction, reset-on-scope-change, runner-block tests |

---

### Task 1: Frozen bundle schema + loader + golden bundle

**Files:**
- Create: `src/voiceagent/deploy/__init__.py`
- Create: `src/voiceagent/deploy/bundle.py`
- Create: `data/deployments/_example/v1/bundle.json`
- Create: `data/deployments/_example/v1/tools.json`
- Create: `data/deployments/_example/v1/policies.yaml`
- Create: `data/deployments/_example/v1/evals.json`
- Create: `tests/test_bundle.py`

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces: `SCHEMA_VERSION: int = 1`; `load_bundle(path: str | Path) -> Bundle` (raises `ValueError` on unknown `schema_version`); `save_bundle(bundle: Bundle, path) -> None`; `diff_bundles(old: Bundle, new: Bundle) -> list[dict]` (each `{section, kind: added|removed|changed, detail}`); `read_live(deploy_dir) -> str | None`; `write_live(deploy_dir, version: str) -> None` (writes `live` pointer file).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bundle.py
from pathlib import Path
import pytest
from voiceagent.deploy.bundle import (
    SCHEMA_VERSION, load_bundle, save_bundle, diff_bundles,
    read_live, write_live,
)

GOLDEN = Path("data/deployments/_example/v1")

def test_schema_version_frozen_at_1():
    assert SCHEMA_VERSION == 1

def test_golden_bundle_loads():
    b = load_bundle(GOLDEN)
    assert b.schema_version == 1
    assert b.deploy_id == "example"
    assert len(b.tools) >= 1
    assert all(t.state in ("PROPOSED", "APPROVED", "CONNECTED") for t in b.tools)

def test_unknown_schema_version_rejected(tmp_path):
    import json
    bad = tmp_path / "bundle.json"
    bad.write_text(json.dumps({"schema_version": 99, "deploy_id": "x"}))
    with pytest.raises(ValueError, match="schema_version"):
        load_bundle(tmp_path)

def test_diff_detects_tool_added_and_policy_changed():
    old = load_bundle(GOLDEN)
    new = load_bundle(GOLDEN)
    new.policies["refund"]["max_without_approval"] = 9999
    d = diff_bundles(old, new)
    assert any(x["section"] == "policies" and x["kind"] == "changed" for x in d)

def test_live_pointer_roundtrip(tmp_path):
    assert read_live(tmp_path) is None
    write_live(tmp_path, "v2")
    assert read_live(tmp_path) == "v2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bundle.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.deploy'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/voiceagent/deploy/__init__.py
from voiceagent.deploy.bundle import (
    SCHEMA_VERSION, Bundle, ToolEntry, EvalCheck,
    load_bundle, save_bundle, diff_bundles, read_live, write_live,
)
__all__ = ["SCHEMA_VERSION", "Bundle", "ToolEntry", "EvalCheck",
           "load_bundle", "save_bundle", "diff_bundles",
           "read_live", "write_live"]
```

```python
# src/voiceagent/deploy/bundle.py
"""Frozen bundle schema v1. Loader rejects unknown versions, never guesses."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA_VERSION = 1
TOOL_STATES = ("PROPOSED", "APPROVED", "CONNECTED")
LIVE_POINTER = "live"

@dataclass
class ToolEntry:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    state: str = "PROPOSED"
    connection_ref: str | None = None
    policy_action: str = ""
    scopes: list[str] = field(default_factory=list)

@dataclass
class EvalCheck:
    name: str
    turns: list[dict] = field(default_factory=list)  # [{user: str}]
    assert_: dict = field(default_factory=dict)      # {contains?, action?, verdict?}

@dataclass
class Bundle:
    schema_version: int
    deploy_id: str
    spec: dict = field(default_factory=dict)
    tools: list[ToolEntry] = field(default_factory=list)
    policies: dict = field(default_factory=dict)
    knowledge: list[dict] = field(default_factory=list)
    evals: list[EvalCheck] = field(default_factory=list)

def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def load_bundle(path: str | Path) -> Bundle:
    d = Path(path)
    meta = _read_json(d / "bundle.json")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema_version {meta.get('schema_version')!r}; "
            f"this code reads {SCHEMA_VERSION}")
    tools = [ToolEntry(**t) for t in _read_json(d / "tools.json")]
    for t in tools:
        if t.state not in TOOL_STATES:
            raise ValueError(f"tool {t.name!r} has bad state {t.state!r}")
    evals = [EvalCheck(name=e["name"], turns=e.get("turns", []),
                       assert_=e.get("assert", {}))
             for e in _read_json(d / "evals.json")]
    import json as _json
    pol_path = d / "policies.yaml"
    policies = _load_policies_yaml(pol_path)  # see note below
    kdir = d / "knowledge"
    knowledge = [_read_json(p) for p in sorted(kdir.glob("*.json"))] if kdir.exists() else []
    return Bundle(schema_version=1, deploy_id=meta.get("deploy_id", d.parent.name),
                  spec=meta.get("spec", {}), tools=tools,
                  policies=policies, knowledge=knowledge, evals=evals)
```

Note for implementer: the repo has no PyYAML dependency — read `scripts/validate_policies.py` and `src/voiceagent/policy.py` first and implement `_load_policies_yaml` by mirroring the existing YAML loading pattern exactly (do not add a dep; if they vendor a tiny parser, reuse it). `knowledge/` loads each `*.json` chunk `{text, source, crawled_at}`. Keep `load_bundle` strict: missing files raise `FileNotFoundError`, never synthesize defaults.

```python
def save_bundle(bundle: Bundle, path: str | Path) -> None:
    d = Path(path); d.mkdir(parents=True, exist_ok=True)
    (d / "bundle.json").write_text(json.dumps(
        {"schema_version": SCHEMA_VERSION, "deploy_id": bundle.deploy_id,
         "spec": bundle.spec}, indent=2, sort_keys=True), encoding="utf-8")
    (d / "tools.json").write_text(json.dumps(
        [asdict(t) for t in bundle.tools], indent=2, sort_keys=True), encoding="utf-8")
    (d / "evals.json").write_text(json.dumps(
        [{"name": e.name, "turns": e.turns, "assert": e.assert_}
         for e in bundle.evals], indent=2, sort_keys=True), encoding="utf-8")
    kdir = d / "knowledge"; kdir.mkdir(exist_ok=True)
    for i, ch in enumerate(bundle.knowledge):
        (kdir / f"{i:03d}.json").write_text(
            json.dumps(ch, indent=2, sort_keys=True), encoding="utf-8")

def diff_bundles(old: Bundle, new: Bundle) -> list[dict]:
    out: list[dict] = []
    if old.spec != new.spec:
        out.append({"section": "spec", "kind": "changed",
                    "detail": sorted(set(new.spec) | set(old.spec))})
    old_t, new_t = {t.name: t for t in old.tools}, {t.name: t for t in new.tools}
    for n in new_t.keys() - old_t.keys():
        out.append({"section": "tools", "kind": "added", "detail": n})
    for n in old_t.keys() - new_t.keys():
        out.append({"section": "tools", "kind": "removed", "detail": n})
    for n in old_t.keys() & new_t.keys():
        if asdict(old_t[n]) != asdict(new_t[n]):
            out.append({"section": "tools", "kind": "changed", "detail": n})
    if old.policies != new.policies:
        out.append({"section": "policies", "kind": "changed",
                    "detail": sorted(set(new.policies) | set(old.policies))})
    if old.knowledge != new.knowledge:
        out.append({"section": "knowledge", "kind": "changed",
                    "detail": f"{len(old.knowledge)}->{len(new.knowledge)} chunks"})
    return out

def read_live(deploy_dir: str | Path) -> str | None:
    p = Path(deploy_dir) / LIVE_POINTER
    return p.read_text(encoding="utf-8").strip() if p.exists() else None

def write_live(deploy_dir: str | Path, version: str) -> None:
    Path(deploy_dir, LIVE_POINTER).write_text(version + "\n", encoding="utf-8")
```

Golden files (write exactly): `bundle.json` `{"schema_version": 1, "deploy_id": "example", "spec": {"role": "example support", "tone": "concise", "patterns": ["answer"], "disclosures": []}}`; `tools.json` one entry `{"name": "fetch_status", "description": "Look up an order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}, "state": "PROPOSED", "connection_ref": null, "policy_action": "order_status", "scopes": ["read"]}`; `policies.yaml` minimal (`order_status:\n  allow: true\n` plus `refund:\n  require_auth: true\n  max_without_approval: 5000\n` — required because the Task 1 test mutates `policies["refund"]["max_without_approval"]`); `evals.json` two entries with one turn each.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bundle.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/deploy/__init__.py src/voiceagent/deploy/bundle.py data/deployments/_example/v1 tests/test_bundle.py
git commit -m "feat: frozen bundle schema v1 with golden bundle and diff"
```

---

### Task 2: Scoped ingestion (crawl caps + allowlist + paste-first)

**Files:**
- Create: `src/voiceagent/deploy/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Bundle` knowledge chunk shape `{text, source, crawled_at}` from Task 1.
- Produces: `fetch_site(seed_url: str, allowlist: list[str] | None = None, fetcher=None) -> list[dict]`; `ingest_owner_paste(text: str, label: str = "owner_paste") -> dict`; `rank_chunks(pasted: list[dict], crawled: list[dict]) -> list[dict]` (pasted first). fetcher signature: `fetch(url: str) -> tuple[str, str]` returning `(html, final_url)`; default fetcher uses stdlib `urllib` with 15s timeout. Constants: `MAX_PAGES = 50`, `MAX_DEPTH = 3`, `PAGE_TIMEOUT_S = 15`, `MAX_ALLOWLIST = 3`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
from voiceagent.deploy import ingest

def _stub_fetcher_factory(pages):
    def fetch(url):
        if url not in pages:
            raise OSError(f"blocked or missing: {url}")
        return pages[url], url
    return fetch

def test_caps_and_same_origin():
    pages = {"https://acme.test/": '<a href="/a">a</a><a href="https://evil.test/x">x</a><p>Hello</p>',
             "https://acme.test/a": "<p>Page A</p>"}
    chunks = ingest.fetch_site("https://acme.test/", fetcher=_stub_fetcher_factory(pages))
    sources = [c["source"] for c in chunks]
    assert sources and all(s.startswith("https://acme.test/") for s in sources)
    assert not any("evil.test" in s for s in sources)

def test_allowlist_capped_at_3():
    import pytest
    with pytest.raises(ValueError, match="allowlist"):
        ingest.fetch_site("https://acme.test/",
                          allowlist=["https://a.test/", "https://b.test/",
                                     "https://c.test/", "https://d.test/"],
                          fetcher=_stub_fetcher_factory({}))

def test_max_pages_cap():
    pages = {f"https://acme.test/p{i}": "<p>t</p>" for i in range(200)}
    pages["https://acme.test/"] = "".join(
        f'<a href="/p{i}">x</a>' for i in range(200)) + "<p>home</p>"
    chunks = ingest.fetch_site("https://acme.test/", fetcher=_stub_fetcher_factory(pages))
    assert len(chunks) <= ingest.MAX_PAGES

def test_owner_paste_ranks_first():
    pasted = [ingest.ingest_owner_paste("Refund window is 30 days", label="policy-doc")]
    crawled = [{"text": "old text", "source": "https://acme.test/", "crawled_at": "2020-01-01T00:00:00Z"}]
    ranked = ingest.rank_chunks(pasted, crawled)
    assert ranked[0]["source"] == "owner_paste:policy-doc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -q`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` (no `ingest` module)

- [ ] **Step 3: Write minimal implementation**

```python
# src/voiceagent/deploy/ingest.py
"""Scoped ingestion: owner domain + allowlist only. Text-only, capped."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib import robotparser, request

MAX_PAGES = 50
MAX_DEPTH = 3
PAGE_TIMEOUT_S = 15
MAX_ALLOWLIST = 3

class _TextLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text: list[str] = []
        self.links: list[str] = []
        self._skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)
    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
    def handle_data(self, data):
        if not self._skip and data.strip():
            self.text.append(data.strip())

def _default_fetcher(url: str) -> tuple[str, str]:
    req = request.Request(url, headers={"User-Agent": "VoiceAgent-deploy/1.0"})
    with request.urlopen(req, timeout=PAGE_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", errors="replace"), resp.geturl()

def _allowed(url: str, seed_host: str, extra: set[str]) -> bool:
    host = urlparse(url).netloc
    return host == seed_host or host in extra

def fetch_site(seed_url: str, allowlist: list[str] | None = None,
               fetcher=None) -> list[dict]:
    allowlist = allowlist or []
    if len(allowlist) > MAX_ALLOWLIST:
        raise ValueError(f"allowlist capped at {MAX_ALLOWLIST}")
    fetch = fetcher or _default_fetcher
    seed_host = urlparse(seed_url).netloc
    extra = {urlparse(u).netloc for u in allowlist}
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(seed_url, 0)]
    chunks: list[dict] = []
    gaps: list[str] = []
    while queue and len(chunks) < MAX_PAGES:
        url, depth = queue.pop(0)
        if url in seen or depth > MAX_DEPTH:
            continue
        seen.add(url)
        try:
            html, final_url = fetch(url)
        except Exception as e:  # gap, never half-parse
            gaps.append(f"{url}: {e}")
            continue
        if not _allowed(final_url, seed_host, extra):
            continue
        p = _TextLinks()
        p.feed(html)
        text = " ".join(p.text)[:4000]
        if text:
            chunks.append({"text": text, "source": final_url,
                           "crawled_at": datetime.now(timezone.utc).isoformat()})
        if depth < MAX_DEPTH:
            for href in p.links:
                nxt = urljoin(final_url, href)
                if nxt.startswith("http") and nxt not in seen:
                    queue.append((nxt, depth + 1))
    for g in gaps:
        chunks.append({"text": "", "source": f"gap:{g}",
                       "crawled_at": datetime.now(timezone.utc).isoformat()})
    return chunks

def ingest_owner_paste(text: str, label: str = "owner_paste") -> dict:
    return {"text": text[:8000], "source": f"owner_paste:{label}",
            "crawled_at": datetime.now(timezone.utc).isoformat()}

def rank_chunks(pasted: list[dict], crawled: list[dict]) -> list[dict]:
    real = [c for c in crawled if not c["source"].startswith("gap:")]
    return list(pasted) + real
```

Implementer note: check `urllib.robotparser` fetch of `/robots.txt` for the seed host before crawling (best-effort, 5s timeout; on failure proceed — log a gap chunk). Keep it inside `fetch_site` so tests with stub fetchers skip network entirely.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ingest.py tests/test_bundle.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/deploy/ingest.py tests/test_ingest.py
git commit -m "feat: scoped ingestion with caps, allowlist, paste-first ranking"
```

---

### Task 3: Compiler (sources + interview → bundle vN+1)

**Files:**
- Create: `src/voiceagent/deploy/compiler.py`
- Test: `tests/test_compiler.py`

**Interfaces:**
- Consumes: `ingest.rank_chunks` output; `Bundle`, `ToolEntry`, `EvalCheck`, `SCHEMA_VERSION` from Task 1.
- Produces: `compile_bundle(deploy_id: str, chunks: list[dict], interview: dict) -> Bundle` where `interview = {offering: str, top_asks: list[str], never_promise: list[str], handoff_triggers: list[str]}`. Guarantees: every tool `state == "PROPOSED"`; policies default external actions to `require_approval: true`; exactly 10 evals.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compiler.py
from voiceagent.deploy.compiler import compile_bundle
from voiceagent.deploy.ingest import ingest_owner_paste

INTERVIEW = {
    "offering": "Sharma Realty sells 2/3BHK flats in Whitefield",
    "top_asks": ["2BHK price?", "site visit slot?", "loan help?",
                 "floor plans?", "possession date?"],
    "never_promise": ["never promise loan approval", "never quote final price"],
    "handoff_triggers": ["customer asks legal", "budget above 2cr"],
}

def test_compiler_emits_gated_bundle_with_10_evals():
    chunks = [ingest_owner_paste("2BHK from 85L. Site visits 10am-6pm.")]
    b = compile_bundle("sharma-realty", chunks, INTERVIEW)
    assert b.schema_version == 1 and b.deploy_id == "sharma-realty"
    assert b.tools and all(t.state == "PROPOSED" for t in b.tools)
    assert len(b.evals) == 10
    assert "answer" in b.spec.get("patterns", [])
    for name in ("escalate_to_human",):
        assert b.policies[name] == {"allow": True}

def test_external_tools_default_require_approval():
    chunks = [ingest_owner_paste("We book site visits.")]
    b = compile_bundle("x", chunks, INTERVIEW)
    # escalate_to_human is always-allowed by design; external tools default gated
    for t in b.tools[1:]:
        action = t.policy_action or t.name
        assert b.policies.get(action, {}).get("require_approval") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compiler.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.deploy.compiler'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/voiceagent/deploy/compiler.py
"""Deterministic compiler: sources + interview -> Bundle. No LLM calls in v1;
slot-filling + templates only (brain-assisted extraction is a later upgrade)."""
from __future__ import annotations

import re
from voiceagent.deploy.bundle import (
    Bundle, EvalCheck, ToolEntry, SCHEMA_VERSION)

_STOP = {"price", "cost", "book", "slot", "visit", "loan", "plan",
        "possession", "refund", "status", "cancel", "timing", "contact"}

def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "ask"

def compile_bundle(deploy_id: str, chunks: list[dict], interview: dict) -> Bundle:
    offering = interview.get("offering", "")
    top_asks = list(interview.get("top_asks", []))[:5]
    never = list(interview.get("never_promise", []))
    handoffs = list(interview.get("handoff_triggers", []))
    knowledge = [c for c in chunks if c.get("text")]
    tools = [ToolEntry(name="escalate_to_human",
                       description="Hand off to a human with full context",
                       parameters={"type": "object", "properties": {}},
                       state="PROPOSED", policy_action="escalate_to_human",
                       scopes=[])]
    for ask in top_asks:
        name = _slug(ask)
        if name in {t.name for t in tools} or name in _STOP and len(name) < 4:
            name = f"ask_{name}"
        tools.append(ToolEntry(
            name=name, description=f"Handle: {ask}",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string"}}},
            state="PROPOSED", policy_action=name, scopes=["read"]))
    policies: dict = {"escalate_to_human": {"allow": True}}
    for t in tools[1:]:
        policies[t.policy_action] = {"require_approval": True}
    spec = {"role": offering[:200], "tone": "concise, no invented facts",
            "patterns": ["answer", "qualify", "follow_up", "draft_action"],
            "never_promise": never, "handoff_triggers": handoffs,
            "disclosures": ["I am an AI assistant; a human signs off actions."]}
    evals = [EvalCheck(name=f"selfcheck-{i+1:02d}",
                       turns=[{"user": top_asks[i % len(top_asks)]}] if top_asks else [{"user": "Hello"}],
                       assert_={"contains": top_asks[i % len(top_asks)][:12]} if top_asks else {"contains": "Hello"})
             for i in range(10)]
    return Bundle(schema_version=SCHEMA_VERSION, deploy_id=deploy_id,
                  spec=spec, tools=tools, policies=policies,
                  knowledge=knowledge, evals=evals)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_compiler.py tests/test_bundle.py tests/test_ingest.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/deploy/compiler.py tests/test_compiler.py
git commit -m "feat: deterministic bundle compiler with gated defaults"
```

---

### Task 4: Approval gate + dry-run + runner enforcement

**Files:**
- Create: `src/voiceagent/deploy/gate.py`
- Modify: `src/voiceagent/tools.py` (additive state check in `GovernedToolRunner.run`)
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `Bundle`, `ToolEntry`, `load_bundle`, `save_bundle`, `diff_bundles` (Tasks 1–3); `GovernedToolRunner.run(...)` and `DecisionLog` semantics in `tools.py` (read the file first — mirror its `PolicyContext`/verdict plumbing exactly).
- Produces: `approve_knowledge(bundle) -> Bundle` (returns copy with `spec["knowledge_approved"] = True`); `approve_tool(bundle, name) -> Bundle` (`PROPOSED → APPROVED`, raises `ValueError` on unknown name); `tool_state(bundle, name) -> str`; `get_dry_run(bundle, name) -> dict | None`; `record_dry_run(bundle, name, probe: dict, confirmed_by: str) -> Bundle`; `widen_scope(bundle, name, scopes) -> Bundle` (resets `CONNECTED → APPROVED`, clears `dry_run`); `redact(obj)` (replaces values whose key matches `key|token|secret|password|auth`, case-insensitive, with `"[REDACTED]"`, applied to stored dry-run probes). Data change: add `dry_run: dict | None = None` field to `ToolEntry` in `bundle.py` (default `None`, included in `asdict` diffing). `record_dry_run` validates probe has `auth_ok is True` plus `benign_call: {request, response}`, requires non-empty `confirmed_by`, redacts secrets before storing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate.py
import pytest
from voiceagent.deploy.bundle import load_bundle
from voiceagent.deploy import gate

GOLDEN = "data/deployments/_example/v1"

def test_lifecycle_proposed_approved_connected():
    b = load_bundle(GOLDEN)
    b = gate.approve_tool(b, "fetch_status")
    assert gate.tool_state(b, "fetch_status") == "APPROVED"
    probe = {"auth_ok": True,
             "benign_call": {"request": "GET /status?limit=1",
                             "response": {"ok": True}},
             "api_key": "sk-live-123"}
    b = gate.record_dry_run(b, "fetch_status", probe, confirmed_by="owner")
    assert gate.tool_state(b, "fetch_status") == "CONNECTED"
    stored = gate.get_dry_run(b, "fetch_status")
    assert stored["api_key"] == "[REDACTED]"

def test_dry_run_requires_auth_and_confirmation():
    b = load_bundle(GOLDEN)
    b = gate.approve_tool(b, "fetch_status")
    with pytest.raises(ValueError):
        gate.record_dry_run(b, "fetch_status",
                            {"auth_ok": False, "benign_call": {}}, confirmed_by="owner")
    with pytest.raises(ValueError):
        gate.record_dry_run(b, "fetch_status",
                            {"auth_ok": True, "benign_call": {"request": "x", "response": "y"}},
                            confirmed_by="")

def test_scope_widening_resets_to_approved():
    b = load_bundle(GOLDEN)
    b = gate.approve_tool(b, "fetch_status")
    b = gate.record_dry_run(b, "fetch_status",
                            {"auth_ok": True, "benign_call": {"request": "r", "response": "s"}},
                            confirmed_by="owner")
    b2 = gate.widen_scope(b, "fetch_status", ["read", "write"])
    assert gate.tool_state(b2, "fetch_status") == "APPROVED"
    assert gate.get_dry_run(b2, "fetch_status") is None

def test_runner_blocks_unconnected_tool():
    # Behavioral: a PROPOSED-state tool must block + decision-log, never execute.
    # Read src/voiceagent/tools.py GovernedToolRunner.run signature first and
    # write the exact call (pass tool_states={"fetch_status": "PROPOSED"} plus
    # whatever context/policy/log args the real signature requires), then assert
    # the outcome is blocked (ok is False) and the decision log gained an entry
    # with verdict BLOCKED_UNCONNECTED.
```

For the runner assertion, read `src/voiceagent/tools.py` `GovernedToolRunner.run` first, then extend the test to call it with a `PROPOSED`-state tool and assert the outcome is blocked (`ok is False`, verdict `DENY` or `BLOCKED_UNCONNECTED`) and a decision-log entry exists. Write the exact call against the real signature — no mocks of the runner itself.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gate.py -q`
Expected: FAIL with `ModuleNotFoundError` (no `gate` module)

- [ ] **Step 3: Write minimal implementation**

`gate.py`: pure functions over `Bundle` (copy-on-write via `copy.deepcopy`), plus `redact`. `tools.py` change: at the top of `GovernedToolRunner.run`, after resolving the tool spec, look up `tool_state` — the runner receives it via an optional `tool_states: dict[str, str] | None = None` parameter defaulting to `None` (legacy callers unaffected: `None` means "pre-gate deployment, enforce policy only" — document this). If state is present and != `"CONNECTED"`: append a decision-log entry (`verdict="BLOCKED_UNCONNECTED"`, reason `tool {name} is {state}, owner approval required`) and return the blocked outcome shape the file already uses. `CONNECTED` and `None` proceed to existing policy evaluation untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gate.py tests/test_bundle.py tests/test_compiler.py tests/test_ingest.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/deploy/gate.py src/voiceagent/tools.py tests/test_gate.py
git commit -m "feat: approval gate lifecycle with dry-run and runner enforcement"
```

---

### Task 5: Runnable self-checks + go-live mechanics (§7.1 gates 2–3 harness)

**Files:**
- Create: `src/voiceagent/deploy/selfcheck.py`
- Test: `tests/test_selfcheck.py`

**Interfaces:**
- Consumes: `Bundle` + `EvalCheck` (Task 1); `Orchestrator`, `Deployment` from `src/voiceagent/orchestrator.py` (read `deploy()` + `handle_turn` signatures first); gate `tool_state` (Task 4).
- Produces: `run_self_checks(bundle: Bundle, make_brain=None, live_spot: bool = False) -> list[dict]` each `{name, passed: bool, detail}`; `go_live(deploy_dir: str, version: str, results: list[dict]) -> bool` (writes live pointer only if 10/10 passed, returns `True`; else `False`, pointer untouched). Default `make_brain` builds the deterministic stub used in `tests/test_orchestrator.py` (read that file — reuse its `FakeBrain` pattern, do not invent a new harness).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selfcheck.py
from voiceagent.deploy.bundle import load_bundle
from voiceagent.deploy import selfcheck

GOLDEN = "data/deployments/_example/v1"

def test_self_checks_run_and_gate_go_live(tmp_path):
    b = load_bundle(GOLDEN)
    results = selfcheck.run_self_checks(b)
    assert len(results) == len(b.evals) and len(results) >= 2
    assert all(set(r) == {"name", "passed", "detail"} for r in results)

def test_go_live_requires_ten_of_ten(tmp_path):
    ok = [{"name": f"c{i}", "passed": True, "detail": ""} for i in range(10)]
    assert selfcheck.go_live(str(tmp_path), "v3", ok) is True
    bad = list(ok); bad[0] = {"name": "c0", "passed": False, "detail": "x"}
    assert selfcheck.go_live(str(tmp_path), "v4", bad) is False
    from voiceagent.deploy.bundle import read_live
    assert read_live(str(tmp_path)) == "v3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_selfcheck.py -q`
Expected: FAIL with `ModuleNotFoundError` (no `selfcheck` module)

- [ ] **Step 3: Write minimal implementation**

```python
# src/voiceagent/deploy/selfcheck.py
"""Runnable self-checks: each eval is real Orchestrator.handle_turn calls."""
from __future__ import annotations
from voiceagent.deploy.bundle import read_live, write_live

def _check_turn(reply: str, action, verdict, assertion: dict) -> tuple[bool, str]:
    if "contains" in assertion and assertion["contains"] not in reply:
        return False, f"missing {assertion['contains']!r} in {reply!r}"
    if "action" in assertion and action != assertion["action"]:
        return False, f"action {action!r} != {assertion['action']!r}"
    if "verdict" in assertion and verdict != assertion["verdict"]:
        return False, f"verdict {verdict!r} != {assertion['verdict']!r}"
    return True, "ok"

def run_self_checks(bundle, make_brain=None, live_spot: bool = False) -> list[dict]:
    from voiceagent.orchestrator import Deployment, Orchestrator
    from voiceagent.memory import InMemoryMemory
    out: list[dict] = []
    for ev in bundle.evals:
        if make_brain is None:
            from tests.test_orchestrator import FakeBrain  # reuse, don't reinvent
            brain = FakeBrain()
        else:
            brain = make_brain()
        dep = Deployment(name=bundle.deploy_id,
                         system_prompt=bundle.spec.get("role", ""),
                         knowledge={str(i): c["text"] for i, c in enumerate(bundle.knowledge)})
        orch = Orchestrator(brain=brain, deployment=dep, memory=InMemoryMemory())
        ok_all, details = True, []
        for t in ev.turns:
            r = orch.handle_turn(session_id=f"selfcheck-{ev.name}", user_text=t["user"])
            ok, d = _check_turn(r.reply, (r.actions or [{}])[0].get("action"), None, ev.assert_)
            ok_all = ok_all and ok
            details.append(d)
        out.append({"name": ev.name, "passed": ok_all, "detail": "; ".join(details)})
    return out

def go_live(deploy_dir: str, version: str, results: list[dict]) -> bool:
    if len(results) >= 10 and all(r["passed"] for r in results):
        write_live(deploy_dir, version)
        return True
    return False
```

Implementer: verify `TurnResult` field names (`reply`, `actions`), `Orchestrator.__init__` params, and `FakeBrain` location/name against the real files before running — adjust names, never signatures of existing classes. If `FakeBrain` lives elsewhere, import from its real path.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_selfcheck.py tests/test_gate.py tests/test_bundle.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/deploy/selfcheck.py tests/test_selfcheck.py
git commit -m "feat: runnable self-checks with mechanical go-live gate"
```

---

## Out of scope (explicitly later plans)

- Instant-learn + per-person profiles + TTL/delete (spec §4.4–4.5) — Plan 2.
- Batch-learn nightly job + proposal queue (spec §4.6) — Plan 3.
- Operator-pattern packs (spec §9-5) and 50-turn adversarial harness + ≤30min onboarding measurement (spec §7.1-1,3,4) — Plan 4.
- Runtime tool-mining, auto-close, fine-tune — v2, not planned.
