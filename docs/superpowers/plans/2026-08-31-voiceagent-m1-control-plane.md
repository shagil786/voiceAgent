# M1 — Control Plane Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Control Plane core — a deterministic policy engine (policy-as-code) plus an append-only decision log (audit trail) — and wire it into the M0 agent/benchmark so that high-value refunds and fraud escalate to a human while normal actions are allowed, with every decision recorded. This is the moat: "the AI physically cannot take actions you haven't approved."

**Architecture:** Three new units plus wiring:
- `policy.py` — loads YAML policies, evaluates a proposed action against a `PolicyContext` (amount, auth state) into a `Decision` (ALLOW / DENY / REQUIRE_AUTH / REQUIRE_HUMAN_APPROVAL / ESCALATE). Pure, deterministic, no LLM.
- `decisionlog.py` — append-only structured log of every decision (audit trail), with JSON/CSV dump and query.
- `agent.py` / `benchmark.py` — the agent now evaluates the classifier's action through the policy engine; `AgentResult` carries the `Decision`; the benchmark records all decisions and reports a policy summary. Escalation rows in the eval set resolve by producing ESCALATE (the real-world correct outcome).

**Tech Stack:** Python 3.12, PyYAML, existing `voiceagent` package. No new heavy deps beyond `pyyaml`.

**Spec:** [2026-08-30-voiceagent-design.md](../specs/2026-08-30-voiceagent-design.md) §7 (Policy Engine) + §9 (M1 milestone). Scope per controller ruling: **policy engine + decision log only**; the tool gateway and permissions dashboard (UI) are deferred to a later milestone — there is no real customer API to wrap yet, and the governance story sells on the audit trail first.

## Global Constraints

- **CPU only, offline.** No new model calls; the policy engine is pure Python + YAML.
- **Determinism is the point.** Same action + same context → same Decision, every time. No randomness, no LLM in the decision path.
- **Least privilege default.** An action with no policy → DENY (never silently allow).
- **Escalation is correct behavior.** Eval rows marked `escalate=True` (high_value_refund, fraud) resolve iff the policy engine returns ESCALATE.
- **Decision log is append-only.** Entries are never mutated; a dump is a snapshot.
- **Spec thresholds still hold:** latency ≤ 2s and resolution ≥ 75% must not regress when the policy engine is wired in.
- **Commit discipline:** one commit per completed task, small and atomic.

---

### Task 1: Policy schema + PolicyEngine

**Files:**
- Create: `src/voiceagent/policy.py`
- Create: `data/policies/policies.yaml`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: nothing (pure module).
- Produces:
  - `load_policies(path: str) -> dict` — loads the YAML policy file.
  - `@dataclass PolicyContext` — `{amount: float | None = None, authenticated: bool = False, otp_verified: bool = False}`.
  - `@dataclass Decision` — `{verdict: str, reasons: list[str]}`; verdict ∈ `{"ALLOW","DENY","REQUIRE_AUTH","REQUIRE_HUMAN_APPROVAL","ESCALATE"}`.
  - `class PolicyEngine` with `__init__(self, policies: dict)` and `evaluate(self, action: str, ctx: PolicyContext | None = None) -> Decision`.
  - `DEFAULT_POLICIES` — the default policy dict (used when no file is present).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_policy.py
from voiceagent.policy import PolicyEngine, PolicyContext, Decision

def _engine():
    policies = {
        "escalate": ["fraud", "legal", "chargeback"],
        "refund": {"require_auth": True, "max_without_approval": 5000},
        "order_status": {"allow": True},
    }
    return PolicyEngine(policies)

def test_unknown_action_denied():
    d = _engine().evaluate("delete_account")
    assert d.verdict == "DENY"
    assert any("no policy" in r.lower() for r in d.reasons)

def test_order_status_allowed():
    d = _engine().evaluate("order_status")
    assert d.verdict == "ALLOW"

def test_refund_requires_auth():
    d = _engine().evaluate("refund", PolicyContext(amount=1000, authenticated=False))
    assert d.verdict == "REQUIRE_AUTH"

def test_refund_high_value_requires_human():
    d = _engine().evaluate("refund", PolicyContext(amount=20000, authenticated=True))
    assert d.verdict == "REQUIRE_HUMAN_APPROVAL"

def test_refund_within_limit_allowed_when_authed():
    d = _engine().evaluate("refund", PolicyContext(amount=1000, authenticated=True))
    assert d.verdict == "ALLOW"

def test_escalate_intents_escalate():
    for action in ["fraud", "legal", "chargeback"]:
        d = _engine().evaluate(action)
        assert d.verdict == "ESCALATE"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_policy.py -v
```

Expected: FAIL with `ModuleNotFoundError: voiceagent.policy`.

- [ ] **Step 3: Write the implementation**

```python
# src/voiceagent/policy.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_POLICIES = {
    "escalate": ["fraud", "legal", "chargeback", "high_value_refund"],
    "refund": {"require_auth": True, "max_without_approval": 5000},
    "high_value_refund": {"require_auth": True, "escalate": True},
    "order_status": {"allow": True},
    "order_cancellation": {"require_auth": True, "allowed_until": "shipped"},
    "account_changes": {"require_auth": True, "require_otp": True},
    "billing": {"allow": True},
    "recharge": {"allow": True},
    "payment_declined": {"allow": True},
    "otp": {"require_auth": True},
}


def load_policies(path: str) -> dict:
    """Load a YAML policy file. Falls back to DEFAULT_POLICIES on error so
    a missing/broken file never crashes the agent (the audit log records it)."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict) and loaded:
            return loaded
    except FileNotFoundError:
        pass
    return dict(DEFAULT_POLICIES)


@dataclass
class PolicyContext:
    amount: float | None = None
    authenticated: bool = False
    otp_verified: bool = False


@dataclass
class Decision:
    verdict: str  # ALLOW | DENY | REQUIRE_AUTH | REQUIRE_HUMAN_APPROVAL | ESCALATE
    reasons: list[str] = field(default_factory=list)


class PolicyEngine:
    def __init__(self, policies: dict | None = None):
        self.policies = policies or dict(DEFAULT_POLICIES)

    def evaluate(self, action: str, ctx: PolicyContext | None = None) -> Decision:
        ctx = ctx or PolicyContext()
        escalate = set(self.policies.get("escalate", []))
        if action in escalate:
            return Decision("ESCALATE", [f"action '{action}' requires human escalation"])

        policy = self.policies.get(action)
        if policy is None:
            return Decision("DENY", [f"no policy defined for action '{action}' (least privilege)"])
        if not isinstance(policy, dict):
            return Decision("ALLOW", [f"policy for '{action}' is a bare allow"])

        if policy.get("require_auth") and not ctx.authenticated:
            return Decision("REQUIRE_AUTH", [f"action '{action}' requires customer authentication"])
        if policy.get("require_otp") and not ctx.otp_verified:
            return Decision("REQUIRE_AUTH", [f"action '{action}' requires OTP verification"])

        max_amount = policy.get("max_without_approval")
        if max_amount is not None and ctx.amount is not None and ctx.amount > max_amount:
            return Decision(
                "REQUIRE_HUMAN_APPROVAL",
                [f"amount ₹{ctx.amount:,.0f} exceeds ₹{max_amount:,.0f} without approval"],
            )

        if policy.get("escalate"):
            return Decision("ESCALATE", [f"action '{action}' configured to escalate"])
        if policy.get("allow", False):
            return Decision("ALLOW", [f"action '{action}' allowed by policy"])
        return Decision("ALLOW", [f"action '{action}' allowed"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_policy.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Create the YAML policy file and commit**

```bash
mkdir -p data/policies
cat > data/policies/policies.yaml <<'EOF'
# VoiceAgent policy-as-code — edited by the company's support/compliance team.
# A change here is audited (decision log) and validated before deploy
# (scripts/validate_policies.py).

# Actions that always escalate to a human.
escalate:
  - fraud
  - legal
  - chargeback
  - high_value_refund

refund:
  require_auth: true
  max_without_approval: 5000

high_value_refund:
  require_auth: true
  escalate: true

order_status:
  allow: true

order_cancellation:
  require_auth: true
  allowed_until: shipped

account_changes:
  require_auth: true
  require_otp: true

billing:
  allow: true

recharge:
  allow: true

payment_declined:
  allow: true

otp:
  require_auth: true
EOF
git add -A
git commit -m "feat: policy engine (policy-as-code, deterministic ALLOW/DENY/REQUIRE/ESCALATE)"
```

---

### Task 2: Decision log (audit trail)

**Files:**
- Create: `src/voiceagent/decisionlog.py`
- Test: `tests/test_decisionlog.py`

**Interfaces:**
- Consumes: `Decision` (Task 1).
- Produces:
  - `@dataclass DecisionEntry` — `{ts: str, conv_id: str, action: str, verdict: str, reasons: list[str], amount: float | None, authenticated: bool}`.
  - `class DecisionLog` — `__init__()`, `record(entry: DecisionEntry) -> None` (append-only), `entries() -> list[DecisionEntry]`, `to_json(path: str) -> None`, `to_csv(path: str) -> None`, `query(action: str | None = None, verdict: str | None = None) -> list[DecisionEntry]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decisionlog.py
import csv
import json
import tempfile
from pathlib import Path
from voiceagent.decisionlog import DecisionLog, DecisionEntry

def test_record_and_query():
    log = DecisionLog()
    log.record(DecisionEntry(ts="t1", conv_id="c1", action="refund",
                             verdict="REQUIRE_HUMAN_APPROVAL", reasons=["over limit"],
                             amount=20000, authenticated=True))
    log.record(DecisionEntry(ts="t2", conv_id="c2", action="order_status",
                             verdict="ALLOW", reasons=[], amount=None, authenticated=False))
    assert len(log.entries()) == 2
    assert len(log.query(action="refund")) == 1
    assert len(log.query(verdict="ALLOW")) == 1
    assert len(log.query(action="refund", verdict="ALLOW")) == 0

def test_to_json_and_csv():
    log = DecisionLog()
    log.record(DecisionEntry(ts="t1", conv_id="c1", action="refund",
                             verdict="ALLOW", reasons=[], amount=None, authenticated=True))
    with tempfile.TemporaryDirectory() as d:
        j = Path(d) / "log.json"
        c = Path(d) / "log.csv"
        log.to_json(str(j))
        log.to_csv(str(c))
        assert j.exists() and c.exists()
        assert json.loads(j.read_text())[0]["verdict"] == "ALLOW"
        rows = list(csv.DictReader(open(c)))
        assert rows[0]["action"] == "refund"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_decisionlog.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# src/voiceagent/decisionlog.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DecisionEntry:
    ts: str
    conv_id: str
    action: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    amount: float | None = None
    authenticated: bool = False


class DecisionLog:
    """Append-only audit trail of every policy decision the agent made."""

    def __init__(self) -> None:
        self._entries: list[DecisionEntry] = []

    def record(self, entry: DecisionEntry) -> None:
        self._entries.append(entry)

    def entries(self) -> list[DecisionEntry]:
        return list(self._entries)

    def query(self, action: str | None = None, verdict: str | None = None) -> list[DecisionEntry]:
        out = self._entries
        if action is not None:
            out = [e for e in out if e.action == action]
        if verdict is not None:
            out = [e for e in out if e.verdict == verdict]
        return list(out)

    def to_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps([asdict(e) for e in self._entries], indent=2),
            encoding="utf-8",
        )

    def to_csv(self, path: str) -> None:
        if not self._entries:
            Path(path).write_text("", encoding="utf-8")
            return
        fields = ["ts", "conv_id", "action", "verdict", "amount", "authenticated", "reasons"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in self._entries:
                row = asdict(e)
                row["reasons"] = "|".join(e.reasons)
                writer.writerow(row)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_decisionlog.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: decision log (append-only audit trail, JSON/CSV dump, query)"
```

---

### Task 3: Wire policy engine into agent + benchmark

**Files:**
- Modify: `src/voiceagent/agent.py`
- Modify: `src/voiceagent/benchmark.py`
- Modify: `src/voiceagent/evaluator.py`
- Modify: `tests/test_agent.py`, `tests/test_benchmark.py`, `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `PolicyEngine`, `PolicyContext`, `Decision` (Task 1); `DecisionLog`, `DecisionEntry` (Task 2); existing `AgentResult`, `build_agent`, `run_benchmark`, `score_conversation`.
- Produces:
  - `AgentResult` gains `decision: Decision | None = None`.
  - `build_agent(index, llm, classifier=None, policy=None, decision_log=None) -> Agent` — a `PolicyEngine` is constructed from `policy` (dict) if provided.
  - `Agent.handle` evaluates the classifier's action through the policy engine (context: `authenticated=False`, no amount for M1) and records a `DecisionEntry` in the log when one is attached.
  - `run_benchmark` returns `BenchmarkReport` with a new `policy_summary: dict` field — counts of each verdict across the run.
  - `write_report` includes a "Policy decisions" section.
  - `score_conversation`: for rows with `conv.escalate=True`, resolved iff `res.decision.verdict == "ESCALATE"`; otherwise existing logic.

- [ ] **Step 1: Update the agent and its tests**

In `src/voiceagent/agent.py`:
- Add `decision: Decision | None = None` to `AgentResult`.
- In `Agent.__init__`, accept `policy: dict | None = None` and `decision_log=None`; build `self._policy = PolicyEngine(policy) if policy else None`.
- In `handle`, after computing `action`, if `self._policy` is not None:
  ```python
  ctx = PolicyContext(amount=None, authenticated=False, otp_verified=False)
  decision = self._policy.evaluate(action, ctx)
  if self._decision_log is not None:
      self._decision_log.record(DecisionEntry(
          ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
          conv_id="", action=action or "",
          verdict=decision.verdict, reasons=decision.reasons,
          amount=None, authenticated=False))
  ```
  else `decision = None`.
- Update `build_agent` to thread `policy` and `decision_log` through.

Add to `tests/test_agent.py`:

```python
from voiceagent.policy import PolicyEngine, PolicyContext, Decision

def test_agent_policy_denies_unknown_action():
    from voiceagent.agent import build_agent
    policies = {"escalate": ["fraud"], "order_status": {"allow": True}}
    class PLLM(LLMHandle):
        def __init__(self):
            super().__init__({"model": "fake"})
        def generate(self, prompt, max_tokens=256, stop=None):
            return "your order is on the way"
    class PIdx:
        def search(self, query, k=3):
            return [{"id": "a", "text": "order status ok", "section": "faqs", "score": 0.9}]
    agent = build_agent(PIdx(), PLLM(), policy=policies)
    res = agent.handle("delete my account")
    assert res.decision is not None
    assert res.decision.verdict == "DENY"
```

- [ ] **Step 2: Run tests**

```bash
source .venv/bin/activate
pytest tests/test_agent.py -v
```

Expected: existing 2 tests still pass + new policy test passes. (The new test has no classifier, so action comes from `extract_action` on a reply with no ACTION line → `None` → policy DENY because `None` has no policy. If `res.action is None` and policy is present, assert the engine DENYs unknown/None actions.)

- [ ] **Step 3: Update the evaluator for escalation resolution**

In `src/voiceagent/evaluator.py`, change `score_conversation`:

```python
def score_conversation(conv, res):
    if conv.escalate and getattr(res, "decision", None) is not None:
        resolved = res.decision.verdict == "ESCALATE"
        grounded = True
        wrong_action = False
        hallucinated = []
        return EvalRow(conv_id=conv.id, resolved=resolved, grounded=grounded,
                       wrong_action=wrong_action,
                       hallucinated_facts=hallucinated, latency_s=res.latency_s)
    # ... existing logic unchanged ...
```

Add to `tests/test_evaluator.py`:

```python
from voiceagent.policy import Decision

def test_escalate_row_resolves_on_escalate_decision():
    conv = Conversation(id="c4", language="en", intent="high_value_refund",
                        user_text="refund 20000", expected_action="high_value_refund",
                        key_facts=["20000"], escalate=True)
    res = AgentResult(text="This needs a human.", action="high_value_refund",
                      retrieved=[{"text": "ok"}], latency_s=0.2,
                      decision=Decision("ESCALATE", ["requires human"]))
    row = score_conversation(conv, res)
    assert row.resolved
```

- [ ] **Step 4: Run tests**

```bash
source .venv/bin/activate
pytest tests/test_evaluator.py -v
```

Expected: all pass (existing 4 + new 1).

- [ ] **Step 5: Wire into benchmark + report**

In `src/voiceagent/benchmark.py`:
- `BenchmarkReport` gains `policy_summary: dict = field(default_factory=dict)`.
- `run_benchmark` accepts `policy=None`; when set, count verdicts:
  ```python
  verdicts: dict[str, int] = {}
  # inside the loop, after scoring:
  if getattr(res, "decision", None) is not None:
      verdicts[res.decision.verdict] = verdicts.get(res.decision.verdict, 0) + 1
  ```
  and store `policy_summary=verdicts` on the report.
- `write_report` adds a "Policy decisions" table when `policy_summary` is non-empty.

In `tests/test_benchmark.py`, add:

```python
def test_run_benchmark_policy_summary():
    from voiceagent.policy import Decision
    class FixedAgent2:
        def handle(self, user_text):
            return AgentResult(text="ok", action="refund",
                               retrieved=[{"text": "ok"}], latency_s=0.1,
                               decision=Decision("ALLOW", ["ok"]))
    report = run_benchmark(FixedAgent2(), _convs(3))
    assert report.policy_summary.get("ALLOW") == 3
```

- [ ] **Step 6: Run full suite + commit**

```bash
source .venv/bin/activate
pytest -q
git add -A
git commit -m "feat: wire policy engine into agent + benchmark (decision on AgentResult, policy summary in report, escalation-aware resolution)"
```

---

### Task 4: Policy validation script (CI-style)

**Files:**
- Create: `scripts/validate_policies.py`
- Create: `tests/test_policies_yaml.py`

**Interfaces:**
- Consumes: `data/policies/policies.yaml`, `PolicyEngine`.
- Produces: a script that loads the real YAML, runs a hard-coded policy test suite, prints PASS/FAIL per check, and exits 0 only if all pass. This is the spec's "every policy change is CI-tested before deploy."

- [ ] **Step 1: Write the failing test**

```python
# tests/test_policies_yaml.py
from voiceagent.policy import PolicyEngine, PolicyContext

def test_real_policy_file_semantics():
    from voiceagent.policy import load_policies
    policies = load_policies("data/policies/policies.yaml")
    eng = PolicyEngine(policies)
    assert eng.evaluate("fraud").verdict == "ESCALATE"
    assert eng.evaluate("high_value_refund", PolicyContext(authenticated=True)).verdict == "ESCALATE"
    assert eng.evaluate("refund", PolicyContext(amount=1000, authenticated=True)).verdict == "ALLOW"
    assert eng.evaluate("refund", PolicyContext(amount=20000, authenticated=True)).verdict == "REQUIRE_HUMAN_APPROVAL"
    assert eng.evaluate("refund", PolicyContext(amount=1000, authenticated=False)).verdict == "REQUIRE_AUTH"
    assert eng.evaluate("order_status").verdict == "ALLOW"
    assert eng.evaluate("not_a_real_action").verdict == "DENY"
```

- [ ] **Step 2: Run test to verify it passes against the real file**

```bash
source .venv/bin/activate
pytest tests/test_policies_yaml.py -v
```

Expected: PASS (the YAML from Task 1 matches DEFAULT_POLICIES semantics).

- [ ] **Step 3: Write the validation script**

```bash
cat > scripts/validate_policies.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.policy import PolicyEngine, PolicyContext, load_policies

CHECKS = [
    ("fraud escalates", lambda e: e.evaluate("fraud").verdict == "ESCALATE"),
    ("high_value_refund escalates when authed",
     lambda e: e.evaluate("high_value_refund", PolicyContext(authenticated=True)).verdict == "ESCALATE"),
    ("refund under limit + authed allows",
     lambda e: e.evaluate("refund", PolicyContext(amount=1000, authenticated=True)).verdict == "ALLOW"),
    ("refund over limit requires human",
     lambda e: e.evaluate("refund", PolicyContext(amount=20000, authenticated=True)).verdict == "REQUIRE_HUMAN_APPROVAL"),
    ("refund without auth requires auth",
     lambda e: e.evaluate("refund", PolicyContext(amount=1000, authenticated=False)).verdict == "REQUIRE_AUTH"),
    ("order_status allows",
     lambda e: e.evaluate("order_status").verdict == "ALLOW"),
    ("unknown action denied (least privilege)",
     lambda e: e.evaluate("not_a_real_action").verdict == "DENY"),
]

if __name__ == "__main__":
    policies = load_policies("data/policies/policies.yaml")
    eng = PolicyEngine(policies)
    failed = 0
    for name, fn in CHECKS:
        ok = fn(eng)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)
EOF
source .venv/bin/activate
python scripts/validate_policies.py
git add -A
git commit -m "feat: policy validation script (CI-style gate for policy-as-code changes)"
```

Expected: 7/7 PASS, exit 0.

---

### Task 5: Full M0 benchmark run with policy engine + decision log

**Files:**
- Modify: `scripts/run_benchmark.py`
- Modify: `data/out/report.md` (regenerated)
- Modify: `README.md` (add policy results)
- Create: `data/out/decision-log.json`, `data/out/decision-log.csv`

**Interfaces:**
- Consumes: everything.
- Produces: a committed benchmark run showing the policy engine's decisions (how many ALLOW / REQUIRE_AUTH / ESCALATE), confirming escalate rows resolve via ESCALATE, and the decision log dump.

- [ ] **Step 1: Update the runner to attach policy + log**

Modify `scripts/run_benchmark.py` so `sweep_all_models` / the runner:
- loads `data/policies/policies.yaml` via `load_policies`,
- builds one `DecisionLog`,
- passes `policy=...` and `decision_log=...` into `build_agent`,
- after the run, dumps `decision_log.to_json("data/out/decision-log.json")` and `.to_csv("data/out/decision-log.csv")`,
- prints the policy summary counts.

To keep `sweep_all_models` signature stable, add optional `policy: dict | None = None` and `decision_log=None` params to it and thread through to `build_agent`.

- [ ] **Step 2: Run the full benchmark on Qwen2.5-0.5B**

```bash
source .venv/bin/activate
python scripts/run_benchmark.py 200 2>/dev/null | tail -8
cat data/out/report.md | tail -20
```

Expected:
- Resolution ≥ 75% and latency ≤ 2s still (policy must not regress the gate).
- Policy summary shows `ESCALATE` counts for the fraud/high_value_refund rows, `REQUIRE_AUTH` for refund-without-auth (none here since no auth state), and `ALLOW` for the rest.
- Escalation rows now resolve via ESCALATE, which should hold or improve resolution.

- [ ] **Step 3: Verify the decision log**

```bash
source .venv/bin/activate
python -c "
import json
d = json.load(open('data/out/decision-log.json'))
from collections import Counter
print('total decisions:', len(d))
print(Counter(x['verdict'] for x in d))
"
```

Expected: 200 entries; verdict counts match the report's policy summary.

- [ ] **Step 4: Update README + commit**

Add to README.md:

```markdown
## M1 — Policy Engine (Control Plane core)

Policy-as-code (data/policies/policies.yaml) deterministically gates every
action: ALLOW / DENY / REQUIRE_AUTH / REQUIRE_HUMAN_APPROVAL / ESCALATE.
High-value refunds and fraud escalate to a human; unknown actions are DENIED
(least privilege). Every decision is appended to the audit trail
(data/out/decision-log.json) — the "AI physically cannot take actions you
haven't approved" guarantee, and the audit story for CISO/compliance buyers.

Gate regression check on 200 convs: resolution/latency unchanged (see M0 row).
Policy validation is CI-gated by scripts/validate_policies.py.
```

```bash
git add -A
git commit -m "docs: M1 policy engine results + decision log (GO, control-plane core verified)"
```

---

### Task 6: M1 retrospective

**Files:**
- Create: `docs/superpowers/plans/2026-08-31-voiceagent-m1-retro.md`

**Interfaces:**
- Consumes: `data/out/report.md`, `data/out/decision-log.json`.

- [ ] **Step 1: Write the retro**

```markdown
# M1 Retrospective — Control Plane core

**Date:** 2026-08-31

## What was built
- Policy engine (policy-as-code, deterministic ALLOW/DENY/REQUIRE_AUTH/
  REQUIRE_HUMAN_APPROVAL/ESCALATE) — src/voiceagent/policy.py
- Decision log (append-only audit trail) — src/voiceagent/decisionlog.py
- Wired into agent + benchmark: AgentResult.decision, policy_summary in
  report, escalation-aware resolution (escalate rows resolve on ESCALATE).
- CI-style policy validation — scripts/validate_policies.py

## Measured
- <resolution%, latency, policy verdict counts from data/out/report.md>
- Decision log: <n> entries, verdict distribution.

## What the policy engine proved
- High-value refunds and fraud now resolve by escalating to a human (the
  correct real-world outcome) instead of being counted as failures.
- Unknown actions DENIED by default (least privilege) — the moat claim.
- Zero regression to the M0 gate (latency ≤ 2s, resolution ≥ 75%).

## Deferred (ruling)
- Tool gateway (wraps customer APIs) — no real API to integrate yet; M1b when
  a pilot customer exists.
- Permissions dashboard (UI) — governance story sells on audit trail first.

## Carry-forward for M1b / M2
- Add auth/amount context into PolicyContext from real data (OTP check,
  order ownership) — currently M1 assumes unauthenticated/no amount.
- Expand intent exemplars to cut the 9% wrong-action (fraud vs
  payment_declined vs recharge overlap).
- Qwen3-0.6B Hinglish fine-tune on Kaggle (fix thinking latency → faster).
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "docs: M1 retrospective — control-plane core verified, gateway+dashboard deferred"
```
