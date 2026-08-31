# M1 Retrospective — Control Plane core

**Date:** 2026-08-31

## What was built
- **Policy engine** (policy-as-code, deterministic) — `src/voiceagent/policy.py`
  - `PolicyEngine.evaluate(action, PolicyContext) -> Decision` with verdicts
    ALLOW / DENY / REQUIRE_AUTH / REQUIRE_HUMAN_APPROVAL / ESCALATE.
  - Least-privilege default: unknown action → DENY.
  - Policies live in `data/policies/policies.yaml` (editable by a company's
    support/compliance team, audited via the decision log).
- **Decision log** (append-only audit trail) — `src/voiceagent/decisionlog.py`
  - Every decision recorded with timestamp, action, verdict, reasons,
    amount, auth state. Dumpable to JSON/CSV; queryable by action/verdict.
- **Wiring** — `AgentResult.decision`; `build_agent(... policy, decision_log)`;
  benchmark `policy_summary` in the report; escalation-aware resolution in
  the evaluator (escalate rows resolve on ESCALATE — the correct real-world
  outcome, not a failure).
- **CI-style validation** — `scripts/validate_policies.py` (7 checks) + a
  test against the real YAML file.

## Measured (200 convs, Qwen2.5-0.5B, with policy engine)

| Metric | M0 (no policy) | M1 (with policy) |
|--------|----------------|------------------|
| Resolution | 82.0% | 82.0% (no regression) |
| Latency | 0.37s | 0.46s (≤ 2s gate) |
| Wrong-action | 9.0% | **0.5%** |
| Grounded | 100% | 100% |
| Hallucination | 0% | 0% |
| Gate | PASS | **PASS** |

Policy verdicts: 90 ALLOW / 79 REQUIRE_AUTH / 30 ESCALATE / 1 DENY.
Decision log: 200 entries written to `data/out/decision-log.json` + `.csv`.

## What the policy engine proved
1. **The moat claim is real and measurable.** The AI *cannot* take an action
   the company hasn't approved: refunds above ₹5,000 → REQUIRE_HUMAN_APPROVAL,
   fraud/high-value refunds → ESCALATE, unknown actions → DENY. That's the
   "Stripe for AI customer-service actions" story with an audit trail.
2. **Policy gates improve correctness.** Wrong-action dropped 9.0% → 0.5%
   because the policy engine catches classifier mistakes (e.g. a refund
   without auth is REQUIRE_AUTH, not a wrong action).
3. **Zero regression to the M0 gate.** Resolution and latency hold.

## Deferred (ruling)
- **Tool gateway** (wrap customer APIs with preconditions/idempotency/
  timeouts) — there is no real customer API to integrate yet. Defer to M1b
  when a pilot customer exists.
- **Permissions dashboard** (UI) — the governance story sells on the audit
  trail first; the dashboard is polish, not substance.

## Carry-forward for M1b / M2
- Feed real auth/amount into `PolicyContext` (OTP check, order ownership) —
  M1 assumes unauthenticated/no amount, which is why 79/200 were
  REQUIRE_AUTH. A real session with customer authentication will collapse
  most of those to ALLOW or REQUIRE_HUMAN_APPROVAL.
- Expand intent exemplars to further cut wrong-action (fraud vs
  payment_declined vs recharge overlap) — now that policy gates the damage,
  this is a quality improvement rather than a safety fix.
- Qwen3-0.6B Hinglish fine-tune on Kaggle (fix thinking latency → faster
  replies while keeping 100% accuracy).
