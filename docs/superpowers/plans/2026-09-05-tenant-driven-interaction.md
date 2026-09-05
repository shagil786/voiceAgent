# Sprint: Tenant-Driven Interaction Layer (bot→orchestration rework)

Branch: `feat/livekit-limb`. Base: ca08ee8. Suite baseline: 566 passed, 2 skipped, 1 xfailed.

External audit (2026-09-05) found the interaction layer behaves like a rule bot:
canned templates REPLACE frontier output, core-hardcoded action vocabulary,
business thresholds in Python, keyword guardrails hardcoded. Orchestration core
(policy engine, governed tools, decision log, tenant bundles) is fine and stays.

## Hard constraints (non-negotiable)

1. **No-frontier mode must keep working**: BASE-tier deterministic templates are
   the product's free tier and degraded-mode floor. They become LAST resort,
   never first choice.
2. **Language invariants stay hard**: never-Hindi for global tenants, replies
   only in tenant-declared languages. Mechanism may change; invariant may not.
3. **Policy-first execution unchanged**: PolicyEngine verdict → ALLOW-only →
   DecisionLog. The sprint changes what the brain can say/propose, not the
   governance spine.
4. **Fail-open only at the surface**: every new failure path falls back to the
   previous behavior (current templates), never to an exception mid-call.
5. Tests first; suite must stay green; new behavior needs new tests.

## Tasks

### A. Tenant-declared vocabularies + data-driven thresholds + tool-contract facts
- `DEFAULT_ACTIONS` (agent.py) moves out of core: tenant bundle declares its
  action vocabulary (derive from tenant intents dir + tools.yaml; tenant.json
  may add info-only actions). Core keeps the current list ONLY as the
  default/demo tenant's data, clearly labeled as demo content.
- `high_value_refund` threshold (agent.py `amount > 5000`) moves to
  `policies.yaml` (per-tenant, currency-aware via existing PolicyEngine
  currency param). Python reads policy, never hardcodes.
- `KEYWORD_FACTS` (agent.py) — echo-guardrail required facts become ToolSpec
  contract data (`facts`/`guarantees` per tool in tools.yaml + DEFAULT_TOOL_SPECS),
  derived at guard time from the executed tool's spec, not a hardcoded dict.
- Update data/tenants/example-acme accordingly (threshold, facts) + validator
  (scripts/validate_tenant.py) asserts the new fields.

### B. Guardrails guide, not replace + clarify-and-dig escalation ladder
- On language/echo guardrail violation with a frontier configured: governed
  re-render (one retry) — prompt the frontier to restate within constraints
  (allowed languages, required facts, persona never-say). Template fallback
  only when no frontier OR retry still violates. Deterministic BASE mode
  (no frontier) unchanged.
- Not-found ladder (policies.yaml-driven, default 2 attempts): tool says
  not-found → ask to reconfirm ID → offer alternate lookup (phone/email per
  tenant tools) → offer known options → escalate only after ladder exhausted.
  DialogueTracker gains a NOT_FOUND state machine; escalation stays the
  mandatory terminal.
- Voice path unchanged: same brain, same directives (directive-anchored
  prompts), telephony worker untouched.

### C. Real tenant bundle (BLOCKED — needs user's domain content)
- Build user's actual tenant from their business description + KB docs +
  clarify-dig ladder demo. Separate task once the user supplies the domain.

## Verification
- Full suite green; new tests for: action vocabulary from tenant data,
  threshold from policy, facts from tool spec, re-render path (mock frontier),
  template fallback path, ladder end-to-end (multi-turn reschedule-style test).
- validate_tenant.py gates the new schema fields.
