# ADR-001: Declared control plane vs learned memory plane

## Context
The agent makes promises on live phone calls. Tenant bundles originally mixed
two kinds of knowledge: compliance facts (what the agent may promise/never say)
and understanding facts (intents, exemplars, KB). External review correctly
flagged that bundles are configuration, not learning.

## Decision
Two planes, permanently:
- **Declared (small, versioned, human-owned)**: identity, persona, languages,
  policy rules, tool surface composition, seed KB documents. Lives in the
  tenant bundle. CI-validated (`scripts/validate_tenant.py`). Changes are diffs.
- **Learned (growing, agent-owned)**: intent prototypes, episodic fragments,
  correction outcomes. Lives in the memory store (SQL + vectors). Bounded by
  consolidation (top-K prototypes, TTL episodes, decay).

Policy is NEVER learned. Understanding is NEVER hardcoded. A tool may read
memory; no memory write may change what the agent is allowed to promise.

## Consequences
+ Compliance stays auditable; understanding adapts without deploys.
+ The onboarding story is a tiny config; the agent learns the long tail itself.
− Requires consolidation machinery (ADR-002).
