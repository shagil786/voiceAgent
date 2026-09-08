# ADR-005: Owner-approved tool proposals — the safe path to tool adaptability

## Context
ADR-004 rejected *agent-created executable tools* (runtime codegen +
sandboxing): it trades the platform's core safety property — a governed,
human-authored executable surface (ADR-003: "tenants never invent tool
bindings") — for speculative adaptability, with security rated Critical and
hallucination Medium by the proposing review.

The underlying product pull remains real: onboarding a new business today
means a code-authored domain module (`demo_repairs.py`, `demo_clinic.py`)
plus a tenant bundle. Fast for a platform team; not zero-touch.

What is genuinely desirable is NOT "the agent invents and runs tools" but
"the platform can PROPOSE new surface from evidence, and a human approves
before anything executes." Every executable tool stays human-approved; the
change is that the *proposal* can be generated (from an OpenAPI spec, from
a KB, from an onboarding interview) instead of hand-written.

## Decision
Adopt a **tool-proposal lifecycle with a mandatory human approval gate** —
as a DESIGNED, gated capability, not runtime tool codegen:

```
DISCOVERED -> PROPOSED -> REVIEWED/APPROVED -> REGISTERED -> MONITORED
                       (human decision; rejected => DROPPED)
```

Concretely, and deliberately NOT yet built:
1. **Discovery is offline and bounded** — a proposal generator reads an
   explicit artifact the operator supplies (OpenAPI/AsyncAPI spec, or a
   declared KB file). It does NOT roam a live system and does NOT run code.
2. **Proposal = declaration, not code** — the artifact is a structured
   ToolProposal (name, params, resource/operation mapping, risk class),
   stored as data. It can never execute by itself.
3. **Approval gate is human, in the platform's existing review channel** —
   a `proposals/` bundle section the operator commits, exactly like
   tools.yaml today; nothing is REGISTERED until a human merges it. This is
   the same trust boundary as the current code review, moved to data.
4. **Risk classification is explicit data** — cancel/refund-class
   operations demand the same review depth as today; read-only proposals
   are lighter review. No operation is exempt from the gate.
5. **Registration uses ADR-004's existing mechanism** — an approved
   proposal compiles (in CI) to `register_binding` + ToolSpec calls over
   the deployment's GenericBackend, which is precisely the surface a
   proposal references. MONITORED = the existing DecisionLog + audit trail.

Explicitly rejected in this ADR (same reasoning as ADR-004):
- Runtime tool code generation (the agent writes an executable binding
  mid-call).
- Sandboxed execution of agent-authored code (a full sandbox is a
  research-grade security product; the platform's value is not shipping
  one).
- Approving tools from the agent's own chat (an agent must not be the
  approver of its own surface).

## Consequences
+ New businesses can be onboarded from their API spec + interview without
  a platform-code change — the proposal compiles to the same governed
  registrations a hand-written module would have made.
+ Every executable tool still passes a human gate; ADR-003's invariant
  ("tenants never invent bindings") is preserved: the OPERATOR approves,
  never the tenant's LLM.
+ Reuses ADR-004's registration + GenericBackend + ToolSpec.resource —
  no new execution machinery, no sandbox.
- A real build (~1-2 weeks: proposal schema, OpenAPI parser, compile-to-
  registration in CI, proposal review UX). Not started; this ADR records
  the design so the work is decision-ready when self-serve onboarding or a
  second real tenant demands it.
- Proposal generators can hallucinate a WRONG mapping (API field -> tool
  param): mitigation is the human gate + CI validation that the proposal
  compiles against the declared GenericBackend surface before review.

## Verification gate (when built)
A proposal for a NEW domain must, through the pipeline, produce exactly
the governed surface a hand-written domain module would: same specs, same
preconditions, same policy gating, same DecisionLog trail — and every
proposal must be REJECTABLE with zero residue (nothing registered).
