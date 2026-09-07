# ADR-003: Tool surface derives from ToolSpec; metadata lives at the binding

## Context
The brain's proposal surface was a hand-maintained dict and silently drifted
from the execution bindings twice (initiate_refund and order_lookup were
invisible to the brain for their first day).

## Decision
- A tool exists in exactly ONE place: its binding + ToolSpec in
  `voiceagent/tools.py` (or a backend adapter). The ToolSpec carries params,
  preconditions, contract facts, policy action, side_effects, description.
- The brain's proposal surface is COMPUTED from DEFAULT_TOOL_SPECS
  (`runtime._auto_gateway_tools`). Runtime has zero per-tool entries.
- Tenant bundles compose the surface per deployment via tools.yaml
  (which tools + description overrides). `escalate_to_human` must always be
  proposeable — the safety valve.
- Tenants NEVER invent tool bindings; bindings are code against real backends.

## Consequences
+ Proposal surface cannot drift from execution bindings (pin-tested).
+ Adding a tool anywhere = one place.
− Tool authors must declare metadata once at the binding (acceptable:
  contracts belong to their owners).

## Adding tools for a NON-default business

A new business's tools arrive as **code + declaration, two halves of one
tool**:
1. **Binding + ToolSpec** — a backend adapter module implementing the
   `SupportBackend` protocol (or extending `tools.py`), with the ToolSpec
   carrying params/preconditions/facts/action/side_effects/description.
   This is developer code by design: it executes real side effects against
   a real system.
2. **Declaration** — the tenant's `tools.yaml` composes the surface: which
   of the available bindings this tenant exposes, plus description/
   parameter overrides. See `data/tenants/default/tools.yaml` (generated
   from the bindings) and `data/tenants/example-acme/tools.yaml` for both
   shapes.

Tenants never invent bindings at runtime; developers never hand-edit the
brain's surface. The default bundle's tools.yaml is the worked example of
the composition mechanism, and its equivalence to the derived builtin
surface is the drift tripwire.

### Worked example: a second domain on the unchanged core

`src/voiceagent/demo_clinic.py` + `data/tenants/example-clinic/` are the
second-domain worked example: a CLINIC runs end-to-end (backend adapter →
ToolGateway → policy → brain prompt) with ZERO changes to the domain-neutral
core (`tools.py` execute chain, `runtime.py`, `policy.py`). The adapter maps
clinic semantics onto the seven generic `SupportBackend` verbs; the domain
language ("appointment") enters only as DATA — intents/ exemplars, tools.yaml
`action:` renames + description overrides, persona, knowledge. Known
limitation (not fixed here): a domain verb with NO generic-binding
equivalent (e.g. booking a new appointment) cannot be composed at all —
tools.yaml rejects unknown names by design — so such intents resolve through
the governed handoff/policy path, and a real first-class binding would
require the deliberate core seam (a new `SupportBackend` verb + ToolGateway
binding), exactly the "bindings are code" trade-off this ADR accepts.
