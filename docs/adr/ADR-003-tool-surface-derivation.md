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
