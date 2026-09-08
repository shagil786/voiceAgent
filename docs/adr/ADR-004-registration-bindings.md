# ADR-004: Registration-based tool bindings; domain backends stay structural

## Context
ADR-003 fixed surface/binding drift by making the ToolGateway the single
home of tool bindings — but the bindings are an if/elif chain inside
`ToolGateway.execute`, and the `SupportBackend` protocol names domain verbs
(`get_order`, `cancel_order`, ...). Two consequences:

1. Adding any tool requires editing platform code (tools.py), in TWO places
   (spec + binding branch) — mechanical but platform-coupled.
2. A non-e-commerce domain must either map its nouns to order-words
   (clinic: appointments->orders) or wait for platform protocol changes.

An external review proposed a full 5-phase/20-week migration: resource-
oriented generic CRUD backends, LLM-driven tool discovery, and agent-created
tools with sandboxed execution. Verified against this codebase:

- Preconditions/lifecycle values are ALREADY tenant data (tools.yaml
  preconditions override code defaults; SHIPPED/DELIVERED is the demo
  default, not a platform rule) — "lifecycle hardcoding" is mostly solved.
- The clinic proves the structural-protocol path works (12 tests, zero core
  change) — the cost is a domain adapter, which is exactly where domain
  vocabulary belongs.
- Agent-created tools (the report's Phase 4/5) would trade the platform's
  strongest property — a governed, human-authored surface (ADR-003: "tenants
  NEVER invent tool bindings") — for speculative adaptability, with the
  report itself rating the security risk "Critical" and hallucination risk
  "Medium". Rejected until self-serve tenant onboarding is a product
  requirement.

## Decision
1. **Bindings become registrations, not branches.** `ToolGateway` gains
   `register_binding(tool_name, executor: Callable[[Any, dict], Any])` (and
   a `bindings` mapping). `execute()` looks up the registration; the
   historical if/elif chain becomes DEFAULT bindings registered at
   construction against the classic `SupportBackend` surface. Behavior,
   error contract, idempotency, and precondition flow are unchanged.
2. **New domains get a resource-verb protocol** — `GenericBackend`
   (`get_resource`, `list_resources`, `create_resource`, `update_resource`,
   `execute_operation`, `get_lifecycle_states`) — implemented alongside
   `SupportBackend`, NOT replacing it. `EcommerceAdapter` implements
   GenericBackend by delegating to a wrapped SupportBackend, proving the
   mapping; new domains implement GenericBackend directly and never see
   order-words.
3. **Governance unchanged.** Policy engine gates by tool name; ToolSpecs
   (params, preconditions, facts, param_types) stay the authority; tools
   are still declared by CODE AUTHORS, only the binding MECHANISM becomes
   registration. Tenants still never invent bindings (ADR-003 holds).
4. **Lifecycle states remain data.** `get_lifecycle_states` enables
   backends that CAN describe their own state machine; preconditions keep
   coming from tools.yaml/specs. No state discovery is invented at runtime.

## Consequences
+ Adding a tool = register binding + ToolSpec (one place, no chain edit).
+ New domains implement resource verbs; no semantic noun-mapping forced.
+ Existing `SupportBackend` backends (MockERP, HttpERP, clinic) keep
  working unchanged — the default registrations target that surface.
- Two protocols now exist; GenericBackend is the forward path, SupportBackend
  stays supported (deprecation NOT scheduled — real deployments depend on it).
- Dynamic tool creation is explicitly out of scope; revisit only when
  self-serve tenant onboarding becomes a product requirement.

## Verification gate
Every change here ships behind the existing suite (byte-identical governed
behavior: same ToolResult values, same error strings, same idempotency
semantics) plus new tests proving: registration API works, default
registrations cover the classic surface, GenericBackend adapters route, and
policy gating still fires before any backend call.
