# OpenAPI Ingestion — the Adaptation Front Door

**Date:** 2026-09-13
**Status:** Approved design, pending implementation plan
**Sprint:** "Adaptation Front Door" — drop an OpenAPI spec, get a governed voice agent

## Summary

The platform's thesis is true drop-in adaptation: hand the running system a
Business Package (instructions + knowledge + APIs) and it becomes an
operational voice agent with **zero domain-specific changes to core Python**
(the sprint itself adds generic, domain-neutral platform code — the claim is
that onboarding the *next* business is data only). The ADR-005
proposal machinery (`proposals.py`: `ToolProposal`, `load_proposals_yaml`,
`compile_approved`) and the ADR-004 `GenericBackend` protocol already form
two-thirds of the pipeline. What is missing is the front door: nothing parses
a real OpenAPI 3.x document into the operations shape `draft_from_api_spec`
consumes, and no demo backend exists that can serve an arbitrary new domain
without a platform-side adapter.

This sprint builds that front door, deterministically, and proves it
end-to-end with a hotel tenant whose entire domain is bundle data.

## Goals

1. A deterministic OpenAPI 3.x → ToolProposal drafter (stdlib parser, no LLM
   in the trust chain; an optional prose-enrichment seam for later).
2. A discovery report that speaks the onboarding moment: operation counts by
   risk, drafted tool names, flagged ambiguities.
3. Emission of the ADR-005 approval artifact (`proposals.yaml`, all
   `status: proposed`) plus a draft `policies.yaml` starter — so onboarding
   output is fail-closed by construction (nothing executes until a human
   approves and commits).
4. A domain-neutral `FixtureGenericBackend` so any tenant is demoable from
   JSON data, selected like every other backend (env var, fail-closed).
5. Proof: a hotel tenant bundle + scripted e2e conversation through the
   governed orchestrator — `search_rooms` (read, no confirmation),
   `create_booking` (mutating → confirmation gate), with policy verdicts and
   DecisionLog records asserted — and **no edits to core modules for the
   hotel domain**.

## Non-goals (this sprint)

- Capability-graph workflow generation (hotel→clinic diagram); needs real
  drafted proposals to build against.
- Interactive approval TUI; committing the reviewed file remains the gate.
- The actual LLM prose enricher (the seam ships; the enricher is a follow-up).
- audit-report → machine-readable proposals loop.
- De-ecommerce cleanup of `DEFAULT_TOOL_SPECS` preconditions / `order_id`
  ladder hardcode / `DEMO_TENANT_ACTIONS` (separate sprint; does not block
  this one — hotel tools come from proposals, not the legacy specs).
- Real HTTP execution for drafted tools (FixtureGenericBackend is the demo
  connector; `HttpERP` remains the production path).

## Architecture

```
hotel-openapi.yaml
      │
      ▼
scripts/onboard.py  ──►  openapi_draft.parse_openapi()      (deterministic)
      │                       │
      │                       ▼
      │                 discovery_report()  ──► stdout ("I found 12
      │                                        operations: 8 read…")
      │                       │
      ▼                       ▼
data/tenants/hotel-demo/   write_proposals_yaml()  ──► proposals.yaml
  proposals.yaml   (owner reviews, edits prose,      (all status: proposed)
  policies.yaml     flips approved entries, commits)      │
  tools.yaml (universal only)                             │
  tenant.json / intents/ / knowledge/                     ▼
                                      runtime._bundle_proposals() ──►
                                      validate_proposal() ──►
                                      compile_approved() ──► ToolGateway
                                                                  │
        VOICEAGENT_FIXTURE_BACKEND=data/fixtures/hotel.json       │
                                      │                           ▼
                                      ▼                  GovernedToolRunner
                              FixtureGenericBackend       (Policy decides)
                                      │                           │
                                      └──────────► governed call ◄┘
```

Everything below the owner's commit already exists (ADR-005 state machine,
`GovernedToolRunner`, PolicyEngine). The sprint adds the top half and one
backend.

## Component design

### 1. `src/voiceagent/openapi_draft.py` (new, pure functions)

**`parse_openapi(doc: dict) -> list[dict]`** — lowers OpenAPI 3.x into the
pre-digested operations shape `proposals.draft_from_api_spec` consumes.
Requires `openapi: 3.*`; Swagger 2.0 is rejected with an explicit message.

Per operation under `paths` (methods get/put/post/patch/delete only):

- **tool name**: `operationId` when present (validated snake_case; non-
  conforming names are slugified); otherwise synthesized
  `{method}_{singular(resource)}` and listed in the report's
  "synthesized names — review" section. No new ToolProposal field; the
  report is the review surface.
- **risk class** (deterministic verb map, checked in this exact order):
  `DELETE` → `high` regardless of name; `GET` → `read`; a `POST`/`PUT`/
  `PATCH` whose operationId or path contains refund/payment/charge/payout
  → `high` (money tokens are checked BEFORE the read prefixes, so a
  `POST getRefund` can never draft as a read); otherwise a `POST`/`PUT`/
  `PATCH` operationId starting with a read prefix (`search|list|find|
  fetch|get|lookup|query`) drafts as `read` with `side_effects=False` —
  POST-search endpoints (`POST /rooms/search`) are reads, and
  `validate_proposal` forbids `side_effects=true` with `risk_class=read`;
  anything else → `mutating`.
- **params**: path parameters + required query parameters + required
  `requestBody` (`application/json`) schema properties, in that order.
  Optional parameters are dropped in v1 and noted in the report
  (`ToolProposal.params` is an all-required surface;
  `gateway_tool_meta` marks every param required). `$ref`s resolve within
  the document only.
- **param_types**: OpenAPI `type` (`string|number|integer|boolean`) mapped
  1:1 onto `PARAM_TYPES`; `format` narrows per the obvious mapping
  (`int32/int64` → `integer`, `float/double` → `number`).
- **resource_type / id_param**: first path segment, naively singularized
  (`/reservations` → `reservation`; documented as naive, owner-reviewable).
  A trailing `{param}` path segment becomes `id_param`.
- **operation (executor routing)**: `GET /res/{id}` → `__fetch__` with
  `resource_type` + `id_param`; `GET /res` → `__list__` with the single
  required filter param as `filter_param` (a collection GET without a
  required filter is *not draftable* — reported, not guessed);
  everything else → `execute_operation(operationId)`.
- **preconditions / facts**: none auto-generated (lifecycle semantics are
  the owner's knowledge). The report suggests where they belong.
- Every draft: `provenance="ai"`, `status="proposed"`, `side_effects=True`
  iff method != GET, `description` from `summary`/`description` (falls back
  to `{method} {path}`).

**`discovery_report(ops) -> str`** — the onboarding moment as text:
total/read/mutating/high counts, per-resource grouping of drafted tools,
synthesized-name flags, dropped-optional-param notes, undraftable-operation
notes.

**`write_proposals_yaml(ops, path)`** — emits the approval artifact with
every entry `status: proposed`. Round-trips through
`proposals.load_proposals_yaml` (the loader's unknown-key rejection is the
schema contract; the writer emits only known keys).

**`enrich(ops, enricher=None)`** — the prose seam. The enricher callable
receives drafted ops and may rewrite **only** `description` fields and may
draft intent-exemplar text; it never touches name/params/operation/risk/
status. Default `None` = identity. The LLM enricher is a future
implementation behind this signature.

### 2. `ToolProposal.param_types` (small extension, backward-compatible)

- `ToolProposal` gains `param_types: dict[str, str] = field(default_factory=dict)`.
- `load_proposals_yaml`: `param_types` added to the allowed-keys set,
  validated with `tools.parse_param_types`.
- `compile_approved`: threads `param_types` into the lowered `ToolSpec`.
- `gateway_tool_meta`: emits typed `properties` (param absent from the map
  stays `string`).
- `validate_proposal`: a `param_types` key naming a param not in `params`
  is a validation error (same discipline as `id_param`).

### 3. `src/voiceagent/fixture_backend.py` (new, core, domain-neutral)

`FixtureGenericBackend` implements the `GenericBackend` protocol over a JSON
file — platform code with zero domain nouns; tenant data supplies the
domain:

```json
{
  "resources": {
    "room":     {"RM-101": {"room_id": "RM-101", "type": "deluxe", ...}},
    "booking":  {"B-1001": {"booking_id": "B-1001", "status": "BOOKED", ...}}
  },
  "operations": {
    "search_rooms":     {"response": {"results": ["RM-101", "RM-102"]}},
    "create_booking":   {"id_prefix": "B-", "resource": "booking",
                         "response": {"status": "BOOKED"}}
  },
  "lifecycles": {"booking": ["HELD", "BOOKED", "CANCELLED"]}
}
```

Semantics:

- `get_resource(rt, id)` → deep copy from `resources` or `None`.
- `list_resources(rt, filters)` → exact-match filtering over the resource
  map (empty filter = error, mirroring the EcommerceAdapter's refusal of
  unbounded listing).
- `create_resource(rt, data)` → mints an id from `id_prefix` + counter,
  stores, returns the record. `update_resource` merges and returns.
- `execute_operation(name, params)` → returns a deep copy of the canned
  `response`; an optional `create` block (`{"resource", "id_prefix",
  "id_key"}`) first creates the record from `params` and merges it into
  the response; an optional `patch` block (`{"resource", "id_from",
  "set"}`) applies `set` to the record named by `params[id_from]` and
  merges the updated record into the response (unknown id →
  `GenericBackendError`) — deterministic demo state changes, no scripting
  language, and a cancel that actually mutates rather than lying with a
  canned response.
- Unknown resource types / operations raise `GenericBackendError` (fail-
  closed, TimeoutError-compatible — the governed timeout path handles it,
  exactly like every other adapter). `get_lifecycle_states` reads the
  `lifecycles` map (absent → `NotImplementedError`, the historical model).

**Runtime selection** (`runtime._erp_from_env` extension): explicit `erp`
arg wins → `VOICEAGENT_ERP_URL` (HttpERP) → `VOICEAGENT_FIXTURE_BACKEND`
(FixtureGenericBackend) → MockERP with the existing loud warning. An
explicitly configured fixture path that is missing or invalid raises — an
operator-named backend must never silently swap to a mock.

### 4. `scripts/onboard.py` (minimal CLI, no interactivity)

```
python scripts/onboard.py --spec hotel-openapi.yaml \
    --out data/tenants/hotel-demo --name "Grand Hotel"
```

- Parses the spec, prints `discovery_report` to stdout.
- Writes `proposals.yaml` (all `proposed`), a **draft** `policies.yaml`
  starter (reads → `ALLOW`; mutating → `ALLOW`, governed by ToolSpec
  side-effect confirmation; high-risk → `ESCALATE` — owner relaxes
  deliberately), `tenant.json` scaffold (name from `--name` or slugified
  `info.title`, empty persona fields for the owner), empty `intents/` and
  `knowledge/` dirs, and a README listing the review steps.
- Never touches runtime state; output is data. Because every proposal is
  `proposed`, pointing `VOICEAGENT_TENANT` at the scaffold immediately is
  safe: only universal tools (escalate/end_call/feedback) are live.

### 5. Hotel proof: `data/tenants/hotel-demo/` + `data/fixtures/hotel.json`

The committed bundle represents the **post-review** state (owner approved
and committed):

- `proposals.yaml`: `search_rooms`, `get_booking` (read), `create_booking`,
  `modify_booking` (mutating), `cancel_booking` (high) — all `approved`
  with param_types, `create_booking` carrying the owner-written
  confirmation-relevant description; `cancel_booking` demonstrates the high
  class with a precondition stub the owner would fill from the hotel's
  cancellation policy.
- `tools.yaml`: declares **only the universal tools**
  (`escalate_to_human`, `end_call`, `record_feedback`). This is what keeps
  the brain's proposal surface free of e-commerce legacy:
  `_bundle_gateway_tools` returns exactly the declared tools, and the
  domain surface arrives solely through approved proposals folded in at
  `runtime.py:556–559`.
- `policies.yaml`: hotel actions with least-privilege verdicts;
  `tenant.json` hotel persona; small `intents/` exemplar files;
  `knowledge/` with cancellation/check-in policy notes.

**E2E test** (`tests/test_hotel_adaptation.py`): scripted multi-turn
conversation through `build_orchestrator(tenant="hotel-demo",
erp=FixtureGenericBackend(...), env=<stub frontier>)`:

1. guest asks for a room → `search_rooms` executes, no confirmation
   required;
2. guest books → `create_booking` **requires confirmation** (side_effects)
   before the gateway mutates the fixture; booking id lands in
   `resources.booking`;
3. guest asks to cancel → high-risk path observable (policy verdict
   recorded in DecisionLog);
4. governance asserts: every verdict DecisionLog-recorded; no tool outside
   `search_rooms/get_booking/create_booking/modify_booking/cancel_booking`
   + universal three was proposed or registered; a `proposed`-status
   variant of the bundle registers nothing (the human gate holds);
5. neutrality assertion (behavioral, not grep): the deployed brain surface
   contains no e-commerce legacy tool (no `fetch_order_status`,
   `cancel_order`, …) — the hotel domain exists only as bundle data.

## Error handling

- Parser: unknown construct types (e.g. `oneOf` at top level of a required
  schema) → parameter dropped + report note, never a guess; Swagger 2.0 →
  hard error naming the version.
- `write_proposals_yaml` → `load_proposals_yaml` round-trip is the schema
  gate; the onboard CLI runs `validate_proposal` on its own output before
  writing and refuses to emit invalid drafts.
- FixtureGenericBackend: every miss raises `GenericBackendError` (governed
  timeout/degradation path), never a fabricated ok.
- Runtime: invalid fixture file → raise at assembly (fail-closed), same
  discipline as a typo'd `VOICEAGENT_TENANT`.

## Testing

- Unit (`tests/test_openapi_draft.py`): clean hotel spec (full pipeline
  shape assertions) + messy spec (missing operationIds → synthesized names
  flagged; `$ref` chains; optional params dropped with notes; collection
  GET without filter → undraftable; DELETE → high; refund-named POST →
  high; Swagger 2.0 rejected). Round-trip write→load equality.
- Unit (`tests/test_proposals.py` extension): `param_types` validation,
  threading into `ToolSpec`, typed `gateway_tool_meta`.
- Unit (`tests/test_fixture_backend.py`): protocol conformance, fail-closed
  errors, create/update/execute semantics, lifecycle introspection.
- Integration: the hotel e2e above; all 737 existing tests stay green
  (ToolProposal change is additive; default-tenant behavior byte-identical).
- `scripts/validate_tenant.py`: verify the existing proposals.yaml gate,
  and extend it for the new `param_types` field — the CI gate stays the
  single source of "is this bundle well-formed".

## ADR alignment

- **ADR-001** (declared vs learned): the drafter is deterministic
  mechanics; semantic prose stays owner-reviewed; nothing learned mutates
  the surface.
- **ADR-003/004** (bindings are code, proposals are declarations):
  `FixtureGenericBackend` is platform code with no domain nouns; drafted
  tools are declarations lowered by existing `compile_approved`.
- **ADR-005** (human gate): onboard output is all-`proposed`; committing
  the reviewed file is the approval; the agent can never approve its own
  surface.

## Follow-up sprints (explicitly out of scope here)

1. LLM prose enricher behind `enrich()`.
2. Capability-graph workflow derivation (slot-gathering flows from request
   schemas, the hotel→clinic capability diagram).
3. audit-report → machine-readable proposals (the learning loop re-entering
   the same approval artifact).
4. De-ecommerce-ization of core defaults (`DEFAULT_TOOL_SPECS` preconditions,
   `order_id` ladder, `DEMO_TENANT_ACTIONS`).
5. Interactive approval TUI + `voiceagent onboard` as a first-class command.
