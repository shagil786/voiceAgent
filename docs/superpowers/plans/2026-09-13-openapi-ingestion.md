# OpenAPI Ingestion — Adaptation Front Door Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop an OpenAPI 3.x spec into the platform and get a governed voice agent — deterministic parser → ToolProposal drafts → human approval → compiled tools executing over a data-only fixture backend, proven end-to-end with a hotel tenant that adds zero domain code.

**Architecture:** A pure `openapi_draft.py` module lowers OpenAPI into the operations shape `proposals.draft_from_api_spec` already consumes (ADR-005 state machine unchanged). A domain-neutral `FixtureGenericBackend` implements the ADR-004 `GenericBackend` protocol from a JSON file. `scripts/onboard.py` emits the fail-closed approval artifacts. The hotel tenant (`data/tenants/hotel-demo/`) is pure data.

**Tech Stack:** Python 3 stdlib + PyYAML (both already used); pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-13-openapi-ingestion-design.md` — the plan argues from it; read both.

## Global Constraints

- Test interpreter: `.venv/bin/python -m pytest` (never bare `python`).
- Proposals are DECLARATION ONLY; nothing executes below `status: approved` (ADR-005). The committed `proposals.yaml` file IS the human approval.
- `ToolGateway`/`GenericBackend` failures raise `TimeoutError`-compatible errors (`GenericBackendError`); never fabricate an ok result.
- No domain nouns (`hotel`, `room`, `booking`) in `src/voiceagent/` core modules EXCEPT where this plan places them: none — hotel content lives only in `data/` and `tests/`.
- Run the full suite before every final commit of a task: `.venv/bin/python -m pytest -q` (737+ tests must stay green).
- Commit after every task. Messages follow repo style (`feat:`, `test:`, `data:` prefixes).

---

### Task 1: `ToolProposal.param_types` extension

**Files:**
- Modify: `src/voiceagent/proposals.py`
- Test: `tests/test_proposals.py`

**Interfaces:**
- Consumes: `tools.parse_param_types(value, where)` (exists, `tools.py:296`), `ToolSpec.param_types` (exists, `tools.py:222`).
- Produces: `ToolProposal.param_types: dict[str, str]` (new field, default `{}`); `gateway_tool_meta(prop)` now emits typed `properties`; `compile_approved` threads `param_types` into the lowered `ToolSpec`; `draft_from_api_spec` reads an optional `"param_types"` key from its operations dicts (later tasks rely on all four).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_proposals.py` (reuse the file's existing imports of `ToolProposal`, `validate_proposal`, `gateway_tool_meta`, `compile_approved`, `RISK_MUTATING`, `RISK_READ`, `PROPOSED`, `APPROVED`; add `import pytest` and `from voiceagent.tools import ToolGateway` if not already imported):

```python
# --- param_types: OpenAPI types ride the declaration (front-door sprint) -----

def _typed_prop(**over):
    kw = dict(name="create_booking", description="book a room",
              params=("guest_name", "nights"),
              action="create_booking", operation="createBooking",
              param_types={"nights": "integer"},
              side_effects=True, risk_class=RISK_MUTATING, provenance="ai")
    kw.update(over)
    return ToolProposal(**kw)


def test_param_types_roundtrip_to_gateway_tool_meta():
    meta = gateway_tool_meta(_typed_prop())
    assert meta["parameters"]["properties"]["nights"] == {"type": "integer"}
    assert meta["parameters"]["properties"]["guest_name"] == {"type": "string"}


def test_param_types_absent_defaults_to_strings():
    prop = ToolProposal(name="x", description="d", params=("a",),
                        action="x", operation="x")
    meta = gateway_tool_meta(prop)
    assert meta["parameters"]["properties"]["a"] == {"type": "string"}


def test_param_types_unknown_param_is_a_validation_error():
    prop = _typed_prop(param_types={"ghost": "integer"})
    assert any("ghost" in e and "not in params" in e
               for e in validate_proposal(prop))


def test_param_types_bad_type_value_is_a_validation_error():
    prop = _typed_prop(param_types={"nights": "float"})
    assert validate_proposal(prop)  # parse_param_types rejects "float"


def test_compile_approved_threads_param_types_into_toolspec():
    gw = ToolGateway(erp=object(), specs={})
    registered = compile_approved(gw, object(), [_typed_prop(status=APPROVED)])
    assert registered == ["create_booking"]
    spec = gw.specs["create_booking"]
    assert spec.param_types == {"nights": "integer"}


def test_load_proposals_yaml_accepts_param_types(tmp_path):
    (tmp_path / "proposals.yaml").write_text(
        "proposals:\n"
        "  - name: book_room\n"
        "    description: book\n"
        "    params: [nights]\n"
        "    action: book_room\n"
        "    operation: createBooking\n"
        "    param_types: {nights: integer}\n"
        "    status: proposed\n", encoding="utf-8")
    props = load_proposals_yaml(tmp_path / "proposals.yaml")
    assert props[0].param_types == {"nights": "integer"}


def test_draft_from_api_spec_carries_param_types():
    ops = [{"operation": "createBooking", "tool_name": "create_booking",
            "description": "d", "params": ["nights"],
            "param_types": {"nights": "integer"}, "side_effects": True,
            "risk_class": RISK_MUTATING}]
    (prop,) = draft_from_api_spec(ops)
    assert prop.param_types == {"nights": "integer"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_proposals.py -q -k param_types`
Expected: FAIL — `ToolProposal` has no `param_types` field (TypeError: unexpected keyword argument).

- [ ] **Step 3: Implement in `src/voiceagent/proposals.py`**

1. In the `ToolProposal` dataclass (after the `operation_params` field), add:

```python
    param_types: dict[str, str] = field(default_factory=dict)
```

2. In `validate_proposal`, add (after the `id_param` check):

```python
    if prop.param_types:
        bad = sorted(set(prop.param_types) - set(prop.params))
        if bad:
            errs.append(f"{prop.name}: param_types name(s) not in params: "
                        f"{bad}")
        try:
            from voiceagent.tools import parse_param_types
            parse_param_types(dict(prop.param_types), where=prop.name)
        except ValueError as exc:
            errs.append(str(exc))
```

3. In `compile_approved`, extend the `ToolSpec(...)` construction with:

```python
            param_types=dict(prop.param_types),
```

4. In `gateway_tool_meta`, replace the string-hardcoded loop:

```python
    properties = {}
    for pname in prop.params:
        properties[pname] = {"type": prop.param_types.get(pname, "string")}
```

5. In `load_proposals_yaml`: add `"param_types"` to the `allowed` set, and add `param_types=dict(e.get("param_types", {})),` to the `ToolProposal(...)` construction.

6. In `draft_from_api_spec`, add `param_types=dict(op.get("param_types", {})),` to the `ToolProposal(...)` construction.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_proposals.py tests/test_proposals_deploy.py -q`
Expected: PASS (new tests + all existing proposal tests).

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest -q`
Expected: all green (additive change; default-tenant behavior byte-identical).

```bash
git add src/voiceagent/proposals.py tests/test_proposals.py
git commit -m "feat(proposals): param_types rides the declaration — typed tools from OpenAPI"
```

---

### Task 2: `openapi_draft.parse_openapi` — the deterministic drafter

**Files:**
- Create: `src/voiceagent/openapi_draft.py`
- Create: `tests/test_openapi_draft.py`

**Interfaces:**
- Consumes: nothing from earlier tasks at runtime (pure stdlib module).
- Produces (used by Tasks 3, 6, 7):
  - `DraftResult` frozen dataclass: `operations: tuple[dict, ...]`, `notes: tuple[str, ...]`
  - `parse_openapi(doc: dict) -> DraftResult` — raises `ValueError` on non-3.x
  - Each operations dict has exactly the keys `draft_from_api_spec` reads: `operation, tool_name, description, params, side_effects, risk_class, param_types, action` plus optional `resource_type, id_param, filter_param`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_openapi_draft.py`:

```python
# tests/test_openapi_draft.py — the deterministic OpenAPI drafter.
"""Pure mechanics: verb→risk, operationId→name, schemas→params. The trust
chain has no LLM; every draft is status: proposed for human review."""
import pytest

from voiceagent.openapi_draft import DraftResult, parse_openapi

HOTEL_SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Grand Hotel PMS", "version": "1.0.0"},
    "paths": {
        "/rooms/search": {
            "post": {
                "operationId": "searchRooms",
                "summary": "Search available rooms for a stay window.",
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "check_in": {"type": "string"},
                            "check_out": {"type": "string"},
                            "guests": {"type": "integer"},
                        },
                        "required": ["check_in", "check_out"],
                    }}}}},
        },
        "/rooms/{room_id}": {
            "get": {"operationId": "getRoom",
                    "summary": "Fetch one room.",
                    "parameters": [{"name": "room_id", "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"}}]},
        },
        "/bookings": {
            "post": {
                "operationId": "createBooking",
                "summary": "Create a booking.",
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "guest_name": {"type": "string"},
                            "room_id": {"type": "string"},
                            "check_in": {"type": "string"},
                            "guests": {"type": "integer"},
                        },
                        "required": ["guest_name", "room_id", "check_in"],
                    }}}}},
        },
        "/bookings/{booking_id}": {
            "get": {"operationId": "getBooking",
                    "summary": "Fetch a booking.",
                    "parameters": [{"name": "booking_id", "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"}}]},
            "patch": {"operationId": "modifyBooking",
                      "summary": "Move a booking's check-out date.",
                      "parameters": [{"name": "booking_id", "in": "path",
                                      "required": True,
                                      "schema": {"type": "string"}}],
                      "requestBody": {"required": True, "content": {
                          "application/json": {"schema": {
                              "type": "object",
                              "properties": {"new_check_out":
                                             {"type": "string"}},
                              "required": ["new_check_out"],
                          }}}}},
            "delete": {"operationId": "cancelBooking",
                       "summary": "Cancel a booking.",
                       "parameters": [{"name": "booking_id", "in": "path",
                                       "required": True,
                                       "schema": {"type": "string"}}]},
        },
    },
}


def by_name(result):
    return {op["tool_name"]: op for op in result.operations}


def test_hotel_happy_path_full_shape():
    ops = by_name(parse_openapi(HOTEL_SPEC))
    assert set(ops) == {"search_rooms", "get_room", "create_booking",
                        "get_booking", "modify_booking", "cancel_booking"}


def test_post_search_drafts_as_read_with_no_side_effects():
    op = by_name(parse_openapi(HOTEL_SPEC))["search_rooms"]
    assert op["risk_class"] == "read"
    assert op["side_effects"] is False
    assert op["operation"] == "searchRooms"
    assert op["params"] == ["check_in", "check_out"]  # optional guests dropped
    assert op["description"] == "Search available rooms for a stay window."


def test_optional_request_body_property_dropped_and_noted():
    result = parse_openapi(HOTEL_SPEC)
    assert by_name(result)["create_booking"]["params"] == \
        ["guest_name", "room_id", "check_in"]
    assert any("guests" in n and "optional" in n for n in result.notes)


def test_fetch_routing_get_with_trailing_id():
    op = by_name(parse_openapi(HOTEL_SPEC))["get_room"]
    assert op["operation"] == "__fetch__"
    assert op["resource_type"] == "room"
    assert op["id_param"] == "room_id"
    assert op["risk_class"] == "read"


def test_resource_singularized_from_first_segment():
    ops = by_name(parse_openapi(HOTEL_SPEC))
    assert ops["get_booking"]["resource_type"] == "booking"
    assert ops["get_booking"]["operation"] == "__fetch__"


def test_patch_is_mutating_delete_is_high():
    ops = by_name(parse_openapi(HOTEL_SPEC))
    assert ops["modify_booking"]["risk_class"] == "mutating"
    assert ops["modify_booking"]["side_effects"] is True
    assert ops["modify_booking"]["params"] == ["booking_id", "new_check_out"]
    assert ops["cancel_booking"]["risk_class"] == "high"
    assert ops["cancel_booking"]["side_effects"] is True


def test_param_types_extracted_from_schemas():
    ops = by_name(parse_openapi(HOTEL_SPEC))
    assert ops["create_booking"]["param_types"] == \
        {"guest_name": "string", "room_id": "string", "check_in": "string"}
    assert ops["search_rooms"]["param_types"] == \
        {"check_in": "string", "check_out": "string"}


def test_swagger_2_rejected_with_clear_error():
    with pytest.raises(ValueError, match="OpenAPI 3"):
        parse_openapi({"swagger": "2.0", "paths": {}})


def test_missing_operation_id_synthesized_and_noted():
    spec = {"openapi": "3.1.0",
            "paths": {"/widgets": {"post": {
                "summary": "Make a widget.",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "properties": {"size": {"type": "string"}},
                    "required": ["size"]}}}}}}}}
    result = parse_openapi(spec)
    (op,) = result.operations
    assert op["tool_name"] == "post_widget"
    assert op["operation"] == "post_widget"
    assert any("synthesized" in n for n in result.notes)


def test_collection_get_without_single_filter_is_undraftable():
    spec = {"openapi": "3.1.0",
            "paths": {"/widgets": {"get": {"operationId": "listWidgets"}}}}
    result = parse_openapi(spec)
    assert result.operations == ()
    assert any("exactly one required filter" in n for n in result.notes)


def test_collection_get_with_one_required_filter_becomes_list_lookup():
    spec = {"openapi": "3.1.0",
            "paths": {"/widgets": {"get": {
                "operationId": "findWidgets",
                "parameters": [{"name": "owner_phone", "in": "query",
                                "required": True,
                                "schema": {"type": "string"}}]}}}}
    result = parse_openapi(spec)
    (op,) = result.operations
    assert op["operation"] == "__list__"
    assert op["filter_param"] == "owner_phone"
    assert op["resource_type"] == "widget"


def test_refund_named_post_escalates_to_high():
    spec = {"openapi": "3.1.0",
            "paths": {"/payments/refund": {"post": {
                "operationId": "processRefund",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "properties": {"payment_id": {"type": "string"}},
                    "required": ["payment_id"]}}}}}}}}
    (op,) = parse_openapi(spec).operations
    assert op["risk_class"] == "high"


def test_refs_resolved_and_unscalar_required_property_dropped_with_note():
    spec = {"openapi": "3.1.0",
            "components": {"schemas": {"Guest": {
                "type": "object",
                "properties": {"guest_name": {"type": "string"}},
                "required": ["guest_name"]}}},
            "paths": {"/bookings": {"post": {
                "operationId": "createBooking",
                "requestBody": {"content": {"application/json": {"schema": {
                    "$ref": "#/components/schemas/Guest"}}}}}}}}
    result = parse_openapi(spec)
    (op,) = result.operations
    assert op["params"] == ["guest_name"]
    assert op["param_types"] == {"guest_name": "string"}


def test_post_without_any_params_is_skipped_with_note():
    spec = {"openapi": "3.1.0",
            "paths": {"/ping": {"post": {"operationId": "ping"}}}}
    result = parse_openapi(spec)
    assert result.operations == ()
    assert any("no required parameters" in n for n in result.notes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_openapi_draft.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'voiceagent.openapi_draft'`.

- [ ] **Step 3: Create `src/voiceagent/openapi_draft.py`**

```python
# src/voiceagent/openapi_draft.py — OpenAPI 3.x → ToolProposal drafts.
"""The adaptation front door: a DETERMINISTIC OpenAPI 3.x reader that
lowers a customer's API spec into ADR-005 ToolProposal drafts.

Trust chain (ADR-001/005): pure mechanics — verb→risk, operationId→name,
schemas→params. No LLM is in the chain; the only semantic freedoms are
documented, deterministic, and every draft lands in proposals.yaml as
status: proposed for HUMAN approval. The enrich() seam lets a future LLM
rewrite DESCRIPTIONS — never names, params, operations, or risk.

Drafting rules (pinned in docs/superpowers/specs/
2026-09-13-openapi-ingestion-design.md):
  - risk: DELETE -> high (regardless of name); GET -> read;
    operationId starting with a read prefix (search/list/find/fetch/get/
    lookup/query) -> read with side_effects=False (POST-search endpoints
    are reads); operationId or path containing refund/payment/charge/
    payout -> high; otherwise mutating.
  - params: path params + REQUIRED query params + REQUIRED requestBody
    (application/json) properties. Optional params are dropped and noted.
  - routing: GET /res/{id} -> __fetch__; GET /res with exactly one
    required query param -> __list__ (filter_param); everything else ->
    execute_operation(operationId). A collection GET without exactly one
    required query param is NOT draftable — noted, never guessed.
  - preconditions/facts: never auto-generated (lifecycle semantics are
    the owner's knowledge; the discovery report points at the review).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

_HTTP_METHODS = ("get", "put", "post", "patch", "delete")
_READ_PREFIXES = ("search", "list", "find", "fetch", "get", "lookup",
                  "query")
_HIGH_TOKENS = ("refund", "payment", "charge", "payout")
_SCALAR_TYPES = ("string", "number", "integer", "boolean")
_FORMAT_MAP = {"int32": "integer", "int64": "integer",
               "float": "number", "double": "number"}


@dataclass(frozen=True)
class DraftResult:
    """Operations drafts + human-review notes from one parse."""

    operations: tuple[dict, ...]
    notes: tuple[str, ...]


def _snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    out = re.sub(r"[^a-zA-Z0-9]+", "_", s2).lower().strip("_")
    return out or "tool"


def _singularize(segment: str) -> str:
    return segment[:-1] if len(segment) > 1 and segment.endswith("s") \
        else segment


def _resolve_ref(doc: dict, node: Any, depth: int = 0) -> Any:
    """Resolve local '#/...' $refs (chained, depth-capped). External refs
    are left as-is and fail the scalar checks downstream."""
    if depth > 10 or not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return node
    cur: Any = doc
    for part in ref[2:].split("/"):
        cur = cur.get(part.replace("~1", "/").replace("~0", "~")) \
            if isinstance(cur, dict) else None
    if cur is None:
        return node
    return _resolve_ref(doc, cur, depth + 1)


def _openapi_type(schema: dict) -> str:
    t = schema.get("type", "string")
    if t in ("integer", "number") and schema.get("format") in _FORMAT_MAP:
        return _FORMAT_MAP[schema["format"]]
    return t if t in _SCALAR_TYPES else "string"


def _risk_for(method: str, op_id: str, path: str) -> str:
    if method == "delete":
        return "high"
    if method == "get":
        return "read"
    if op_id.lower().startswith(_READ_PREFIXES):
        return "read"
    low = f"{op_id} {path}".lower()
    if any(tok in low for tok in _HIGH_TOKENS):
        return "high"
    return "mutating"


def parse_openapi(doc: dict) -> DraftResult:
    """Lower an OpenAPI 3.x document into draft operations. Raises
    ValueError on anything that is not OpenAPI 3.x."""
    version = str(doc.get("openapi", ""))
    if not version.startswith("3."):
        raise ValueError(
            f"unsupported spec: 'openapi: {version or 'missing'}' — this "
            "drafter reads OpenAPI 3.x; Swagger 2.0 is not accepted")
    ops: list[dict] = []
    notes: list[str] = []
    for path, item in (doc.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        segments = [s for s in str(path).strip("/").split("/") if s]
        plain = [s for s in segments
                 if not (s.startswith("{") and s.endswith("}"))]
        resource = _singularize(plain[0]) if plain else "resource"
        trailing_id = None
        if segments and segments[-1].startswith("{") \
                and segments[-1].endswith("}"):
            trailing_id = segments[-1][1:-1]
        shared = [p for p in item.get("parameters", [])
                  if isinstance(p, dict)]
        for method in _HTTP_METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            raw_id = op.get("operationId")
            op_id = str(raw_id) if raw_id else f"{method}_{resource}"
            name = _snake(op_id)
            if not raw_id:
                notes.append(
                    f"{method.upper()} {path}: no operationId — tool name "
                    f"synthesized as {name!r}; review the name")
            risk = _risk_for(method, op_id, str(path))
            params: list[str] = []
            types: dict[str, str] = {}
            optionals: list[str] = []
            for p in shared + [p for p in op.get("parameters", [])
                               if isinstance(p, dict)]:
                pname = p.get("name")
                if not pname or pname in params:
                    continue
                if p.get("in") == "path" or (p.get("in") == "query"
                                             and p.get("required")):
                    params.append(pname)
                    types[pname] = _openapi_type(p.get("schema") or {})
                elif p.get("in") == "query":
                    optionals.append(pname)
            body = _resolve_ref(doc, op.get("requestBody") or {})
            content = (body.get("content") or {}) \
                if isinstance(body, dict) else {}
            media = content.get("application/json") or {}
            schema = _resolve_ref(doc, media.get("schema"))
            if isinstance(schema, dict):
                props = schema.get("properties") or {}
                required = schema.get("required") or []
                for req in required:
                    sub = _resolve_ref(doc, props.get(req))
                    if isinstance(sub, dict) \
                            and sub.get("type", "string") in _SCALAR_TYPES:
                        if req not in params:
                            params.append(req)
                            types[req] = _openapi_type(sub)
                    else:
                        notes.append(
                            f"{method.upper()} {path}: required body "
                            f"property {req!r} has no scalar schema — "
                            "dropped")
                for prop in props:
                    if prop not in required and prop not in optionals:
                        optionals.append(prop)
            if optionals:
                notes.append(
                    f"{name}: dropped optional parameter(s) {optionals} — "
                    "add them back manually if needed")
            if method == "get":
                if trailing_id:
                    routing = {"operation": "__fetch__",
                               "resource_type": resource,
                               "id_param": trailing_id}
                elif len(params) == 1:
                    routing = {"operation": "__list__",
                               "resource_type": resource,
                               "filter_param": params[0]}
                else:
                    notes.append(
                        f"skipped GET {path}: a collection lookup needs "
                        "exactly one required filter parameter — author "
                        "this tool manually")
                    continue
            else:
                if not params:
                    notes.append(
                        f"skipped {method.upper()} {path}: no required "
                        "parameters — nothing to govern; author manually")
                    continue
                routing = {"operation": op_id}
            ops.append({
                "operation": routing.pop("operation"),
                "tool_name": name,
                "description": op.get("summary") or op.get("description")
                               or f"{method.upper()} {path}",
                "params": params,
                "side_effects": risk != "read",
                "risk_class": risk,
                "param_types": types,
                "action": name,
                **routing,
            })
    return DraftResult(operations=tuple(ops), notes=tuple(notes))
```

Note: `Callable` is imported here for Task 3's `enrich` seam; if the linter objects at this task, remove it and re-add in Task 3.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_openapi_draft.py -q`
Expected: PASS (all 13).

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/openapi_draft.py tests/test_openapi_draft.py
git commit -m "feat(adaptation): deterministic OpenAPI 3.x drafter (parse_openapi)"
```

---

### Task 3: Discovery report, proposals.yaml writer, enrich seam

**Files:**
- Modify: `src/voiceagent/openapi_draft.py` (append)
- Test: `tests/test_openapi_draft.py` (append)

**Interfaces:**
- Consumes: `DraftResult`/`parse_openapi` (Task 2); `proposals.draft_from_api_spec`, `proposals.validate_proposal`, `proposals.PROPOSED` (exists).
- Produces (used by Task 6): `discovery_report(result: DraftResult, title: str = "") -> str`; `write_proposals_yaml(result: DraftResult, path) -> list[str]`; `enrich(result: DraftResult, enricher: Enricher | None = None) -> DraftResult` where `Enricher = Callable[[list[dict]], list[dict]]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_openapi_draft.py` (add imports `json`, `yaml`, `Path`, and `from voiceagent.proposals import load_proposals_yaml`):

```python
# --- report, artifact writer, enrich seam ------------------------------------

from voiceagent.openapi_draft import discovery_report, enrich, \
    write_proposals_yaml


def test_discovery_report_speaks_the_onboarding_moment():
    report = discovery_report(parse_openapi(HOTEL_SPEC),
                              title="Grand Hotel PMS")
    assert "Discovered API: Grand Hotel PMS" in report
    assert "6 operations: 3 read, 2 mutating, 1 high-risk" in report
    assert "[HIGH RISK] cancel_booking(booking_id)" in report
    assert "[read] search_rooms(check_in, check_out)" in report
    assert "status: proposed" in report


def test_write_then_load_roundtrip_is_the_approval_artifact(tmp_path):
    result = parse_openapi(HOTEL_SPEC)
    out = tmp_path / "proposals.yaml"
    names = write_proposals_yaml(result, out)
    assert set(names) == {"search_rooms", "get_room", "create_booking",
                          "get_booking", "modify_booking", "cancel_booking"}
    props = load_proposals_yaml(out)
    assert all(p.status == "proposed" and p.provenance == "ai"
               for p in props)
    by = {p.name: p for p in props}
    assert by["create_booking"].param_types == \
        {"guest_name": "string", "room_id": "string", "check_in": "string"}
    assert by["get_room"].operation == "__fetch__"
    assert by["get_room"].resource_type == "room"
    assert by["cancel_booking"].risk_class == "high"


def test_writer_refuses_invalid_drafts(tmp_path):
    bad = DraftResult(
        operations=({"operation": "x", "tool_name": "Bad Name!",
                     "description": "d", "params": [], "side_effects": False,
                     "risk_class": "read", "param_types": {},
                     "action": "x"},),
        notes=())
    with pytest.raises(ValueError, match="refusing to write invalid draft"):
        write_proposals_yaml(bad, tmp_path / "proposals.yaml")


def test_enrich_default_is_identity():
    result = parse_openapi(HOTEL_SPEC)
    assert enrich(result) is result


def test_enrich_may_only_touch_descriptions():
    result = parse_openapi(HOTEL_SPEC)

    def _enricher(ops):
        out = [{**op, "description": f"OWNER COPY: {op['description']}"}
               for op in ops]
        return out

    enriched = enrich(result, _enricher)
    assert enriched.operations[0]["description"].startswith("OWNER COPY")
    assert enriched.operations[0]["params"] == \
        result.operations[0]["params"]
    assert enriched.notes == result.notes


def test_enrich_changing_mechanics_is_refused():
    result = parse_openapi(HOTEL_SPEC)

    def _sneaky(ops):
        out = [dict(op) for op in ops]
        out[0]["risk_class"] = "read"  # NOT allowed — risk is deterministic
        return out

    with pytest.raises(ValueError, match="may only rewrite 'description'"):
        enrich(result, _sneaky)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_openapi_draft.py -q -k "report or write or enrich"`
Expected: FAIL — ImportError: cannot import name 'discovery_report'.

- [ ] **Step 3: Append the implementation to `src/voiceagent/openapi_draft.py`**

```python
# --- report, artifact writer, prose seam -------------------------------------

Enricher = Callable[[list[dict]], list[dict]]


def discovery_report(result: DraftResult, title: str = "") -> str:
    """The onboarding moment: what was found, how it was classified, what
    needs review. Output is text for the OPERATOR — never parsed by code."""
    counts = {"read": 0, "mutating": 0, "high": 0}
    for op in result.operations:
        counts[op["risk_class"]] += 1
    mark = {"read": "[read]", "mutating": "[mutating]",
            "high": "[HIGH RISK]"}
    lines = []
    if title:
        lines.append(f"Discovered API: {title}")
    lines.append(
        f"I found {len(result.operations)} operations: "
        f"{counts['read']} read, {counts['mutating']} mutating, "
        f"{counts['high']} high-risk.")
    lines.append("Drafted tools:")
    for op in result.operations:
        lines.append(f"  {mark[op['risk_class']]} {op['tool_name']}"
                     f"({', '.join(op['params'])})")
    if result.notes:
        lines.append("Review notes:")
        for n in result.notes:
            lines.append(f"  ! {n}")
    lines.append(
        "All drafts are status: proposed — nothing executes until you "
        "review proposals.yaml and mark entries approved.")
    return "\n".join(lines)


_PROPOSAL_KEYS = ("name", "description", "params", "action", "operation",
                  "operation_params", "resource_type", "id_param",
                  "filter_param", "filter_key", "preconditions", "facts",
                  "side_effects", "risk_class", "param_types",
                  "provenance", "status")


def write_proposals_yaml(result: DraftResult, path) -> list[str]:
    """Emit the ADR-005 approval artifact with every draft status: proposed.
    The output must always load through load_proposals_yaml, so every draft
    is validated FIRST — invalid drafts refuse to write. Returns the names
    written."""
    from voiceagent.proposals import (PROPOSED, draft_from_api_spec,
                                      validate_proposal)
    proposals = draft_from_api_spec(list(result.operations),
                                    provenance="ai", status=PROPOSED)
    for prop in proposals:
        errs = validate_proposal(prop)
        if errs:
            raise ValueError(f"refusing to write invalid draft "
                             f"{prop.name}: {errs}")
    entries = []
    for p in proposals:
        entry = {k: getattr(p, k) for k in _PROPOSAL_KEYS}
        for seq in ("params", "preconditions", "facts"):
            entry[seq] = list(entry[seq])
        entries.append(entry)
    header = (
        "# DRAFT — generated by scripts/onboard.py from an OpenAPI spec.\n"
        "# Every entry is status: proposed. Review, edit descriptions, add\n"
        "# preconditions where the business rules demand them, then flip\n"
        "# approved entries' status and COMMIT — the commit is the human\n"
        "# approval (ADR-005).\n")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        header + yaml.safe_dump({"proposals": entries}, sort_keys=False),
        encoding="utf-8")
    return [p.name for p in proposals]


def enrich(result: DraftResult,
           enricher: Enricher | None = None) -> DraftResult:
    """The prose seam: an enricher may rewrite ONLY each operation's
    'description' and must return the same operations in the same order.
    Mechanics (name/params/operation/risk/status) are deterministic and
    never LLM-touched. Default (None) is the identity."""
    if enricher is None:
        return result
    out = enricher(list(result.operations))
    if len(out) != len(result.operations):
        raise ValueError("enricher must return the same number of operations")
    for orig, new in zip(result.operations, out):
        for key, val in new.items():
            if key != "description" and val != orig.get(key):
                raise ValueError(
                    f"enricher may only rewrite 'description' — it changed "
                    f"{key!r} on {orig.get('tool_name')!r}")
    return DraftResult(operations=tuple(out), notes=result.notes)
```

Also add to the module's imports at the top: `from pathlib import Path` and `import yaml` (moving `json` there if you prefer — one import block, linter-clean).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_openapi_draft.py tests/test_proposals.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/openapi_draft.py tests/test_openapi_draft.py
git commit -m "feat(adaptation): discovery report, proposals.yaml writer, enrich seam"
```

---

### Task 4: `FixtureGenericBackend` — data-only GenericBackend

**Files:**
- Create: `src/voiceagent/fixture_backend.py`
- Create: `tests/test_fixture_backend.py`

**Interfaces:**
- Consumes: `generic_backend.GenericBackendError` (exists, `generic_backend.py:55`).
- Produces: `FixtureGenericBackend(path: str | Path)` implementing the full `GenericBackend` protocol (`get_resource`, `list_resources`, `create_resource`, `update_resource`, `execute_operation`, `get_lifecycle_states`); raises `ValueError` at construction for missing/invalid files. Task 5 wires it into runtime; Task 8 uses it as the hotel backend.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fixture_backend.py`:

```python
# tests/test_fixture_backend.py — the data-only GenericBackend.
"""Platform code with zero domain nouns: ANY tenant is demoable from a
JSON fixture. Guarded: protocol conformance, fail-closed errors, the
create/patch operation effects, lifecycle introspection."""
import json

import pytest

from voiceagent.fixture_backend import FixtureGenericBackend
from voiceagent.generic_backend import GenericBackendError

FIXTURE = {
    "resources": {
        "gadget": {"G-1": {"gadget_id": "G-1", "kind": "basic",
                           "owner": "+15550001"},
                   "G-2": {"gadget_id": "G-2", "kind": "pro",
                           "owner": "+15550002"}},
        "order": {},
    },
    "operations": {
        "searchGadgets": {"response": {"available": ["G-1", "G-2"]}},
        "createOrder": {"create": {"resource": "order", "id_prefix": "O-",
                                   "id_key": "order_id"},
                        "response": {"status": "PLACED"}},
        "cancelOrder": {"patch": {"resource": "order",
                                  "id_from": "order_id",
                                  "set": {"status": "CANCELLED"}},
                        "response": {}},
    },
    "lifecycles": {"order": ["PLACED", "CANCELLED"]},
}


@pytest.fixture()
def be(tmp_path):
    f = tmp_path / "fx.json"
    f.write_text(json.dumps(FIXTURE), encoding="utf-8")
    return FixtureGenericBackend(f)


def test_get_resource_deep_copies(be):
    rec = be.get_resource("gadget", "G-1")
    rec["kind"] = "MUTATED"
    assert be.get_resource("gadget", "G-1")["kind"] == "basic"


def test_get_resource_missing_is_none(be):
    assert be.get_resource("gadget", "G-999") is None


def test_unknown_resource_type_fails_closed(be):
    with pytest.raises(GenericBackendError, match="resource_type"):
        be.get_resource("spacecraft", "X-1")


def test_list_requires_filter_and_exact_matches(be):
    with pytest.raises(GenericBackendError, match="filter"):
        be.list_resources("gadget")
    out = be.list_resources("gadget", {"kind": "pro"})
    assert [r["gadget_id"] for r in out] == ["G-2"]


def test_execute_operation_canned_response(be):
    assert be.execute_operation("searchGadgets", {}) == \
        {"available": ["G-1", "G-2"]}


def test_execute_operation_unknown_fails_closed(be):
    with pytest.raises(GenericBackendError, match="unsupported operation"):
        be.execute_operation("launchRocket", {})


def test_create_effect_mints_record_and_merges_response(be):
    out = be.execute_operation("createOrder",
                               {"gadget_id": "G-1", "owner": "+15550001"})
    assert out["order_id"] == "O-1001"
    assert out["status"] == "PLACED"
    stored = be.get_resource("order", "O-1001")
    assert stored["gadget_id"] == "G-1"
    out2 = be.execute_operation("createOrder", {"gadget_id": "G-2"})
    assert out2["order_id"] == "O-1002"  # counter increments


def test_patch_effect_mutates_and_returns_updated_record(be):
    be.execute_operation("createOrder", {"gadget_id": "G-1"})
    out = be.execute_operation("cancelOrder", {"order_id": "O-1001"})
    assert out["status"] == "CANCELLED"
    assert be.get_resource("order", "O-1001")["status"] == "CANCELLED"


def test_patch_unknown_id_fails_closed(be):
    with pytest.raises(GenericBackendError, match="order_not_found"):
        be.execute_operation("cancelOrder", {"order_id": "O-404"})


def test_lifecycle_introspection(be):
    assert be.get_lifecycle_states("order") == ["PLACED", "CANCELLED"]
    with pytest.raises(NotImplementedError):
        be.get_lifecycle_states("gadget")


def test_protocol_conformance(be):
    from voiceagent.generic_backend import GenericBackend
    assert isinstance(be, GenericBackend)


def test_missing_or_invalid_file_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        FixtureGenericBackend(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        FixtureGenericBackend(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fixture_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'voiceagent.fixture_backend'`.

- [ ] **Step 3: Create `src/voiceagent/fixture_backend.py`**

```python
# src/voiceagent/fixture_backend.py — data-only GenericBackend (ADR-004).
"""A GenericBackend served entirely from a JSON fixture file — platform
code with ZERO domain nouns, so ANY tenant is demoable as bundle data
without writing an adapter (ADR-003/004: bindings are platform code,
domain data is tenant data). This is the demo tier between HttpERP
(production) and MockERP (e-commerce demo); selected via
VOICEAGENT_FIXTURE_BACKEND (see runtime._erp_from_env).

Fixture shape:
    {
      "resources": {"room": {"RM-101": {...}}, "booking": {}},
      "operations": {
        "searchRooms": {"response": {"available": [...]}},
        "createBooking": {"create": {"resource": "booking",
                                     "id_prefix": "B-",
                                     "id_key": "booking_id"},
                          "response": {"status": "BOOKED"}},
        "cancelBooking": {"patch": {"resource": "booking",
                                    "id_from": "booking_id",
                                    "set": {"status": "CANCELLED"}},
                          "response": {}}
      },
      "lifecycles": {"booking": ["HELD", "BOOKED", "CANCELLED"]}
    }

Semantics (deterministic, no scripting language):
  - get_resource/list_resources read the resource maps (deep copies);
    list requires a non-empty filter and exact-matches record fields.
  - create_resource/update_resource implement the protocol generically
    (ids minted "<RT[:2].upper()>-<n>").
  - execute_operation returns a deep copy of "response"; an optional
    "create" first creates the record from params (merged into the
    response); an optional "patch" applies "set" to the record named by
    params[id_from] and merges the updated record in (unknown id ->
    GenericBackendError). A cancel MUTATES — a canned response that lied
    would break the governance story.
  - EVERY miss raises GenericBackendError (TimeoutError-compatible, so
    the governed timeout path handles it like any adapter).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from voiceagent.generic_backend import GenericBackendError


class FixtureGenericBackend:
    """GenericBackend over a JSON fixture — the demo connector for any
    domain. Construction fails closed: a missing or invalid file raises
    ValueError (an operator-named backend must never silently swap)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(
                f"fixture backend file not found: {self._path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"fixture backend file is not valid JSON: {self._path}: "
                f"{exc}") from exc
        if not isinstance(doc, dict):
            raise ValueError(
                f"fixture backend must be a JSON object: {self._path}")
        self._resources: dict[str, dict] = \
            copy.deepcopy(doc.get("resources") or {})
        self._operations: dict = doc.get("operations") or {}
        self._lifecycles: dict = doc.get("lifecycles") or {}
        self._counters: dict[str, int] = {}

    def _require(self, resource_type: str) -> None:
        if resource_type not in self._resources:
            raise GenericBackendError(
                f"unsupported resource_type {resource_type!r} "
                f"(fixture declares: {sorted(self._resources)})")

    # --- GenericBackend protocol --------------------------------------------

    def get_resource(self, resource_type: str,
                     resource_id: str) -> dict | None:
        self._require(resource_type)
        rec = self._resources[resource_type].get(resource_id)
        return copy.deepcopy(rec) if rec is not None else None

    def list_resources(self, resource_type: str,
                       filters: dict[str, Any] | None = None) -> list[dict]:
        self._require(resource_type)
        filters = filters or {}
        if not filters:
            raise GenericBackendError(
                "list_resources requires a filter — unbounded listing is "
                "not a governed operation")
        out = []
        for rec in self._resources[resource_type].values():
            if all(rec.get(k) == v for k, v in filters.items()):
                out.append(copy.deepcopy(rec))
        return out

    def create_resource(self, resource_type: str, data: dict) -> dict:
        self._require(resource_type)
        n = self._counters.get(resource_type, 1000) + 1
        self._counters[resource_type] = n
        rid = f"{resource_type[:2].upper()}-{n}"
        rec = {f"{resource_type}_id": rid, **data}
        self._resources[resource_type][rid] = rec
        return copy.deepcopy(rec)

    def update_resource(self, resource_type: str, resource_id: str,
                        data: dict) -> dict:
        self._require(resource_type)
        rec = self._resources[resource_type].get(resource_id)
        if rec is None:
            raise GenericBackendError(
                f"{resource_type}_not_found: {resource_id}")
        rec.update(data)
        return copy.deepcopy(rec)

    def execute_operation(self, operation_name: str,
                          params: dict) -> dict:
        params = dict(params or {})
        op = self._operations.get(operation_name)
        if not isinstance(op, dict):
            raise GenericBackendError(
                f"unsupported operation {operation_name!r} "
                f"(fixture declares: {sorted(self._operations)})")
        result = copy.deepcopy(op.get("response") or {})
        create = op.get("create")
        if create:
            rt = create["resource"]
            n = self._counters.get(rt, 1000) + 1
            self._counters[rt] = n
            rid = f"{create.get('id_prefix', 'X-')}{n}"
            id_key = create.get("id_key", f"{rt}_id")
            rec = {id_key: rid, **params}
            self._resources.setdefault(rt, {})[rid] = rec
            result.update(copy.deepcopy(rec))
        patch = op.get("patch")
        if patch:
            rt = patch["resource"]
            self._require(rt)
            rid = params.get(patch["id_from"])
            rec = self._resources[rt].get(rid)
            if rec is None:
                raise GenericBackendError(
                    f"{rt}_not_found: {rid}")
            rec.update(patch.get("set") or {})
            result.update(copy.deepcopy(rec))
        return result

    def get_lifecycle_states(self, resource_type: str) -> list[str]:
        if resource_type not in self._lifecycles:
            raise NotImplementedError(
                f"fixture declares no lifecycle for {resource_type!r}; "
                "preconditions stay with the ToolSpecs")
        return list(self._lifecycles[resource_type])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fixture_backend.py -q`
Expected: PASS (all 12).

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/fixture_backend.py tests/test_fixture_backend.py
git commit -m "feat(adaptation): FixtureGenericBackend — any domain demoable from JSON data"
```

---

### Task 5: Runtime selection — `VOICEAGENT_FIXTURE_BACKEND`

**Files:**
- Modify: `src/voiceagent/runtime.py:456-464` (`_erp_from_env`)
- Test: `tests/test_fixture_backend.py` (append)

**Interfaces:**
- Consumes: `FixtureGenericBackend` (Task 4).
- Produces: backend selection order everywhere `build_orchestrator` resolves a backend: explicit `erp` arg → `VOICEAGENT_ERP_URL` (HttpERP) → `VOICEAGENT_FIXTURE_BACKEND` (FixtureGenericBackend, fail-closed on bad file) → MockERP with the loud warning. Task 8's e2e can run via env alone.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fixture_backend.py`:

```python
# --- runtime selection --------------------------------------------------------

def test_runtime_selects_fixture_backend_from_env(tmp_path):
    from voiceagent.runtime import _erp_from_env
    f = tmp_path / "fx.json"
    f.write_text(json.dumps(FIXTURE), encoding="utf-8")
    be = _erp_from_env({"VOICEAGENT_FIXTURE_BACKEND": str(f)})
    assert isinstance(be, FixtureGenericBackend)


def test_runtime_fixture_backend_missing_file_raises_named_error():
    from voiceagent.runtime import _erp_from_env
    with pytest.raises(ValueError, match="VOICEAGENT_FIXTURE_BACKEND"):
        _erp_from_env({"VOICEAGENT_FIXTURE_BACKEND": "/nonexistent/fx.json"})


def test_runtime_erp_url_wins_over_fixture():
    from voiceagent.erp_http import HttpERP
    from voiceagent.runtime import _erp_from_env
    be = _erp_from_env({"VOICEAGENT_ERP_URL": "https://erp.example",
                        "VOICEAGENT_ERP_TOKEN": "t",
                        "VOICEAGENT_FIXTURE_BACKEND": "/nonexistent/fx.json"})
    assert isinstance(be, HttpERP)


def test_runtime_no_config_returns_none():
    from voiceagent.runtime import _erp_from_env
    assert _erp_from_env({}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fixture_backend.py -q -k runtime`
Expected: FAIL — fixture env var is not read (`_erp_from_env` returns `None`).

- [ ] **Step 3: Extend `_erp_from_env` in `src/voiceagent/runtime.py`**

Replace the function body (currently `runtime.py:456-464`) with:

```python
def _erp_from_env(env: dict[str, str] | None = None):
    """Backend selection by tier: real HTTP ERP when VOICEAGENT_ERP_URL is
    configured; else a data-only FixtureGenericBackend when
    VOICEAGENT_FIXTURE_BACKEND names a fixture file (the demo tier for any
    new domain — fails CLOSED on a missing/invalid file, never a silent
    swap); None when neither is set (callers keep their offline default).
    Live entry points (the LiveKit worker) REQUIRE the ERP URL — no silent
    mock serving."""
    e = os.environ if env is None else env
    if not e.get("VOICEAGENT_ERP_URL"):
        fixture = e.get("VOICEAGENT_FIXTURE_BACKEND")
        if fixture:
            from voiceagent.fixture_backend import FixtureGenericBackend
            try:
                return FixtureGenericBackend(fixture)
            except ValueError as exc:
                raise ValueError(
                    f"VOICEAGENT_FIXTURE_BACKEND={fixture!r}: {exc}") from exc
        return None
    from voiceagent.erp_http import HttpERP
    return HttpERP(env=e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fixture_backend.py tests/test_runtime.py tests/test_demo_warning.py -q`
Expected: PASS (selection order preserved for all existing callers).

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/runtime.py tests/test_fixture_backend.py
git commit -m "feat(runtime): VOICEAGENT_FIXTURE_BACKEND — data-only backend tier, fail-closed"
```

---

### Task 6: `scripts/onboard.py` — the front-door CLI

**Files:**
- Create: `scripts/onboard.py`
- Test: `tests/test_onboard_cli.py`

**Interfaces:**
- Consumes: `parse_openapi`, `discovery_report`, `write_proposals_yaml` (Tasks 2–3); `yaml` for spec loading (YAML is a superset of JSON, so both file types load the same way).
- Produces: a bundle scaffold on disk — `proposals.yaml` (all proposed), `policies.yaml` (starter verdicts), `tenant.json`, `intents/`, `knowledge/`, `README.md`; exit code 0; report on stdout. Task 7's committed hotel bundle is the hand-reviewed descendant of exactly this output.

- [ ] **Step 1: Write the failing test**

Create `tests/test_onboard_cli.py`:

```python
# tests/test_onboard_cli.py — the adaptation front door CLI.
"""Running onboard on the committed hotel spec produces a fail-closed
scaffold: every draft proposed, high-risk policy ESCALATE, tenant.json
parseable — and the output passes through load_proposals_yaml unchanged."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data" / "fixtures" / "hotel-openapi.yaml"
CLI = ROOT / "scripts" / "onboard.py"


def _run(tmp_out):
    return subprocess.run(
        [sys.executable, str(CLI), "--spec", str(SPEC),
         "--out", str(tmp_out), "--name", "Grand Hotel"],
        capture_output=True, text=True, cwd=ROOT)


def test_onboard_prints_discovery_report_and_writes_bundle(tmp_path):
    out = tmp_path / "hotel-demo"
    r = _run(out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Discovered API: Grand Hotel PMS" in r.stdout
    assert "6 operations" in r.stdout
    assert "[HIGH RISK] cancel_booking" in r.stdout
    for f in ("proposals.yaml", "policies.yaml", "tenant.json",
              "README.md"):
        assert (out / f).exists(), f
    assert (out / "intents").is_dir() and (out / "knowledge").is_dir()


def test_onboard_output_is_fail_closed_by_construction(tmp_path):
    out = tmp_path / "hotel-demo"
    _run(out)
    from voiceagent.proposals import load_proposals_yaml
    props = load_proposals_yaml(out / "proposals.yaml")
    assert len(props) == 6
    assert all(p.status == "proposed" for p in props)
    policies = (out / "policies.yaml").read_text(encoding="utf-8")
    assert "cancel_booking:\n  escalate: true" in policies
    tenant = json.loads((out / "tenant.json").read_text(encoding="utf-8"))
    assert tenant["name"] == "Grand Hotel"
    assert tenant["persona"]


def test_onboard_scaffold_loads_through_the_bundle_loader(tmp_path):
    out = tmp_path / "hotel-demo"
    _run(out)
    from voiceagent.tenant import Tenant
    t = Tenant.load(out)
    assert t.config.name == "Grand Hotel"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_onboard_cli.py -q`
Expected: FAIL — `data/fixtures/hotel-openapi.yaml` does not exist yet; create it now as part of this task (it is the CLI's input fixture, needed by the test):

Create `data/fixtures/hotel-openapi.yaml`:

```yaml
# Grand Hotel PMS — the OpenAPI spec the owner hands the platform.
# This is the ADAPTATION DEMO's source artifact: scripts/onboard.py drafts
# the hotel tenant's tools from this file; data/fixtures/hotel.json serves
# the same domain at runtime (FixtureGenericBackend).
openapi: 3.0.3
info:
  title: Grand Hotel PMS
  version: "1.0.0"
paths:
  /rooms/search:
    post:
      operationId: searchRooms
      summary: Search available rooms for a stay window.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                check_in: {type: string}
                check_out: {type: string}
                guests: {type: integer}
              required: [check_in, check_out]
  /rooms/{room_id}:
    get:
      operationId: getRoom
      summary: Fetch one room by id.
      parameters:
        - name: room_id
          in: path
          required: true
          schema: {type: string}
  /bookings:
    post:
      operationId: createBooking
      summary: Create a booking for a room.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                guest_name: {type: string}
                room_id: {type: string}
                check_in: {type: string}
                guests: {type: integer}
              required: [guest_name, room_id, check_in]
  /bookings/{booking_id}:
    get:
      operationId: getBooking
      summary: Fetch a booking by id.
      parameters:
        - name: booking_id
          in: path
          required: true
          schema: {type: string}
    patch:
      operationId: modifyBooking
      summary: Move a booking's check-out date.
      parameters:
        - name: booking_id
          in: path
          required: true
          schema: {type: string}
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                new_check_out: {type: string}
              required: [new_check_out]
    delete:
      operationId: cancelBooking
      summary: Cancel a booking.
      parameters:
        - name: booking_id
          in: path
          required: true
          schema: {type: string}
```

Then re-run: `.venv/bin/python -m pytest tests/test_onboard_cli.py -q`
Expected: FAIL — `scripts/onboard.py` does not exist.

- [ ] **Step 3: Create `scripts/onboard.py`**

```python
# scripts/onboard.py — the adaptation front door.
"""Draft a tenant bundle from an OpenAPI 3.x spec:

    .venv/bin/python scripts/onboard.py --spec data/fixtures/hotel-openapi.yaml \
        --out data/tenants/hotel-demo --name "Grand Hotel"

Prints the discovery report, then writes fail-closed data only:

  proposals.yaml   every draft status: proposed — NOTHING executes until
                   you review and commit approvals (ADR-005)
  policies.yaml    starter verdicts: reads ALLOW, mutating ALLOW (the
                   brain still gathers parameters and confirms side
                   effects), high-risk ESCALATE
  tenant.json      name/currency scaffold; persona fields for the owner
  intents/ knowledge/  empty dirs (owner fills)
  README.md        the review checklist

Then: scripts/validate_tenant.py <out> — the CI gate.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from voiceagent.openapi_draft import (discovery_report, parse_openapi,
                                      write_proposals_yaml)

_README = """# Draft tenant bundle (generated by scripts/onboard.py)

Review checklist before this bundle goes live:

1. proposals.yaml — every entry is `status: proposed`. For each tool:
   - read the description; rewrite it in YOUR voice (the brain hears it)
   - add `preconditions:` where business rules demand them (e.g. a booking
     can only be cancelled before check-in)
   - flip `status: approved` only for tools you want LIVE, then COMMIT —
     the commit is the human approval (ADR-005)
2. policies.yaml — high-risk actions start ESCALATE; relax deliberately.
   Undeclared actions are DENY (least privilege).
3. tenant.json — fill the persona (role, tone, may_promise, never_say).
4. intents/ — add one YAML list of exemplar phrases per intent.
5. knowledge/ — add markdown docs (policies, FAQs) the agent may quote.
6. Validate: .venv/bin/python scripts/validate_tenant.py <this dir>
"""


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "tenant"


def _starter_policies(ops) -> str:
    lines = [
        "# DRAFT policy verdicts generated by scripts/onboard.py.",
        "# Reads: allowed. Mutating: allowed (the brain gathers parameters",
        "# and confirms side effects before executing). High-risk: ESCALATE",
        "# — review each and relax deliberately. Undeclared actions are",
        "# DENY (least privilege). The platform valves (escalate_to_human,",
        "# end_call) are always allowed by code, not by this file.", "",
    ]
    seen: set[str] = set()
    for op in ops:
        action = op["action"]
        if action in seen:
            continue
        seen.add(action)
        if op["risk_class"] == "high":
            lines.append(f"{action}:")
            lines.append("  escalate: true   # DRAFT — review")
        else:
            lines.append(f"{action}:")
            lines.append("  allow: true")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Draft a governed tenant bundle from an OpenAPI spec.")
    ap.add_argument("--spec", required=True,
                    help="OpenAPI 3.x file (.yaml or .json)")
    ap.add_argument("--out", required=True,
                    help="bundle directory to scaffold (e.g. "
                         "data/tenants/my-business)")
    ap.add_argument("--name", default=None,
                    help="tenant name (defaults to slugified info.title)")
    args = ap.parse_args(argv)

    spec_path = Path(args.spec)
    doc = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    result = parse_openapi(doc)
    title = (doc.get("info") or {}).get("title", "")
    print(discovery_report(result, title=title))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = write_proposals_yaml(result, out / "proposals.yaml")
    (out / "policies.yaml").write_text(
        _starter_policies(result.operations), encoding="utf-8")
    bundle_name = args.name or title or "tenant"
    (out / "tenant.json").write_text(json.dumps({
        "name": bundle_name,
        "currency": "USD",
        "persona": f"{bundle_name}'s voice agent — help customers with "
                   "their requests.",
        "languages": ["en"],
        "greeting": f"Thank you for calling {bundle_name}. How can I help?",
    }, indent=2) + "\n", encoding="utf-8")
    (out / "intents").mkdir(exist_ok=True)
    (out / "knowledge").mkdir(exist_ok=True)
    (out / "README.md").write_text(_README, encoding="utf-8")

    print(f"\nWrote bundle scaffold to {out} with drafts: "
          f"{', '.join(names)}")
    print("Next: review proposals.yaml + policies.yaml, fill persona/"
          "intents/knowledge, then validate:")
    print(f"  .venv/bin/python scripts/validate_tenant.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_onboard_cli.py -q`
Expected: PASS (all 3).

- [ ] **Step 5: Commit**

```bash
git add scripts/onboard.py tests/test_onboard_cli.py data/fixtures/hotel-openapi.yaml
git commit -m "feat(onboard): adaptation front-door CLI — spec in, fail-closed bundle scaffold out"
```

---

### Task 7: Hotel tenant bundle + fixture (data only)

**Files:**
- Create: `data/fixtures/hotel.json`
- Create: `data/tenants/hotel-demo/tenant.json`
- Create: `data/tenants/hotel-demo/tools.yaml`
- Create: `data/tenants/hotel-demo/policies.yaml`
- Create: `data/tenants/hotel-demo/proposals.yaml`
- Create: `data/tenants/hotel-demo/intents/book_room.yaml`
- Create: `data/tenants/hotel-demo/intents/cancel_booking.yaml`
- Create: `data/tenants/hotel-demo/intents/room_info.yaml`
- Create: `data/tenants/hotel-demo/knowledge/policies.md`
- Test: `tests/test_hotel_adaptation.py` (first test only — the validator gate; the file grows in Task 8)

**Interfaces:**
- Consumes: the Task 6 CLI output shape (the committed bundle is the reviewed descendant of that scaffold: descriptions owner-edited, preconditions added by the owner, statuses flipped to approved).
- Produces: `FixtureGenericBackend("data/fixtures/hotel.json")` + `build_orchestrator(tenant="hotel-demo", erp=...)` — the exact wiring Task 8's conversation tests drive. Operation names in the fixture (`searchRooms`, `createBooking`, `cancelBooking`) match the committed proposals' `operation:` values.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hotel_adaptation.py` with just the validator gate for now:

```python
# tests/test_hotel_adaptation.py — the adaptation front door, end to end.
"""The hotel tenant exists ONLY as bundle + fixture data. Guarded here:
the bundle validates; the approved proposals compile onto a
FixtureGenericBackend; a scripted conversation searches, books, and
cancels governed; the brain surface contains NO e-commerce legacy tools;
a proposed-status variant registers nothing. Zero domain code anywhere —
this file only wires what exists."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTEL = ROOT / "data" / "tenants" / "hotel-demo"
FIXTURE = ROOT / "data" / "fixtures" / "hotel.json"
VALIDATOR = ROOT / "scripts" / "validate_tenant.py"
FRONTIER_URL = {"VOICEAGENT_FRONTIER_URL": "https://fake/v1"}


def test_hotel_bundle_validates():
    r = subprocess.run([sys.executable, str(VALIDATOR), str(HOTEL)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[PASS]" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hotel_adaptation.py -q`
Expected: FAIL — the bundle directory does not exist.

- [ ] **Step 3: Create the fixture and bundle (data only)**

`data/fixtures/hotel.json`:

```json
{
  "resources": {
    "room": {
      "RM-101": {"room_id": "RM-101", "type": "deluxe", "rate_per_night": 5500, "sleeps": 2},
      "RM-102": {"room_id": "RM-102", "type": "standard", "rate_per_night": 3500, "sleeps": 2}
    },
    "booking": {}
  },
  "operations": {
    "searchRooms": {
      "response": {"available": [
        {"room_id": "RM-101", "type": "deluxe", "rate_per_night": 5500},
        {"room_id": "RM-102", "type": "standard", "rate_per_night": 3500}
      ]}
    },
    "createBooking": {
      "create": {"resource": "booking", "id_prefix": "B-", "id_key": "booking_id"},
      "response": {"status": "BOOKED"}
    },
    "cancelBooking": {
      "patch": {"resource": "booking", "id_from": "booking_id",
                "set": {"status": "CANCELLED"}},
      "response": {}
    },
    "modifyBooking": {
      "patch": {"resource": "booking", "id_from": "booking_id",
                "set_from": "new_check_out"},
      "response": {}
    }
  },
  "lifecycles": {"booking": ["BOOKED", "CHECKED_IN", "CHECKED_OUT", "CANCELLED"]}
}
```

Note: `set_from` (take the value from this param) is NOT part of `FixtureGenericBackend`'s patch semantics (it supports only `set` with fixed values). Either (a) drop `modifyBooking` from the fixture and the committed proposals (the e2e does not exercise it), or (b) extend the fixture backend's patch with an optional `set_from` param copy. **Choose (a)** — YAGNI; the e2e covers search/create/cancel and `modify_booking` stays a declared-but-fixture-simple tool by giving it a canned response only:

Replace `modifyBooking` in the fixture with:

```json
    "modifyBooking": {
      "response": {"status": "BOOKED"}
    }
```

`data/tenants/hotel-demo/tenant.json`:

```json
{
  "name": "hotel-demo",
  "currency": "USD",
  "persona": "The Grand Hotel's front-desk voice agent — help guests find rooms, book stays, and manage their reservations.",
  "languages": ["en"],
  "greeting": "Thank you for calling the Grand Hotel. How may I help you today?"
}
```

`data/tenants/hotel-demo/tools.yaml` — universal valves ONLY (this is what keeps e-commerce legacy off the brain surface):

```yaml
# This deployment's surface is proposals-driven (ADR-005): the DOMAIN tools
# arrive via proposals.yaml (human-approved from the OpenAPI draft). tools.yaml
# composes ONLY the platform valves every deployment needs — the classic demo
# tools are deliberately NOT declared, so they are not proposeable here.
tools:
  escalate_to_human:
    action: escalate_to_human
    side_effects: true
    description: "Page the front-desk supervisor to take over this call. Provide a short reason — use immediately for safety concerns or anything you cannot resolve."
  end_call:
    action: end_call
    side_effects: true
    description: "End this call politely after the guest's request is resolved."
  record_feedback:
    action: record_feedback
    side_effects: true
    description: "Record the guest's satisfaction rating (1-10) for this call."
    parameters: {'type': 'object', 'properties': {rating: {"type": "string"}}, 'required': ['rating']}
```

`data/tenants/hotel-demo/policies.yaml` — committed post-review state (the owner relaxed the onboard `escalate` draft deliberately):

```yaml
# Grand Hotel — least-privilege action verdicts. Only actions declared here
# may execute; everything else is DENY (policy engine). cancel_booking was
# ESCALATE in the onboard draft; the owner reviewed and allowed it here, with
# a precondition in proposals.yaml (never cancel an already-cancelled booking).
search_rooms:
  allow: true
get_room:
  allow: true
get_booking:
  allow: true
create_booking:
  allow: true
modify_booking:
  allow: true
cancel_booking:
  allow: true
end_call:
  allow: true
```

`data/tenants/hotel-demo/proposals.yaml` — the reviewed artifact (descriptions edited by the owner, `cancel_booking` precondition added, statuses approved):

```yaml
# Grand Hotel — APPROVED tool surface, drafted by scripts/onboard.py from
# data/fixtures/hotel-openapi.yaml, then reviewed by the owner: descriptions
# rewritten in house voice, the cancel precondition added, statuses approved.
# Committing this file IS the human approval for these entries (ADR-005).
proposals:
  - name: search_rooms
    description: "Search available rooms for a stay (check_in and check_out dates, YYYY-MM-DD)."
    params: [check_in, check_out]
    action: search_rooms
    operation: searchRooms
    param_types: {check_in: string, check_out: string}
    side_effects: false
    risk_class: read
    provenance: ai
    status: approved
  - name: get_room
    description: "Fetch one room's details by its room id (e.g. RM-101)."
    params: [room_id]
    action: get_room
    operation: __fetch__
    resource_type: room
    id_param: room_id
    side_effects: false
    risk_class: read
    provenance: ai
    status: approved
  - name: get_booking
    description: "Look up a guest's reservation by its booking id (e.g. B-1001)."
    params: [booking_id]
    action: get_booking
    operation: __fetch__
    resource_type: booking
    id_param: booking_id
    side_effects: false
    risk_class: read
    provenance: ai
    status: approved
  - name: create_booking
    description: "Book a room for a guest (guest_name, room_id, check_in). Confirm every detail with the guest BEFORE calling — this mutates their reservation."
    params: [guest_name, room_id, check_in]
    action: create_booking
    operation: createBooking
    param_types: {guest_name: string, room_id: string, check_in: string}
    side_effects: true
    risk_class: mutating
    provenance: ai
    status: approved
  - name: modify_booking
    description: "Move a reservation's check-out date (booking_id, new_check_out). Confirm the new date with the guest BEFORE calling."
    params: [booking_id, new_check_out]
    action: modify_booking
    operation: modifyBooking
    resource_type: booking
    id_param: booking_id
    side_effects: true
    risk_class: mutating
    provenance: ai
    status: approved
  - name: cancel_booking
    description: "Cancel a reservation (booking_id). Confirm with the guest BEFORE calling — cancellations may carry charges per the hotel policy."
    params: [booking_id]
    action: cancel_booking
    operation: cancelBooking
    resource_type: booking
    id_param: booking_id
    preconditions:
      - {field: status, op: not_in, value: [CANCELLED]}
    side_effects: true
    risk_class: high
    provenance: ai
    status: approved
```

`data/tenants/hotel-demo/intents/book_room.yaml`:

```yaml
- "I'd like to book a room"
- "do you have a room available for next weekend"
- "reserve the deluxe room for two nights"
- "I want to make a reservation"
```

`data/tenants/hotel-demo/intents/cancel_booking.yaml`:

```yaml
- "cancel my booking"
- "I need to cancel my reservation"
- "please cancel booking B-1001"
```

`data/tenants/hotel-demo/intents/room_info.yaml`:

```yaml
- "how much is the deluxe room"
- "what time is check-in"
- "is breakfast included"
```

`data/tenants/hotel-demo/knowledge/policies.md`:

```markdown
# Grand Hotel — policies

## Cancellation
Cancellations are free until 24 hours before the check-in date.
Within 24 hours, one night's rate is charged.

## Check-in / check-out
Check-in from 15:00. Check-out by 11:00. Early check-in on request,
subject to availability.

## Breakfast
Breakfast is served 07:00-10:00 in the atrium; included with every booking.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hotel_adaptation.py -q`
Expected: PASS (validator accepts the bundle, including `param_types` through `validate_proposal`).

- [ ] **Step 5: Commit**

```bash
git add data/fixtures/hotel.json data/tenants/hotel-demo tests/test_hotel_adaptation.py
git commit -m "data(hotel): tenant bundle + fixture — the domain is data, zero core code"
```

---

### Task 8: Hotel e2e — the governed conversation proof

**Files:**
- Modify: `tests/test_hotel_adaptation.py` (append)

**Interfaces:**
- Consumes: everything prior. Test vocabulary from `tests/test_orchestrator.py`: `ScriptedBrain`, `reply(content, calls=[])`, `tc(call_id, name, **args)` — import them exactly like the clinic suite does (`from tests.test_orchestrator import ScriptedBrain, reply, tc`); `orch.brain.client = ScriptedBrain([...])` swaps the stub in; `orch.handle_turn(session_id, text, authenticated=True)` returns a result with `.actions` (list of dicts with `tool/action/verdict/ok/value/error`) and `.reply`. `DecisionLog.query(action=..., verdict=...)` reads recorded verdicts.
- Produces: the sprint's acceptance evidence.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hotel_adaptation.py`:

```python
# --- the wired system ---------------------------------------------------------

from voiceagent.decisionlog import DecisionLog
from voiceagent.fixture_backend import FixtureGenericBackend
from voiceagent.runtime import build_orchestrator

from tests.test_orchestrator import ScriptedBrain, reply, tc


def _orch(**kw):
    return build_orchestrator(dict(FRONTIER_URL), tenant="hotel-demo",
                              erp=FixtureGenericBackend(FIXTURE), **kw)


def test_surface_is_hotel_only_no_ecommerce_legacy():
    orch = _orch()
    surface = set(orch._deployment.gateway_tools)
    assert {"search_rooms", "get_room", "get_booking", "create_booking",
            "modify_booking", "cancel_booking",
            "escalate_to_human", "end_call"} <= surface
    assert not any(n.startswith(("fetch_order", "order_", "cancel_order",
                                 "initiate_refund", "reschedule_delivery"))
                   for n in surface)
    meta = orch._deployment.gateway_tools["create_booking"]
    assert meta["side_effects"] is True
    assert meta["parameters"]["properties"]["guest_name"] == {"type": "string"}


def test_search_is_a_governed_read():
    orch = _orch(decision_log=DecisionLog())
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t1", "search_rooms", check_in="2026-09-20",
                        check_out="2026-09-21")]),
        reply("We have the deluxe room at $55 and the standard at $35 "
              "per night for those dates."),
    ])
    res = orch.handle_turn("s-search", "any rooms for September 20th?",
                           authenticated=True)
    assert res.actions[0]["tool"] == "search_rooms"
    assert res.actions[0]["verdict"] == "ALLOW" and res.actions[0]["ok"]
    assert res.actions[0]["value"]["available"][0]["room_id"] == "RM-101"


def test_booking_requires_confirmation_and_mutates_fixture():
    log = DecisionLog()
    orch = _orch(decision_log=log)
    be = orch.runner.gateway.erp
    # Turn 1: the brain asks for confirmation (no tool call) — the
    # side-effects contract on the surface is what drives that behavior.
    orch.brain.client = ScriptedBrain([
        reply("Shall I book the deluxe room for Ravi Kumar, "
              "check-in September 20th?"),
    ])
    res1 = orch.handle_turn("s-book", "please book the deluxe room",
                            authenticated=True)
    assert not res1.actions  # nothing executed without confirmation
    assert be.get_resource("booking", "B-1001") is None  # fixture untouched
    # Turn 2: guest confirms — NOW the brain calls the governed tool.
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t2", "create_booking", guest_name="Ravi Kumar",
                        room_id="RM-101", check_in="2026-09-20")]),
        reply("You're all set — booking B-1001, deluxe room, "
              "check-in September 20th."),
    ])
    res2 = orch.handle_turn("s-book", "yes, please book it",
                            authenticated=True)
    act = res2.actions[0]
    assert act["tool"] == "create_booking" and act["verdict"] == "ALLOW"
    assert act["ok"] and act["value"]["booking_id"] == "B-1001"
    assert act["value"]["status"] == "BOOKED"
    # The fixture backend MUTATED — the demo data is real state.
    assert be.get_resource("booking", "B-1001")["guest_name"] == "Ravi Kumar"
    assert log.query(action="create_booking", verdict="ALLOW")


def test_cancel_is_high_risk_governed_with_precondition():
    log = DecisionLog()
    orch = _orch(decision_log=log)
    gw = orch.runner.gateway
    # Seed a booking directly through the same governed surface.
    gw.execute("create_booking", {"guest_name": "Nina Roy",
                                  "room_id": "RM-102",
                                  "check_in": "2026-09-20"})
    orch.brain.client = ScriptedBrain([
        reply(calls=[tc("t1", "cancel_booking", booking_id="B-1001")]),
        reply("Your reservation B-1001 has been cancelled."),
    ])
    res = orch.handle_turn("s-cancel", "cancel my reservation",
                           authenticated=True)
    act = res.actions[0]
    assert act["tool"] == "cancel_booking" and act["verdict"] == "ALLOW"
    assert act["ok"] and act["value"]["status"] == "CANCELLED"
    # The precondition the OWNER added during review is enforced: a second
    # cancel of the same booking is refused before any execution.
    res2 = gw.execute("cancel_booking", {"booking_id": "B-1001"})
    assert not res2.ok and "precondition_failed" in res2.error
    assert log.query(action="cancel_booking")


def test_proposed_variant_registers_nothing(tmp_path):
    import shutil
    import yaml
    bundle = tmp_path / "hotel-proposed"
    shutil.copytree(HOTEL, bundle)
    pf = bundle / "proposals.yaml"
    doc = yaml.safe_load(pf.read_text(encoding="utf-8"))
    for entry in doc["proposals"]:
        entry["status"] = "proposed"
    pf.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    orch = build_orchestrator(dict(FRONTIER_URL), tenant=str(bundle),
                              erp=FixtureGenericBackend(FIXTURE))
    surface = set(orch._deployment.gateway_tools)
    assert not ({"search_rooms", "get_room", "get_booking",
                 "create_booking", "modify_booking",
                 "cancel_booking"} & surface)
    assert {"escalate_to_human", "end_call"} <= surface  # valves remain
```

- [ ] **Step 2: Run tests to verify they fail or reveal wiring gaps**

Run: `.venv/bin/python -m pytest tests/test_hotel_adaptation.py -q`
Expected: the Task 7 validator test still PASSES; the new tests may pass immediately (they assert existing machinery wired through new data) — if any fail, the failure is a real wiring bug in the bundle (operation name mismatch, missing policy action, wrong param name). Fix the DATA, never the core.

- [ ] **Step 3: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green — 737 prior tests + the new ~30.

- [ ] **Step 4: Neutrality spot-check (manual, 30 seconds)**

Run: `git diff --stat 585d2c6..HEAD -- src/voiceagent/`
Expected: only `proposals.py` (param_types), `openapi_draft.py` (new), `fixture_backend.py` (new), `runtime.py` (backend selection) — no other core file touched. Domain words may appear in docstring EXAMPLES (the pattern `generic_backend.py` already established: resource types are data — "orders", "appointments", "rooms"); they must not appear in code identifiers. Spot-check identifiers only:

Run: `grep -n "hotel\|grand" src/voiceagent/openapi_draft.py src/voiceagent/fixture_backend.py`
Expected: no output at all (no hotel-specific identifiers anywhere).

- [ ] **Step 5: Commit**

```bash
git add tests/test_hotel_adaptation.py
git commit -m "test(hotel): governed e2e proof — search, book with confirmation, high-risk cancel; zero domain code"
```

---

## Self-review notes (filled at plan time)

- Spec coverage: parser (Task 2), report/writer/enrich (Task 3), param_types (Task 1), fixture backend + runtime tier (Tasks 4–5), onboard CLI (Task 6), hotel bundle + fixture (Task 7), e2e incl. neutrality + human-gate assertions (Task 8). Non-goals untouched.
- Known judgment calls the executor must not "fix": `modify_booking` rides a canned fixture response (no `set_from` machinery — YAGNI); `record_feedback` IS declared in the hotel `tools.yaml` (universal valve, consistent with the repairs bundle — the feedback flywheel needs it).
- The hotel `tools.yaml` requires `escalate_to_human` (runtime mandates it); `end_call` declared to keep the call-lifecycle valve proposeable.
