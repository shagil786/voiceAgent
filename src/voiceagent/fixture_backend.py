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
            self._require(rt)
            n = self._counters.get(rt, 1000) + 1
            self._counters[rt] = n
            rid = f"{create.get('id_prefix', 'X-')}{n}"
            id_key = create.get("id_key", f"{rt}_id")
            rec = {id_key: rid, **params}
            self._resources[rt][rid] = rec
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
