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
  - risk (checked in this exact order): DELETE -> high (regardless of
    name); GET -> read; non-GET whose operationId or path contains
    refund/payment/charge/payout -> high (money tokens beat the read
    prefixes — a POST getRefund is never a read); non-GET operationId
    starting with a read prefix (search/list/find/fetch/get/lookup/query)
    -> read with side_effects=False (POST-search endpoints are reads);
    otherwise mutating.
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
import yaml
from dataclasses import dataclass
from pathlib import Path
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
    """Resolve local '#/...' $refs (chained, depth-capped). External or
    broken refs resolve to the original node; a required body property
    whose schema carries no scalar type then drafts as a plain string
    param — visible in proposals.yaml for owner review."""
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
    # Order matters: money tokens are checked BEFORE the read prefixes so
    # a mutating POST/PUT/PATCH named getRefund/fetchPayout can never
    # draft as a side-effect-free read (the safe direction — a misjudged
    # read loses its confirmation gate and review marker).
    if method == "delete":
        return "high"
    if method == "get":
        return "read"
    low = f"{op_id} {path}".lower()
    if any(tok in low for tok in _HIGH_TOKENS):
        return "high"
    if op_id.lower().startswith(_READ_PREFIXES):
        return "read"
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
    proposals = draft_from_api_spec(
        {"operations": list(result.operations)},
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
    never LLM-touched. The guard compares against a snapshot taken before
    the enricher runs, so a DROPPED key is refused exactly like a changed
    value, and the enricher works on copies — the input DraftResult is
    never mutated. Default (None) is the identity."""
    if enricher is None:
        return result
    snapshot = [dict(op) for op in result.operations]
    out = enricher([dict(op) for op in snapshot])
    if len(out) != len(result.operations):
        raise ValueError("enricher must return the same number of operations")
    for orig, new in zip(snapshot, out):
        if set(new) != set(orig):
            raise ValueError(
                f"enricher may only rewrite 'description' — it changed the "
                f"key set of {orig.get('tool_name')!r}: removed "
                f"{sorted(set(orig) - set(new))}, added "
                f"{sorted(set(new) - set(orig))}")
        for key, val in new.items():
            if key != "description" and val != orig.get(key):
                raise ValueError(
                    f"enricher may only rewrite 'description' — it changed "
                    f"{key!r} on {orig.get('tool_name')!r}")
    return DraftResult(operations=tuple(out), notes=result.notes)
