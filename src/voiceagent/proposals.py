# src/voiceagent/proposals.py — tool proposals: hybrid authorship, human gate.
"""Tool proposals (ADR-005): the HYBRID authorship path with a mandatory
human approval gate.

Two channels produce proposals for a deployment's GenericBackend:
  - AI-drafted   (provenance="ai")       — an LLM/parser reads an operator-
    supplied API spec / KB and drafts tool proposals (draft_from_api_spec is
    the deterministic hook; the LLM fills descriptions/params above it).
  - operator-authored (provenance="operator") — an operator writes the
    proposal YAML directly.

BOTH channels land in the SAME state machine:

    PROPOSED --(human approves: operator commits proposals.yaml with the
                tool marked approved)--> APPROVED --compile--> REGISTERED
        |                                                            |
        +--(human rejects)--> REJECTED ---------------------> DROPPED (never)

Nothing executes from PROPOSED or REJECTED. `compile_approved()` registers
ONLY tools whose committed status is APPROVED — committing the file IS the
human approval (the same trust boundary as tools.yaml today: an operator
merges it, never the tenant's LLM). An agent can never approve its own
surface: the approval artifact is out-of-band data.

A proposal is DECLARATION ONLY — no code, no sandbox, no runtime codegen.
Compilation (compile_approved) lowers an approved proposal onto ADR-004's
existing machinery: a ToolSpec (with ToolSpec.resource for precondition
fetch) + a registered executor over the deployment's GenericBackend. The
result is byte-for-byte the same governed surface a hand-written domain
module (demo_repairs.py) would have produced — checked by tests.

Risk classification (`risk_class`) is review metadata for the human gate
(read tools = lighter review; cancel/refund-class = full review), declared
by the proposal author and validated against the operation kind; it never
replaces policy (the PolicyEngine still gates by action).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from voiceagent.tools import ToolGateway, ToolSpec

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Proposal lifecycle statuses (stored in the committed YAML; the file is the
# approval artifact). Only APPROVED ever compiles to a registered tool.
PROPOSED = "proposed"
APPROVED = "approved"
REJECTED = "rejected"

RISK_READ = "read"          # no side effects — lighter review
RISK_MUTATING = "mutating"  # cancel/reschedule-class — full review
RISK_HIGH = "high"          # refund/payment/delete-class — full review + policy


@dataclass(frozen=True)
class ToolProposal:
    """One proposed tool: declaration only. Never executes by itself."""
    name: str
    description: str
    params: tuple[str, ...]
    action: str                      # policy key (PolicyEngine gates on it)
    operation: str                   # GenericBackend.execute_operation name
    operation_params: dict[str, str] = field(default_factory=dict)
    # tool param -> backend operation param (identity when absent)
    resource_type: str | None = None
    id_param: str | None = None
    # __list__ lookups (e.g. find-by-phone): tool param carrying the key +
    # backend filter key (defaults to the param name when absent).
    filter_param: str | None = None
    filter_key: str | None = None
    preconditions: tuple[dict, ...] = ()
    facts: tuple[str, ...] = ()
    side_effects: bool = False
    risk_class: str = RISK_READ
    provenance: str = "operator"     # "ai" | "operator"
    status: str = PROPOSED           # proposed | approved | rejected


def validate_proposal(prop: ToolProposal) -> list[str]:
    """Structural validation (pure — no backend needed). Errors list, empty
    = valid. A proposal that fails this never reaches the approval gate."""
    errs: list[str] = []
    if not _NAME_RE.match(prop.name):
        errs.append(f"invalid tool name {prop.name!r} "
                    f"(lowercase snake_case)")
    if not prop.description.strip():
        errs.append(f"{prop.name}: description required")
    if not prop.params:
        errs.append(f"{prop.name}: params required")
    if not _NAME_RE.match(prop.action):
        errs.append(f"{prop.name}: invalid action {prop.action!r}")
    if prop.status not in (PROPOSED, APPROVED, REJECTED):
        errs.append(f"{prop.name}: invalid status {prop.status!r}")
    if prop.provenance not in ("ai", "operator"):
        errs.append(f"{prop.name}: provenance must be 'ai' or 'operator'")
    if prop.risk_class not in (RISK_READ, RISK_MUTATING, RISK_HIGH):
        errs.append(f"{prop.name}: invalid risk_class {prop.risk_class!r}")
    if prop.side_effects and prop.risk_class == RISK_READ:
        errs.append(f"{prop.name}: side_effects=true but risk_class=read")
    if prop.id_param and prop.id_param not in prop.params:
        errs.append(f"{prop.name}: id_param {prop.id_param!r} not in params")
    if prop.operation == "__list__":
        if not prop.resource_type:
            errs.append(f"{prop.name}: __list__ requires resource_type")
        if not prop.filter_param or prop.filter_param not in prop.params:
            errs.append(f"{prop.name}: __list__ requires filter_param "
                        f"in params")
    return errs


def compile_approved(gw: ToolGateway, backend: Any,
                     proposals: list[ToolProposal]) -> list[str]:
    """Lower APPROVED proposals onto the gateway (ADR-004 machinery):
    a ToolSpec (+ ToolSpec.resource for precondition fetch) and a registered
    executor calling backend.execute_operation. Returns the registered tool
    names. PROPOSED/REJECTED proposals are skipped — never registered.
    The executor maps tool params -> operation params and raises the
    backend's errors (TimeoutError-compatible -> governed timeout path)."""
    registered: list[str] = []
    for prop in proposals:
        if prop.status != APPROVED:
            continue  # the human gate: nothing registers unapproved
        errs = validate_proposal(prop)
        if errs:
            raise ValueError(
                f"cannot compile approved proposal {prop.name}: {errs}")
        executor_params = {
            prop.operation_params.get(k, k): v
            for k, v in prop.params
        } if False else None
        # executor signature: (backend, tool_params) -> value

        def _make_executor(op: str, mapping: dict[str, str],
                           rt: str | None, idp: str | None,
                           fp: str | None = None, fk: str | None = None):
            if rt is not None and idp is not None and op == "__fetch__":
                return lambda erp, p: erp.get_resource(rt, p[idp])
            if rt is not None and op == "__list__":
                # Alternate lookup (e.g. find-by-phone): unknown keys yield
                # an empty list, never an error — the ladder's alternate leg.
                return lambda erp, p: erp.list_resources(
                    rt, {(fk or fp or "phone"): p[fp or "phone"]})
            return lambda erp, p: erp.execute_operation(
                op, {mapping.get(k, k): v for k, v in p.items()})

        resource = None
        if prop.resource_type and prop.id_param:
            resource = (prop.resource_type, prop.id_param,
                        lambda erp, rid, rt=prop.resource_type:
                        erp.get_resource(rt, rid))
        spec = ToolSpec(
            params=prop.params,
            preconditions=prop.preconditions,
            facts=prop.facts,
            side_effects=prop.side_effects,
            action=prop.action,
            description=prop.description,
            resource=resource,
        )
        gw.specs[prop.name] = spec
        gw.register_binding(
            prop.name,
            _make_executor(prop.operation, prop.operation_params,
                           prop.resource_type, prop.id_param,
                           prop.filter_param, prop.filter_key))
        registered.append(prop.name)
    return registered


def load_proposals_yaml(path: str | Path) -> list[ToolProposal]:
    """Load the deployment's proposals artifact (proposals.yaml) — the
    approval document. Shape:

        proposals:
          - name: cancel_visit
            description: Cancel a booking (approved by ops on 2026-09-09).
            params: [booking_id, reason]
            action: cancel_booking
            operation: cancel_booking
            operation_params: {}
            resource_type: booking
            id_param: booking_id
            preconditions:
              - {field: status, op: not_in, value: [IN_PROGRESS, DONE]}
            facts: [booking]
            side_effects: true
            risk_class: mutating
            provenance: ai        # ai | operator
            status: approved      # proposed | approved | rejected

    Committing this file IS the human approval for the approved entries;
    proposed/rejected entries are recorded (and validated) but never
    compile. Unknown keys are rejected so a typo'd field cannot silently
    drop a tool from the surface."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or "proposals" not in raw:
        raise ValueError(f"{path}: expected a top-level 'proposals' list")
    entries = raw["proposals"]
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'proposals' must be a list")
    out: list[ToolProposal] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"{path}: proposal #{i} must be a mapping")
        allowed = {"name", "description", "params", "action", "operation",
                   "operation_params", "resource_type", "id_param",
                   "filter_param", "filter_key",
                   "preconditions", "facts", "side_effects", "risk_class",
                   "provenance", "status"}
        unknown = set(e) - allowed
        if unknown:
            raise ValueError(
                f"{path}: proposal #{i} has unknown key(s) "
                f"{sorted(unknown)} — refusing to drop a tool silently")
        out.append(ToolProposal(
            name=str(e["name"]),
            description=str(e.get("description", "")),
            params=tuple(e.get("params", [])),
            action=str(e.get("action", e["name"])),
            operation=str(e.get("operation", "")),
            operation_params=dict(e.get("operation_params", {})),
            resource_type=e.get("resource_type"),
            id_param=e.get("id_param"),
            filter_param=e.get("filter_param"),
            filter_key=e.get("filter_key"),
            preconditions=tuple(e.get("preconditions", [])),
            facts=tuple(e.get("facts", [])),
            side_effects=bool(e.get("side_effects", False)),
            risk_class=e.get("risk_class", RISK_READ),
            provenance=e.get("provenance", "operator"),
            status=e.get("status", PROPOSED),
        ))
    return out


def gateway_tool_meta(prop: ToolProposal) -> dict:
    """The deployment.tools.yaml-style entry for an approved proposal —
    what the brain's proposal surface needs (description, action, facts,
    parameters schema). Compile-approved proposals land in
    Deployment.gateway_tools so deploy() registers them with the brain;
    execution goes through the gateway spec/binding compile_approved made."""
    properties = {}
    for pname in prop.params:
        ptype = "string"  # proposals declare string params by default;
        properties[pname] = {"type": ptype}  # typed coercion is a ToolSpec
    return {
        "action": prop.action,
        "description": prop.description,
        "facts": list(prop.facts),
        "side_effects": prop.side_effects,
        "parameters": {"type": "object",
                       "properties": properties,
                       "required": list(prop.params)},
    }


def draft_from_api_spec(spec: dict[str, Any], *,
                        provenance: str = "ai",
                        status: str = PROPOSED) -> list[ToolProposal]:
    """Draft proposals from a tiny declarative API spec — the deterministic
    hook under an LLM-assisted drafter. `spec` shape:

        {"operations": [
            {"operation": "cancel_booking",
             "tool_name": "cancel_booking",
             "description": "...",
             "params": ["booking_id", "reason"],
             "side_effects": true,
             "risk_class": "mutating",
             "resource_type": "booking",
             "id_param": "booking_id",
             "action": "cancel_booking"}
        ]}

    An LLM-assisted drafter would parse a real OpenAPI document INTO this
    shape (mapping operations -> params); this pure function then yields
    the proposals. Drafted proposals are PROPOSED — they need human
    approval like any other."""
    out: list[ToolProposal] = []
    for op in spec.get("operations", []):
        out.append(ToolProposal(
            name=op["tool_name"],
            description=op.get("description", ""),
            params=tuple(op.get("params", [])),
            action=op.get("action", op["tool_name"]),
            operation=op["operation"],
            operation_params=dict(op.get("operation_params", {})),
            resource_type=op.get("resource_type"),
            id_param=op.get("id_param"),
            filter_param=op.get("filter_param"),
            filter_key=op.get("filter_key"),
            preconditions=tuple(op.get("preconditions", [])),
            facts=tuple(op.get("facts", [])),
            side_effects=bool(op.get("side_effects", False)),
            risk_class=op.get("risk_class", RISK_READ),
            provenance=provenance,
            status=status,
        ))
    return out
