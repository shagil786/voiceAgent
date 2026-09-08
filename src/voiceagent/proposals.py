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
                           rt: str | None, idp: str | None):
            if rt is not None and idp is not None and op == "__fetch__":
                return lambda erp, p: erp.get_resource(rt, p[idp])
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
                           prop.resource_type, prop.id_param))
        registered.append(prop.name)
    return registered


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
            preconditions=tuple(op.get("preconditions", [])),
            facts=tuple(op.get("facts", [])),
            side_effects=bool(op.get("side_effects", False)),
            risk_class=op.get("risk_class", RISK_READ),
            provenance=provenance,
            status=status,
        ))
    return out
