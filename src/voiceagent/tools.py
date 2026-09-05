# src/voiceagent/tools.py — Sprint A / WS2: the production Tool Gateway.
"""The end of the 'phantom action facade': actions the policy engine ALLOWS
now execute against a backend, behind one governed seam.

Architecture (non-negotiable): the LLM/dialogue manager PROPOSES, the
PolicyEngine DISPOSES, and only an ALLOW verdict reaches the ToolGateway —
which itself enforces preconditions, idempotency, and timeout protection
before touching the ERP. Every step lands in the DecisionLog.

MockERP is an in-memory stand-in for the customer's ERP/Shopify/CRM with
failure injection (fail_next) so timeout handling is testable offline. Real
tenant deployments swap MockERP for HTTP connectors declared in their
tenant bundle's tools.yaml; the Gateway/Runner code is identical.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Mock ERP
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportBackend(Protocol):
    """The ERP/CRM binding surface the ToolGateway executes against — the
    exact method set the gateway's bindings call (this IS the ERP
    abstraction; bindings stay code, per "declarations are data, bindings
    are code"). A production deployment implements this protocol (HTTP
    connector, SDK client, ...) and passes it via ToolGateway(erp=...);
    MockERP satisfies it structurally. Structural: no inheritance needed."""

    def get_order(self, order_id: str) -> dict | None: ...

    def orders_for_customer(self, customer_id: str) -> list[str]: ...

    def cancel_order(self, order_id: str, reason: str) -> dict: ...

    def reschedule_delivery(self, order_id: str, new_date: str) -> dict: ...

    def initiate_refund(self, order_id: str, amount: float,
                        reason: str) -> dict: ...

    def mark_return(self, order_id: str, reason: str) -> dict: ...

    def record_handoff(self, reason: str) -> dict: ...


class MockERP:
    """In-memory ERP with the demo customer's orders and failure injection."""

    def __init__(self) -> None:
        self.orders: dict[str, dict] = {
            "ORD-4821": {
                "order_id": "ORD-4821", "customer_id": "CUST-001",
                "status": "CONFIRMED", "amount": 1299.0,
                "items": ["Running Shoes"], "delivery_date": "2026-09-05",
                "address": "12 MG Road, Bangalore",
            },
            "ORD-7734": {
                "order_id": "ORD-7734", "customer_id": "CUST-001",
                "status": "SHIPPED", "amount": 6500.0,
                "items": ["Smart Watch"], "delivery_date": "2026-09-02",
                "tracking_url": "https://track.fake/7734",
            },
        }
        self.customers: dict[str, dict] = {
            "CUST-001": {"name": "Shagil", "phone": "+91-9876543210",
                         "orders": ["ORD-4821", "ORD-7734"]},
        }
        self.refunds: list[dict] = []
        self.handoffs: list[dict] = []
        # Failure injection: the next mutating/reading operation raises like
        # a hung backend, so graceful timeout handling is testable offline.
        self.fail_next = False

    def _check_live(self) -> None:
        if self.fail_next:
            self.fail_next = False  # one-shot: the NEXT operation fails
            raise TimeoutError("erp backend timed out")

    def get_order(self, order_id: str) -> dict | None:
        self._check_live()
        o = self.orders.get(order_id)
        return copy.deepcopy(o) if o else None

    def orders_for_customer(self, customer_id: str) -> list[str]:
        self._check_live()
        c = self.customers.get(customer_id)
        return list(c["orders"]) if c else []

    def cancel_order(self, order_id: str, reason: str) -> dict:
        self._check_live()
        o = self.orders[order_id]
        o["status"] = "CANCELLED"
        o["cancel_reason"] = reason
        return copy.deepcopy(o)

    def reschedule_delivery(self, order_id: str, new_date: str) -> dict:
        self._check_live()
        o = self.orders[order_id]
        o["delivery_date"] = new_date
        return copy.deepcopy(o)

    def initiate_refund(self, order_id: str, amount: float, reason: str) -> dict:
        self._check_live()
        o = self.orders[order_id]
        o["status"] = "REFUND_INITIATED"
        refund = {"order_id": order_id, "amount": amount, "reason": reason,
                  "refund_id": f"RF-{len(self.refunds) + 1:04d}"}
        self.refunds.append(refund)
        return refund

    def mark_return(self, order_id: str, reason: str) -> dict:
        self._check_live()
        o = self.orders[order_id]
        o["status"] = "RETURN_REQUESTED"
        o["return_reason"] = reason
        return copy.deepcopy(o)

    def record_handoff(self, reason: str) -> dict:
        self._check_live()
        self.handoffs.append({"reason": reason,
                              "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return {"handed_off": True, "reason": reason}


# ---------------------------------------------------------------------------
# Tool specs, precondition evaluation, gateway
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    params: tuple[str, ...]
    preconditions: tuple[dict, ...] = ()
    # preconditions entries: {"field": <order field>, "op": in|not_in|eq|ne,
    # "value": ...} evaluated against the fetched order record.


@dataclass
class ToolResult:
    ok: bool
    value: dict | None = None
    error: str | None = None
    idempotent_replay: bool = False


DEFAULT_TOOL_SPECS: dict[str, ToolSpec] = {
    "fetch_order_status": ToolSpec(params=("order_id",)),
    "cancel_order": ToolSpec(
        params=("order_id", "reason"),
        preconditions=({"field": "status", "op": "not_in",
                        "value": ["SHIPPED", "DELIVERED"]},)),
    "reschedule_delivery": ToolSpec(
        params=("order_id", "new_date"),
        preconditions=({"field": "status", "op": "in",
                        "value": ["CONFIRMED", "SHIPPED"]},)),
    "initiate_refund": ToolSpec(params=("order_id", "amount", "reason")),
    # Escalation is always permitted — no preconditions; the point is that
    # the handoff becomes a real, auditable governed action.
    "escalate_to_human": ToolSpec(params=("reason",)),
    # Only shipped/delivered orders can be returned.
    "initiate_return": ToolSpec(
        params=("order_id", "reason"),
        preconditions=({"field": "status", "op": "in",
                        "value": ["SHIPPED", "DELIVERED"]},)),
}


def _check_precondition(order: dict, cond: dict) -> str | None:
    """Return an error string when the condition FAILS, else None."""
    actual = order.get(cond.get("field", ""))
    op, value = cond.get("op"), cond.get("value")
    if op == "in" and actual in value:
        return None
    if op == "not_in" and actual not in value:
        return None
    if op == "eq" and actual == value:
        return None
    if op == "ne" and actual != value:
        return None
    return (f"precondition_failed: {cond.get('field')} is {actual!r} "
            f"(requires {op} {value!r})")


class ToolGateway:
    """Executes tools against the ERP with precondition, idempotency, and
    timeout protection. Specs are declarative (Python defaults, overridable
    from a tenant bundle's tools.yaml); the tool->ERP bindings are code —
    the if/elif chain below maps each tool name to the SupportBackend
    method it calls. A production backend implements SupportBackend and is
    passed via erp= (MockERP is the offline demo fixture)."""

    def __init__(self, erp: SupportBackend | None = None,
                 specs: dict[str, ToolSpec] | None = None):
        self.erp = erp or MockERP()
        self.specs = dict(specs or DEFAULT_TOOL_SPECS)
        self._idempotency: dict[str, ToolResult] = {}

    @classmethod
    def from_yaml(cls, path, erp: MockERP | None = None) -> "ToolGateway":
        """Load spec overrides from a tenant bundle's tools.yaml. Only known
        tool names may be overridden — bindings are code, declarations are
        data. Unknown names are rejected rather than silently ignored."""
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        tools = raw.get("tools", {})
        gw = cls(erp=erp)
        for name, spec in tools.items():
            if name not in DEFAULT_TOOL_SPECS:
                raise ValueError(f"tools.yaml: unknown tool '{name}'")
            params = tuple(spec.get("params",
                                    list(DEFAULT_TOOL_SPECS[name].params)))
            preconds = tuple(spec.get("preconditions", []))
            gw.specs[name] = ToolSpec(params=params, preconditions=preconds)
        return gw

    def execute(self, tool_name: str, params: dict,
                idempotency_key: str | None = None) -> ToolResult:
        spec = self.specs.get(tool_name)
        if spec is None:
            return ToolResult(ok=False, error=f"unknown_tool: {tool_name}")
        missing = [p for p in spec.params if params.get(p) is None]
        if missing:
            return ToolResult(ok=False, error=f"missing_params: {missing}")

        if idempotency_key and idempotency_key in self._idempotency:
            replay = self._idempotency[idempotency_key]
            return ToolResult(ok=replay.ok, value=replay.value,
                              error=replay.error, idempotent_replay=True)

        # Order-scoped tools fetch the record for precondition checks; tools
        # whose spec has no order_id (escalate_to_human) skip the fetch.
        try:
            order = (self.erp.get_order(params["order_id"])
                     if "order_id" in spec.params else None)
        except TimeoutError:
            return ToolResult(ok=False,
                              error="backend_timeout (graceful; ticket issued)")
        if "order_id" in spec.params and order is None:
            return ToolResult(ok=False,
                              error=f"order_not_found: {params['order_id']}")
        for cond in spec.preconditions:
            err = _check_precondition(order, cond)
            if err:
                return ToolResult(ok=False, error=err)

        try:
            if tool_name == "fetch_order_status":
                value = order
            elif tool_name == "cancel_order":
                value = self.erp.cancel_order(params["order_id"],
                                              params["reason"])
            elif tool_name == "reschedule_delivery":
                value = self.erp.reschedule_delivery(params["order_id"],
                                                     params["new_date"])
            elif tool_name == "initiate_refund":
                value = self.erp.initiate_refund(params["order_id"],
                                                 float(params["amount"]),
                                                 params["reason"])
            elif tool_name == "escalate_to_human":
                value = self.erp.record_handoff(params["reason"])
            elif tool_name == "initiate_return":
                value = self.erp.mark_return(params["order_id"],
                                             params["reason"])
            else:  # pragma: no cover — specs and bindings stay in sync
                return ToolResult(ok=False,
                                  error=f"unbound_tool: {tool_name}")
        except TimeoutError:
            # Graceful timeout: NOT cached (a retry may succeed once the
            # backend recovers); the caller tickets instead of retrying
            # blindly.
            return ToolResult(ok=False,
                              error="backend_timeout (graceful; ticket issued)")
        result = ToolResult(ok=True, value=value)
        if idempotency_key:
            self._idempotency[idempotency_key] = result
        return result


# ---------------------------------------------------------------------------
# Governed runner: PolicyEngine in front, gateway behind, log always
# ---------------------------------------------------------------------------

@dataclass
class GovernedOutcome:
    decision_verdict: str
    reasons: list[str] = field(default_factory=list)
    executed: bool = False
    result: ToolResult | None = None


class GovernedToolRunner:
    """The only sanctioned way to execute a tool: policy verdict first, tool
    execution only on ALLOW, decision log ALWAYS (both allows and blocks)."""

    def __init__(self, gateway: ToolGateway, policy, decision_log=None):
        self.gateway = gateway
        self.policy = policy
        self.decision_log = decision_log

    def run(self, action: str, context, tool_name: str, params: dict,
            idempotency_key: str | None = None,
            conv_id: str = "",
            tool_states: dict[str, str] | None = None) -> GovernedOutcome:
        # Deployment gate (additive): when the caller passes the bundle's
        # tool states, only CONNECTED tools may execute. None means a
        # pre-gate deployment — enforce policy only, as before. A passed
        # dict MISSING the tool name means unknown — never executed
        # (spec section 6) — blocked exactly like a non-CONNECTED state.
        if tool_states is not None:
            state = tool_states.get(tool_name)
            if state != "CONNECTED":
                label = state if state is not None else "unknown"
                outcome = GovernedOutcome(
                    decision_verdict="BLOCKED_UNCONNECTED",
                    reasons=[f"tool '{tool_name}' is {label}, "
                             "owner approval required"])
                if self.decision_log is not None:
                    from voiceagent.decisionlog import DecisionEntry
                    self.decision_log.record(DecisionEntry(
                        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        conv_id=conv_id, action=action,
                        verdict="BLOCKED_UNCONNECTED",
                        reasons=list(outcome.reasons),
                        amount=getattr(context, "amount", None),
                        authenticated=getattr(context, "authenticated", False)))
                return outcome
        decision = self.policy.evaluate(action, context)
        outcome = GovernedOutcome(decision_verdict=decision.verdict,
                                  reasons=list(decision.reasons))
        if decision.verdict == "ALLOW":
            result = self.gateway.execute(tool_name, params,
                                          idempotency_key=idempotency_key)
            outcome.result = result
            outcome.executed = result.ok
            outcome.reasons.append(
                f"tool '{tool_name}' "
                + ("executed" if result.ok else f"failed: {result.error}"))
        else:
            outcome.reasons.append(f"tool '{tool_name}' blocked by policy")
        if self.decision_log is not None:
            from voiceagent.decisionlog import DecisionEntry
            self.decision_log.record(DecisionEntry(
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                conv_id=conv_id, action=action,
                verdict=decision.verdict, reasons=list(outcome.reasons),
                amount=getattr(context, "amount", None),
                authenticated=getattr(context, "authenticated", False)))
        return outcome

