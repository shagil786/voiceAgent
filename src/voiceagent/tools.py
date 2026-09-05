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
from dataclasses import dataclass, field, replace
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

    def lookup_orders_by_phone(self, phone: str) -> list[dict]:
        """Fetch orders for a caller-supplied phone number. The agent never
        knows order IDs — it asks for the number and the backend returns the
        matching orders (the not-found ladder's alternate lookup). Phone
        matching is suffix-based on digits, so '+91-9876543210' matches
        '9876543210'."""
        self._check_live()
        want = "".join(ch for ch in str(phone) if ch.isdigit())
        if not want:
            return []
        out = []
        for c in self.customers.values():
            have = "".join(ch for ch in str(c.get("phone", "")) if ch.isdigit())
            if have and (have == want or have.endswith(want) or want.endswith(have)):
                for oid in c.get("orders", []):
                    o = self.orders.get(oid)
                    if o:
                        out.append(copy.deepcopy(o))
        return out

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
    # Contract facts (Sprint A3): the customer-visible guarantees this tool's
    # reply must carry (e.g. the order reference, the word "refund"). The echo
    # guardrail forces a fact into the reply when the CUSTOMER stated it —
    # declared here per tool (code defaults, tools.yaml overrides), never
    # hardcoded in the guard.
    facts: tuple[str, ...] = ()


def parse_facts(value, where: str = "tools.yaml") -> tuple[str, ...]:
    """Validate a `facts` declaration: a non-empty list of non-empty strings.
    Raises ValueError with a deploy-gate-friendly message on bad data."""
    if (not isinstance(value, list) or not value
            or not all(isinstance(x, str) and x.strip() for x in value)):
        raise ValueError(f"{where}: 'facts' must be a non-empty list of "
                         f"non-empty strings, got {value!r}")
    return tuple(value)


def spec_facts(specs: dict[str, "ToolSpec"]) -> list[str]:
    """Union of the contract facts across a spec registry, in declaration
    order, deduplicated — the fact list the echo guardrail scans against the
    customer's own words."""
    out: list[str] = []
    for spec in specs.values():
        for f in spec.facts:
            if f not in out:
                out.append(f)
    return out


@dataclass
class ToolResult:
    ok: bool
    value: dict | None = None
    error: str | None = None
    idempotent_replay: bool = False


DEFAULT_TOOL_SPECS: dict[str, ToolSpec] = {
    # facts = the tool's reply contract (Sprint A3): what the customer must
    # see acknowledged when this tool serves the turn. A tenant bundle
    # overrides per tool via tools.yaml `facts:`.
    # NOTE: the echo guardrail scans FIRST-MATCH-PER-SPEC (one fact per spec,
    # historical KEYWORD_FACTS group semantics), so one keyword must not be
    # split across specs that can both match the same turn — e.g. "delivery"
    # deliberately stays inside the demo delivery_eta group ("order",
    # "delivery") instead of becoming reschedule_delivery's own fact.
    "fetch_order_status": ToolSpec(params=("order_id",), facts=("order",)),
    # Caller without an order ID: the agent asks for the phone number and the
    # BACKEND returns the matching orders — order IDs are never agent data.
    "order_lookup": ToolSpec(params=("phone",), facts=("order",)),
    "cancel_order": ToolSpec(
        params=("order_id", "reason"),
        preconditions=({"field": "status", "op": "not_in",
                        "value": ["SHIPPED", "DELIVERED"]},)),
    "reschedule_delivery": ToolSpec(
        params=("order_id", "new_date"),
        preconditions=({"field": "status", "op": "in",
                        "value": ["CONFIRMED", "SHIPPED"]},)),
    "initiate_refund": ToolSpec(params=("order_id", "amount", "reason"),
                                facts=("refund",)),
    # Escalation is always permitted — no preconditions; the point is that
    # the handoff becomes a real, auditable governed action.
    "escalate_to_human": ToolSpec(params=("reason",)),
    # Only shipped/delivered orders can be returned.
    "initiate_return": ToolSpec(
        params=("order_id", "reason"),
        preconditions=({"field": "status", "op": "in",
                        "value": ["SHIPPED", "DELIVERED"]},)),
}


def specs_with_yaml_facts(path: str | Path,
                          base: dict[str, ToolSpec] | None = None
                          ) -> dict[str, ToolSpec]:
    """Merge optional per-tool `facts:` declarations from a bundle's tools.yaml
    into a COPY of the base specs (DEFAULT_TOOL_SPECS). Tools the file does
    not mention keep their base spec untouched (params AND preconditions);
    unknown tool names are rejected — bindings are code, declarations are
    data. This is the facts-only view of the DEPLOYMENT tools.yaml shape
    (action/description/...) — never run it through ToolGateway.from_yaml,
    which would wipe default preconditions."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tools = raw.get("tools") or {}
    out = dict(base or DEFAULT_TOOL_SPECS)
    for name, meta in tools.items():
        if name not in out:
            raise ValueError(f"tools.yaml: unknown tool '{name}'")
        if isinstance(meta, dict) and "facts" in meta:
            out[name] = replace(out[name],
                                facts=parse_facts(meta["facts"],
                                                  f"tools.yaml '{name}'"))
    return out


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
            facts = (parse_facts(spec["facts"], f"tools.yaml '{name}'")
                     if "facts" in spec
                     else DEFAULT_TOOL_SPECS[name].facts)
            gw.specs[name] = ToolSpec(params=params, preconditions=preconds,
                                      facts=facts)
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
            elif tool_name == "order_lookup":
                value = self.erp.lookup_orders_by_phone(params["phone"])
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

