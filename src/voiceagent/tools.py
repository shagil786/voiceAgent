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
import json
import math
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable


# The default (demo) tenant's ERP fixture: a file in the COMMITTED default
# bundle (data/tenants/default/erp_fixture.json — {"orders": ..., "customers":
# ...}) — demo ERP data is tenant data, not a Python import (Task E). Anchored
# to the repo root (src/voiceagent/ -> parents[2]) so resolution does not
# depend on the process cwd.
DEFAULT_BUNDLE_ERP_FIXTURE = (Path(__file__).resolve().parents[2]
                              / "data" / "tenants" / "default"
                              / "erp_fixture.json")


def _default_erp_fixture() -> tuple[dict, dict]:
    """(orders, customers) from the committed default bundle; empty dicts
    when the bundle file is unavailable (never a hard import-time failure)."""
    p = DEFAULT_BUNDLE_ERP_FIXTURE
    if not p.exists():
        return {}, {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return (data.get("orders", {}) or {}, data.get("customers", {}) or {})


# ---------------------------------------------------------------------------
# Mock ERP
# ---------------------------------------------------------------------------

# ADR-004: new domains implement the resource-verb GenericBackend
# (voiceagent.generic_backend) instead of mapping nouns onto this protocol;
# SupportBackend stays the supported surface for existing deployments.
# (Forward reference only — no behavior change here.)

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


def _id_shape(order_id: str) -> str:
    """Comparison shape for order IDs: alphanumeric uppercase, so 'ORD4821',
    'ord-4821' and 'ORD-4821' all match."""
    return "".join(ch for ch in str(order_id) if ch.isalnum()).upper()


class MockERP:
    """In-memory ERP with the demo customer's orders and failure injection."""

    def __init__(self) -> None:
        # Demo fixture data is the committed default tenant bundle's
        # erp_fixture.json (loaded lazily, deep-copied so one instance's
        # mutations never leak into another).
        orders, customers = _default_erp_fixture()
        self.orders: dict[str, dict] = copy.deepcopy(orders)
        self.customers: dict[str, dict] = copy.deepcopy(customers)
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
        if o is None:
            # Callers and brains spell IDs loosely ('ORD4821' vs the stored
            # 'ORD-4821') — match on the alphanumeric-uppercase shape, the
            # same normalization idea as phone lookup. Exact key always wins.
            want = _id_shape(order_id)
            for k, v in self.orders.items():
                if _id_shape(k) == want:
                    return copy.deepcopy(v)
            return None
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
    # Optional numeric bounds: {param: (min, max)} enforced after coercion —
    # a rating must be 1..10, a partial refund 0..cap, etc. Declared data.
    param_bounds: dict = field(default_factory=dict)
    # Tool metadata the brain's proposal surface needs — declared HERE, next
    # to the binding, so adding a tool never requires touching runtime.py:
    # side_effects (mutating? default True = safest assumption) drives the
    # confirmation/governance hints; description is what the frontier sees.
    side_effects: bool = True
    description: str = ""
    # The POLICY/intent action name this tool is governed under (policies.yaml
    # rules are written against actions, e.g. 'order_status', 'refund').
    # Default: the tool name itself.
    action: str = ""
    # Task D2: light param-type validation at the governed boundary. Maps a
    # declared param name to one of PARAM_TYPES ("string" | "number" |
    # "integer" | "boolean"); params WITHOUT an entry default to "string".
    # The gateway coerces safe cases (amount "200" -> 200.0) and rejects
    # impossible ones with `invalid_param: <name>` before any ERP call.
    param_types: dict[str, str] = field(default_factory=dict)


# The only param types the boundary understands (Task D2).
PARAM_TYPES = ("string", "number", "integer", "boolean")


def _coerce_param(name: str, value, declared: str | None) -> tuple[bool, object]:
    """Validate/coerce one param against its declared type (None -> the
    "string" default). Returns (ok, coerced_value); ok=False means the value
    can never be that type -> the caller rejects with `invalid_param: <name>`.

    Safe coercions only: bool is NEVER a number/integer (it subclasses int);
    strings parse for number/integer/boolean; anything else coerces to str
    under the "string" default so a brain that quotes a value still works."""
    t = declared or "string"
    if t == "string":
        return True, (value if isinstance(value, str) else str(value))
    if t == "number":
        if isinstance(value, bool):
            return False, None
        if isinstance(value, (int, float)):
            f = float(value)
            return (True, f) if math.isfinite(f) else (False, None)
        if isinstance(value, str):
            # reject underscore literals / nan / inf before float() accepts them
            if not re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", value.strip()):
                return False, None
        if isinstance(value, str):
            try:
                return True, float(value)
            except ValueError:
                return False, None
        return False, None
    if t == "integer":
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            return True, value
        if isinstance(value, float) and not math.isfinite(value):
            return False, None
        if isinstance(value, float) and value.is_integer():
            return True, int(value)
        if isinstance(value, str):
            try:
                return True, int(value)
            except ValueError:
                return False, None
        return False, None
    if t == "boolean":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes"):
                return True, True
            if low in ("false", "0", "no"):
                return True, False
        return False, None
    # An unknown declared type: pass through unchanged (declaration bug, not
    # a runtime rejection — the boundary never invents constraints).
    return True, value


def parse_facts(value, where: str = "tools.yaml") -> tuple[str, ...]:
    """Validate a `facts` declaration: a non-empty list of non-empty strings.
    Raises ValueError with a deploy-gate-friendly message on bad data."""
    if (not isinstance(value, list) or not value
            or not all(isinstance(x, str) and x.strip() for x in value)):
        raise ValueError(f"{where}: 'facts' must be a non-empty list of "
                         f"non-empty strings, got {value!r}")
    return tuple(value)


def parse_param_types(value, where: str = "tools.yaml") -> dict[str, str]:
    """Validate a `param_types` declaration: a non-empty mapping of param
    name -> one of PARAM_TYPES (the same set the gateway boundary coerces
    against). Raises ValueError with a deploy-gate-friendly message on bad
    data — Task E threads these through tools.yaml."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where}: 'param_types' must be a non-empty "
                         f"mapping of param -> one of {PARAM_TYPES}, "
                         f"got {value!r}")
    out: dict[str, str] = {}
    for p, t in value.items():
        if not isinstance(p, str) or not p.strip() or t not in PARAM_TYPES:
            raise ValueError(f"{where}: 'param_types' entries must map a "
                             f"param name to one of {PARAM_TYPES}, got "
                             f"{p!r}: {t!r}")
        out[p] = t
    return out


def parse_param_bounds(value,
                       where: str = "tools.yaml") -> dict[str, tuple]:
    """Validate a `param_bounds` declaration: a non-empty mapping of param
    name -> [lo, hi] with NUMERIC lo < hi (parsed to floats, stored as a
    tuple the gateway enforces after coercion: lo <= value <= hi). Raises
    ValueError with a deploy-gate-friendly message on bad data — Task E
    threads these through tools.yaml."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where}: 'param_bounds' must be a non-empty "
                         f"mapping of param -> [lo, hi], got {value!r}")
    out: dict[str, tuple] = {}
    for p, b in value.items():
        if (not isinstance(p, str) or not p.strip()
                or not isinstance(b, (list, tuple)) or len(b) != 2):
            raise ValueError(f"{where}: 'param_bounds' entries must be "
                             f"param: [lo, hi], got {p!r}: {b!r}")
        try:
            lo, hi = float(b[0]), float(b[1])
        except (TypeError, ValueError):
            raise ValueError(f"{where}: param_bounds for {p!r} must be "
                             f"numeric [lo, hi], got {b!r}") from None
        if math.isnan(lo) or math.isnan(hi) or not lo < hi:
            raise ValueError(f"{where}: param_bounds for {p!r} need numeric "
                             f"lo < hi, got {b!r}")
        out[p] = (lo, hi)
    return out


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
    "fetch_order_status": ToolSpec(
        params=("order_id",), facts=("order",), side_effects=False,
        action="order_status",
        description="Fetch the current status of an order by its order ID "
                    "(e.g. ORD-4821)."),
    # Caller without an order ID: the agent asks for the phone number and the
    # BACKEND returns the matching orders — order IDs are never agent data.
    "order_lookup": ToolSpec(
        params=("phone",), facts=("order",), side_effects=False,
        description="Find a caller's orders by the phone number they ordered "
                    "with — use when the caller does not know their order ID."),
    "cancel_order": ToolSpec(
        params=("order_id", "reason"),
        preconditions=({"field": "status", "op": "not_in",
                        "value": ["SHIPPED", "DELIVERED"]},)),
    "reschedule_delivery": ToolSpec(
        params=("order_id", "new_date"),
        preconditions=({"field": "status", "op": "in",
                        "value": ["CONFIRMED", "SHIPPED"]},)),
    "initiate_refund": ToolSpec(params=("order_id", "amount", "reason"),
                                facts=("refund",), action="refund",
                                param_types={"amount": "number"}),
    # Escalation is always permitted — no preconditions; the point is that
    # the handoff becomes a real, auditable governed action.
    "escalate_to_human": ToolSpec(
        params=("reason",),
        description="Page a human agent to take over this call. Provide a "
                    "short reason for the handoff."),
    # Call lifecycle: the caller's own call ends when THEY are done — the
    # brain proposes end_call on farewell or after resolution + rating; the
    # telephony session observes the executed action and hangs up.
    "end_call": ToolSpec(
        params=("reason",),
        description="End this call politely. Propose when the caller says "
                    "goodbye/thanks-and-bye, or after their issue is resolved "
                    "and any feedback captured."),
    "record_feedback": ToolSpec(
        params=("rating",),
        param_types={"rating": "number", "comment": "string"},
        param_bounds={"rating": (1, 10)},
        description="Record the caller's satisfaction rating (1-10) for this "
                    "call. Ask for it once the issue is resolved; 0 or >10 is "
                    "invalid."),
    # Only shipped/delivered orders can be returned.
    "initiate_return": ToolSpec(
        params=("order_id", "reason"),
        preconditions=({"field": "status", "op": "in",
                        "value": ["SHIPPED", "DELIVERED"]},),
        action="return",
        description="Request a return for a shipped or delivered order "
                    "(params: order_id, reason)."),
}


def specs_with_yaml_facts(path: str | Path,
                          base: dict[str, ToolSpec] | None = None
                          ) -> dict[str, ToolSpec]:
    """Merge optional per-tool constraint declarations from a bundle's
    tools.yaml into a COPY of the base specs (DEFAULT_TOOL_SPECS): `facts:`
    (Sprint A3) plus `param_types:` / `param_bounds:` (Task E). Tools the
    file does not mention keep their base spec untouched (params AND
    preconditions); unknown tool names are rejected — bindings are code,
    declarations are data. This is the constraints view of the DEPLOYMENT
    tools.yaml shape (action/description/...) — never run it through
    ToolGateway.from_yaml, which rebuilds the execution spec shape."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tools = raw.get("tools") or {}
    out = dict(base or DEFAULT_TOOL_SPECS)
    for name, meta in tools.items():
        if name not in out:
            raise ValueError(f"tools.yaml: unknown tool '{name}'")
        if not isinstance(meta, dict):
            continue
        kw: dict = {}
        if "facts" in meta:
            kw["facts"] = parse_facts(meta["facts"],
                                      f"tools.yaml '{name}'")
        if "param_types" in meta:
            kw["param_types"] = parse_param_types(meta["param_types"],
                                                  f"tools.yaml '{name}'")
        if "param_bounds" in meta:
            kw["param_bounds"] = parse_param_bounds(meta["param_bounds"],
                                                    f"tools.yaml '{name}'")
        if kw:
            out[name] = replace(out[name], **kw)
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
        data. Unknown names are rejected rather than silently ignored.

        Per-tool keys: params, preconditions, facts, param_types,
        param_bounds. A key ABSENT from the yaml entry keeps the base spec's
        value — most importantly preconditions: a bundle that does not
        declare them keeps the code-default ones (relaxing a precondition
        requires an explicit `preconditions: []`), so a tools.yaml entry can
        never silently widen what the gateway executes. Declared keys
        override; validation runs through the shared parse_* helpers so the
        gateway and the CI gate accept exactly the same shapes."""
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        tools = raw.get("tools", {})
        gw = cls(erp=erp)
        for name, spec in tools.items():
            if name not in DEFAULT_TOOL_SPECS:
                raise ValueError(f"tools.yaml: unknown tool '{name}'")
            if not isinstance(spec, dict):
                raise ValueError(f"tools.yaml: '{name}' must be a mapping "
                                 "of declared keys")
            base = DEFAULT_TOOL_SPECS[name]
            kw: dict = {}
            if "params" in spec:
                kw["params"] = tuple(spec["params"])
            if "preconditions" in spec:
                kw["preconditions"] = tuple(spec["preconditions"])
            if "facts" in spec:
                kw["facts"] = parse_facts(spec["facts"],
                                          f"tools.yaml '{name}'")
            if "param_types" in spec:
                kw["param_types"] = parse_param_types(spec["param_types"],
                                                      f"tools.yaml '{name}'")
            if "param_bounds" in spec:
                kw["param_bounds"] = parse_param_bounds(spec["param_bounds"],
                                                        f"tools.yaml '{name}'")
            # Deployment-facing metadata keys also override when declared
            # (data beats defaults); anything undeclared stays base.
            if "description" in spec:
                kw["description"] = str(spec["description"])
            if "action" in spec:
                kw["action"] = str(spec["action"])
            if "side_effects" in spec:
                kw["side_effects"] = bool(spec["side_effects"])
            gw.specs[name] = replace(base, **kw)
        return gw

    def execute(self, tool_name: str, params: dict,
                idempotency_key: str | None = None) -> ToolResult:
        spec = self.specs.get(tool_name)
        if spec is None:
            return ToolResult(ok=False, error=f"unknown_tool: {tool_name}")
        missing = [p for p in spec.params if params.get(p) is None]
        if missing:
            return ToolResult(ok=False, error=f"missing_params: {missing}")

        # Task D2: light param-type validation at the governed boundary —
        # AFTER the presence check, BEFORE idempotency/ERP/preconditions, so
        # a bad-typed param can never reach the backend (or poison the
        # idempotency cache). Safe coercions are applied on a copy; the
        # binding sees the coerced values.
        coerced = dict(params)
        for p in spec.params:
            ok, cv = _coerce_param(p, params.get(p),
                                   spec.param_types.get(p))
            if not ok:
                return ToolResult(ok=False, error=f"invalid_param: {p}")
            coerced[p] = cv
        # declared numeric bounds (e.g. rating 1..10) — after coercion
        for p, (lo, hi) in (spec.param_bounds or {}).items():
            v = coerced.get(p)
            if isinstance(v, (int, float)) and not (lo <= v <= hi):
                return ToolResult(
                    ok=False,
                    error=f"out_of_range: {p}={v} (expected {lo}..{hi})")
        params = coerced

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
            elif tool_name == "end_call":
                value = {"call_ended": True,
                         "reason": params.get("reason", "resolved")}
            elif tool_name == "record_feedback":
                value = {"feedback_recorded": True,
                         "rating": params["rating"],
                         "comment": params.get("comment", "")}
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

