# src/voiceagent/generic_backend.py — resource-verb backend protocol (ADR-004).
"""The domain-agnostic backend surface for NEW domains (ADR-004).

`SupportBackend` (tools.py) names domain verbs — get_order, cancel_order —
which forces every non-e-commerce domain to map its nouns to order-words
(the clinic maps appointments->orders). `GenericBackend` is the forward
protocol for new domains: resource TYPES are data ("orders", "appointments",
"tickets"), the verbs are resource-shaped (get/list/create/update/
execute_operation), and a backend that CAN describe its lifecycle exposes
it (`get_lifecycle_states`) — preconditions keep coming from ToolSpecs/
tools.yaml either way.

Contract (mirrors SupportBackend's discipline):
- Structural protocol: duck-typed, no inheritance required.
- Data-in/data-out: dicts in, dicts/lists out, None for absent resources;
  failures RAISE (TimeoutError-compatible) so the ToolGateway's governed
  timeout path handles them — never fabricate an ok result.
- Tenants never implement this themselves; a platform-side CODE AUTHOR
  writes the adapter per domain (ADR-003 holds).

`EcommerceAdapter` is the bridge proof: it implements GenericBackend over
any wrapped SupportBackend (MockERP, HttpERP, the clinic backend), so
existing deployments keep working while new code can target GenericBackend.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GenericBackend(Protocol):
    """Resource-verb backend surface (ADR-004). All dicts are JSON-safe."""

    def get_resource(self, resource_type: str, resource_id: str) -> dict | None: ...

    def list_resources(self, resource_type: str,
                       filters: dict[str, Any] | None = None) -> list[dict]: ...

    def create_resource(self, resource_type: str, data: dict) -> dict: ...

    def update_resource(self, resource_type: str, resource_id: str,
                        data: dict) -> dict: ...

    def execute_operation(self, operation_name: str,
                          params: dict) -> dict: ...

    def get_lifecycle_states(self, resource_type: str) -> list[str]:
        """States this backend's lifecycle for `resource_type` moves through
        (e.g. ["CONFIRMED", "SHIPPED", "DELIVERED"]). Backends that cannot
        describe one raise NotImplementedError — preconditions then stay
        fully with the ToolSpecs (the historical model)."""
        ...


class GenericBackendError(TimeoutError):
    """GenericBackend failure. Subclasses TimeoutError deliberately, exactly
    like erp_http.ErpHttpError: the ToolGateway's governed backend_timeout
    path catches TimeoutError and only TimeoutError — every adapter failure
    degrades the same graceful, ticket-issuing way. Never raise anything
    else from an adapter; never let a failure masquerade as a value."""


class EcommerceAdapter:
    """GenericBackend over a wrapped SupportBackend — the ADR-004 bridge.

    Maps resource verbs to the classic order-verb surface:
        get_resource("order", id)          -> get_order(id)
        list_resources("order", {phone})   -> lookup_orders_by_phone(phone)
        list_resources("order", {customer_id}) -> orders_for_customer(cid)
        execute_operation("cancel_order", ...) -> cancel_order(...)
        execute_operation("reschedule_delivery", ...) -> reschedule_delivery(...)
        execute_operation("initiate_refund", ...)  -> initiate_refund(...)
        execute_operation("mark_return", ...)      -> mark_return(...)
        execute_operation("record_handoff", ...)   -> record_handoff(...)

    Unknown resource types / operations raise GenericBackendError (the
    adapter is explicit about what its wrapped backend supports — never a
    silent None). get_lifecycle_states raises NotImplementedError: the
    wrapped SupportBackend has no lifecycle introspection; preconditions
    stay with the ToolSpecs (the historical model)."""

    RESOURCE_ORDER = "order"

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    # --- resource verbs -----------------------------------------------------

    def get_resource(self, resource_type: str, resource_id: str) -> dict | None:
        if resource_type != self.RESOURCE_ORDER:
            raise GenericBackendError(
                f"unsupported resource_type {resource_type!r} for the "
                f"e-commerce adapter (expected 'order')")
        return self._backend.get_order(resource_id)

    def list_resources(self, resource_type: str,
                       filters: dict[str, Any] | None = None) -> list[dict]:
        if resource_type != self.RESOURCE_ORDER:
            raise GenericBackendError(
                f"unsupported resource_type {resource_type!r} for the "
                f"e-commerce adapter (expected 'order')")
        filters = filters or {}
        phone = filters.get("phone")
        if phone is not None:
            return self._backend.lookup_orders_by_phone(phone)
        customer_id = filters.get("customer_id")
        if customer_id is not None:
            return [self._backend.get_order(oid)
                    for oid in self._backend.orders_for_customer(customer_id)]
        raise GenericBackendError(
            "list_resources('order') requires a 'phone' or 'customer_id' "
            "filter — unbounded order listing is not a governed operation")

    def create_resource(self, resource_type: str, data: dict) -> dict:
        raise GenericBackendError(
            "create_resource is not supported by the e-commerce adapter: "
            "orders are created by the merchant's own funnel, not the "
            "support agent (governed surface = read/cancel/reschedule/"
            "refund/return/handoff)")

    def update_resource(self, resource_type: str, resource_id: str,
                        data: dict) -> dict:
        raise GenericBackendError(
            "update_resource is not supported by the e-commerce adapter: "
            "order mutations go through governed operations (cancel/"
            "reschedule), not free-form field writes")

    def execute_operation(self, operation_name: str, params: dict) -> dict:
        params = params or {}
        if operation_name == "cancel_order":
            return self._backend.cancel_order(
                params["order_id"], params["reason"])
        if operation_name == "reschedule_delivery":
            return self._backend.reschedule_delivery(
                params["order_id"], params["new_date"])
        if operation_name == "initiate_refund":
            return self._backend.initiate_refund(
                params["order_id"], float(params["amount"]),
                params["reason"])
        if operation_name == "mark_return":
            return self._backend.mark_return(
                params["order_id"], params["reason"])
        if operation_name == "record_handoff":
            return self._backend.record_handoff(params["reason"])
        raise GenericBackendError(
            f"unsupported operation {operation_name!r} for the e-commerce "
            f"adapter (cancel_order, reschedule_delivery, initiate_refund, "
            f"mark_return, record_handoff)")

    def get_lifecycle_states(self, resource_type: str) -> list[str]:
        raise NotImplementedError(
            "the wrapped SupportBackend has no lifecycle introspection; "
            "preconditions stay with the ToolSpecs (tools.yaml overrides)")
