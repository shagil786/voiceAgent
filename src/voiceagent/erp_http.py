# src/voiceagent/erp_http.py — real ERP adapter: HTTP connector.
"""The production SupportBackend: an HTTP ERP/CRM connector.

ADR-003's contract — "declarations are data, bindings are code" — means the
ToolGateway's tool->ERP bindings stay Python; a real deployment supplies a
REAL backend object that structurally satisfies SupportBackend. This module
is that backend: the same 8-method surface MockERP satisfies in-memory,
marshaled over HTTP to a tenant's ERP/CRM service.

Design contract
---------------
Transport    stdlib urllib.request (the repo has no `requests` dependency;
             importing this module stays zero-heavy-deps).
Auth         static bearer token via VOICEAGENT_ERP_TOKEN (Authorization:
             Bearer <token> header on every request). Token refresh, OAuth
             flows: not built (forward work).
Timeouts     every call bounded by VOICEAGENT_ERP_TIMEOUT seconds (default
             5.0). urllib raises socket.timeout / URLError; both map to
             ErpHttpError.
Failure      FAIL-CLOSED: connection refused, DNS failure, timeout, HTTP
             >= 400, non-JSON body, or a payload violating the method's
             contract all raise ErpHttpError. ErpHttpError SUBCLASSES
             TimeoutError, so the ToolGateway's existing governed timeout
             path (tools.py backend_timeout, graceful ticket — NOT cached
             idempotency) handles it with zero platform changes. This
             adapter NEVER fabricates an ok result and NEVER returns a
             dict for a failed call.

Method -> REST mapping (path style: lowercase, hyphen-free)
-----------------------------------------------------------
    get_order               GET    /orders/{id}
    orders_for_customer     GET    /customers/{id}/orders
    cancel_order            POST   /orders/{id}/cancel        {reason}
    reschedule_delivery     POST   /orders/{id}/reschedule    {new_date}
    initiate_refund         POST   /orders/{id}/refund        {amount, reason}
    mark_return             POST   /orders/{id}/return        {reason}
    record_handoff          POST   /handoffs                  {reason}
    lookup_orders_by_phone  GET    /orders?phone={digits}

Response normalization (documented contract)
--------------------------------------------
    get_order:              200 + JSON object -> dict; 404 -> None
                            (order-not-found is a NORMAL answer — the
                            gateway's not-found ladder uses it; only
                            transport/server failures raise).
    orders_for_customer:    200 + JSON array -> list[str] of order ids
                            (entries coerced str; non-array 200 is an error).
    lookup_orders_by_phone: 200 + JSON array of order objects -> list[dict].
    Mutating methods:       200 + JSON object -> dict (the tool's value,
                            verbatim).
Order-id / phone normalization: order ids are percent-encoded as given (the
backend owns id format); phones are reduced to digits before the query
string (MockERP's lookup semantics).

Config (env > code defaults, via config.load_config):
    VOICEAGENT_ERP_URL      base URL, e.g. https://erp.example.com/api/v1
    VOICEAGENT_ERP_TOKEN    bearer token (optional; header omitted when None)
    VOICEAGENT_ERP_TIMEOUT  seconds, default 5.0

Instantiation is explicit and opt-in: ToolGateway(erp=HttpERP(cfg)) —
erp=None keeps the MockERP offline default. One instance per server start;
instances are stateless apart from read-only config.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from voiceagent.config import RuntimeConfig

__all__ = ["ErpHttpError", "HttpERP"]


class ErpHttpError(TimeoutError):
    """ERP HTTP failure (transport, HTTP >= 400, or contract violation).

    Subclasses TimeoutError deliberately: the ToolGateway's governed
    backend_timeout path catches TimeoutError — and ONLY TimeoutError — so
    every failure of this adapter degrades the same graceful, ticket-issuing
    way MockERP's injected timeouts do. Never raise anything else from the
    adapter; never let a failure masquerade as a successful value."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status  # HTTP status when the failure was an HTTP one


_DEFAULT_TIMEOUT_S = 5.0
_USER_AGENT = "voiceagent-erp-adapter/1.0"


def _digits(s: Any) -> str:
    """Phone -> digits only (MockERP's lookup semantics)."""
    return "".join(ch for ch in str(s) if ch.isdigit())


def _id_path(value: Any) -> str:
    """Order/customer id -> single URL path segment (percent-encoded)."""
    return urllib.parse.quote(str(value), safe="")


class HttpERP:
    """SupportBackend over HTTP. Opt-in: pass erp=HttpERP(...) explicitly;
    ToolGateway(erp=None) keeps the MockERP offline default."""

    def __init__(self, config: RuntimeConfig | None = None, *,
                 env: dict[str, str] | None = None) -> None:
        cfg = config if config is not None else _config_from_env(env)
        if not getattr(cfg, "erp_url", None):
            raise ValueError(
                "HttpERP requires config.erp_url (VOICEAGENT_ERP_URL) — "
                "refusing to build an adapter with no backend")
        self.base_url = str(cfg.erp_url).rstrip("/")
        self.token = getattr(cfg, "erp_token", None) or None
        try:
            self.timeout_s = float(getattr(cfg, "erp_timeout", None)
                                   or _DEFAULT_TIMEOUT_S)
        except (TypeError, ValueError):
            self.timeout_s = _DEFAULT_TIMEOUT_S

    # --- transport ----------------------------------------------------------

    def _request(self, method: str, path: str, *, query: dict | None = None,
                 body: dict | None = None) -> Any:
        """One HTTP call -> decoded JSON. Any failure raises ErpHttpError
        (never a sentinel); a 404 on a GET returns None — not-found is a
        normal read answer (the gateway's not-found ladder), while a 404 on
        a mutating call is still a failure."""
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        data = (json.dumps(body).encode("utf-8") if body is not None
                else None)
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _USER_AGENT)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                status = resp.status
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and method == "GET" and body is None:
                return None
            raise ErpHttpError(
                f"erp http {exc.code} on {method} {path}",
                status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # URLError wraps connection refused / DNS / SSL; socket.timeout
            # arrives as TimeoutError (py3.10+) or OSError subclass.
            raise ErpHttpError(
                f"erp transport failure on {method} {path}: "
                f"{type(exc).__name__}") from exc
        if status < 200 or status >= 300:
            raise ErpHttpError(
                f"erp http {status} on {method} {path}", status=status)
        if not raw:
            raise ErpHttpError(f"erp empty body on {method} {path}")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ErpHttpError(
                f"erp non-JSON body on {method} {path}") from exc

    # --- SupportBackend surface ---------------------------------------------

    def get_order(self, order_id: str) -> dict | None:
        payload = self._request("GET", f"/orders/{_id_path(order_id)}")
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ErpHttpError(
                f"erp contract violation: get_order expected object, got "
                f"{type(payload).__name__}")
        return payload

    def orders_for_customer(self, customer_id: str) -> list[str]:
        payload = self._request(
            "GET", f"/customers/{_id_path(customer_id)}/orders")
        if not isinstance(payload, list):
            raise ErpHttpError(
                f"erp contract violation: orders_for_customer expected "
                f"array, got {type(payload).__name__}")
        return [str(entry) for entry in payload]

    def cancel_order(self, order_id: str, reason: str) -> dict:
        return self._mutate(
            "POST", f"/orders/{_id_path(order_id)}/cancel", {"reason": reason})

    def reschedule_delivery(self, order_id: str, new_date: str) -> dict:
        return self._mutate(
            "POST", f"/orders/{_id_path(order_id)}/reschedule",
            {"new_date": new_date})

    def initiate_refund(self, order_id: str, amount: float,
                        reason: str) -> dict:
        return self._mutate(
            "POST", f"/orders/{_id_path(order_id)}/refund",
            {"amount": amount, "reason": reason})

    def mark_return(self, order_id: str, reason: str) -> dict:
        return self._mutate(
            "POST", f"/orders/{_id_path(order_id)}/return",
            {"reason": reason})

    def record_handoff(self, reason: str) -> dict:
        return self._mutate("POST", "/handoffs", {"reason": reason})

    def lookup_orders_by_phone(self, phone: str) -> list[dict]:
        payload = self._request("GET", "/orders",
                                query={"phone": _digits(phone)})
        if not isinstance(payload, list):
            raise ErpHttpError(
                f"erp contract violation: lookup_orders_by_phone expected "
                f"array, got {type(payload).__name__}")
        out: list[dict] = []
        for entry in payload:
            if not isinstance(entry, dict):
                raise ErpHttpError(
                    "erp contract violation: lookup_orders_by_phone entries "
                    "must be objects")
            out.append(entry)
        return out

    # --- helpers --------------------------------------------------------------

    def _mutate(self, method: str, path: str, body: dict) -> dict:
        """Mutating call: success must yield a JSON object (the tool's
        value); anything else is a contract violation / failure."""
        payload = self._request(method, path, body=body)
        if not isinstance(payload, dict):
            raise ErpHttpError(
                f"erp contract violation: {method} {path} expected object, "
                f"got {type(payload).__name__}")
        return payload


def _config_from_env(env: dict[str, str] | None) -> RuntimeConfig:
    """Resolve the ERP config slice through the standard loader (injectable
    env keeps tests off os.environ; None reads os.environ)."""
    from voiceagent.config import load_config
    return load_config(env=env)
