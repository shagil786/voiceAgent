# ERP HTTP adapter — API contract (`erp_http.HttpERP`)

The real-ERP adapter behind the platform's `SupportBackend` surface
(ADR-003: declarations are data, bindings are code — the ToolGateway's
tool->ERP bindings stay Python; a deployment supplies a real backend
object). `MockERP` remains the offline demo default; `HttpERP` is opt-in:

```python
from voiceagent.config import load_config
from voiceagent.erp_http import HttpERP
from voiceagent.tools import ToolGateway

cfg = load_config()                      # reads VOICEAGENT_ERP_* env vars
gw = ToolGateway(erp=HttpERP(cfg))       # erp=None keeps MockERP
```

## Configuration

| Env var | Meaning | Default |
|---|---|---|
| `VOICEAGENT_ERP_URL` | ERP base URL (e.g. `https://erp.example.com/api/v1`) | none — `HttpERP` refuses to build without it |
| `VOICEAGENT_ERP_TOKEN` | Static bearer token; `Authorization: Bearer <token>` on every request | none — header omitted |
| `VOICEAGENT_ERP_TIMEOUT` | Per-call timeout, seconds | 5.0 |

## Endpoint mapping

| SupportBackend method | HTTP | Path | Body |
|---|---|---|---|
| `get_order` | GET | `/orders/{id}` | — |
| `orders_for_customer` | GET | `/customers/{id}/orders` | — |
| `cancel_order` | POST | `/orders/{id}/cancel` | `{"reason": ...}` |
| `reschedule_delivery` | POST | `/orders/{id}/reschedule` | `{"new_date": ...}` |
| `initiate_refund` | POST | `/orders/{id}/refund` | `{"amount": ..., "reason": ...}` |
| `mark_return` | POST | `/orders/{id}/return` | `{"reason": ...}` |
| `record_handoff` | POST | `/handoffs` | `{"reason": ...}` |
| `lookup_orders_by_phone` | GET | `/orders?phone={digits}` | — |

Path style: lowercase, hyphen-free; path ids are percent-encoded single
segments; phone numbers are reduced to digits before the query string.

## Failure contract (fail-closed)

`ErpHttpError` (subclasses `TimeoutError`) is raised on: connection
refused / DNS failure / timeout, any HTTP >= 400 except `404 on GET`
(reads treat not-found as a normal answer so the gateway's not-found
ladder works; a 404 on a mutating POST is still an error), non-JSON or
empty body, and payload-contract violations (wrong JSON shape). The
ToolGateway catches `TimeoutError` and degrades to its governed
`backend_timeout` path — graceful, ticket-issuing, never idempotency-
cached. The adapter NEVER fabricates a success value.

Responses expected: `get_order`/mutations -> JSON object;
`orders_for_customer` -> JSON array of ids (coerced str);
`lookup_orders_by_phone` -> JSON array of objects.

## Not built (forward work)

Retry/backoff, pagination, token refresh / OAuth (static bearer only),
bulk endpoints, webhook-driven ERP event ingestion.
