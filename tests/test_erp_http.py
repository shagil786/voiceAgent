# tests/test_erp_http.py — HttpERP contract tests (offline, stdlib fixture).
"""Every test runs against a local threaded http.server — no live network.
Covers: the 8-method request shape, auth header, timeout/error mapping,
JSON contract violations, config resolution, and the ToolGateway seams
(erp=HttpERP accepted; erp=None still defaults to MockERP)."""
from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voiceagent.config import load_config
from voiceagent.erp_http import ErpHttpError, HttpERP
from voiceagent.tools import MockERP, ToolGateway


# --- fixture server -----------------------------------------------------------

class _Recorder:
    """Captures the last request; scripted response for the next one."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.method = None
        self.path = None
        self.headers = None
        self.body = None
        self.next_status = 200
        self.next_payload: object = {"ok": True}
        self.next_raw: bytes | None = None

    def respond(self, status: int, payload: object) -> None:
        with self.lock:
            self.next_status = status
            self.next_payload = payload
            self.next_raw = None


_rec = _Recorder()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request stderr
        pass

    def _record_and_respond(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        with _rec.lock:
            _rec.method = self.command
            _rec.path = self.path
            _rec.headers = dict(self.headers.items())
            _rec.body = body
            status, payload, raw = (_rec.next_status, _rec.next_payload,
                                    _rec.next_raw)
        if raw is not None:
            data = raw
        elif payload is None:
            data = b""
        else:
            data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    do_GET = do_POST = _record_and_respond


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _rec.respond(200, {"ok": True})
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def erp(server):
    cfg = load_config(env={"VOICEAGENT_ERP_URL": server,
                           "VOICEAGENT_ERP_TOKEN": "tok-123"})
    return HttpERP(cfg)


def _last():
    with _rec.lock:
        return (_rec.method, _rec.path, _rec.headers, _rec.body)


# --- request shapes ------------------------------------------------------------

def test_get_order_shape_and_payload(erp):
    _rec.respond(200, {"order_id": "ORD-4821", "status": "CONFIRMED"})
    got = erp.get_order("ORD-4821")
    method, path, headers, body = _last()
    assert (method, path) == ("GET", "/orders/ORD-4821")
    assert body is None
    assert got == {"order_id": "ORD-4821", "status": "CONFIRMED"}


def test_auth_header_present(erp):
    _rec.respond(200, {"ok": True})
    erp.get_order("X1")
    _, _, headers, _ = _last()
    assert headers["Authorization"] == "Bearer tok-123"


def test_no_token_omits_auth_header(server):
    erp = HttpERP(load_config(env={"VOICEAGENT_ERP_URL": server}))
    _rec.respond(200, {"ok": True})
    erp.record_handoff("r")
    _, _, headers, _ = _last()
    assert "Authorization" not in headers


def test_orders_for_customer_parses_ids(erp):
    _rec.respond(200, ["ORD-1", "ORD-2", 7])
    assert erp.orders_for_customer("CUST-9") == ["ORD-1", "ORD-2", "7"]
    method, path, _, _ = _last()
    assert (method, path) == ("GET", "/customers/CUST-9/orders")


def test_lookup_by_phone_digits_and_query(erp):
    _rec.respond(200, [{"order_id": "ORD-1"}])
    out = erp.lookup_orders_by_phone("+91-98765 43210")
    method, path, _, _ = _last()
    assert method == "GET"
    qs = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
    assert qs["phone"] == ["919876543210"]  # digits only, '+91-98765 43210'
    assert out == [{"order_id": "ORD-1"}]


def test_cancel_order_posts_json_body(erp):
    _rec.respond(200, {"cancelled": True})
    out = erp.cancel_order("ORD-4821", "changed mind")
    method, path, headers, body = _last()
    assert (method, path) == ("POST", "/orders/ORD-4821/cancel")
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body) == {"reason": "changed mind"}
    assert out == {"cancelled": True}


def test_reschedule_posts_new_date(erp):
    _rec.respond(200, {"rescheduled": True})
    erp.reschedule_delivery("ORD-1", "2026-09-10")
    method, path, _, body = _last()
    assert (method, path) == ("POST", "/orders/ORD-1/reschedule")
    assert json.loads(body) == {"new_date": "2026-09-10"}


def test_refund_posts_amount_and_reason(erp):
    _rec.respond(200, {"refund_id": "RF-1"})
    erp.initiate_refund("ORD-1", 499.0, "damaged")
    _, path, _, body = _last()
    assert path == "/orders/ORD-1/refund"
    assert json.loads(body) == {"amount": 499.0, "reason": "damaged"}


def test_mark_return_and_handoff_paths(erp):
    _rec.respond(200, {"ok": True})
    erp.mark_return("ORD-2", "size")
    _, path, _, body = _last()
    assert (path, json.loads(body)) == ("/orders/ORD-2/return",
                                        {"reason": "size"})
    erp.record_handoff("angry caller")
    method, path, _, body = _last()
    assert (method, path, json.loads(body)) == ("POST", "/handoffs",
                                                {"reason": "angry caller"})


def test_order_id_percent_encoded(erp):
    _rec.respond(200, {"order_id": "ORD 1/2"})
    erp.get_order("ORD 1/2")
    _, path, _, _ = _last()
    assert "/" not in path[len("/orders/"):]  # single segment, encoded


# --- 404 / errors / contract violations -----------------------------------------

def test_get_order_404_is_none_not_error(erp):
    _rec.respond(404, {"detail": "nope"})
    assert erp.get_order("ORD-MISSING") is None


def test_mutate_404_is_error(erp):
    _rec.respond(404, {"detail": "nope"})
    with pytest.raises(ErpHttpError) as ei:
        erp.cancel_order("ORD-X", "r")
    assert ei.value.status == 404


def test_http_500_raises(erp):
    _rec.respond(500, {"detail": "boom"})
    with pytest.raises(ErpHttpError) as ei:
        erp.get_order("ORD-1")
    assert ei.value.status == 500


def test_non_json_body_raises(erp):
    _rec.respond(200, None, )
    with _rec.lock:
        _rec.next_raw = b"<html>not json</html>"
    with pytest.raises(ErpHttpError):
        erp.get_order("ORD-1")


def test_empty_body_raises(erp):
    with _rec.lock:
        _rec.next_raw = b""
    with pytest.raises(ErpHttpError):
        erp.get_order("ORD-1")


def test_contract_violation_non_object_get_order(erp):
    _rec.respond(200, ["not", "an", "object"])
    with pytest.raises(ErpHttpError):
        erp.get_order("ORD-1")


def test_contract_violation_non_array_lookup(erp):
    _rec.respond(200, {"unexpected": "object"})
    with pytest.raises(ErpHttpError):
        erp.lookup_orders_by_phone("123")


def test_lookup_entry_not_object_raises(erp):
    _rec.respond(200, ["scalar"])
    with pytest.raises(ErpHttpError):
        erp.lookup_orders_by_phone("123")


def test_transport_failure_connection_refused():
    # Port 1 on loopback: nothing listens; must raise ErpHttpError (URLError).
    erp = HttpERP(load_config(env={"VOICEAGENT_ERP_URL": "http://127.0.0.1:1"}))
    with pytest.raises(ErpHttpError, match="transport failure"):
        erp.get_order("ORD-1")


def test_timeout_raises_erp_error(server, monkeypatch):
    erp = HttpERP(load_config(env={"VOICEAGENT_ERP_URL": server,
                                   "VOICEAGENT_ERP_TIMEOUT": "0.01"}))
    _rec.respond(200, {"ok": True})

    class _Slow:
        def __enter__(self):
            import time
            time.sleep(0.2)
            raise AssertionError("should not complete")

        def __exit__(self, *a):
            return False

    import urllib.request as _ur
    real = _ur.urlopen

    def slow_urlopen(req, timeout=None):
        import time
        time.sleep(0.2)
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(_ur, "urlopen", slow_urlopen)
    with pytest.raises(ErpHttpError, match="transport failure"):
        erp.get_order("ORD-1")
    monkeypatch.setattr(_ur, "urlopen", real)


# --- config seams -----------------------------------------------------------------

def test_config_env_resolution():
    c = load_config(env={"VOICEAGENT_ERP_URL": "http://e",
                         "VOICEAGENT_ERP_TOKEN": "t",
                         "VOICEAGENT_ERP_TIMEOUT": "7.5"})
    assert (c.erp_url, c.erp_token, c.erp_timeout) == ("http://e", "t", 7.5)


def test_config_defaults_when_unset():
    c = load_config(env={})
    assert c.erp_url is None and c.erp_token is None
    assert c.erp_timeout == 5.0


def test_config_bad_timeout_falls_back():
    c = load_config(env={"VOICEAGENT_ERP_TIMEOUT": "soon"})
    assert c.erp_timeout == 5.0


def test_hfterp_fails_fast_without_url():
    with pytest.raises(ValueError, match="VOICEAGENT_ERP_URL"):
        HttpERP(env={})


# --- ToolGateway seams -------------------------------------------------------------

def test_gateway_accepts_http_erp(server):
    erp = HttpERP(load_config(env={"VOICEAGENT_ERP_URL": server,
                                   "VOICEAGENT_ERP_TOKEN": "t"}))
    _rec.respond(200, {"order_id": "ORD-4821", "status": "CONFIRMED"})
    gw = ToolGateway(erp=erp)
    res = gw.execute("fetch_order_status", {"order_id": "ORD-4821"})
    assert res.ok and res.value["order_id"] == "ORD-4821"


def test_gateway_http_failure_is_governed_backend_timeout(server):
    erp = HttpERP(load_config(env={"VOICEAGENT_ERP_URL": server}))
    _rec.respond(500, {"detail": "boom"})
    gw = ToolGateway(erp=erp)
    res = gw.execute("fetch_order_status", {"order_id": "ORD-1"})
    assert not res.ok
    assert res.error.startswith("backend_timeout")


def test_gateway_default_still_mockerp():
    gw = ToolGateway(erp=None)
    assert isinstance(gw.erp, MockERP)
