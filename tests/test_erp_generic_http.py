# tests/test_erp_generic_http.py — the any-org transport contract.
"""GenericHttpERP (GenericBackend over HTTP) + the erp_server generic
routes: any backend speaking GET /resources/... + POST /operations/...
works, whatever its domain nouns. Client shapes are pinned against a stub
server (mirroring test_erp_http.py); the live round-trip runs against the
REAL erp_server on a temp DB — including through EcommerceAdapter, which
proves the bridge runs both directions (classic surface over generic
transport)."""
from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from voiceagent.config import load_config
from voiceagent.erp_http import ErpHttpError, GenericHttpERP
from voiceagent.generic_backend import EcommerceAdapter, GenericBackendError

ROOT = Path(__file__).resolve().parents[1]


# --- stub server: request shapes ------------------------------------------------

class _Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.method = None
        self.path = None
        self.next_status = 200
        self.next_payload: object = {"ok": True}

    def respond(self, status: int, payload: object) -> None:
        with self.lock:
            self.next_status = status
            self.next_payload = payload


_rec = _Recorder()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _rr(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        with _rec.lock:
            _rec.method, _rec.path = self.command, self.path
            status, payload = _rec.next_status, _rec.next_payload
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = _rr


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _rec.respond(200, {"ok": True})
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def erp(server):
    return GenericHttpERP(load_config(env={"VOICEAGENT_ERP_URL": server}))


def _last():
    with _rec.lock:
        return _rec.method, _rec.path


def test_get_resource_shape(erp):
    _rec.respond(200, {"booking_id": "BK-1", "status": "BOOKED"})
    assert erp.get_resource("booking", "BK-1") == {
        "booking_id": "BK-1", "status": "BOOKED"}
    assert _last() == ("GET", "/resources/booking/BK-1")


def test_get_resource_404_is_none(erp):
    _rec.respond(404, {"error": "not found"})
    assert erp.get_resource("booking", "BK-NOPE") is None


def test_list_resources_phone_filter(erp):
    _rec.respond(200, [{"booking_id": "BK-1"}])
    assert erp.list_resources("booking", {"phone": "+91-98"}) == [
        {"booking_id": "BK-1"}]
    method, path = _last()
    assert method == "GET" and path.startswith("/resources/booking?")
    assert "phone=91" in path


def test_list_resources_rejects_non_phone_filters(erp):
    with pytest.raises(GenericBackendError):
        erp.list_resources("booking", {"customer_id": "C-1"})
    with pytest.raises(GenericBackendError):
        erp.list_resources("booking", {})


def test_execute_operation_shape(erp):
    _rec.respond(200, {"status": "CANCELLED"})
    assert erp.execute_operation(
        "cancel_booking",
        {"booking_id": "BK-1", "reason": "x"}) == {"status": "CANCELLED"}
    assert _last() == ("POST", "/operations/cancel_booking")


def test_create_update_have_no_http_verb(erp):
    with pytest.raises(GenericBackendError):
        erp.create_resource("booking", {})
    with pytest.raises(GenericBackendError):
        erp.update_resource("booking", "BK-1", {})


def test_transport_failure_is_governed(erp):
    _rec.respond(500, {"error": "boom"})
    with pytest.raises(ErpHttpError):
        erp.get_resource("booking", "BK-1")


# --- live round-trip against the real server ------------------------------------

def _load_server_module(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "erp_server_live", str(ROOT / "scripts" / "erp_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DB_PATH = tmp_path / "erp.sqlite"
    return mod


@pytest.fixture()
def live_url(tmp_path):
    mod = _load_server_module(tmp_path)
    conn = mod._connect(mod.DB_PATH)
    try:
        mod._seed_if_empty(conn)
    finally:
        conn.close()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_live_generic_round_trip(live_url):
    erp = GenericHttpERP(load_config(env={"VOICEAGENT_ERP_URL": live_url}))
    order = erp.get_resource("order", "ORD-4821")
    assert order["status"] == "CONFIRMED"
    assert erp.get_resource("order", "ORD-NOPE") is None
    hits = erp.list_resources("order", {"phone": "9876543210"})
    assert {o["order_id"] for o in hits} == {"ORD-4821", "ORD-7734"}
    cancelled = erp.execute_operation(
        "cancel_order", {"order_id": "ORD-4821", "reason": "live test"})
    assert cancelled["status"] == "CANCELLED"
    assert erp.get_resource("order", "ord4821")["status"] == "CANCELLED"
    with pytest.raises(ErpHttpError):
        erp.execute_operation("no_such_op", {})


def test_live_legacy_routes_unchanged(live_url):
    # The refactor shares mutation helpers — legacy shapes must be identical.
    from voiceagent.erp_http import HttpERP
    classic = HttpERP(load_config(env={"VOICEAGENT_ERP_URL": live_url}))
    assert classic.get_order("ORD-7734")["status"] == "SHIPPED"
    out = classic.cancel_order("ORD-7734", "live test")
    assert out["status"] == "CANCELLED" and out["cancel_reason"] == "live test"


def test_live_classic_surface_over_generic_transport(live_url):
    # EcommerceAdapter(GenericHttpERP): the classic order verbs execute over
    # the domain-neutral routes — the bridge runs both directions.
    adapter = EcommerceAdapter(
        GenericHttpERP(load_config(env={"VOICEAGENT_ERP_URL": live_url})))
    assert adapter.get_resource("order", "ORD-9021")["status"] == "CONFIRMED"
    out = adapter.execute_operation(
        "cancel_order", {"order_id": "ORD-9021", "reason": "bridge test"})
    assert out["status"] == "CANCELLED"
