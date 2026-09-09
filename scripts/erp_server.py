#!/usr/bin/env python3
"""scripts/erp_server.py — real ERP HTTP service (stdlib + SQLite).

Serves the SupportBackend HTTP surface that voiceagent.erp_http.HttpERP
calls: GET /orders/{id}, GET /customers/{id}/orders, GET /orders?phone=...,
POST /orders/{id}/cancel|reschedule|refund|return, POST /handoffs.

Storage is SQLite (data/erp/erp.sqlite, gitignored) — real persistence,
real query semantics, survives restarts. On first boot the store seeds
from the COMMITTED tenant fixtures (data/tenants/*/erp_fixture.json):
that is deployment DATA (versioned, reviewable), not agent code, and not
an in-process mock. Production deployments never run this service — they
point VOICEAGENT_ERP_URL at their own ERP and the agent code is unchanged.

Run:  .venv/bin/python scripts/erp_server.py [--port 8090]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "erp" / "erp.sqlite"
FIXTURE_PATTERN = ROOT / "data" / "tenants" / "*" / "erp_fixture.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY, customer_id TEXT, status TEXT, amount REAL,
  items TEXT, delivery_date TEXT, address TEXT, tracking_url TEXT,
  cancel_reason TEXT, return_reason TEXT);
CREATE TABLE IF NOT EXISTS customers (
  customer_id TEXT PRIMARY KEY, name TEXT, phone TEXT);
CREATE TABLE IF NOT EXISTS customer_orders (
  customer_id TEXT, order_id TEXT);
CREATE TABLE IF NOT EXISTS refunds (
  refund_id TEXT PRIMARY KEY, order_id TEXT, amount REAL, reason TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS handoffs (
  handoff_id TEXT PRIMARY KEY, reason TEXT, status TEXT, ts TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _row_to_order(r: sqlite3.Row) -> dict:
    d = {k: r[k] for k in r.keys()}
    try:
        d["items"] = json.loads(d.get("items") or "[]")
    except Exception:
        d["items"] = []
    for drop in ("cancel_reason", "return_reason", "tracking_url", "address"):
        if not d.get(drop):
            d.pop(drop, None)
    return d


def _seed_if_empty(conn: sqlite3.Connection) -> None:
    n = conn.execute("SELECT COUNT(*) AS c FROM orders").fetchone()["c"]
    if n:
        return
    loaded = 0
    for p in sorted(ROOT.glob("data/tenants/*/erp_fixture.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"erp seed: skipping {p}: {e}", file=sys.stderr)
            continue
        for cid, c in (data.get("customers") or {}).items():
            conn.execute("INSERT OR IGNORE INTO customers "
                         "(customer_id, name, phone) VALUES (?, ?, ?)",
                         (cid, c.get("name"), c.get("phone")))
            for oid in c.get("orders") or []:
                conn.execute("INSERT OR IGNORE INTO customer_orders "
                             "(customer_id, order_id) VALUES (?, ?)",
                             (cid, oid))
        for oid, o in (data.get("orders") or {}).items():
            conn.execute(
                "INSERT OR IGNORE INTO orders (order_id, customer_id, status,"
                " amount, items, delivery_date, address, tracking_url)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (oid, o.get("customer_id"), o.get("status"), o.get("amount"),
                 json.dumps(o.get("items") or []), o.get("delivery_date"),
                 o.get("address"), o.get("tracking_url")))
        loaded += 1
    conn.commit()
    print(f"erp seed: loaded {loaded} fixture file(s) into {DB_PATH}")


def _shape(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _find_order(conn: sqlite3.Connection, order_id: str) -> sqlite3.Row | None:
    row = conn.execute("SELECT * FROM orders WHERE order_id = ?",
                       (order_id,)).fetchone()
    if row is not None:
        return row
    want = _shape(order_id)
    if not want:
        return None
    for r in conn.execute("SELECT * FROM orders"):
        if _shape(r["order_id"]) == want:
            return r
    return None


def _orders_by_phone(conn: sqlite3.Connection, phone: str) -> list[dict]:
    want = "".join(ch for ch in str(phone) if ch.isdigit())
    if not want:
        return []
    out: list[dict] = []
    for c in conn.execute("SELECT * FROM customers"):
        have = "".join(ch for ch in str(c["phone"] or "") if ch.isdigit())
        if have and (have == want or have.endswith(want) or want.endswith(have)):
            for oid in conn.execute(
                    "SELECT order_id FROM customer_orders WHERE customer_id = ?",
                    (c["customer_id"],)):
                row = _find_order(conn, oid["order_id"])
                if row is not None:
                    out.append(_row_to_order(row))
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "voiceagent-erp/1"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args) -> None:
        print(f"[erp] {self.address_string()} - {fmt % args}", file=sys.stderr)

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        conn = _connect(DB_PATH)
        try:
            if parts and parts[0] == "health":
                return self._send(200, {"ok": True})
            if parts[:1] == ["orders"] and len(parts) == 1:
                phone = (query.get("phone") or [""])[0]
                return self._send(200, _orders_by_phone(conn, phone))
            if parts[:1] == ["orders"] and len(parts) == 2:
                row = _find_order(conn, parts[1])
                if row is None:
                    return self._send(404, {"error": "not found"})
                return self._send(200, _row_to_order(row))
            if parts[:1] == ["customers"] and len(parts) == 3 \
                    and parts[2] == "orders":
                rows = conn.execute(
                    "SELECT order_id FROM customer_orders WHERE customer_id = ?",
                    (parts[1],)).fetchall()
                return self._send(200, [r["order_id"] for r in rows])
            return self._send(404, {"error": f"no route {self.path}"})
        finally:
            conn.close()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        body = self._body()
        conn = _connect(DB_PATH)
        try:
            if parts[:1] == ["handoffs"] and len(parts) == 1:
                reason = body.get("reason") or ""
                hid = f"HO-{conn.execute('SELECT COUNT(*) AS c FROM handoffs').fetchone()['c'] + 1:04d}"
                conn.execute("INSERT INTO handoffs (handoff_id, reason, status, ts)"
                             " VALUES (?, ?, 'OPEN', ?)", (hid, reason, _now()))
                conn.commit()
                return self._send(200, {"handoff_id": hid, "reason": reason,
                                        "status": "OPEN"})
            if parts[:1] == ["orders"] and len(parts) == 2:
                action = None
            elif parts[:1] == ["orders"] and len(parts) == 3:
                action = parts[2]
            else:
                return self._send(404, {"error": f"no route {self.path}"})
            row = _find_order(conn, parts[1])
            if row is None:
                return self._send(404, {"error": "order not found"})
            oid = row["order_id"]
            if action == "cancel":
                reason = body.get("reason") or ""
                conn.execute("UPDATE orders SET status = 'CANCELLED',"
                             " cancel_reason = ? WHERE order_id = ?",
                             (reason, oid))
                conn.commit()
                return self._send(200, _row_to_order(
                    conn.execute("SELECT * FROM orders WHERE order_id = ?",
                                 (oid,)).fetchone()))
            if action == "reschedule":
                new_date = body.get("new_date") or body.get("delivery_date") or ""
                conn.execute("UPDATE orders SET delivery_date = ?"
                             " WHERE order_id = ?", (new_date, oid))
                conn.commit()
                return self._send(200, _row_to_order(
                    conn.execute("SELECT * FROM orders WHERE order_id = ?",
                                 (oid,)).fetchone()))
            if action == "refund":
                amount = body.get("amount") or 0.0
                reason = body.get("reason") or ""
                conn.execute("UPDATE orders SET status = 'REFUND_INITIATED'"
                             " WHERE order_id = ?", (oid,))
                rid = f"RF-{conn.execute('SELECT COUNT(*) AS c FROM refunds').fetchone()['c'] + 1:04d}"
                conn.execute("INSERT INTO refunds (refund_id, order_id, amount,"
                             " reason, ts) VALUES (?, ?, ?, ?, ?)",
                             (rid, oid, amount, reason, _now()))
                conn.commit()
                return self._send(200, {"order_id": oid, "amount": amount,
                                        "reason": reason, "refund_id": rid})
            if action == "return":
                reason = body.get("reason") or ""
                conn.execute("UPDATE orders SET status = 'RETURN_REQUESTED',"
                             " return_reason = ? WHERE order_id = ?",
                             (reason, oid))
                conn.commit()
                return self._send(200, _row_to_order(
                    conn.execute("SELECT * FROM orders WHERE order_id = ?",
                                 (oid,)).fetchone()))
            return self._send(404, {"error": f"no action {action}"})
        finally:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    conn = _connect(args.db)
    try:
        _seed_if_empty(conn)
    finally:
        conn.close()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"erp server on http://127.0.0.1:{args.port} (db: {args.db})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
