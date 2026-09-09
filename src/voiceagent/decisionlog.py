# src/voiceagent/decisionlog.py
from __future__ import annotations

import csv
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DecisionEntry:
    ts: str
    conv_id: str
    action: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    amount: float | None = None
    authenticated: bool = False


class DecisionLog:
    """Append-only audit trail of every policy decision the agent made."""

    def __init__(self) -> None:
        self._entries: list[DecisionEntry] = []

    def record(self, entry: DecisionEntry) -> None:
        self._entries.append(entry)

    def entries(self) -> list[DecisionEntry]:
        return list(self._entries)

    def query(self, action: str | None = None, verdict: str | None = None) -> list[DecisionEntry]:
        out = self._entries
        if action is not None:
            out = [e for e in out if e.action == action]
        if verdict is not None:
            out = [e for e in out if e.verdict == verdict]
        return list(out)

    def to_json(self, path: str) -> None:
        _entries_to_json_file(self.entries(), path)

    def to_csv(self, path: str) -> None:
        _entries_to_csv_file(self.entries(), path)


# --- shared export helpers (in-memory and SQLite logs) ------------------------

_CSV_FIELDS = ["ts", "conv_id", "action", "verdict", "amount",
               "authenticated", "reasons"]


def _entries_to_json_file(entries: list[DecisionEntry], path: str) -> None:
    Path(path).write_text(
        json.dumps([asdict(e) for e in entries], indent=2),
        encoding="utf-8",
    )


def _entries_to_csv_file(entries: list[DecisionEntry], path: str) -> None:
    if not entries:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for e in entries:
            row = asdict(e)
            row["reasons"] = "|".join(e.reasons)
            writer.writerow(row)


class SqliteDecisionLog:
    """Durable DecisionLog (Task D3): the SAME append-only interface as
    DecisionLog (record / entries / query / to_json / to_csv) backed by a
    SQLite table. Schema: ts, conv_id, action, verdict, reasons (JSON),
    plus the tool-outcome context fields (amount, authenticated).

    One connection guarded by a lock: writers (voice turn thread, runner,
    concurrent tests) serialize safely without a connection per write, and
    every record() commits, so the trail survives process restarts. Wired by
    build_orchestrator when VOICEAGENT_AUDIT_DB is set; the default stays
    in-memory (zero config change)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS decision_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        conv_id TEXT NOT NULL DEFAULT '',
        action TEXT NOT NULL DEFAULT '',
        verdict TEXT NOT NULL DEFAULT '',
        reasons TEXT NOT NULL DEFAULT '[]',
        amount REAL,
        authenticated INTEGER NOT NULL DEFAULT 0
    )
    """

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self._conn.execute(self._SCHEMA)
            self._conn.commit()

    def record(self, entry: DecisionEntry) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO decision_log"
                " (ts, conv_id, action, verdict, reasons, amount,"
                "  authenticated)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (entry.ts, entry.conv_id, entry.action, entry.verdict,
                 json.dumps(list(entry.reasons)), entry.amount,
                 1 if entry.authenticated else 0))
            self._conn.commit()

    def _select(self, action: str | None, verdict: str | None
                ) -> list[DecisionEntry]:
        sql = ("SELECT ts, conv_id, action, verdict, reasons, amount,"
               " authenticated FROM decision_log")
        clauses, args = [], []
        if action is not None:
            clauses.append("action = ?")
            args.append(action)
        if verdict is not None:
            clauses.append("verdict = ?")
            args.append(verdict)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        rows = self._conn.execute(sql, args).fetchall()
        return [DecisionEntry(ts=r[0], conv_id=r[1], action=r[2],
                              verdict=r[3], reasons=json.loads(r[4]),
                              amount=r[5], authenticated=bool(r[6]))
                for r in rows]

    def entries(self) -> list[DecisionEntry]:
        with self._lock:
            return self._select(None, None)

    def query(self, action: str | None = None,
              verdict: str | None = None) -> list[DecisionEntry]:
        with self._lock:
            return self._select(action, verdict)

    def to_json(self, path: str) -> None:
        _entries_to_json_file(self.entries(), path)

    def to_csv(self, path: str) -> None:
        _entries_to_csv_file(self.entries(), path)

    def delete_conv(self, conv_id: str) -> int:
        """Right-to-be-forgotten: delete every entry for one conversation.
        Returns the row count removed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM decision_log WHERE conv_id = ?", (conv_id,))
            self._conn.commit()
            return cur.rowcount

    def prune_before(self, cutoff_iso: str) -> int:
        """Retention: delete entries older than an ISO timestamp (ts < cutoff).
        Returns the row count removed. Timestamps are ISO-8601 UTC, so lexic
        comparison is chronological."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM decision_log WHERE ts < ?", (cutoff_iso,))
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
