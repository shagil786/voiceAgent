# src/voiceagent/memory.py
"""Short-term working memory (M4a): per-conversation turn history.

Two stdlib-only backends behind one protocol:
- InMemoryMemory: dict of deques — tests, CLI, ephemeral demos.
- SQLiteMemory: one file in WAL mode — survives restarts. Concurrency model:
  a single shared connection (check_same_thread=False) guarded by one lock.
  The stdlib server serializes requests anyway, and every method holds the
  lock for one short statement, so worker threads never interleave reads
  and writes.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def now_ts() -> str:
    """Timestamp string for Turn.ts (same format as the decision log)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Turn:
    ts: str
    role: str  # "user" | "agent"
    text: str
    action: str | None = None
    verdict: str | None = None
    refs: list[str] = field(default_factory=list)


@runtime_checkable
class ConversationMemory(Protocol):
    def append(self, conv_id: str, turn: Turn) -> None: ...
    def history(self, conv_id: str, last_n: int | None = None) -> list[Turn]: ...
    def clear(self, conv_id: str) -> None: ...


class InMemoryMemory:
    """dict[conv_id -> deque[Turn]], oldest evicted at maxlen_per_conv."""

    def __init__(self, maxlen_per_conv: int = 100):
        self._maxlen = maxlen_per_conv
        self._convs: dict[str, deque[Turn]] = {}

    def append(self, conv_id: str, turn: Turn) -> None:
        self._convs.setdefault(conv_id, deque(maxlen=self._maxlen)).append(turn)

    def history(self, conv_id: str, last_n: int | None = None) -> list[Turn]:
        turns = list(self._convs.get(conv_id, ()))
        return turns[-last_n:] if last_n is not None else turns

    def clear(self, conv_id: str) -> None:
        self._convs.pop(conv_id, None)


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS turns ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " conv_id TEXT NOT NULL,"
    " ts TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL,"
    " action TEXT, verdict TEXT, refs_json TEXT NOT NULL DEFAULT '[]')",
    "CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conv_id)",
)


class SQLiteMemory:
    """SQLite-backed ConversationMemory. refs are stored as a JSON list in
    refs_json; history() reads oldest-first (last_n takes the newest rows)."""

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # WAL's recommended pairing: commits no longer fsync per write.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        self._conn.commit()

    def append(self, conv_id: str, turn: Turn) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (conv_id, ts, role, text, action, verdict,"
                " refs_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conv_id, turn.ts, turn.role, turn.text, turn.action,
                 turn.verdict, json.dumps(turn.refs)))
            self._conn.commit()

    def history(self, conv_id: str, last_n: int | None = None) -> list[Turn]:
        sql = ("SELECT ts, role, text, action, verdict, refs_json FROM turns"
               " WHERE conv_id = ? ORDER BY id")
        params: list = [conv_id]
        if last_n is not None:
            sql += " DESC LIMIT ?"
            params.append(last_n)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        if last_n is not None:
            rows.reverse()  # DESC LIMIT gives the newest n; restore order
        return [Turn(ts=r[0], role=r[1], text=r[2], action=r[3], verdict=r[4],
                     refs=json.loads(r[5])) for r in rows]

    def clear(self, conv_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM turns WHERE conv_id = ?", (conv_id,))
            self._conn.commit()


def public_dict(turn: Turn) -> dict:
    """API view of a turn (the /api/history shape; internal refs omitted)."""
    return {"ts": turn.ts, "role": turn.role, "text": turn.text,
            "action": turn.action, "verdict": turn.verdict}
