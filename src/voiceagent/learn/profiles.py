# src/voiceagent/learn/profiles.py
"""Per-person profile store (Instant-Learn + Profiles, Task 1).

Two stdlib-only backends behind one protocol:
- InMemoryProfiles: dicts — tests, CLI, ephemeral demos.
- SQLiteProfiles: one file in WAL mode — survives restarts. Concurrency model
  mirrors memory.SQLiteMemory: a single shared connection
  (check_same_thread=False) guarded by one lock.

Keys: E.164-ish phone (`+<digits>`) or `cid:<customer_id>` fallback.
TTL: profiles older than PROFILE_TTL_DAYS are treated as expired.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from voiceagent.memory import now_ts
from voiceagent.swarm.blackboard import CallerProfile

PROFILE_TTL_DAYS: int = 365


def normalize_phone(raw: str) -> str:
    """Strip spaces/dashes/parens/dots (anything non-digit); return +<digits>.

    v1: assume already-E.164-or-local-digits; do NOT validate country codes.
    Empty (no digits) -> "".
    """
    digits = "".join(ch for ch in (raw or "") if ch.isdigit())
    if not digits:
        return ""
    return "+" + digits


def contact_key(profile: CallerProfile) -> str:
    """Contact key for a caller: normalized phone, else cid:<customer_id>."""
    phone = normalize_phone(profile.phone or "")
    if phone:
        return phone
    return f"cid:{profile.customer_id or 'unknown'}"


@dataclass
class Profile:
    key: str
    alias: str = ""
    prefs: list[str] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    open_items: list[str] = field(default_factory=list)
    pending_global: list[dict] = field(default_factory=list)
    consent: dict = field(default_factory=dict)
    updated_at: str = ""


def _expired(updated_at: str, now: str | None = None) -> bool:
    """True when updated_at is more than PROFILE_TTL_DAYS before now.

    Both stamps are "%Y-%m-%dT%H:%M:%S"; only the date part is compared.
    now=None means real current date (used by get); prune_expired injects now.
    """
    try:
        updated = datetime.date.fromisoformat((updated_at or "")[:10])
    except ValueError:
        return False
    today = (datetime.date.fromisoformat(now[:10]) if now is not None
             else datetime.date.today())
    return (today - updated).days > PROFILE_TTL_DAYS


def _profile_dict(p: Profile) -> dict[str, Any]:
    return {"key": p.key, "alias": p.alias, "prefs": list(p.prefs),
            "corrections": list(p.corrections),
            "open_items": list(p.open_items),
            "pending_global": list(p.pending_global),
            "consent": dict(p.consent), "updated_at": p.updated_at}


@runtime_checkable
class ProfileStore(Protocol):
    def get(self, key: str) -> Profile | None: ...
    def put(self, profile: Profile) -> None: ...
    def set_alias(self, alias: str, key: str) -> None: ...
    def resolve(self, alias_or_key: str) -> str: ...
    def link_session(self, key: str, session_id: str) -> None: ...
    def sessions_for(self, key: str) -> list[str]: ...
    def delete_contact(self, key: str) -> dict: ...
    def export_contact(self, key: str) -> dict: ...
    def prune_expired(self, now: str | None = None) -> int: ...


class InMemoryProfiles:
    """dict-backed ProfileStore (profiles + alias map + contact->sessions)."""

    def __init__(self) -> None:
        self._profiles: dict[str, Profile] = {}
        self._aliases: dict[str, str] = {}
        self._links: dict[str, set[str]] = {}

    def get(self, key: str) -> Profile | None:
        p = self._profiles.get(key)
        if p is None:
            return None
        if _expired(p.updated_at):
            self._drop(key)
            return None
        return p

    def put(self, profile: Profile) -> None:
        if not profile.updated_at:
            profile.updated_at = now_ts()
        self._profiles[profile.key] = profile

    def set_alias(self, alias: str, key: str) -> None:
        self._aliases[alias] = key

    def resolve(self, alias_or_key: str) -> str:
        if alias_or_key in self._aliases:
            return self._aliases[alias_or_key]
        return alias_or_key

    def link_session(self, key: str, session_id: str) -> None:
        self._links.setdefault(key, set()).add(session_id)

    def sessions_for(self, key: str) -> list[str]:
        return sorted(self._links.get(key, ()))

    def delete_contact(self, key: str) -> dict:
        sessions = sorted(self._links.pop(key, ()))
        self._drop(key)
        for alias, target in [i for i in self._aliases.items()
                              if i[1] == key]:
            del self._aliases[alias]
        return {"sessions": sessions}

    def export_contact(self, key: str) -> dict:
        p = self.get(key)
        if p is None:
            raise KeyError(key)
        return _profile_dict(p)

    def prune_expired(self, now: str | None = None) -> int:
        expired = [k for k, p in self._profiles.items()
                   if _expired(p.updated_at, now)]
        for k in expired:
            self.delete_contact(k)
        return len(expired)

    def _drop(self, key: str) -> None:
        self._profiles.pop(key, None)


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS profiles("
    " key TEXT PRIMARY KEY, alias TEXT NOT NULL DEFAULT '',"
    " prefs_json TEXT NOT NULL DEFAULT '[]',"
    " corrections_json TEXT NOT NULL DEFAULT '[]',"
    " open_items_json TEXT NOT NULL DEFAULT '[]',"
    " pending_json TEXT NOT NULL DEFAULT '[]',"
    " consent_json TEXT NOT NULL DEFAULT '{}',"
    " updated_at TEXT NOT NULL DEFAULT '')",
    "CREATE TABLE IF NOT EXISTS aliases(alias TEXT PRIMARY KEY, key TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS links(key TEXT NOT NULL, session_id TEXT NOT NULL,"
    " PRIMARY KEY(key, session_id))",
)


def _row_to_profile(row: tuple) -> Profile:
    return Profile(key=row[0], alias=row[1], prefs=json.loads(row[2]),
                   corrections=json.loads(row[3]), open_items=json.loads(row[4]),
                   pending_global=json.loads(row[5]), consent=json.loads(row[6]),
                   updated_at=row[7])


class SQLiteProfiles:
    """SQLite-backed ProfileStore. Lists/dicts are stored as JSON columns."""

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # WAL's recommended pairing: commits no longer fsync per write.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        self._conn.commit()

    def get(self, key: str) -> Profile | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT key, alias, prefs_json, corrections_json,"
                " open_items_json, pending_json, consent_json, updated_at"
                " FROM profiles WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            if _expired(row[7]):
                self._delete_locked(key)
                self._conn.commit()
                return None
            return _row_to_profile(row)

    def put(self, profile: Profile) -> None:
        if not profile.updated_at:
            profile.updated_at = now_ts()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO profiles (key, alias, prefs_json,"
                " corrections_json, open_items_json, pending_json,"
                " consent_json, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (profile.key, profile.alias, json.dumps(profile.prefs),
                 json.dumps(profile.corrections), json.dumps(profile.open_items),
                 json.dumps(profile.pending_global), json.dumps(profile.consent),
                 profile.updated_at))
            self._conn.commit()

    def set_alias(self, alias: str, key: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO aliases (alias, key) VALUES (?, ?)",
                (alias, key))
            self._conn.commit()

    def resolve(self, alias_or_key: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT key FROM aliases WHERE alias = ?",
                (alias_or_key,)).fetchone()
        if row is not None:
            return row[0]
        return alias_or_key

    def link_session(self, key: str, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO links (key, session_id) VALUES (?, ?)",
                (key, session_id))
            self._conn.commit()

    def sessions_for(self, key: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id FROM links WHERE key = ? ORDER BY session_id",
                (key,)).fetchall()
        return [r[0] for r in rows]

    def delete_contact(self, key: str) -> dict:
        with self._lock:
            sessions = [r[0] for r in self._conn.execute(
                "SELECT session_id FROM links WHERE key = ? ORDER BY session_id",
                (key,)).fetchall()]
            self._delete_locked(key)
            self._conn.commit()
        return {"sessions": sessions}

    def export_contact(self, key: str) -> dict:
        p = self.get(key)
        if p is None:
            raise KeyError(key)
        return _profile_dict(p)

    def prune_expired(self, now: str | None = None) -> int:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, updated_at FROM profiles").fetchall()
            expired = [k for k, ts in rows if _expired(ts, now)]
            for k in expired:
                self._delete_locked(k)
            self._conn.commit()
        return len(expired)

    def _delete_locked(self, key: str) -> None:
        self._conn.execute("DELETE FROM profiles WHERE key = ?", (key,))
        self._conn.execute("DELETE FROM aliases WHERE key = ?", (key,))
        self._conn.execute("DELETE FROM links WHERE key = ?", (key,))
