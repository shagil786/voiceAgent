# src/voiceagent/retention.py — data retention + right-to-be-forgotten.
"""Operator tooling for the platform's persistent caller-data stores.

Two mechanisms, both explicit (nothing deletes by default):

- Retention (`purge_expired`): delete audit-log entries and memory episodes
  older than VOICEAGENT_DATA_RETENTION_DAYS. Run on a schedule (cron) or
  from scripts/forget_caller.py. The in-memory defaults are untouched —
  only the SQLite paths (VOICEAGENT_AUDIT_DB / VOICEAGENT_MEMORY_DB) purge.
- Erasure (`erase_session`): right-to-be-forgotten for one conversation —
  removes its audit entries, memory episodes and ratings. Keyed by conv_id
  / session_id (the platform never persists caller phone numbers, so there
  is no phone-keyed row to delete — see docs/DATA_FLOWS.md).

The org's own ERP/CRM stays the system of record: erasure there is the
org's procedure, not this module (the agent never writes PII anywhere
else — inventory in docs/DATA_FLOWS.md).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone


def retention_days(env: dict[str, str] | None = None) -> int | None:
    """Configured retention window, or None (keep — the default)."""
    e = os.environ if env is None else env
    raw = (e.get("VOICEAGENT_DATA_RETENTION_DAYS") or "").strip()
    if not raw:
        return None
    try:
        days = int(raw)
    except ValueError:
        raise ValueError(
            "VOICEAGENT_DATA_RETENTION_DAYS must be an integer number of "
            f"days, got {raw!r}")
    if days < 1:
        raise ValueError(
            "VOICEAGENT_DATA_RETENTION_DAYS must be >= 1, "
            f"got {days}")
    return days


def cutoff_iso(days: int,
               now: datetime | None = None) -> str:
    """UTC ISO cutoff: rows older than this are expired. Matches the
    stores' timestamp format (%Y-%m-%dT%H:%M:%S) so lexic comparison is
    chronological."""
    moment = now or datetime.now(timezone.utc)
    return (moment - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


def purge_expired(*, audit_db: str | None = None,
                  memory_db: str | None = None,
                  days: int | None = None,
                  env: dict[str, str] | None = None) -> dict[str, int]:
    """Delete rows older than the retention window. Returns per-store
    counts. Days defaults to VOICEAGENT_DATA_RETENTION_DAYS; None means
    keep (no-op, returns {}). Missing DB paths are skipped, not errors."""
    if days is None:
        days = retention_days(env)
    if days is None:
        return {}
    cutoff = cutoff_iso(days)
    out: dict[str, int] = {}
    e = os.environ if env is None else env
    audit_db = audit_db or e.get("VOICEAGENT_AUDIT_DB")
    memory_db = memory_db or e.get("VOICEAGENT_MEMORY_DB")
    # sqlite3.connect creates missing files — never create a store just to
    # purge it; absent paths are skipped, not errors.
    if audit_db and os.path.exists(audit_db):
        from voiceagent.decisionlog import SqliteDecisionLog
        log = SqliteDecisionLog(audit_db)
        try:
            out["audit_log"] = log.prune_before(cutoff)
        finally:
            log.close()
    if memory_db and os.path.exists(memory_db):
        from voiceagent.memory import IntentMemoryStore
        store = IntentMemoryStore(memory_db)
        try:
            out["memory_episodes"] = store.prune_episodes_before(cutoff)
        finally:
            store.close()
    return out


def erase_session(conv_id: str, *,
                  audit_db: str | None = None,
                  memory_db: str | None = None,
                  tenant: str | None = None,
                  env: dict[str, str] | None = None) -> dict[str, int]:
    """Erase one conversation everywhere the platform stores caller data.
    Returns per-store counts. Missing DB paths are skipped, not errors."""
    out: dict[str, int] = {}
    e = os.environ if env is None else env
    audit_db = audit_db or e.get("VOICEAGENT_AUDIT_DB")
    memory_db = memory_db or e.get("VOICEAGENT_MEMORY_DB")
    if audit_db and os.path.exists(audit_db):
        from voiceagent.decisionlog import SqliteDecisionLog
        log = SqliteDecisionLog(audit_db)
        try:
            out["audit_log"] = log.delete_conv(conv_id)
        finally:
            log.close()
    if memory_db and os.path.exists(memory_db):
        from voiceagent.memory import IntentMemoryStore
        store = IntentMemoryStore(memory_db)
        try:
            counts = store.erase_session(conv_id, tenant=tenant)
            out["memory_episodes"] = counts.get("episodes", 0)
            out["memory_ratings"] = counts.get("ratings", 0)
        finally:
            store.close()
    return out
