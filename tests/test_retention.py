# tests/test_retention.py — retention + right-to-be-forgotten.
"""Proves the compliance seams: audit-log prune/erase, memory
episode/rating erase + prune, env resolution, and the CLI contract.
All stores are temp files; nothing touches real deployment DBs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from voiceagent.decisionlog import DecisionEntry, SqliteDecisionLog
from voiceagent.memory import IntentMemoryStore
from voiceagent.retention import cutoff_iso, erase_session, purge_expired

ROOT = Path(__file__).resolve().parents[1]


def _audit(path, conv="s-1", ts="2026-01-01T10:00:00"):
    log = SqliteDecisionLog(str(path))
    log.record(DecisionEntry(ts=ts, conv_id=conv, action="a",
                             verdict="ALLOW", reasons=[], amount=None,
                             authenticated=True))
    log.close()


def _memory(path, ts="2026-01-01T10:00:00"):
    store = IntentMemoryStore(str(path), eager=False)
    store.capture("t1", "where is my order", "order_status", 0.9,
                  outcome="ok", ts=ts, session_id="s-1")
    store.record_rating("t1", "s-1", 5.0, "great")
    store.close()


def test_audit_prune_and_erase(tmp_path):
    db = tmp_path / "audit.sqlite"
    _audit(db, conv="s-old", ts="2020-01-01T00:00:00")
    _audit(db, conv="s-new", ts="2099-01-01T00:00:00")
    log = SqliteDecisionLog(str(db))
    assert log.prune_before("2026-01-01T00:00:00") == 1
    assert [e.conv_id for e in log.entries()] == ["s-new"]
    assert log.delete_conv("s-new") == 1
    assert log.entries() == []
    assert log.delete_conv("s-new") == 0  # idempotent
    log.close()


def test_memory_erase_and_prune(tmp_path):
    prune_db = tmp_path / "prune.sqlite"
    _memory(prune_db)
    store = IntentMemoryStore(str(prune_db), eager=False)
    assert store.prune_episodes_before("2020-01-01T00:00:00") == 0
    assert store.prune_episodes_before("2099-01-01T00:00:00") == 1
    store.close()
    erase_db = tmp_path / "erase.sqlite"
    _memory(erase_db)
    store = IntentMemoryStore(str(erase_db), eager=False)
    # A distilled prototype (caller text in exemplar JSON) — episode
    # deletion alone must not leave it behind. Seeded via SQL so the test
    # never loads embedding models.
    import sqlite3
    raw = sqlite3.connect(str(erase_db))
    raw.execute(
        "INSERT INTO prototypes (tenant, label, centroid_json,"
        " exemplars_json, hit_count, last_seen, confidence)"
        " VALUES ('t1', 'order_status', '[0.1]', '[\"where is my order\"]',"
        " 1, '2026-01-01T10:00:00', 0.9)")
    raw.commit()
    raw.close()
    out = store.erase_session("s-1", tenant="t1")
    assert out == {"episodes": 1, "ratings": 1, "prototypes": 1}
    assert store.erase_session("s-1", tenant="t1") == {
        "episodes": 0, "ratings": 0, "prototypes": 0}
    store.close()


def test_purge_and_erase_end_to_end(tmp_path):
    audit_db, mem_db = tmp_path / "a.sqlite", tmp_path / "m.sqlite"
    _audit(audit_db, conv="s-old", ts="2020-01-01T00:00:00")
    _audit(audit_db, conv="s-new", ts="2099-01-01T00:00:00")
    _memory(mem_db, ts="2099-01-01T10:00:00")
    env = {"VOICEAGENT_AUDIT_DB": str(audit_db),
           "VOICEAGENT_MEMORY_DB": str(mem_db),
           "VOICEAGENT_DATA_RETENTION_DAYS": "30"}
    out = purge_expired(env=env)
    assert out["audit_log"] == 1
    assert out["memory_ratings"] == 0  # fresh rating survives a 30d window
    out = erase_session("s-new", env=env)
    assert out["audit_log"] == 1 and out["memory_episodes"] == 0
    out = erase_session("s-1", env=env)
    assert out["audit_log"] == 0
    assert out["memory_episodes"] == 1 and out["memory_ratings"] == 1


def test_purge_no_window_is_noop(tmp_path):
    assert purge_expired(env={}) == {}
    with pytest.raises(ValueError):
        purge_expired(env={"VOICEAGENT_DATA_RETENTION_DAYS": "soon"})
    with pytest.raises(ValueError):
        purge_expired(env={"VOICEAGENT_DATA_RETENTION_DAYS": "0"})


def test_purge_skips_missing_paths_without_creating(tmp_path):
    missing = tmp_path / "nope.sqlite"
    out = purge_expired(audit_db=str(missing), memory_db=str(missing),
                        days=30)
    assert out == {}
    assert not missing.exists()


def test_forget_caller_cli(tmp_path):
    audit_db = tmp_path / "a.sqlite"
    missing_mem = tmp_path / "missing-mem.sqlite"
    _audit(audit_db, conv="s-1", ts="2020-01-01T00:00:00")
    # Explicit store flags: the CLI must never touch the deployment DBs
    # from .env during tests (or accidents).
    env = {"PATH": "/usr/bin:/bin", "SYSTEMROOT": "x"}
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "forget_caller.py"),
         "--purge", "--days", "30", "--audit-db", str(audit_db),
         "--memory-db", str(missing_mem)],
        capture_output=True, text=True, env=env, cwd=ROOT,
        timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["removed"] == {"audit_log": 1}
    assert not missing_mem.exists()
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "forget_caller.py")],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    assert r.returncode == 2  # no action = usage error, nothing deleted


def test_cutoff_iso_format():
    assert cutoff_iso(30).count("T") == 1 and len(cutoff_iso(30)) == 19


def test_ratings_prune_with_episodes(tmp_path):
    from voiceagent.memory import IntentMemoryStore
    db = tmp_path / "r.sqlite"
    _memory(db)  # rating ts = now -> survives; backdate it, then prune
    import sqlite3
    raw = sqlite3.connect(str(db))
    raw.execute("UPDATE ratings SET ts = '2020-01-01T00:00:00'")
    raw.commit()
    raw.close()
    store = IntentMemoryStore(str(db), eager=False)
    assert store.prune_ratings_before("2026-01-01T00:00:00") == 1
    assert store.prune_ratings_before("2026-01-01T00:00:00") == 0
    store.close()
