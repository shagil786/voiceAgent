# tests/test_decisionlog.py
import csv
import json
import tempfile
from pathlib import Path
from voiceagent.decisionlog import DecisionLog, DecisionEntry

def test_record_and_query():
    log = DecisionLog()
    log.record(DecisionEntry(ts="t1", conv_id="c1", action="refund",
                             verdict="REQUIRE_HUMAN_APPROVAL", reasons=["over limit"],
                             amount=20000, authenticated=True))
    log.record(DecisionEntry(ts="t2", conv_id="c2", action="order_status",
                             verdict="ALLOW", reasons=[], amount=None, authenticated=False))
    assert len(log.entries()) == 2
    assert len(log.query(action="refund")) == 1
    assert len(log.query(verdict="ALLOW")) == 1
    assert len(log.query(action="refund", verdict="ALLOW")) == 0

def test_to_json_and_csv():
    log = DecisionLog()
    log.record(DecisionEntry(ts="t1", conv_id="c1", action="refund",
                             verdict="ALLOW", reasons=[], amount=None, authenticated=True))
    with tempfile.TemporaryDirectory() as d:
        j = Path(d) / "log.json"
        c = Path(d) / "log.csv"
        log.to_json(str(j))
        log.to_csv(str(c))
        assert j.exists() and c.exists()
        assert json.loads(j.read_text())[0]["verdict"] == "ALLOW"
        rows = list(csv.DictReader(open(c)))
        assert rows[0]["action"] == "refund"


# ---------------------------------------------------------------------------
# Task D3: the PERSISTENT DecisionLog. SqliteDecisionLog implements the same
# append-only interface over SQLite (single connection + lock), so an audit
# trail survives process restarts. Schema: ts, conv_id, action, verdict,
# reasons (JSON), amount, authenticated.
# ---------------------------------------------------------------------------

def test_sqlite_record_and_read_back_round_trip(tmp_path):
    from voiceagent.decisionlog import SqliteDecisionLog
    db = tmp_path / "audit.db"
    log = SqliteDecisionLog(str(db))
    e1 = DecisionEntry(ts="t1", conv_id="c1", action="refund",
                       verdict="REQUIRE_HUMAN_APPROVAL",
                       reasons=["over limit", "high value"],
                       amount=20000.0, authenticated=True)
    e2 = DecisionEntry(ts="t2", conv_id="c2", action="order_status",
                       verdict="ALLOW", reasons=[], amount=None,
                       authenticated=False)
    log.record(e1)
    log.record(e2)
    entries = log.entries()
    assert entries == [e1, e2]
    assert log.query(action="refund") == [e1]
    assert log.query(verdict="ALLOW") == [e2]
    assert log.query(action="refund", verdict="ALLOW") == []
    log.close()


def test_sqlite_persists_across_reopens(tmp_path):
    from voiceagent.decisionlog import SqliteDecisionLog
    db = tmp_path / "audit.db"
    w = SqliteDecisionLog(str(db))
    w.record(DecisionEntry(ts="t1", conv_id="c1", action="refund",
                           verdict="DENY", reasons=["not connected"],
                           amount=10.0, authenticated=False))
    w.close()
    r = SqliteDecisionLog(str(db))
    assert [e.action for e in r.entries()] == ["refund"]
    assert r.entries()[0].reasons == ["not connected"]
    assert r.entries()[0].amount == 10.0
    r.close()


def test_sqlite_export_helpers(tmp_path):
    from voiceagent.decisionlog import SqliteDecisionLog
    log = SqliteDecisionLog(str(tmp_path / "audit.db"))
    log.record(DecisionEntry(ts="t1", conv_id="c1", action="refund",
                             verdict="ALLOW", reasons=["ok"],
                             amount=1.0, authenticated=True))
    j = tmp_path / "log.json"
    c = tmp_path / "log.csv"
    log.to_json(str(j))
    log.to_csv(str(c))
    assert json.loads(j.read_text())[0]["verdict"] == "ALLOW"
    rows = list(csv.DictReader(open(c)))
    assert rows[0]["action"] == "refund"
    log.close()


def test_sqlite_concurrent_writes_all_landed(tmp_path):
    """Concurrent-ish writers share one connection guarded by a lock — no
    entry may be lost and the ordering stays total."""
    import threading
    from voiceagent.decisionlog import SqliteDecisionLog
    log = SqliteDecisionLog(str(tmp_path / "audit.db"))
    def writer(tid):
        for i in range(10):
            log.record(DecisionEntry(ts=f"t{tid}-{i}", conv_id=f"c{tid}",
                                     action="refund", verdict="ALLOW",
                                     reasons=[], amount=float(i),
                                     authenticated=True))
    threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.entries()) == 50
    assert len({e.ts for e in log.entries()}) == 50
    log.close()


def test_sqlite_log_satisfies_decisionlog_duck_type(tmp_path):
    """Same interface as the in-memory DecisionLog (record/entries/query)."""
    from voiceagent.decisionlog import DecisionLog, SqliteDecisionLog
    mem = DecisionLog()
    sql = SqliteDecisionLog(str(tmp_path / "audit.db"))
    for log in (mem, sql):
        log.record(DecisionEntry(ts="t", conv_id="c", action="a",
                                 verdict="ALLOW", reasons=["r"]))
        assert len(log.entries()) == 1
        assert log.query(action="a") == log.entries()
        assert log.query(verdict="DENY") == []
    sql.close()


def test_build_orchestrator_wires_sqlite_log_when_env_set(tmp_path, monkeypatch):
    """VOICEAGENT_AUDIT_DB=<path> makes build_orchestrator wire the SQLite
    log (runner AND orchestrator audit seams); unset keeps the in-memory
    default with zero config change."""
    from voiceagent.decisionlog import DecisionLog, SqliteDecisionLog
    from voiceagent.runtime import build_orchestrator
    env = {"VOICEAGENT_FRONTIER_URL": "https://fake/v1",
           "VOICEAGENT_AUDIT_DB": str(tmp_path / "audit.db")}
    orch = build_orchestrator(env=env)
    assert isinstance(orch.decision_log, SqliteDecisionLog)
    assert orch.runner.decision_log is orch.decision_log
    # a real record round-trips through the wired log
    from voiceagent.decisionlog import DecisionEntry
    orch.decision_log.record(DecisionEntry(
        ts="t", conv_id="c", action="refund", verdict="ALLOW", reasons=[]))
    assert len(orch.decision_log.entries()) == 1

    # no env -> the in-memory default (zero config change)
    monkeypatch.delenv("VOICEAGENT_AUDIT_DB", raising=False)
    orch2 = build_orchestrator(env={"VOICEAGENT_FRONTIER_URL": "https://fake/v1"})
    assert isinstance(orch2.decision_log, DecisionLog)
    assert not isinstance(orch2.decision_log, SqliteDecisionLog)
