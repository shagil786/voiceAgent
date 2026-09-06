import json
import sqlite3

from voiceagent.audit_report import MIN_CALLS, build_report


def _seed(audit_db, memory_db, calls=30, deny_escalations=0):
    a = sqlite3.connect(audit_db)
    a.execute("CREATE TABLE IF NOT EXISTS decision_log (id INTEGER PRIMARY KEY"
              " AUTOINCREMENT, ts TEXT, conv_id TEXT, action TEXT, verdict"
              " TEXT, reasons TEXT DEFAULT '[]', amount REAL, authenticated"
              " INTEGER DEFAULT 0)")
    for i in range(calls):
        conv = f"c{i}"
        a.execute("INSERT INTO decision_log (ts, conv_id, action, verdict,"
                  " reasons) VALUES ('2026-11-01T10:00:00', ?, 'order_lookup',"
                  " 'ALLOW', '[]')", (conv,))
        if i < deny_escalations:
            a.execute("INSERT INTO decision_log (ts, conv_id, action, verdict,"
                      " reasons) VALUES ('2026-11-01T10:01:00', ?,"
                      " 'initiate_refund', 'DENY', '[\"policy\"]')", (conv,))
            a.execute("INSERT INTO decision_log (ts, conv_id, action, verdict,"
                      " reasons) VALUES ('2026-11-01T10:05:00', ?,"
                      " 'escalate_to_human', 'ALLOW', '[]')", (conv,))
    a.commit()
    a.close()
    m = sqlite3.connect(memory_db)
    m.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY"
              " AUTOINCREMENT, tenant TEXT, ts TEXT, text TEXT, label TEXT,"
              " confidence REAL DEFAULT 0.0, outcome TEXT DEFAULT '',"
              " session_id TEXT DEFAULT '')")
    m.execute("CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY"
              " AUTOINCREMENT, tenant TEXT, session_id TEXT, ts TEXT,"
              " rating REAL, comment TEXT DEFAULT '')")
    for i in range(calls):
        m.execute("INSERT INTO episodes (tenant, ts, text, label, confidence,"
                  " outcome) VALUES ('pizzapal', '2026-11-01T10:00:00', ?,"
                  " 'order_status', 0.8, 'order_status')", (f"phrase {i}",))
    m.execute("INSERT INTO ratings (tenant, session_id, ts, rating, comment)"
              " VALUES ('pizzapal', 'c0', '2026-11-01T10:06:00', 9.0, 'fast')")
    m.commit()
    m.close()


def test_insufficient_data_state_is_honest(tmp_path):
    adb, mdb = str(tmp_path / "a.db"), str(tmp_path / "m.db")
    _seed(adb, mdb, calls=3)
    r = build_report(adb, mdb)
    assert not r.sufficient
    assert "Insufficient data" in r.markdown
    assert r.suggestions == []


def test_full_report_flags_deny_then_escalate_pattern(tmp_path):
    adb, mdb = str(tmp_path / "a.db"), str(tmp_path / "m.db")
    _seed(adb, mdb, calls=MIN_CALLS, deny_escalations=6)  # 6/6 = 100% pattern
    r = build_report(adb, mdb)
    assert r.sufficient and r.calls == MIN_CALLS
    assert "Deny-then-escalate" in r.markdown
    assert any("initiate_refund" in s for s in r.suggestions)


def test_feedback_section_and_no_pattern_below_threshold(tmp_path):
    adb, mdb = str(tmp_path / "a.db"), str(tmp_path / "m.db")
    _seed(adb, mdb, calls=MIN_CALLS, deny_escalations=1)  # 1/1? no: 1 of 25
    r = build_report(adb, mdb)
    assert r.suggestions == []          # 1 deny -> below the 5-call threshold
    assert "9.0/10" in r.markdown or "average **9.0" in r.markdown


def test_missing_dbs_render_empty_sections_without_error(tmp_path):
    r = build_report(None, None)
    assert not r.sufficient
    assert "0" in r.markdown
