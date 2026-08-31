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
