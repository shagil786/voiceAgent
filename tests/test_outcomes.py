# tests/test_outcomes.py
from voiceagent.learn.outcomes import InMemoryOutcomes, JsonlOutcomes, OutcomeLabel

def test_record_query_and_labels():
    s = InMemoryOutcomes()
    s.record(OutcomeLabel(session_id="s1", label="resolved", ts="2026-09-04T00:00:00"))
    s.record(OutcomeLabel(session_id="s2", label="escalated", ts="2026-09-04T00:01:00"))
    assert [o.session_id for o in s.query(label="resolved")] == ["s1"]
    assert len(s.query()) == 2
    assert s.query(session_id="s9") == []

def test_jsonl_roundtrip(tmp_path):
    p = str(tmp_path / "sub" / "outcomes.jsonl")
    s = JsonlOutcomes(p)
    s.record(OutcomeLabel(session_id="s1", label="thumbs_down", ts="t", note="wrong fee"))
    assert JsonlOutcomes(p).query(label="thumbs_down")[0].note == "wrong fee"
