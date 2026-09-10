from voiceagent.metrics import Metrics


def test_snapshot_math():
    m = Metrics()
    m.record(0.2, "ALLOW"); m.record(0.4, "DENY")
    s = m.snapshot()
    assert s == {"turns": 2, "avg_latency_ms": 300,
                 "verdicts": {"ALLOW": 1, "DENY": 1}, "events": {}}


def test_events_counter_survives_snapshot():
    m = Metrics()
    m.note("rag_fallback", 2)
    m.note("rag_fallback")
    assert m.snapshot()["events"] == {"rag_fallback": 3}
