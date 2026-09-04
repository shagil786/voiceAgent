from voiceagent.metrics import Metrics


def test_snapshot_math():
    m = Metrics()
    m.record(0.2, "ALLOW"); m.record(0.4, "DENY")
    s = m.snapshot()
    assert s == {"turns": 2, "avg_latency_ms": 300, "verdicts": {"ALLOW": 1, "DENY": 1}}
