# tests/test_onboard_measure.py
import sys
sys.path.insert(0, "scripts")
from onboard_measure import measure_onboard
INTERVIEW = {"offering": "Acme sells widgets", "top_asks": ["price?", "hours?"],
             "never_promise": ["same-day"], "handoff_triggers": ["legal"]}

def test_drill_reports_phases(tmp_path):
    def fetcher(url):
        return ('<p>Acme widgets $9. <a href="/h">h</a></p>', url)
    out = measure_onboard("https://acme.test/", INTERVIEW, fetcher=fetcher)
    assert out["pages"] >= 1 and out["evals"] == 10 and out["total_ms"] >= 0
    assert set(out) >= {"pages", "chunks", "compile_ms", "checks_ms", "total_ms", "evals", "tools"}
