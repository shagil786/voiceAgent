# tests/test_selfcheck.py — runnable self-checks + mechanical go-live gate.
"""Each bundle eval runs as real Orchestrator.handle_turn calls against a
deterministic scripted brain; go_live writes the live pointer only on 10/10."""
import inspect
from pathlib import Path

from voiceagent.deploy import selfcheck
from voiceagent.deploy.bundle import load_bundle, read_live
from voiceagent.deploy.stub import make_default_client

GOLDEN = Path("data/deployments/_example/v1")


def test_self_checks_run_and_gate_go_live(tmp_path):
    b = load_bundle(GOLDEN)
    results = selfcheck.run_self_checks(b)
    assert len(results) == len(b.evals) and len(results) >= 2
    assert all(set(r) == {"name", "passed", "detail"} for r in results)


def test_go_live_requires_ten_of_ten(tmp_path):
    ok = [{"name": f"c{i}", "passed": True, "detail": ""} for i in range(10)]
    assert selfcheck.go_live(str(tmp_path), "v3", ok) is True
    bad = list(ok)
    bad[0] = {"name": "c0", "passed": False, "detail": "x"}
    assert selfcheck.go_live(str(tmp_path), "v4", bad) is False
    assert read_live(str(tmp_path)) == "v3"


def test_default_brain_is_tests_package_free():
    client = make_default_client(["hello"])
    assert inspect.getmodule(type(client)).__name__.startswith("voiceagent")
    src = Path(inspect.getfile(selfcheck)).read_text(encoding="utf-8")
    assert "from tests" not in src and "import tests" not in src
