# tests/test_benchmark.py
import json
import tempfile
from pathlib import Path
from voiceagent.benchmark import (run_benchmark, write_report, evaluate_gate,
                                  estimate_vps_cost, sweep_all_models)
from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult

class FixedAgent:
    def __init__(self, action, text):
        self.action = action
        self.text = text
    def handle(self, user_text):
        return AgentResult(text=self.text, action=self.action,
                           retrieved=[{"text": "ok"}], latency_s=0.5)

def _convs(n=10, action="refund"):
    return [Conversation(id=f"c{i}", language="en", intent="refund",
                         user_text="refund ORD-1", expected_action=action,
                         key_facts=["ORD-1"], escalate=False) for i in range(n)]

def test_run_benchmark_reports_by_language_and_gate():
    agent = FixedAgent("refund", "refund for ORD-1 done")
    report = run_benchmark(agent, _convs(10))
    assert report.summary.n == 10
    assert report.summary.resolution_rate == 1.0
    assert "en" in report.per_language

def test_evaluate_gate_flags_failures():
    report = run_benchmark(FixedAgent("wrong", "x"), _convs(10))
    ok, reasons = evaluate_gate(report)
    assert ok is False
    assert any("resolution" in r for r in reasons)

def test_write_report_emits_md_and_json():
    agent = FixedAgent("refund", "refund for ORD-1 done")
    report = run_benchmark(agent, _convs(10))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.md"
        write_report(report, str(out), {"model": "fake", "params": "0.5B"})
        assert out.exists()
        assert (Path(d) / "report.json").exists()
        j = json.loads((Path(d) / "report.json").read_text())
        assert j["summary"]["n"] == 10

def test_estimate_vps_cost_returns_tier():
    cost = estimate_vps_cost({"model": "qwen2.5-0.5b-q4", "size_mb": 400})
    assert cost["vps_tier"] == "2-4vCPU/8GB"
    assert cost["vps_cost_rs_estimate"] > 0

def test_sweep_all_models_returns_one_report_per_available_model(monkeypatch):
    # Patch at the source modules: sweep_all_models imports these inside the
    # function, which binds at call time, so patching the module attributes
    # is enough to avoid real model downloads / loads.
    import voiceagent.llm as llm_mod
    import voiceagent.knowledge as kb_mod

    class FakeLLM:
        def __init__(self, path):
            self.specs = {"model": "fake", "params": "0.5B",
                          "quant": "Q4_K_M", "model_path": path, "size_mb": 400}
        def generate(self, prompt, max_tokens=256, stop=None):
            return "refund for ORD-1 done\nACTION: refund"

    monkeypatch.setattr(
        llm_mod, "list_available_models",
        lambda model_dir: [
            {"name": "qwen2.5-0.5b-q4", "model_path": "/x",
             "params": "0.5B", "quant": "Q4_K_M", "size_mb": 400},
            {"name": "qwen2.5-1.5b-q4", "model_path": "/y",
             "params": "1.5B", "quant": "Q4_K_M", "size_mb": 1100},
        ])
    monkeypatch.setattr(llm_mod, "load_llm", lambda path, **kw: FakeLLM(path))
    monkeypatch.setattr(kb_mod, "load_docs", lambda d: [{"id": "a",
                         "text": "Refunds processed in 5-7 days.",
                         "section": "Refunds"}])
    monkeypatch.setattr(kb_mod, "build_index", lambda docs: _FakeIndex())

    reports = sweep_all_models(_convs(4), knowledge_dir="data/knowledge",
                               model_dir="data/models", max_rows=4)
    assert len(reports) == 2
    assert all(r.summary.n == 4 for r in reports)
    assert all(r.summary.resolution_rate == 1.0 for r in reports)

class _FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "Refunds processed in 5-7 days.",
                 "section": "Refunds", "score": 0.9}]
