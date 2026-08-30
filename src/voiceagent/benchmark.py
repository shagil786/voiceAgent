# src/voiceagent/benchmark.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult
from voiceagent.evaluator import EvalRow, EvalSummary, score_conversation, aggregate

LATENCY_MAX_S = 2.0
RESOLUTION_MIN = 0.75

# Per-model fixed overheads on the VPS: embedding model, whisper, piper, base.
BASE_RAM_MB = 850  # embedding ~300 + whisper tiny ~200 + piper ~150 + overhead ~200

VPS_TIERS = [
    {"name": "2-4vCPU/8GB", "ram_mb": 8192, "cost_rs": 3000},
    {"name": "4-8vCPU/16GB", "ram_mb": 16384, "cost_rs": 5000},
]


@dataclass
class BenchmarkReport:
    summary: EvalSummary
    per_language: dict[str, EvalSummary]
    rows: list[EvalRow]
    model_specs: dict


def run_benchmark(agent, conversations: list[Conversation],
                  max_rows: int | None = None,
                  model_specs: dict | None = None) -> BenchmarkReport:
    rows: list[EvalRow] = []
    by_lang: dict[str, list[EvalRow]] = {}
    for conv in conversations[:max_rows]:
        res = agent.handle(conv.user_text)
        row = score_conversation(conv, res)
        rows.append(row)
        by_lang.setdefault(conv.language, []).append(row)
    return BenchmarkReport(
        summary=aggregate(rows),
        per_language={k: aggregate(v) for k, v in by_lang.items()},
        rows=rows,
        model_specs=model_specs or {},
    )


def estimate_vps_cost(model_specs: dict) -> dict:
    """Pick the cheapest VPS tier that fits the model + fixed pipeline overheads."""
    size_mb = model_specs.get("size_mb", 0)
    total = size_mb + BASE_RAM_MB
    for tier in VPS_TIERS:
        if total <= tier["ram_mb"]:
            return {"vps_tier": tier["name"], "ram_mb": total,
                    "vps_cost_rs_estimate": tier["cost_rs"]}
    # even 16GB doesn't fit -> report the 16GB tier anyway (LLM mmap can swap)
    tier = VPS_TIERS[-1]
    return {"vps_tier": tier["name"], "ram_mb": total,
            "vps_cost_rs_estimate": tier["cost_rs"]}


def estimate_cost_per_conversation(model_specs: dict, avg_latency_s: float,
                                   resolution_rate: float) -> dict:
    """₹ per turn and ₹ per resolved conversation.

    Assumes: one conversation at a time (sequential), ~4 turns per
    conversation, VPS billed per month at the tier cost."""
    tier = estimate_vps_cost(model_specs)
    cost_rs = tier["vps_cost_rs_estimate"]
    hours_per_month = 720
    cost_rs_per_hour = cost_rs / hours_per_month
    turns_per_conv = 4
    turn_cost = cost_rs_per_hour / 3600 * avg_latency_s * turns_per_conv
    resolved_cost = turn_cost / max(resolution_rate, 1e-9)
    return {"cost_per_turn_rs": round(turn_cost / turns_per_conv, 4),
            "cost_per_conversation_rs": round(turn_cost, 4),
            "cost_per_resolved_rs": round(resolved_cost, 4),
            "vps_tier": tier["vps_tier"]}


def sweep_all_models(conversations: list[Conversation], knowledge_dir: str,
                     model_dir: str, max_rows: int | None = None,
                     max_conversations: int = 200) -> list[BenchmarkReport]:
    """Run the full pipeline once per downloaded model. Return sorted by
    (passed gate, resolution desc, size asc). Uses at most max_conversations
    so a sweep over 3 models stays fast."""
    from voiceagent.knowledge import load_docs, build_index
    from voiceagent.agent import build_agent
    from voiceagent.llm import list_available_models, load_llm

    docs = load_docs(knowledge_dir)
    index = build_index(docs)
    models = list_available_models(model_dir)
    if not models:
        raise RuntimeError("no models downloaded — run scripts/smoke_llm.py first")

    reports = []
    for m in models:
        llm = load_llm(m["model_path"], params=m["params"], size_mb=m["size_mb"])
        agent = build_agent(index, llm)
        report = run_benchmark(agent, conversations,
                               max_rows=min(max_rows or len(conversations),
                                            max_conversations),
                               model_specs=llm.specs)
        reports.append(report)

    def sort_key(r: BenchmarkReport):
        passed, _ = evaluate_gate(r)
        return (not passed, -r.summary.resolution_rate, r.model_specs.get("size_mb", 0))
    return sorted(reports, key=sort_key)


def evaluate_gate(report: BenchmarkReport,
                  latency_max_s: float = LATENCY_MAX_S,
                  resolution_min: float = RESOLUTION_MIN) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    s = report.summary
    if s.avg_latency_s > latency_max_s:
        reasons.append(f"latency {s.avg_latency_s:.2f}s > {latency_max_s}s")
    if s.resolution_rate < resolution_min:
        reasons.append(f"resolution {s.resolution_rate:.3f} < {resolution_min}")
    return (not reasons, reasons)


def write_report(report: BenchmarkReport, out_path: str,
                 model_specs: dict | None = None) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = report.summary
    specs = model_specs or report.model_specs or {}
    cost = estimate_cost_per_conversation(specs, s.avg_latency_s, s.resolution_rate)
    tier = estimate_vps_cost(specs)

    lines = [
        "# VoiceAgent M0 Benchmark Report",
        "",
        f"- **Model:** {specs.get('model', '?')} "
        f"({specs.get('params', '?')}, {specs.get('quant', '?')})",
        f"- **Size:** {specs.get('size_mb', '?')} MB",
        f"- **VPS tier:** {tier['vps_tier']} (est ₹{tier['vps_cost_rs_estimate']}/month)",
        f"- **Cost:** ₹{cost['cost_per_resolved_rs']} per resolved conversation "
        f"(₹{cost['cost_per_conversation_rs']} per conversation)",
        f"- **Conversations:** {s.n}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Resolution rate | {s.resolution_rate:.3f} |",
        f"| Grounded rate | {s.grounded_rate:.3f} |",
        f"| Wrong-action rate | {s.wrong_action_rate:.3f} |",
        f"| Hallucination rate | {s.hallucination_rate:.3f} |",
        f"| Avg turn latency (s) | {s.avg_latency_s:.3f} |",
        f"| Est. cost / resolved (₹) | {cost['cost_per_resolved_rs']} |",
        "",
        "## By language",
        "",
        "| Language | n | Resolution | Latency (s) |",
        "|----------|---|------------|-------------|",
    ]
    for lang, ls in sorted(report.per_language.items()):
        lines.append(
            f"| {lang} | {ls.n} | {ls.resolution_rate:.3f} | {ls.avg_latency_s:.3f} |"
        )
    passed, reasons = evaluate_gate(report)
    lines += ["", "## Gate",
              "", f"**PASSED** ({len(reasons)} checks)" if passed
              else "**FAILED**"] + [f"- {r}" for r in reasons]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "summary": s.as_dict(),
        "per_language": {k: v.as_dict() for k, v in report.per_language.items()},
        "model": specs,
        "cost": cost,
    }
    (out.parent / "report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def write_sweep_report(reports: list[BenchmarkReport], out_dir: str) -> str:
    """Write per-model reports + a comparison sweep table. Returns sweep path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(reports):
        name = r.model_specs.get("model", f"model-{i}").replace(".gguf", "")
        write_report(r, str(out_dir / f"report-{name}.md"), r.model_specs)

    lines = [
        "# VoiceAgent M0 Model Sweep",
        "",
        "| Model | Size (MB) | VPS tier | Resolution | Latency (s) | Wrong-action | ₹/resolved | Gate |",
        "|-------|-----------|----------|-----------|-------------|--------------|------------|------|",
    ]
    for r in reports:
        specs = r.model_specs
        cost = estimate_cost_per_conversation(specs, r.summary.avg_latency_s,
                                              r.summary.resolution_rate)
        tier = estimate_vps_cost(specs)
        passed, _ = evaluate_gate(r)
        lines.append(
            f"| {specs.get('model','?')} | {specs.get('size_mb','?')} "
            f"| {tier['vps_tier']} | {r.summary.resolution_rate:.3f} "
            f"| {r.summary.avg_latency_s:.3f} | {r.summary.wrong_action_rate:.3f} "
            f"| {cost['cost_per_resolved_rs']} | {'PASS' if passed else 'FAIL'} |"
        )
    sweep_path = out_dir / "sweep-report.md"
    sweep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(sweep_path)
