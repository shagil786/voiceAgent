import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.dataset import load_conversations
from voiceagent.benchmark import (sweep_all_models, write_sweep_report,
                                  evaluate_gate)
from voiceagent.policy import load_policies
from voiceagent.decisionlog import DecisionLog

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the model sweep benchmark.")
    ap.add_argument("n", nargs="?", type=int, default=200,
                    help="max conversations per model (default 200)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model stems/names to include "
                         "(default: every downloaded model), e.g. "
                         "qwen2.5-0.5b-instruct-q4_k_m,qwen2.5-0.5b-hinglish-q4_k_m")
    ap.add_argument("--sample", choices=["head", "stride"], default="head",
                    help="head = first n conversations (legacy behavior; "
                         "excludes appended conv-2000+ rows); stride = "
                         "deterministic even sample across the whole file "
                         "so appended language blocks are represented")
    ap.add_argument("--max-convs", type=int, default=200,
                    help="per-model conversation cap in sweep_all_models "
                         "(default 200; raise to >= dataset size to evaluate "
                         "every row, including conv-2000+ native rows)")
    args = ap.parse_args()

    convs = load_conversations("data/eval/conversations.csv")
    if args.sample == "stride" and args.n < len(convs):
        step = len(convs) / args.n
        convs = [convs[int(i * step)] for i in range(args.n)]
    policy = load_policies("data/policies/policies.yaml")
    log = DecisionLog()
    wanted = [m for m in args.models.split(",") if m.strip()] \
        if args.models else None
    reports = sweep_all_models(convs, knowledge_dir="data/knowledge",
                               model_dir="data/models", max_rows=args.n,
                               policy=policy, decision_log=log,
                               model_filter=wanted,
                               max_conversations=args.max_convs)
    path = write_sweep_report(reports, "data/out")
    print("wrote", path)
    log.to_json("data/out/decision-log.json")
    log.to_csv("data/out/decision-log.csv")
    print(f"decision log: {len(log.entries())} entries")
    from collections import Counter
    print(Counter(x.verdict for x in log.entries()))
    for r in reports:
        passed, reasons = evaluate_gate(r)
        print(f"{r.model_specs.get('model','?'):40s} "
              f"res={r.summary.resolution_rate:.3f} "
              f"lat={r.summary.avg_latency_s:.3f}s "
              f"policy={r.policy_summary} "
              f"gate={'PASS' if passed else 'FAIL'}")

    # --- M2: handoff sample + billing summary ---
    if len(reports):
        from voiceagent.handoff import HandoffBundle, handoff_markdown
        from voiceagent.billing import compute_billing
        top = reports[0]
        esc = [e for e in log.entries() if e.verdict == "ESCALATE"]
        if esc:
            e = esc[0]
            conv = next((c for c in convs if c.id == e.conv_id), convs[0])
            h = HandoffBundle(conv_id=e.conv_id, user_text=conv.user_text,
                              reply="<see decision log>", action=e.action,
                              decision=e.verdict, decision_reasons=e.reasons,
                              retrieved=[], amount=e.amount, order_id=None,
                              authenticated=e.authenticated)
            Path("data/out/handoff-sample.md").write_text(
                handoff_markdown(h), encoding="utf-8")
        b = compute_billing([r for r in top.rows], log)
        Path("data/out/billing.json").write_text(
            json.dumps(b, indent=2), encoding="utf-8")
        print("billing:", b)
