import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.dataset import load_conversations
from voiceagent.benchmark import (sweep_all_models, write_sweep_report,
                                  evaluate_gate)
from voiceagent.policy import load_policies
from voiceagent.decisionlog import DecisionLog

if __name__ == "__main__":
    convs = load_conversations("data/eval/conversations.csv")
    policy = load_policies("data/policies/policies.yaml")
    log = DecisionLog()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    reports = sweep_all_models(convs, knowledge_dir="data/knowledge",
                               model_dir="data/models", max_rows=n,
                               policy=policy, decision_log=log)
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