import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.dataset import load_conversations
from voiceagent.benchmark import (sweep_all_models, write_sweep_report,
                                  evaluate_gate)

if __name__ == "__main__":
    convs = load_conversations("data/eval/conversations.csv")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    reports = sweep_all_models(convs, knowledge_dir="data/knowledge",
                               model_dir="data/models", max_rows=n)
    path = write_sweep_report(reports, "data/out")
    print("wrote", path)
    for r in reports:
        passed, reasons = evaluate_gate(r)
        print(f"{r.model_specs.get('model','?'):40s} "
              f"res={r.summary.resolution_rate:.3f} "
              f"lat={r.summary.avg_latency_s:.3f}s "
              f"gate={'PASS' if passed else 'FAIL'}")
