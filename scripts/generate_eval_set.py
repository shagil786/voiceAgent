import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.dataset import generate_eval_set

if __name__ == "__main__":
    out = "data/eval/conversations.csv"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"wrote {generate_eval_set(out, n)} rows to {out}")
