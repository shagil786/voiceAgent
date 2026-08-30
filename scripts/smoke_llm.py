import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.llm import CANDIDATE_MODELS, download_model, load_llm

if __name__ == "__main__":
    targets = sys.argv[1:] or [c["name"] for c in CANDIDATE_MODELS]
    for cand in CANDIDATE_MODELS:
        if cand["name"] not in targets:
            continue
        path = download_model(cand["url"])
        llm = load_llm(path, params=cand["params"], size_mb=cand["size_mb"])
        t0 = time.time()
        print(f"[{cand['name']}] ->",
              llm.generate("Reply in one line: order ORD-1 status?"))
        print(f"  first inference: {time.time()-t0:.2f}s (includes model load)")
