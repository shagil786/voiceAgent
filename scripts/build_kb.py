import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.knowledge import load_docs, build_index

if __name__ == "__main__":
    data_dir = "data/knowledge"
    docs = load_docs(data_dir)
    idx = build_index(docs)
    Path("data/index").mkdir(exist_ok=True)
    import pickle
    with open("data/index/handle.pkl", "wb") as f:
        pickle.dump({"ids": [d["id"] for d in docs],
                     "texts": [d["text"] for d in docs],
                     "sections": [d["section"] for d in docs]}, f)
    # verify search
    for q in ["how long do refunds take?", "refund limit approval"]:
        print(q, "->", idx.search(q, k=2)[0]["section"])
