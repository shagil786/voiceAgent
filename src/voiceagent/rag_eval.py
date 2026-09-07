"""Retrieval quality evaluation for the chunked knowledge path.

Offline, deterministic, CI-runnable: a fixture set of (question -> expected
chunk-id or file-id) pairs scored against `retrieve_chunks` with the REAL
encoder. Metrics: hit@k (expected id among the top-k retrieved) and MRR
(mean reciprocal rank). This is the ruler that phase-2 tuning (BM25 blend,
rerank, floor changes) must beat — no retrieval change ships without a
before/after score here.

CLI:  .venv/bin/python -m voiceagent.rag_eval [--k 6] [--suite default]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

# Fixture suite: questions a caller plausibly asks, mapped to the chunk that
# MUST be retrieved (file ids match data/tenants/*/knowledge/ files; chunk
# ids are file:index). Honest mix: paraphrases, multilingual, adversarial
# near-misses (expected: no confident hit).
SUITE_DEFAULT: list[tuple[str, str | None, str]] = [
    # (question, expected_file_id_or_chunk_id, note)
    ("what are your delivery hours", "eta", "plain paraphrase"),
    ("when do deliveries happen", "eta", "paraphrase"),
    ("can I cancel after it shipped", "cancel_policy", "policy question"),
    ("my order already shipped, cancel it", "cancel_policy", "constraint"),
    ("delivery kab hoti hai", "eta", "hindi paraphrase"),
    ("order kab aayega", "eta", "hinglish paraphrase (eta domain)"),
    ("ship ho gaya hai cancel karna hai", "cancel_policy",
     "hinglish constraint (cancel domain)"),
    ("क्या शिप होने के बाद रद्द कर सकते हैं", "cancel_policy", "devanagari"),
    ("ऑर्डर कब डिलीवर होगा", "eta", "devanagari paraphrase"),
    ("what is the capital of France", None, "gap: unrelated must not hit"),
    ("tell me a joke", None, "gap: chit-chat must not hit"),
    ("mazak kar raha tha", None, "gap: hinglish chit-chat must not hit"),
]


@dataclass
class EvalResult:
    suite: str
    k: int
    hits: int = 0
    mrr_sum: float = 0.0
    gaps_correct: int = 0
    gaps_total: int = 0
    rows: list[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        relevant = [r for r in self.rows if "-> expect" in r]
        return self.hits / len(relevant) if relevant else 0.0

    @property
    def mrr(self) -> float:
        relevant = [r for r in self.rows if "-> expect" in r]
        return self.mrr_sum / len(relevant) if relevant else 0.0

    def summary(self) -> str:
        return (f"suite={self.suite} k={self.k} "
                f"hit_rate={self.hit_rate:.2f} mrr={self.mrr:.2f} "
                f"gap_accuracy={self.gaps_correct}/{self.gaps_total}")


def evaluate(files: dict[str, str], suite: list[tuple[str, str | None, str]],
             k: int = 6, floor: float = 0.35,
             build=None, retrieve=None) -> EvalResult:
    """Score `suite` against the chunked index built from `files`.
    build/retrieve injectable for tests; defaults are the real functions."""
    from voiceagent.knowledge_rag import build_chunked_index, retrieve_chunks
    build = build or build_chunked_index
    retrieve = retrieve or retrieve_chunks
    ck = build(files)
    res = EvalResult(suite="default", k=k)
    for question, expected, note in suite:
        hits = retrieve(ck, question, top_k=k, min_similarity=floor)
        retrieved_ids = [c.chunk_id for c, _ in hits] + \
                        [c.source_file_id for c, _ in hits]
        if expected is None:
            res.gaps_total += 1
            ok = not hits
            res.gaps_correct += int(ok)
            res.rows.append(f"{'PASS' if ok else 'FAIL'} gap {question!r} "
                            f"({note}) -> {len(hits)} hits")
            continue
        rank = next((i + 1 for i, rid in enumerate(retrieved_ids)
                     if expected in rid), None)
        if rank:
            res.hits += 1
            res.mrr_sum += 1.0 / rank
            res.rows.append(f"PASS rank={rank} {question!r} -> expect "
                            f"{expected} ({note})")
        else:
            res.rows.append(f"FAIL {question!r} -> expect {expected} "
                            f"({note}); got {[c.chunk_id for c, _ in hits]}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--suite", default="default")
    args = ap.parse_args()
    from pathlib import Path
    kb_dir = Path("data/tenants/default/knowledge")
    files = {p.stem: p.read_text(encoding="utf-8")
             for p in sorted(kb_dir.glob("*.md"))} if kb_dir.exists() else {}
    if not files:
        print("no knowledge files found for the default tenant", file=sys.stderr)
        return 2
    res = evaluate(files, SUITE_DEFAULT, k=args.k)
    print("\n".join(res.rows))
    print(res.summary())
    return 0 if (res.hit_rate >= 0.8 and res.gaps_correct == res.gaps_total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
