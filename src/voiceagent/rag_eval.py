"""Retrieval quality evaluation for the chunked knowledge path.

Offline, deterministic, CI-runnable: a fixture set of (question -> expected
chunk-id or file-id) pairs scored against `retrieve_chunks` with the REAL
encoder. Metrics: hit@k (expected id among the top-k retrieved) and MRR
(mean reciprocal rank). This is the ruler that phase-2 tuning (BM25 blend,
rerank, floor changes) must beat — no retrieval change ships without a
before/after score here.

The ruler deliberately includes romanized-hinglish and Devanagari questions
plus hinglish chit-chat gap probes: the single-space (LaBSE-only) chunk
path failed exactly there (Latin-script hinglish retrieved 0 hits).

Baseline history (real encoders, k=6, per-space floors):
  single-space chunk index : hit_rate=0.83 mrr=0.83 gap=2/2 (original
                             8-fixture suite); hit_rate=0.67 mrr=0.61
                             gap=3/3 on the expanded suite below
  dual-space (phase-2 step): hit_rate=0.89 mrr=0.78 gap=3/3 (expanded
                             suite) — remaining miss 'order kab aayega'
                             ranks a lexically-overlapping wrong chunk
                             first (the deferred BM25/rerank lever's job)
  KB multilingual glosses   : hit_rate=1.00 mrr=0.98 gap=3/3 — the
  2026-09-08               : 'order kab aayega' routing miss was fixed by
                             glossing the KB with the phrasings callers
                             actually use; the ruler then GREW with es/fr/
                             de/pt (latin space) + te/bn (native space)
                             fixtures asserting the global-languages claim.
                             The glosses moved latin-space chit-chat above
                             the old 0.20 floor (gap probes 0.147-0.287,
                             weakest content 0.426), so the latin floor was
                             recalibrated 0.20 -> 0.32 (both clusters now
                             sit ~0.09 from the floor; details in
                             knowledge_rag.MIN_SIMILARITY_LATIN). This
                             ruler now carries 24 fixtures.

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
    # Global-languages ruler (2026-09-08): es/fr/de/pt route to the latin
    # space (MiniLM), te/bn to the native space (LaBSE) — the same languages
    # the data tables claim to serve (langid, reply templates, TTS). A miss
    # here is a live deployment gap: the caller would get no (or the wrong)
    # KB chunk in a language the platform claims to support.
    ("¿cuándo llega mi pedido?", "eta", "es paraphrase"),
    ("quand arrive ma commande ?", "eta", "fr paraphrase"),
    ("wann kommt meine Bestellung an?", "eta", "de paraphrase"),
    ("quando chega meu pedido?", "eta", "pt paraphrase"),
    ("డెలివరీ ఎప్పుడు ఉంటుంది?", "eta", "te paraphrase"),
    ("ডেলিভারি কবে হবে?", "eta", "bn paraphrase"),
    ("¿puedo cancelar después del envío?", "cancel_policy", "es policy"),
    ("puis-je annuler après l'expédition ?", "cancel_policy", "fr policy"),
    ("kann ich nach dem Versand stornieren?", "cancel_policy", "de policy"),
    ("posso cancelar depois do envio?", "cancel_policy", "pt policy"),
    ("షిప్ అయ్యాక రద్దు చేయవచ్చా?", "cancel_policy", "te policy"),
    ("শিপ হওয়ার পর বাতিল করা যাবে?", "cancel_policy", "bn policy"),
    # 2026-09: the routed-but-untested languages — ta/th/mr/gu/kn/ml/pa
    # now have KB glosses (eta.md/cancel_policy.md tail lines). Initial
    # phrasings, same standing as the earlier gloss sets.
    ("டெலிவரி எப்போது?", "eta", "ta paraphrase"),
    ("จัดส่งเมื่อไหร่?", "eta", "th paraphrase"),
    ("डिलिव्हरी कधी होते?", "eta", "mr paraphrase"),
    ("ડિલિવરી ક્યારે થાય?", "eta", "gu paraphrase"),
    ("ಡೆಲಿವರಿ ಯಾವಾಗ?", "eta", "kn paraphrase"),
    ("ഡെലിവറി എപ്പോൾ?", "eta", "ml paraphrase"),
    ("ਡਿਲਿਵਰੀ ਕਦੋਂ ਹੁੰਦੀ ਹੈ?", "eta", "pa paraphrase"),
    ("ஷிப் ஆன பிறகு ரத்து செய்யலாமா?", "cancel_policy", "ta policy"),
    ("จัดส่งแล้วขอยกเลิกได้ไหม?", "cancel_policy", "th policy"),
    ("शिप झाल्यावर रद्द करता येईल का?", "cancel_policy", "mr policy"),
    ("શિપ થયા પછી રદ્દ કરી શકાય?", "cancel_policy", "gu policy"),
    ("ಶಿಪ್ ಆದ ನಂತರ ರದ್ದು ಮಾಡಬಹುದೇ?", "cancel_policy", "kn policy"),
    ("ഷിപ്പ് ചെയ്തതിന് ശേഷം റദ്ദാക്കാമോ?", "cancel_policy", "ml policy"),
    ("ਸ਼ਿਪ ਹੋਣ ਤੋਂ ਬਾਅਦ ਰੱਦ ਕਰ ਸਕਦੇ ਹਾਂ?", "cancel_policy", "pa policy"),
    ("what is the capital of France", None, "gap: unrelated must not hit"),
    ("tell me a joke", None, "gap: chit-chat must not hit"),
    ("mazak kar raha tha", None, "gap: hinglish chit-chat must not hit"),
]


# Clinic-tenant ruler: the non-default-KB gap. Same schema, file ids match
# data/tenants/example-clinic/knowledge/*.md. The clinic KB is English-only
# while the tenant serves hi — the Hindi probes below measure exactly that
# gap (glosses or a recorded miss, never a silent one).
SUITE_CLINIC: list[tuple[str, str | None, str]] = [
    ("what time do you open", "hours", "plain paraphrase"),
    ("are you open on sundays", "hours", "hours question"),
    ("when should I reach for my appointment", "hours", "check-in timing"),
    ("can I cancel my appointment", "cancellation-policy", "policy question"),
    ("reschedule my visit to friday", "cancellation-policy", "reschedule"),
    ("do you take insurance", "billing-insurance", "insurance"),
    ("can I pay by UPI", "billing-insurance", "payment mode"),
    ("I want a bill adjustment", "billing-insurance", "adjustment"),
    ("chest pain what should I do", "emergencies", "emergency redirect"),
    ("is this line for emergencies", "emergencies", "scope"),
    ("need a refill of my prescription", "prescription-refills", "refill"),
    ("where do I collect my medicines", "prescription-refills", "pickup"),
    ("clinic kab khulti hai", "hours", "hinglish hours (gap probe)"),
    ("appointment cancel karna hai", "cancellation-policy",
     "hinglish cancel (gap probe)"),
    ("what is the capital of France", None, "gap: unrelated must not hit"),
    ("tell me a joke", None, "gap: chit-chat must not hit"),
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
             k: int = 6, floor: float | None = None,
             build=None, retrieve=None) -> EvalResult:
    """Score `suite` against the chunked index built from `files`.
    build/retrieve injectable for tests; defaults are the real functions.
    floor=None (default) applies the per-space module floors (the calibrated
    ones — the honest production configuration); an explicit float overrides
    the routed space's floor for experiments."""
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
