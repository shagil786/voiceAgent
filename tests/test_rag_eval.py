import numpy as np

from voiceagent.rag_eval import evaluate


def _fake_build(files):
    """Deterministic keyword-space index: a chunk 'matches' a query iff they
    share a keyword — enough structure to score hit-rate/MRR mechanics."""
    from voiceagent.knowledge_rag import Chunk, ChunkedKnowledge

    chunks, embs, texts = [], [], []
    for fid, text in files.items():
        for i, para in enumerate(text.split("\n\n")):
            chunks.append(Chunk(source_file_id=fid, chunk_index=i, text=para))
            texts.append(para.lower())
    vocab = sorted({w for t in texts for w in t.split()})
    index = {w: j for j, w in enumerate(vocab)}
    for t in texts:
        v = np.zeros(len(vocab))
        for w in t.split():
            v[index[w]] = 1.0
        embs.append(v / (np.linalg.norm(v) or 1.0))
    return ChunkedKnowledge(chunks=chunks, embeddings=np.array(embs),
                            source_texts=dict(files), model_name="fake",
                            embedder=lambda qs: [
                                _fake_vec(q, index, len(vocab)) for q in qs])


def _fake_vec(q, index, dim):
    v = np.zeros(dim)
    for w in q.lower().split():
        if w in index:
            v[index[w]] = 1.0
    return v / (np.linalg.norm(v) or 1.0)


FILES = {
    "eta": "Delivery hours\n\nDeliveries occur between 9:00 and 19:00 local time.",
    "cancel_policy": "Cancellation\n\nOrders that already shipped cannot be cancelled.",
}


def test_eval_scores_hits_and_gaps():
    suite = [
        ("deliveries between what hours", "eta", "paraphrase"),
        ("cancel a shipped order", "cancel_policy", "policy"),
        ("what is the capital of France", None, "gap"),
    ]
    r = evaluate(FILES, suite, build=_fake_build)
    assert r.hits == 2 and r.gaps_correct == 1 and r.gaps_total == 1
    assert 0 < r.mrr <= 1.0
    assert r.summary().startswith("suite=default")


def test_eval_reports_misses():
    suite = [("unrelated gibberish question", "eta", "expected miss")]
    r = evaluate(FILES, suite, build=_fake_build)
    assert r.hits == 0
    assert any(r_.startswith("FAIL") for r_ in r.rows)


def test_ruler_gate_perfect():
    """RAG phase-2 gate: the 24-fixture ruler is at 1.00 hit-rate with gaps
    intact (KB glosses for ETA + cancel-policy phrasings landed 2026-09-08;
    measured 0.89 -> 1.00, then grew with es/fr/de/pt + te/bn fixtures). If
    this drops, a retrieval change regressed the ruler — do not ship without
    beating the previous baseline."""
    from voiceagent.rag_eval import SUITE_DEFAULT, evaluate
    from pathlib import Path as _P
    kb = _P("data/tenants/default/knowledge")
    files = {p.stem: p.read_text(encoding="utf-8") for p in sorted(kb.glob("*.md"))}
    res = evaluate(files, SUITE_DEFAULT, k=6)
    assert res.gaps_correct == res.gaps_total
    assert res.hit_rate >= 1.0, f"ruler regressed: hit_rate={res.hit_rate:.2f} (baseline 1.00)"


def test_clinic_ruler_gate_perfect():
    """Non-default-KB ruler: the clinic tenant's 5-file KB scores 1.00
    hit-rate with gaps intact (measured 0.929 -> 1.00 after a symptom
    line fixed the chest-pain -> emergencies redirect — a safety miss,
    fixed in data). If this drops, a retrieval change or KB edit
    regressed a live tenant's answers."""
    from voiceagent.rag_eval import SUITE_CLINIC, evaluate
    from pathlib import Path as _P
    kb = _P("data/tenants/example-clinic/knowledge")
    files = {p.stem: p.read_text(encoding="utf-8") for p in sorted(kb.glob("*.md"))}
    res = evaluate(files, SUITE_CLINIC, k=6)
    assert res.gaps_correct == res.gaps_total
    assert res.hit_rate >= 1.0, f"clinic ruler regressed: hit_rate={res.hit_rate:.2f} (baseline 1.00)"
