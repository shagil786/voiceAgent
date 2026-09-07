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
