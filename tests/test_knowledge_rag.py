# tests/test_knowledge_rag.py — RAG phase 1: chunked knowledge retrieval with
# per-claim provenance.
"""Covers the four seams of the phase-1 RAG path:

1. Chunking (deterministic, heading-bounded, no mid-sentence splits).
2. The retrieval switch (small KB -> byte-identical whole-file behavior;
   large KB -> chunked index with cached embeddings).
3. Per-turn retrieval into the system prompt (top-K, ordered by similarity,
   fail-open to whole files on embedder error).
4. Provenance + gap detection (TurnResult.retrieved_chunk_ids /
   knowledge_gaps).

Everything runs on an injected deterministic embedder except one
real-encoder test (the shared LaBSE via memory.default_embed).
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from voiceagent.knowledge import CACHE_VERSION as KB_CACHE_VERSION
from voiceagent.knowledge_rag import (
    CHUNK_MAX_CHARS,
    Chunk,
    ChunkedKnowledge,
    build_chunked_index,
    chunk_files,
    chunk_text,
    chunks_hash,
    retrieve_chunks,
)
from voiceagent.memory import InMemoryMemory
from voiceagent.orchestrator import Deployment, Orchestrator, TurnResult
from voiceagent.swarm.frontier import (
    FrontierAgentBridge,
    FrontierReply,
)


# --- deterministic fake embedder ---------------------------------------------

TOPIC_WORDS = {"refund": 0, "delivery": 1, "warranty": 2, "account": 3}


class FakeEmbedder:
    """Deterministic keyword-space embedder: each recognized topic word maps
    to its own axis; unrecognized text lands on a shared "other" axis (so an
    unrelated query is orthogonal to every topical chunk). Counts every batch
    it encodes; `fail=True` makes every call raise (fail-open tests)."""

    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedder down")
        dim = len(TOPIC_WORDS) + 1
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for word, axis in TOPIC_WORDS.items():
                if word in t.lower():
                    out[i, axis] = 1.0
                    break
            else:
                out[i, -1] = 1.0
        return out


def big_kb() -> dict[str, str]:
    """Two large files: faq.md (refund + delivery sections) and policy.md
    (warranty) — together far over the 6000-char budget, while faq.md alone
    still fits (so the whole-file fail-open keeps faq and drops policy)."""
    para = ("Refunds are processed within 5 to 7 business days after the "
            "returned item is received at our warehouse. ")
    faq = ("## Refunds\n\n" + para * 20 + "\n\n"
           "## Delivery\n\n"
           "Delivery takes 3 to 5 business days inside the city limits. "
           "Delivery outside the city takes up to 10 business days. " * 10)
    policy = ("## Warranty\n\n"
              "Every device carries a 24 month warranty covering "
              "manufacturing defects only. " * 35)
    return {"faq": faq, "policy": policy}


# --- 1. chunking ---------------------------------------------------------------

def test_small_file_is_one_chunk_itself():
    text = "## Returns\n\nYou can return any item within 7 days of delivery."
    chunks = chunk_text(text, "faq")
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.source_file_id == "faq"
    assert c.chunk_index == 0
    assert c.chunk_id == "faq:0"
    assert "7 days" in c.text


def test_chunking_splits_on_headings_and_never_spans_them():
    text = ("## Refunds\n\nRefunds take five days. Ask the finance team.\n\n"
            "## Delivery\n\nDelivery takes three days. Ask logistics.\n\n"
            "## Warranty\n\nWarranty covers defects for two years. Keep "
            "the receipt.")
    chunks = chunk_text(text, "faq")
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert any("Refunds" in c.text for c in chunks)
    assert any("Delivery" in c.text for c in chunks)
    assert any("Warranty" in c.text for c in chunks)
    for c in chunks:  # no chunk mixes two sections
        heads = sum(w in c.text for w in ("Refunds", "Delivery", "Warranty"))
        assert heads == 1


def test_chunking_is_deterministic():
    a = chunk_text(big_kb()["faq"], "faq")
    b = chunk_text(big_kb()["faq"], "faq")
    assert a == b
    # file iteration order in chunk_files must not matter either
    files = dict(reversed(list(big_kb().items())))
    assert chunk_files(files) == chunk_files(big_kb())


def test_no_chunk_exceeds_max_and_no_mid_sentence_splits():
    # One section, 8 x ~110-char paragraphs: packing must stay under the
    # 900-char ceiling and only ever cut at paragraph boundaries.
    para = ("The courier partner delivers between 9am and 6pm on weekdays "
            "only. Saturday delivery costs extra. ")
    text = "## Delivery\n\n" + "\n\n".join(para * 3 for _ in range(8))
    chunks = chunk_text(text, "faq")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= CHUNK_MAX_CHARS
    # every chunk ends on a sentence boundary (no mid-sentence cut)
    for c in chunks:
        assert c.text.rstrip().endswith((".", "!", "?"))


def test_oversized_paragraph_splits_on_sentences_not_words():
    sentences = [f"Sentence number {i} explains the refund window in detail "
                 "with plenty of filler words to reach a decent length. "
                 for i in range(12)]
    text = "## Refunds\n\n" + "".join(sentences)  # ~1200 chars, one paragraph
    chunks = chunk_text(text, "faq")
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= CHUNK_MAX_CHARS
        assert c.text.rstrip().endswith(".")
        # reassembly preserves every full sentence, in order
    joined = "\n\n".join(c.text for c in chunks)
    for s in sentences:
        assert s.strip() in joined


def test_chunks_hash_is_order_sensitive_and_stable():
    cs = chunk_files(big_kb())
    assert chunks_hash(cs) == chunks_hash(chunk_files(big_kb()))
    assert chunks_hash(cs) != chunks_hash(chunk_files({"faq": "different"}))


# --- 2. the retrieval switch ---------------------------------------------------

def test_build_chunked_index_embeds_all_chunks_normalized():
    emb = FakeEmbedder()
    ck = build_chunked_index(big_kb(), embedder=emb, model_name="fake",
                             cache_path=None)
    emb2 = FakeEmbedder()
    ck2 = build_chunked_index(big_kb(), embedder=emb2, model_name="fake",
                              cache_path=None)
    assert len(ck.chunks) == len(ck2.chunks)
    assert np.allclose(ck.embeddings, ck2.embeddings)
    norms = np.linalg.norm(ck.embeddings, axis=1)
    assert np.allclose(norms, 1.0)
    assert ck.source_texts == big_kb()   # whole files retained for fail-open
    assert ck.embedder is emb            # query-time encoder is the injected one


def test_cache_hit_skips_reembedding(tmp_path):
    cache = tmp_path / "chunks.pkl"
    emb1 = FakeEmbedder()
    ck1 = build_chunked_index(big_kb(), embedder=emb1, model_name="fake",
                              cache_path=cache)
    assert emb1.calls == 1
    emb2 = FakeEmbedder()
    ck2 = build_chunked_index(big_kb(), embedder=emb2, model_name="fake",
                              cache_path=cache)
    assert emb2.calls == 0  # cache-version respected: no re-embed
    assert np.allclose(ck1.embeddings, ck2.embeddings)
    assert [c.chunk_id for c in ck1.chunks] == [c.chunk_id for c in ck2.chunks]


def test_cache_payload_carries_bumped_version(tmp_path):
    cache = tmp_path / "chunks.pkl"
    build_chunked_index(big_kb(), embedder=FakeEmbedder(), model_name="fake",
                        cache_path=cache)
    payload = pickle.loads(cache.read_bytes())
    assert payload["version"] == KB_CACHE_VERSION + 1


def test_cache_invalidated_when_corpus_changes(tmp_path):
    cache = tmp_path / "chunks.pkl"
    emb1 = FakeEmbedder()
    build_chunked_index(big_kb(), embedder=emb1, model_name="fake",
                        cache_path=cache)
    files = big_kb()
    files["faq"] += "\n\n## Extra\n\nA brand new section about accounts."
    emb2 = FakeEmbedder()
    build_chunked_index(files, embedder=emb2, model_name="fake",
                        cache_path=cache)
    assert emb2.calls == 1  # stale hash -> rebuilt


def test_build_with_failing_embedder_raises_for_switch_to_catch():
    with pytest.raises(RuntimeError):
        build_chunked_index(big_kb(), embedder=FakeEmbedder(fail=True),
                            model_name="fake", cache_path=None)


# --- 3. per-turn retrieval -----------------------------------------------------

def _chunked(embedder=None) -> ChunkedKnowledge:
    return build_chunked_index(big_kb(), embedder=embedder or FakeEmbedder(),
                               model_name="fake", cache_path=None)


def test_retrieve_orders_top_k_by_similarity():
    ck = _chunked()
    hits = retrieve_chunks(ck, "I want a refund for my order", top_k=6)
    assert 0 < len(hits) <= 6
    sims = [s for _, s in hits]
    assert sims == sorted(sims, reverse=True)
    # the refund section outranks everything else
    assert "Refunds" in hits[0][0].text
    assert all(s >= ck.min_similarity for _, s in hits)


def test_retrieve_respects_k_and_floor():
    ck = _chunked()
    # k truncates: the refund topic has >= 2 chunks and all clear the floor
    assert len(retrieve_chunks(ck, "refund", top_k=2)) == 2
    assert len(retrieve_chunks(ck, "refund", top_k=1)) == 1
    # an orthogonal (unrelated) query clears no chunk -> empty, not top-K
    assert retrieve_chunks(ck, "what is the meaning of life", top_k=6) == []


def test_query_embedding_cached_per_turn():
    emb = FakeEmbedder()
    ck = _chunked(emb)
    retrieve_chunks(ck, "refund")
    before = emb.calls
    retrieve_chunks(ck, "refund")  # same query, same index -> cached vector
    assert emb.calls == before
    retrieve_chunks(ck, "delivery")
    assert emb.calls == before + 1


# --- orchestrator integration --------------------------------------------------

def tc_reply(content: str) -> FrontierReply:
    return FrontierReply(content=content, tool_calls=[], model="stub",
                         latency_s=0.001, raw={})


def make_orchestrator(dep: Deployment) -> tuple[Orchestrator, list]:
    captured: list[list[dict]] = []

    class Scripted:
        def chat(self, messages, tools=None, tool_choice="auto",
                 temperature=0.4, max_tokens=512):
            captured.append(messages)
            return tc_reply("Answer.")

    from voiceagent.swarm.frontier import FrontierClient
    orch = Orchestrator(brain=FrontierAgentBridge(FrontierClient.__new__(
        FrontierClient)), runner=None, memory=InMemoryMemory())
    orch.brain.client = Scripted()
    orch.deploy(dep)
    return orch, captured


def chunked_deployment(embedder=None) -> Deployment:
    dep = Deployment(name="big", system_prompt="You are BigCo's agent.")
    dep.chunked_knowledge = _chunked()   # built with a healthy embedder
    if embedder is not None:             # then swap the QUERY-time encoder
        dep.chunked_knowledge.embedder = embedder
    return dep


def test_small_kb_prompt_is_byte_identical_and_knowledge_ids_pinned():
    files = {"returns": "## Returns\n\nReturn within 7 days, no questions "
                        "asked.",
             "eta": "## ETA\n\nOrders arrive in 3 days."}
    dep = Deployment(name="small", system_prompt="You are SmallCo's agent.",
                     knowledge=files)   # <= budget: exactly today's shape
    orch, captured = make_orchestrator(dep)
    res = orch.handle_turn("s1", "what is the refund policy?")
    # byte-identical: the static knowledge block, rendered exactly as today
    expected = ("You are SmallCo's agent.\n\n## Knowledge\n"
                "- [returns] ## Returns\n\nReturn within 7 days, no "
                "questions asked.\n"
                "- [eta] ## ETA\n\nOrders arrive in 3 days.")
    assert captured[0][0]["content"] == expected
    assert res.knowledge_ids == ["returns", "eta"]
    assert res.retrieved_chunk_ids == []          # no chunk mode: no chunk ids
    assert res.knowledge_gaps == []


def test_large_kb_skips_static_block_and_injects_top_k_per_turn():
    dep = chunked_deployment()
    orch, captured = make_orchestrator(dep)
    # deploy() must NOT render a static knowledge block for chunked mode
    assert "## Knowledge" not in orch.brain._system_prompt
    res = orch.handle_turn("s1", "I want a refund for my broken order")
    prompt = captured[0][0]["content"]
    assert "## Knowledge" in prompt
    assert "- [faq:" in prompt                    # "[chunk_id] text" format
    ids = res.retrieved_chunk_ids
    assert ids and len(ids) <= 6
    assert "Refunds" in prompt                    # the matching section came in
    assert res.knowledge_ids == list(dict.fromkeys(   # source files, in order
        cid.rsplit(":", 1)[0] for cid in ids))
    assert res.knowledge_gaps == []


def test_unrelated_query_records_gap_and_notes_no_match():
    dep = chunked_deployment()
    orch, captured = make_orchestrator(dep)
    q = "what is the capital of France and who is its mayor " * 3
    res = orch.handle_turn("s1", q)
    prompt = captured[0][0]["content"]
    assert "No knowledge base entry matched" in prompt
    assert res.retrieved_chunk_ids == []
    assert res.knowledge_gaps == [q[:200]]
    assert res.knowledge_ids == []


def test_embedder_failure_fails_open_to_whole_files_that_fit():
    dep = chunked_deployment(embedder=FakeEmbedder(fail=True))
    orch, captured = make_orchestrator(dep)
    res = orch.handle_turn("s1", "refund please")
    prompt = captured[0][0]["content"]
    assert "## Knowledge" in prompt
    # whole-file fallback: today's _cap_knowledge shape (sorted-id prefix),
    # highest-priority files that fit under the budget
    assert "- [faq] " in prompt
    assert res.retrieved_chunk_ids == []
    assert res.knowledge_ids == ["faq"]           # policy.md did not fit


def test_turn_result_new_fields_default_empty():
    r = TurnResult(reply="x", actions=[], brain_latency_s=0.0, session_id="s",
                   raw_tool_calls=0)
    assert r.retrieved_chunk_ids == [] and r.knowledge_gaps == []


def test_real_encoder_retrieval_ranks_matching_section_first():
    pytest.importorskip("sentence_transformers")
    from voiceagent.memory import default_embed
    # Focused single-fact sections (the shape the 600-900 char chunking
    # produces for a real FAQ): the real encoder must rank the refund chunk
    # first AND clear the 0.35 floor, while unrelated queries do not.
    files = {
        "faq": ("## Refunds\n\n"
                "Refunds are processed within 5-7 business days after we "
                "receive the returned item at our warehouse.\n\n"
                "## Delivery\n\n"
                "Standard delivery takes 3 to 5 business days inside the "
                "city limits."),
        "policy": ("## Warranty\n\n"
                   "Every device carries a 24 month warranty covering "
                   "manufacturing defects only."),
    }
    ck = build_chunked_index(files, embedder=default_embed, cache_path=None)
    hits = retrieve_chunks(ck, "how long do refunds take?", top_k=6)
    assert hits and "Refunds" in hits[0][0].text
    assert hits[0][1] >= 0.35
    assert retrieve_chunks(ck, "what is the capital of France?",
                           top_k=6) == []


