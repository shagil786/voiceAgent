# tests/test_index_cache.py
"""M5a-2: model-aware index cache. data/index/handle.pkl is versioned with
the embedder model name + vector dim + a docs fingerprint; load_or_build_index
rebuilds automatically when the cache was built with a different model (or
different docs), and IndexHandle.search refuses to run when the query vector
dim does not match the stored FAISS index dim.

M5b hybrid: the handle (and its cache) carries TWO embedding spaces over the
same doc chunks — latin (MiniLM, 384-d) for en/hinglish queries and native
(LaBSE, 768-d) for native-script queries. Cache metadata records model+dim
per space ("spaces" dict); the v2-style top-level model_name/dim fields
remain the PRIMARY (native) space provenance so cache_is_valid keeps its
single-space contract. A cache without both spaces is stale and rebuilds.

Stub-model approach: SentenceTransformer is monkeypatched with a deterministic
fake so no second big model is downloaded in tests (real-model behavior is
covered by test_knowledge.py / test_intent.py via the default models)."""
import pickle
from pathlib import Path

import torch  # noqa: F401  — load torch OpenMP before faiss (macOS segfault)
import faiss
import numpy as np
import pytest

import voiceagent.knowledge as kb
from voiceagent.knowledge import (DEFAULT_EMBEDDER, IndexHandle,
                                  cache_is_valid, docs_hash, load_docs,
                                  load_or_build_index, read_cache_metadata,
                                  save_index)


def _tiny_docs(tmp: Path) -> list[dict]:
    (tmp / "faqs.md").write_text(
        "# Returns\nYou can return any item within 7 days of delivery.\n\n"
        "# Refunds\nRefunds are processed within 5-7 business days.\n"
    )
    return load_docs(str(tmp))


class _FakeST:
    """Deterministic fake embedder: vector = hash of the text, 8 dims."""
    dim = 8

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))  # deterministic within a process
            for j in range(self.dim):
                out[i, j] = ((h >> (j * 3)) & 1) * 2.0 - 1.0
        if normalize_embeddings:
            norm = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norm, 1e-9)
        return out


@pytest.fixture
def fake_st(monkeypatch):
    monkeypatch.setattr(kb, "SentenceTransformer", _FakeST)
    return _FakeST


@pytest.fixture
def spy_st(monkeypatch):
    """Like fake_st, but every constructed embedder records the texts it was
    asked to encode — lets routing tests observe WHICH space ran."""
    created = []

    class _SpyST(_FakeST):
        def __init__(self, model_name):
            super().__init__(model_name)
            self.queries: list[str] = []
            created.append(self)

        def encode(self, texts, normalize_embeddings=False):
            self.queries.extend(texts)
            return super().encode(texts, normalize_embeddings=normalize_embeddings)

    monkeypatch.setattr(kb, "SentenceTransformer", _SpyST)
    return created


def _spy_by_name(created, model_name):
    return next(s for s in created if s.model_name == model_name)


# --- pure metadata logic (no model needed at all) --------------------------

def test_cache_is_valid_requires_version_model_and_docs_hash():
    meta = {"version": kb.CACHE_VERSION, "model_name": "m-A", "dim": 8,
            "docs_hash": "h1"}
    assert cache_is_valid(meta, "m-A", "h1", dim=8)
    # different model -> invalid (the M5a-2 core rule)
    assert not cache_is_valid(meta, "m-B", "h1", dim=8)
    # same model, different dim -> invalid (never 384-dim query on 768-dim)
    assert not cache_is_valid(meta, "m-A", "h1", dim=768)
    # docs changed -> invalid
    assert not cache_is_valid(meta, "m-A", "h2", dim=8)
    # legacy unversioned pickle (pre-M5a-2 format) -> invalid
    assert not cache_is_valid({"ids": ["a"]}, "m-A", "h1", dim=8)
    # missing cache -> invalid
    assert not cache_is_valid(None, "m-A", "h1", dim=8)


def test_read_cache_metadata_missing_file_returns_none(tmp_path):
    assert read_cache_metadata(tmp_path / "nope.pkl") is None


def test_read_cache_metadata_reads_legacy_v1_pickle(tmp_path):
    p = tmp_path / "handle.pkl"
    with open(p, "wb") as f:
        pickle.dump({"ids": ["a"], "texts": ["x"], "sections": ["s"]}, f)
    meta = read_cache_metadata(p)
    assert meta is not None and "version" not in meta  # unversioned -> invalid


# --- full roundtrip with the fake embedder (no downloads) ------------------

def test_load_or_build_saves_versioned_cache(fake_st, tmp_path):
    docs = _tiny_docs(tmp_path)
    cache = tmp_path / "handle.pkl"
    handle = load_or_build_index(docs, model_name="fake-native",
                                 latin_model_name="fake-latin", cache_path=cache)
    assert handle.model_name == "fake-native"  # primary (native) space
    assert handle.dim == _FakeST.dim
    meta = read_cache_metadata(cache)
    assert meta["version"] == kb.CACHE_VERSION
    # top-level fields remain the primary (native) space provenance
    assert meta["model_name"] == "fake-native"
    assert meta["dim"] == _FakeST.dim
    assert meta["docs_hash"] == docs_hash(docs)
    # M5b: per-space provenance — both spaces recorded with model + dim
    assert set(meta["spaces"]) == {kb.NATIVE_SPACE, kb.LATIN_SPACE}
    assert meta["spaces"]["native"]["model_name"] == "fake-native"
    assert meta["spaces"]["latin"]["model_name"] == "fake-latin"
    assert meta["spaces"]["native"]["dim"] == _FakeST.dim
    assert meta["spaces"]["latin"]["dim"] == _FakeST.dim


def test_load_or_build_reuses_valid_cache(fake_st, tmp_path):
    docs = _tiny_docs(tmp_path)
    cache = tmp_path / "handle.pkl"
    first = load_or_build_index(docs, model_name="fake-native",
                                latin_model_name="fake-latin", cache_path=cache)

    def _boom(*a, **kw):  # rebuild must NOT happen on a valid cache
        raise AssertionError("rebuild attempted despite valid cache")
    import voiceagent.knowledge as k
    orig, k.build_index = k.build_index, _boom
    try:
        second = load_or_build_index(docs, model_name="fake-native",
                                     latin_model_name="fake-latin",
                                     cache_path=cache)
    finally:
        k.build_index = orig
    assert first.search("refunds how long", k=1)[0]["id"] == \
        second.search("refunds how long", k=1)[0]["id"]


def test_load_or_build_invalidates_on_model_mismatch(fake_st, tmp_path):
    docs = _tiny_docs(tmp_path)
    cache = tmp_path / "handle.pkl"
    # cache claims it was built with a DIFFERENT model than requested
    with open(cache, "wb") as f:
        pickle.dump({"version": kb.CACHE_VERSION, "model_name": "old-model",
                     "dim": 4, "docs_hash": docs_hash(docs),
                     "ids": [d["id"] for d in docs],
                     "texts": [d["text"] for d in docs],
                     "sections": [d["section"] for d in docs],
                     "faiss": b"\x00"}, f)
    rebuilt = []

    import voiceagent.knowledge as k
    real_build = k.build_index

    def _recording_build(docs_, model_name="x", latin_model_name="y"):
        # wrap (not replace) the real builder: kb.SentenceTransformer is
        # already stubbed by the fake_st fixture, so this is download-free
        rebuilt.append(model_name)
        return real_build(docs_, model_name=model_name,
                          latin_model_name=latin_model_name)

    k.build_index = _recording_build
    try:
        handle = load_or_build_index(docs, model_name=DEFAULT_EMBEDDER,
                                     cache_path=cache)
    finally:
        k.build_index = real_build

    assert rebuilt == [DEFAULT_EMBEDDER]  # rebuilt with the requested model
    assert handle.model_name == DEFAULT_EMBEDDER
    meta = read_cache_metadata(cache)  # cache rewritten with new metadata
    assert meta["model_name"] == DEFAULT_EMBEDDER
    assert meta["spaces"]["latin"]["model_name"] == \
        kb.SPACE_EMBEDDERS[kb.LATIN_SPACE]


def test_load_or_build_invalidates_single_space_cache(fake_st, tmp_path):
    # M5b: a payload with a valid native space but NO latin space (e.g. a
    # single-space handle was saved) must not be served as a hybrid cache.
    docs = _tiny_docs(tmp_path)
    cache = tmp_path / "handle.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"version": kb.CACHE_VERSION, "model_name": "fake-native",
                     "dim": 8, "docs_hash": docs_hash(docs),
                     "ids": [d["id"] for d in docs],
                     "texts": [d["text"] for d in docs],
                     "sections": [d["section"] for d in docs],
                     "faiss": b"\x00",
                     "spaces": {"native": {"model_name": "fake-native",
                                           "dim": 8, "faiss": b"\x00"}}}, f)
    rebuilt = []
    import voiceagent.knowledge as k
    real_build = k.build_index

    def _recording_build(docs_, model_name="x", latin_model_name="y"):
        rebuilt.append(model_name)
        return real_build(docs_, model_name=model_name,
                          latin_model_name=latin_model_name)

    k.build_index = _recording_build
    try:
        handle = load_or_build_index(docs, model_name="fake-native",
                                     latin_model_name="fake-latin",
                                     cache_path=cache)
    finally:
        k.build_index = real_build

    assert rebuilt == ["fake-native"]
    assert set(handle._spaces) == {kb.NATIVE_SPACE, kb.LATIN_SPACE}


# --- M5b routing: search picks the space matching the query's script -------

def test_search_routes_latin_queries_to_latin_space(spy_st, tmp_path):
    docs = _tiny_docs(tmp_path)
    cache = tmp_path / "handle.pkl"
    handle = load_or_build_index(docs, model_name="fake-native",
                                 latin_model_name="fake-latin", cache_path=cache)
    native = _spy_by_name(spy_st, "fake-native")
    latin = _spy_by_name(spy_st, "fake-latin")
    for query in ("Where is my order?",          # en
                  "mera order kab aayega"):      # hinglish (Latin script)
        native.queries.clear()
        handle.search(query, k=1)
        assert query in latin.queries
        assert native.queries == [], f"{query!r} leaked into the native space"


def test_search_routes_native_script_queries_to_native_space(spy_st, tmp_path):
    docs = _tiny_docs(tmp_path)
    cache = tmp_path / "handle.pkl"
    handle = load_or_build_index(docs, model_name="fake-native",
                                 latin_model_name="fake-latin", cache_path=cache)
    native = _spy_by_name(spy_st, "fake-native")
    latin = _spy_by_name(spy_st, "fake-latin")
    for query in ("मेरा ऑर्डर कहाँ है",                       # hi (Devanagari)
                  "என் ஆர்டர் எந்த நிலையில் உள்ளது?"):          # ta (Tamil)
        latin.queries.clear()
        handle.search(query, k=1)
        assert query in native.queries
        assert latin.queries == [], f"{query!r} leaked into the latin space"


def test_single_space_handle_searches_its_only_space(spy_st, tmp_path):
    # Legacy constructor: one index + one model. No dual routing — every
    # query goes to the only space (dim guard still enforced). The model is
    # built via the patched constructor so the spy records it.
    model = kb.SentenceTransformer("solo")
    handle = IndexHandle(faiss.IndexFlatIP(8), ["a"], model,
                         model_name="solo", dim=8)
    handle._store = {"texts": ["t"], "sections": ["s"]}
    spy = next(s for s in spy_st if s.model_name == "solo")
    for query in ("Where is my order?", "मेरा ऑर्डर कहाँ है"):
        handle.search(query, k=1)
        assert query in spy.queries


def test_search_refuses_dim_mismatch(monkeypatch):
    # "Never query a 384-dim index with 768-dim vectors": the handle must
    # raise instead of returning garbage when dims disagree.
    class _FakeIndex:
        d = 384
        def search(self, x, k):  # pragma: no cover - must never be reached
            raise AssertionError("search ran on dim-mismatched index")
    class _FakeModel:
        def encode(self, texts, normalize_embeddings=False):
            return np.ones((len(texts), 768), dtype=np.float32)
    handle = IndexHandle(_FakeIndex(), ["a"], _FakeModel())
    handle._store = {"texts": ["t"], "sections": ["s"]}
    with pytest.raises(ValueError, match="dim"):
        handle.search("anything", k=1)
