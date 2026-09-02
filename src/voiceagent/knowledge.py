# src/voiceagent/knowledge.py
from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# NOTE: sentence-transformers (torch) is imported BEFORE faiss deliberately.
# Loading torch's OpenMP runtime first avoids a fatal macOS segfault when
# faiss and torch coexist in one process (test suite + voice server paths).
import faiss

from voiceagent.langid import NATIVE_SCRIPT_LANGS, detect_language

# M5b: hybrid per-language embedder routing. The M5a-2 sweep showed LaBSE
# lifts native-script languages (bn 0.300->1.000, te 0.700->1.000,
# gu 0.800->1.000, ta 0.533->0.967) but COLLAPSES on Romanized code-mixed
# Hindi (hinglish 0.993->0.700): LaBSE aligns native-script queries well
# but handles Latin-script hinglish poorly, while MiniLM is the reverse.
# So the KB is indexed in TWO embedding spaces over the SAME doc chunks and
# queries are routed by script:
#   en / hinglish (Latin script)     -> paraphrase-multilingual-MiniLM-L12-v2
#   native script (hi/ta/te/bn/...)  -> sentence-transformers/LaBSE
# Vectors never cross spaces: a 384-dim query only ever touches the 384-dim
# index, a 768-dim query only the 768-dim index.
LATIN_SPACE = "latin"
NATIVE_SPACE = "native"

SPACE_EMBEDDERS = {
    LATIN_SPACE: "paraphrase-multilingual-MiniLM-L12-v2",
    NATIVE_SPACE: "sentence-transformers/LaBSE",
}

# Primary-space model: the native-script space (kept under the historical
# name so every entry point's default keeps meaning "LaBSE").
DEFAULT_EMBEDDER = SPACE_EMBEDDERS[NATIVE_SPACE]

# Known embedding dims — lets the cache validate dim without loading the model.
EMBEDDER_DIMS = {
    "sentence-transformers/LaBSE": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}

DEFAULT_CACHE_PATH = "data/index/handle.pkl"
CACHE_VERSION = 3  # v2 pickles (M5a-2) carried ONE space; v3 carries both


def route_space(query: str) -> str:
    """Which embedding space serves this query (single source of truth for
    M5b routing, shared by IndexHandle.search and the intent classifier):
    native-script languages search the LaBSE (native) space, Latin-script
    queries (en/hinglish) the MiniLM (latin) space."""
    return (NATIVE_SPACE if detect_language(query) in NATIVE_SCRIPT_LANGS
            else LATIN_SPACE)


def load_docs(data_dir: str) -> list[dict]:
    """Parse every .md in data_dir into docs. A line starting with '# '
    starts a new section; body lines concatenate as one doc."""
    docs: list[dict] = []
    for md_path in sorted(Path(data_dir).glob("*.md")):
        section = md_path.stem
        current_title = None
        current_lines: list[str] = []
        for line in md_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                if current_title and current_lines:
                    docs.append(_make_doc(section, current_title, current_lines))
                current_title = line[2:].strip()
                current_lines = []
            elif line.strip():
                current_lines.append(line.strip())
        if current_title and current_lines:
            docs.append(_make_doc(section, current_title, current_lines))
    return docs


def _make_doc(section: str, title: str, lines: list[str]) -> dict:
    text = f"{title}. " + " ".join(lines)
    return {"id": hashlib.md5(text.encode()).hexdigest()[:12],
            "text": text, "section": section}


class IndexHandle:
    """FAISS-backed KB handle.

    M5b hybrid: a handle built by build_index/load_or_build_index holds one
    FAISS index per embedding space over the SAME doc chunks — the latin
    space (MiniLM, 384-d) for en/hinglish queries and the native space
    (LaBSE, 768-d) for native-script queries. search() routes by
    detect_language via route_space(); vectors never cross spaces.

    The legacy single-index constructor still works: such a handle searches
    its one space for every query (the dim guard below still applies), so
    callers that build a custom single-model handle keep working.
    """

    def __init__(self, index, ids, model,
                 model_name: str = "", dim: int | None = None):
        # Legacy single-space construction; the space is the primary
        # (native) one. Dual-space handles come from build_index /
        # load_or_build_index via _from_spaces.
        self._ids = ids
        self._spaces: dict[str, dict] = {
            NATIVE_SPACE: {
                "index": index, "model": model, "model_name": model_name,
                "dim": dim if dim is not None else getattr(index, "d", None),
            },
        }
        self.model_name = model_name
        self.dim = self._spaces[NATIVE_SPACE]["dim"]

    @classmethod
    def _from_spaces(cls, spaces: dict[str, dict],
                     ids: list[str]) -> "IndexHandle":
        handle = cls.__new__(cls)
        handle._ids = ids
        handle._spaces = spaces
        primary = spaces[NATIVE_SPACE]
        handle.model_name = primary["model_name"]
        handle.dim = primary["dim"]
        return handle

    def space_for(self, query: str) -> str:
        """Which space serves this query: routed by script when the handle
        is dual-space, else the handle's only space."""
        if len(self._spaces) < 2:
            return next(iter(self._spaces))
        return route_space(query)

    def search(self, query: str, k: int = 3) -> list[dict]:
        space = self.space_for(query)
        rec = self._spaces[space]
        emb = np.asarray(
            rec["model"].encode([query], normalize_embeddings=True),
            dtype=np.float32)
        index = rec["index"]
        index_dim = getattr(index, "d", None)
        if index_dim is not None and index_dim != emb.shape[1]:
            # M5a-2 guard, per space: never query a 384-dim MiniLM index
            # with 768-dim LaBSE vectors (or vice versa) — fail loudly.
            raise ValueError(
                f"query embedding dim {emb.shape[1]} != {space}-space index "
                f"dim {index_dim} — index was built with model "
                f"{rec['model_name']!r}; rebuild the index or use that model")
        scores, idxs = index.search(emb, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            out.append({"id": self._ids[idx], "text": self._texts[idx],
                        "section": self._sections[idx], "score": float(score)})
        return out

    @property
    def _texts(self):  # populated in build_index
        return self._store["texts"]

    @property
    def _sections(self):
        return self._store["sections"]


def build_index(docs: list[dict],
                model_name: str = DEFAULT_EMBEDDER,
                latin_model_name: str = SPACE_EMBEDDERS[LATIN_SPACE]) -> IndexHandle:
    """Dual-space index (M5b): docs are embedded and indexed in BOTH the
    native space (model_name, default LaBSE) and the latin space
    (latin_model_name, default MiniLM), over the same chunks. Defaults give
    the hybrid pair; overriding a name swaps that space's encoder."""
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    spaces: dict[str, dict] = {}
    for space, name in ((NATIVE_SPACE, model_name),
                        (LATIN_SPACE, latin_model_name)):
        model = SentenceTransformer(name)
        emb = np.asarray(model.encode(texts, normalize_embeddings=True),
                         dtype=np.float32)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        spaces[space] = {"index": index, "model": model,
                         "model_name": name, "dim": dim}
    handle = IndexHandle._from_spaces(spaces, ids)
    handle._store = {"texts": texts, "sections": [d["section"] for d in docs]}
    return handle


def docs_hash(docs: list[dict]) -> str:
    """Fingerprint of the doc corpus (order-sensitive over doc ids)."""
    return hashlib.md5("\x00".join(d["id"] for d in docs).encode()).hexdigest()


def read_cache_metadata(cache_path: str | Path = DEFAULT_CACHE_PATH) -> dict | None:
    """Load the pickled index cache payload (None if missing/corrupt)."""
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None


def cache_is_valid(meta: dict | None, model_name: str, corpus_hash: str,
                   dim: int | None = None) -> bool:
    """True iff the cache payload's PRIMARY (native) space was built with
    exactly this model, dim and corpus. Legacy v1/v2 pickles are always
    invalid (they lack the M5b per-space 'spaces' record)."""
    if not isinstance(meta, dict):
        return False
    if meta.get("version") != CACHE_VERSION:
        return False
    if meta.get("model_name") != model_name:
        return False
    if meta.get("docs_hash") != corpus_hash:
        return False
    expected_dim = dim if dim is not None else EMBEDDER_DIMS.get(model_name)
    if expected_dim is not None and meta.get("dim") != expected_dim:
        return False
    return True


def hybrid_cache_is_valid(meta: dict | None, model_name: str, corpus_hash: str,
                          latin_model_name: str = SPACE_EMBEDDERS[LATIN_SPACE],
                          dim: int | None = None) -> bool:
    """True iff the cache carries BOTH embedding spaces for this corpus:
    the primary/native space (checked by cache_is_valid's model/dim/docs
    rules) plus a latin space with its own matching model + dim, and a
    serialized FAISS index for each. Any single-space cache is stale."""
    if not cache_is_valid(meta, model_name, corpus_hash, dim=dim):
        return False
    spaces = meta.get("spaces")
    if not isinstance(spaces, dict):
        return False
    for space, name in ((NATIVE_SPACE, model_name),
                        (LATIN_SPACE, latin_model_name)):
        rec = spaces.get(space)
        if not isinstance(rec, dict):
            return False
        if rec.get("model_name") != name:
            return False
        expected_dim = EMBEDDER_DIMS.get(name)
        if expected_dim is not None and rec.get("dim") != expected_dim:
            return False
        if "faiss" not in rec:
            return False
    return True


def save_index(handle: IndexHandle, docs: list[dict],
               cache_path: str | Path = DEFAULT_CACHE_PATH) -> None:
    """Persist the FAISS indexes + v3 provenance: per-space model name, dim
    and serialized index (so a cache hit skips re-embedding the corpus in
    every space), plus the v2-style top-level model_name/dim fields for the
    primary (native) space — cache_is_valid's contract."""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    primary = handle._spaces[NATIVE_SPACE]
    payload = {
        "version": CACHE_VERSION,
        "model_name": primary["model_name"],
        "dim": primary["dim"],
        "docs_hash": docs_hash(docs),
        "ids": [d["id"] for d in docs],
        "texts": [d["text"] for d in docs],
        "sections": [d["section"] for d in docs],
        "spaces": {
            space: {"model_name": rec["model_name"], "dim": rec["dim"],
                    "faiss": np.asarray(faiss.serialize_index(rec["index"]))}
            for space, rec in handle._spaces.items()
        },
    }
    with open(p, "wb") as f:
        pickle.dump(payload, f)


def load_or_build_index(docs: list[dict],
                        model_name: str = DEFAULT_EMBEDDER,
                        cache_path: str | Path = DEFAULT_CACHE_PATH,
                        latin_model_name: str = SPACE_EMBEDDERS[LATIN_SPACE],
                        ) -> IndexHandle:
    """Hybrid index over docs (M5b): reuses the pickled cache when BOTH
    spaces were built with the requested models over the same corpus;
    otherwise rebuilds both spaces and re-saves. A v2 single-space cache
    (or a v3 payload missing a space) is detected as stale and rebuilt, so
    a 384-dim MiniLM-only cache is never served for hybrid queries."""
    corpus_hash = docs_hash(docs)
    meta = read_cache_metadata(cache_path)
    if hybrid_cache_is_valid(meta, model_name, corpus_hash,
                             latin_model_name=latin_model_name):
        spaces: dict[str, dict] = {}
        for space in (NATIVE_SPACE, LATIN_SPACE):
            rec = meta["spaces"][space]
            spaces[space] = {
                "index": faiss.deserialize_index(np.asarray(rec["faiss"])),
                "model": SentenceTransformer(rec["model_name"]),
                "model_name": rec["model_name"],
                "dim": rec["dim"],
            }
        handle = IndexHandle._from_spaces(spaces, meta["ids"])
        handle._store = {"texts": meta["texts"],
                         "sections": meta["sections"]}
        return handle
    handle = build_index(docs, model_name=model_name,
                         latin_model_name=latin_model_name)
    save_index(handle, docs, cache_path)
    return handle
