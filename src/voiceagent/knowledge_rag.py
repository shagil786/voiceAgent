# src/voiceagent/knowledge_rag.py — RAG phase 1: chunked knowledge retrieval.
"""Chunked knowledge retrieval with per-claim provenance.

Today the WHOLE tenant KB (capped at the budget, whole-file granularity) is
injected into every turn's system prompt. That breaks the moment a real
tenant's KB outgrows the budget: dropping a file whole asserts the opposite
of the text it cut. This module adds the retrieval path:

1. **Chunking**: each knowledge markdown file is split deterministically —
   on headings first (a chunk never spans two headings), then paragraphs;
   oversized paragraphs are split on sentences only (never mid-sentence).
   Chunks target ~600-900 chars and carry (source_file_id, chunk_index, text).
2. **Index**: chunk texts are embedded once (DEFAULT_EMBEDDER, L2-normalized)
   and cached on disk like knowledge.py's index cache, with a BUMPED version
   (knowledge.CACHE_VERSION + 1) so a chunk cache is never served for a
   different payload shape or corpus.
3. **Retrieval**: per turn, the top-K chunks by cosine against the user text
   are injected into that turn's system prompt; a similarity floor detects
   knowledge gaps; any retrieval error fails open to the historical
   whole-file cap (cap_knowledge).

Importable with zero heavy deps: numpy is light (memory.py already needs
it); sentence-transformers/faiss load lazily inside the build path only.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# The knowledge budget (characters) — mirrored by runtime.MAX_KNOWLEDGE_CHARS
# so both the whole-file cap and the switch threshold share one number.
KNOWLEDGE_BUDGET_CHARS = 6000

# Chunking shape: chunks pack paragraphs up to CHUNK_MAX_CHARS (hard ceiling,
# never exceeded except by a single unsplittable sentence); files smaller
# than the ceiling stay one chunk (themselves).
CHUNK_TARGET_CHARS = 750
CHUNK_MAX_CHARS = 900

# Retrieval shape (phase 1 choices): top-K chunks per turn, cosine floor
# below which nothing is injected and a knowledge gap is recorded.
# Floor judgment call: with focused 600-900-char FAQ chunks in the LaBSE
# space, a matching question lands ~0.35-0.6 while unrelated text sits
# ~0.10-0.25 — 0.35 separates them (measured against the shipped corpus
# shape). BM25/rerank (phase 2) may revisit this.
TOP_K = 6
MIN_SIMILARITY = 0.35

# Disk cache for chunk embeddings — the knowledge.py pickle-cache pattern
# with a BUMPED version (CACHE_VERSION + 1): a chunk payload is a different
# shape from the FAISS handle payload, and bumping the shared constant means
# any future knowledge.py cache bump invalidates chunk caches too.
DEFAULT_CHUNKS_CACHE_PATH = "data/index/chunks.pkl"

_HEADING_RE = re.compile(r"^#{1,6}\s+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
# Sentence boundary: terminal punctuation (including the Devanagari danda
# for Hindi prose) followed by whitespace. Never split inside a sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")


# --- chunk model ---------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of one knowledge file. `chunk_id` is the
    provenance token rendered into the system prompt ("[chunk_id] text")."""
    source_file_id: str
    chunk_index: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.source_file_id}:{self.chunk_index}"


@dataclass
class ChunkedKnowledge:
    """The deployment-side chunk index: chunks + their (normalized) chunk
    embeddings + the whole-file texts kept ONLY for the fail-open fallback.
    `embedder` is the query-time encoder (injected for tests; None defers to
    the shared lazy default embedder)."""
    chunks: list[Chunk]
    embeddings: np.ndarray              # (n_chunks, dim), L2-normalized
    model_name: str
    source_texts: dict[str, str]
    embedder: Callable[[list[str]], np.ndarray] | None = None
    top_k: int = TOP_K
    min_similarity: float = MIN_SIMILARITY
    _query_cache: tuple[str, int, np.ndarray] | None = field(
        default=None, repr=False, compare=False)


# --- chunking ------------------------------------------------------------------

def _split_sections(text: str) -> list[str]:
    """Split on markdown headings (## first-class, but any #-level so a chunk
    never spans a heading boundary). The heading line leads its section."""
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _HEADING_RE.match(line) and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    tail = "\n".join(current).strip()
    if tail:
        sections.append(tail)
    return [s for s in sections if s.strip()]


def _sentence_groups(paragraph: str) -> list[str]:
    """Split one oversized paragraph into sentence groups, each within the
    ceiling. A single sentence LONGER than the ceiling still becomes its own
    chunk (no mid-sentence split ever) — that chunk may exceed the ceiling
    by construction and that is the honest trade."""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    groups: list[str] = []
    buf: list[str] = []
    blen = 0
    for s in sentences:
        if buf and blen + 1 + len(s) > CHUNK_MAX_CHARS:
            groups.append(" ".join(buf))
            buf, blen = [], 0
        buf.append(s)
        blen += len(s) + (1 if blen else 0)
    if buf:
        groups.append(" ".join(buf))
    return groups


def chunk_text(text: str, source_file_id: str) -> list[Chunk]:
    """Deterministically split one knowledge file into chunks: heading
    sections first, paragraphs inside a section, sentences inside an
    oversized paragraph. A file under the ceiling is one chunk (itself)."""
    chunks: list[Chunk] = []
    index = 0
    for section in _split_sections(text):
        paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(section)
                      if p.strip()]
        buf: list[str] = []
        blen = 0

        def emit() -> None:
            nonlocal index, buf, blen
            if buf:
                chunks.append(Chunk(source_file_id=source_file_id,
                                    chunk_index=index,
                                    text="\n\n".join(buf)))
                index += 1
                buf, blen = [], 0

        for para in paragraphs:
            if len(para) > CHUNK_MAX_CHARS:
                # A pending heading (or small paragraphs) rides with the
                # oversized paragraph's sentence groups — a bare heading is
                # never left as its own retrieval unit.
                if buf:
                    para = "\n\n".join(buf + [para])
                    buf, blen = [], 0
                for group in _sentence_groups(para):
                    chunks.append(Chunk(source_file_id=source_file_id,
                                        chunk_index=index, text=group))
                    index += 1
                continue
            if buf and blen + 2 + len(para) > CHUNK_MAX_CHARS:
                emit()
            buf.append(para)
            blen = len(para) if len(buf) == 1 else blen + 2 + len(para)
        emit()
    return chunks


def chunk_files(files: dict[str, str]) -> list[Chunk]:
    """Chunk every file (ids in sorted order — the iteration order of the
    passed dict is irrelevant, output is deterministic)."""
    chunks: list[Chunk] = []
    for file_id in sorted(files):
        chunks.extend(chunk_text(files[file_id], file_id))
    return chunks


def chunks_hash(chunks: list[Chunk]) -> str:
    """Fingerprint of the chunk corpus (order-sensitive over id+text)."""
    payload = "\x00".join(f"{c.chunk_id}\x00{c.text}" for c in chunks)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# --- embedding + cache ----------------------------------------------------------

def _default_embedder() -> Callable[[list[str]], np.ndarray]:
    """THE shared encoder (memory.default_embed): knowledge.DEFAULT_EMBEDDER
    (LaBSE), normalized — loaded lazily so importing this module stays light."""
    from voiceagent.memory import default_embed
    return default_embed


def _cache_version() -> int:
    """knowledge.CACHE_VERSION + 1 — the bump contract described above."""
    from voiceagent.knowledge import CACHE_VERSION
    return CACHE_VERSION + 1


def _cache_is_valid(meta: Any, model_name: str, corpus_hash: str,
                    n_chunks: int) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("version") != _cache_version():
        return False
    if meta.get("model_name") != model_name:
        return False
    if meta.get("chunks_hash") != corpus_hash:
        return False
    emb = meta.get("embeddings")
    if not isinstance(emb, np.ndarray):
        return False
    expected_dim = emb.shape[1] if emb.ndim == 2 else None
    if expected_dim is None or meta.get("dim") != expected_dim:
        return False
    return emb.shape[0] == n_chunks


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0.0, 1.0, norms)


def build_chunked_index(
    files: dict[str, str],
    *,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
    model_name: str | None = None,
    cache_path: str | Path | None = DEFAULT_CHUNKS_CACHE_PATH,
) -> ChunkedKnowledge:
    """Chunk `files` and embed every chunk (normalized). Embeddings are
    cached on disk at `cache_path` (knowledge.py's pickle-cache pattern,
    version = knowledge.CACHE_VERSION + 1); a valid cache hit skips ALL
    encoding. cache_path=None disables persistence (tests). `embedder`
    injects a query/build encoder (tests); None uses the shared default.

    Raises on embedder failure — callers (runtime's switch) decide the
    fail-open policy; this module stays honest about errors."""
    chunks = chunk_files(files)
    if model_name is None:
        from voiceagent.knowledge import DEFAULT_EMBEDDER
        model_name = DEFAULT_EMBEDDER
    corpus_hash = chunks_hash(chunks)
    embeddings: np.ndarray | None = None
    if cache_path is not None and chunks:
        try:
            with open(cache_path, "rb") as f:
                meta = pickle.load(f)
            if _cache_is_valid(meta, model_name, corpus_hash, len(chunks)):
                embeddings = meta["embeddings"]
        except (OSError, EOFError, pickle.UnpicklingError, ValueError):
            embeddings = None
    if embeddings is not None:
        # Cache hit: still TOUCH the encoder once now — a lazy first encode
        # must never happen on a live caller's turn (M3 lesson from memory).
        _ = embedder or _default_embedder()
    if embeddings is None:
        enc = embedder or _default_embedder()
        if not chunks:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            embeddings = _normalize(enc([c.text for c in chunks]))
            if cache_path is not None:
                _save_cache(cache_path, model_name, corpus_hash, embeddings)
    return ChunkedKnowledge(chunks=chunks, embeddings=embeddings,
                            model_name=model_name,
                            source_texts=dict(files), embedder=embedder)


def _save_cache(cache_path: str | Path, model_name: str, corpus_hash: str,
                embeddings: np.ndarray) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _cache_version(),
        "model_name": model_name,
        "chunks_hash": corpus_hash,
        "dim": int(embeddings.shape[1]),
        "embeddings": embeddings,
    }
    with open(p, "wb") as f:
        pickle.dump(payload, f)


# --- retrieval -------------------------------------------------------------------

def retrieve_chunks(
    ck: ChunkedKnowledge,
    query: str,
    *,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> list[tuple[Chunk, float]]:
    """Top-K chunks by cosine against `query`, similarity-ordered, only hits
    at/above the floor. The query embedding is cached on the index (one
    encode per (index, query) pair — handle_turn retrieves once per turn).
    Raises on embedder failure; the orchestrator fails open."""
    k = ck.top_k if top_k is None else top_k
    floor = (ck.min_similarity if min_similarity is None
             else min_similarity)
    if not ck.chunks:
        return []
    enc = embedder or ck.embedder or _default_embedder()
    cache_key = (query, id(enc))
    if ck._query_cache is not None and ck._query_cache[:2] == (query,
                                                               cache_key):
        q = ck._query_cache[2]
    else:
        q = _normalize(enc([query]))[0]
        ck._query_cache = (query, cache_key, q)
    if q.shape[0] != ck.embeddings.shape[1]:
        raise ValueError(
            f"query embedding dim {q.shape[0]} != chunk index dim "
            f"{ck.embeddings.shape[1]} — index was built with model "
            f"{ck.model_name!r}")
    sims = ck.embeddings @ q
    order = np.argsort(-sims, kind="stable")
    hits: list[tuple[Chunk, float]] = []
    for idx in order[:max(k, 0)]:
        sim = float(sims[idx])
        if sim < floor:
            break  # similarity-ordered: everything after is lower
        hits.append((ck.chunks[idx], sim))
    return hits


# --- the historical whole-file cap (shared with runtime) --------------------------

def cap_knowledge(knowledge: dict[str, str],
                  max_chars: int = KNOWLEDGE_BUDGET_CHARS) -> dict[str, str]:
    """Sorted-id prefix of the knowledge that fits under `max_chars`.
    Whole-file granularity: a file that does not fit is dropped together with
    everything after it — a truncated FAQ could assert the opposite of the
    text it cut off. This is ALSO the fail-open fallback for chunked
    deployments when retrieval errors."""
    capped: dict[str, str] = {}
    total = 0
    for kid in sorted(knowledge):
        text = knowledge[kid]
        if total + len(text) > max_chars:
            break
        capped[kid] = text
        total += len(text)
    return capped
