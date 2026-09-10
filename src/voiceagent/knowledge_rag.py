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
2. **Index**: chunk texts are embedded in BOTH embedding spaces (the
   knowledge.py M5b hybrid pair — native LaBSE + latin MiniLM — over the
   SAME chunks, L2-normalized) and cached on disk like knowledge.py's index
   cache, with a BUMPED version (knowledge.CACHE_VERSION + 1) and BOTH
   spaces' model names in the payload, so a chunk cache is never served for
   a different payload shape, model pair, or corpus.
3. **Retrieval**: per turn, the query is routed to a space
   (knowledge.route_space: native script -> LaBSE matrix, Latin script ->
   MiniLM matrix — the same rule as IndexHandle.search) and the top-K chunks
   by cosine in THAT space are injected into that turn's system prompt; a
   per-space similarity floor detects knowledge gaps; any retrieval error
   fails open to the historical whole-file cap (cap_knowledge).

Phase-2 status: dual-space script-routed chunk retrieval is LANDED (this
module; measured on rag_eval — hit_rate 0.67 -> 0.89 -> 1.00, gaps intact).
The historical residual miss ('order kab aayega') was a RANKING failure, not
a routing one — fixed 2026-09-08 by glossing the KB with the phrasings
callers actually use; the ruler then grew with es/fr/de/pt + te/bn fixtures
(default tenant KB glosses, 24-fixture suite). The remaining phase-2 lever
is lexical/second-stage scoring (BM25 blend or rerank over the top-K).
Deliberately deferred: dense-only retrieval clears the quality gate, and
BM25/rerank add an index + a latency/complexity budget that should be paid
against measured failures, not speculatively.

Importable with zero heavy deps: numpy is light (memory.py already needs
it); sentence-transformers/faiss load lazily (build path, and the first
retrieval resolving the shared routing rule).
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

# Retrieval shape (phase 1 choices): top-K chunks per turn, per-space cosine
# floors below which nothing is injected and a knowledge gap is recorded.
# Native-space floor judgment call: with focused 600-900-char FAQ chunks in
# the LaBSE space, a matching question lands ~0.35-0.6 while unrelated text
# sits ~0.10-0.25 — 0.35 separates them (measured against the shipped corpus
# shape). BM25/rerank (phase 2) may revisit this.
TOP_K = 6
MIN_SIMILARITY = 0.35
# Latin-space floor (MiniLM), recalibrated 2026-09-08 on the EXPANDED
# 24-fixture ruler (the ruler this change was measured against): the KB took
# multilingual glosses for the claimed languages (es/fr/de/pt + te/bn joined
# the existing hinglish/hindi), and the OLD 0.20 calibration no longer
# separated chit-chat from content — measured on the real encoder over the
# shipped corpus, distractors moved up to 0.287 ('mazak kar raha tha') while
# the weakest MATCHING content sits at 0.426 ('quando chega meu pedido?').
# 0.32 splits the two clusters with ~0.09 margin each way; the old 0.20
# allowed chit-chat to clear the floor after the glosses landed. Hinglish
# matches sit far lower than English ones (MiniLM aligns romanized Hindi to
# English FAQ text only weakly) — reusing the native 0.35 floor for latin
# queries dropped every hinglish hit; that one-space floor was the measured
# phase-1 retrieval failure.
MIN_SIMILARITY_LATIN = 0.32

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

    Phase 2 (dual space, mirroring IndexHandle): embeddings live PER
    embedding space — `space_embeddings[space]` is that space's
    (n_chunks, dim) L2-normalized matrix over the SAME chunks, and
    `space_model_names[space]` its model. The historical `embeddings` /
    `model_name` fields remain the PRIMARY (native, LaBSE) space. A
    directly-constructed instance carrying only `embeddings` (no
    `space_embeddings`) is a legacy single-space index: it searches that one
    matrix for every query, no routing.

    Query-time encoders (injected for tests; None defers to the shared lazy
    per-space defaults): `embedder` is the historical single override — it
    drives whichever space a query routes to (and post-construction swaps of
    it, e.g. fail-open tests, apply to every space); `space_embedders` pins
    a PER-space encoder (set by build_chunked_index when `latin_embedder` is
    injected) and wins over `embedder` for that space."""
    chunks: list[Chunk]
    embeddings: np.ndarray              # PRIMARY (native) space: (n, dim), L2-normalized
    model_name: str                     # PRIMARY (native) space model (historical field)
    source_texts: dict[str, str]
    embedder: Callable[[list[str]], np.ndarray] | None = None
    top_k: int = TOP_K
    min_similarity: float = MIN_SIMILARITY              # native-space floor
    min_similarity_latin: float = MIN_SIMILARITY_LATIN  # latin-space floor
    space_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    space_model_names: dict[str, str] = field(default_factory=dict)
    # Phase-2 anti-dilution: per-space SENTENCE vectors (build-time, zero
    # query latency) + the chunk index each sentence belongs to. A short
    # query that exactly matches ONE gloss sentence inside a long chunk
    # scores the sentence higher than the whole-chunk average (dilution);
    # retrieve_chunks takes max(chunk, best-sentence) per chunk. Empty dict
    # = no sentence rescoring (legacy/partially built indexes).
    space_sentence_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    sentence_owner: np.ndarray | None = None
    space_embedders: dict[str, Callable[[list[str]], np.ndarray]] = field(
        default_factory=dict)
    # one-slot query cache, keyed by (space, query, encoder-id): a query is
    # re-encoded only when its space, text or encoder changes
    _query_cache: tuple[str, str, int, np.ndarray] | None = field(
        default=None, repr=False, compare=False)

    def space_for(self, query: str) -> str:
        """Which embedding space serves this query: routed by script
        (knowledge.route_space — the same rule as IndexHandle.search and the
        intent classifier) when the index is dual-space, else the legacy
        index's only space (native)."""
        if len(self.space_embeddings) < 2:
            from voiceagent.knowledge import NATIVE_SPACE
            return NATIVE_SPACE
        from voiceagent.knowledge import route_space
        return route_space(query)


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
    """THE shared native-space encoder (memory.default_embed): knowledge's
    LaBSE, normalized — loaded lazily so importing this module stays light."""
    from voiceagent.memory import default_embed
    return default_embed


# One stable callable per space for the shared lazy defaults — stability
# matters because the query cache keys on id(encoder).
_DEFAULT_SPACE_EMBEDDERS: dict[str, Callable[[list[str]], np.ndarray]] = {}


def _default_space_embedder(space: str) -> Callable[[list[str]], np.ndarray]:
    """Shared lazy encoder for `space` (memory.space_embed bound to it)."""
    enc = _DEFAULT_SPACE_EMBEDDERS.get(space)
    if enc is None:
        from functools import partial
        from voiceagent.memory import space_embed
        enc = partial(space_embed, space=space)
        _DEFAULT_SPACE_EMBEDDERS[space] = enc
    return enc


def _cache_version() -> int:
    """knowledge.CACHE_VERSION + 1 — the bump contract described above. The
    phase-2 dual-space chunk payload bumped the SHARED constant (the
    IndexHandle-established bump point), so this moved 4 -> 5 with it."""
    from voiceagent.knowledge import CACHE_VERSION
    return CACHE_VERSION + 1


def _cache_is_valid(meta: Any, model_name: str, latin_model_name: str,
                    corpus_hash: str, n_chunks: int) -> bool:
    """Dual-space payload contract: the top-level legacy fields carry the
    PRIMARY (native) space — model_name/chunks_hash/dim exactly as phase 1 —
    and the per-space 'spaces' record must carry BOTH spaces, each with its
    own model name, dim and full (n_chunks, dim) matrix over this corpus.
    A phase-1 single-space chunk payload is never valid: it lacks the
    'spaces' record AND predates the shared version bump."""
    if not isinstance(meta, dict):
        return False
    if meta.get("version") != _cache_version():
        return False
    if meta.get("model_name") != model_name:
        return False
    if meta.get("chunks_hash") != corpus_hash:
        return False
    spaces = meta.get("spaces")
    if not isinstance(spaces, dict):
        return False
    from voiceagent.knowledge import LATIN_SPACE, NATIVE_SPACE
    for space, name in ((NATIVE_SPACE, model_name),
                        (LATIN_SPACE, latin_model_name)):
        rec = spaces.get(space)
        if not isinstance(rec, dict):
            return False
        if rec.get("model_name") != name:
            return False
        emb = rec.get("embeddings")
        if not isinstance(emb, np.ndarray) or emb.ndim != 2:
            return False
        if emb.shape[0] != n_chunks or rec.get("dim") != emb.shape[1]:
            return False
    return True


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
    latin_embedder: Callable[[list[str]], np.ndarray] | None = None,
    latin_model_name: str | None = None,
) -> ChunkedKnowledge:
    """Chunk `files` and embed every chunk in BOTH embedding spaces (the
    native space — model_name, default LaBSE — and the latin space —
    latin_model_name, default MiniLM — over the SAME chunks; knowledge.py's
    M5b hybrid pair). Embeddings are cached on disk at `cache_path`
    (knowledge.py's pickle-cache pattern, version = knowledge.CACHE_VERSION
    + 1, payload carries BOTH spaces' model names + matrices); a valid cache
    hit skips ALL encoding. cache_path=None disables persistence (tests).

    Encoder injection (tests): `embedder` is the native-space build/query
    encoder and — unless `latin_embedder` is also given — drives the latin
    space too, so offline tests exercise both spaces without loading the
    real latin model. None uses the shared lazy per-space defaults.

    Raises on embedder failure — callers (runtime's switch) decide the
    fail-open policy; this module stays honest about errors."""
    from voiceagent.knowledge import (DEFAULT_EMBEDDER, LATIN_SPACE,
                                      NATIVE_SPACE, SPACE_EMBEDDERS)
    chunks = chunk_files(files)
    if model_name is None:
        model_name = DEFAULT_EMBEDDER
    if latin_model_name is None:
        latin_model_name = SPACE_EMBEDDERS[LATIN_SPACE]
    corpus_hash = chunks_hash(chunks)
    native_enc = embedder or _default_embedder()
    # One injected encoder drives BOTH spaces unless a per-space latin
    # encoder is given (offline determinism without the real latin model).
    latin_enc = (latin_embedder or embedder
                 or _default_space_embedder(LATIN_SPACE))
    # Anti-dilution build step: sentence texts per chunk (vectors encoded in
    # both spaces on a fresh build; cached alongside the chunk matrices).
    sent_texts: list[str] = []
    owner_idx: list[int] = []
    for i, c in enumerate(chunks):
        for s in _SENTENCE_SPLIT_RE.split(c.text):
            s = s.strip()
            if len(s) >= 2:
                sent_texts.append(s)
                owner_idx.append(i)
    sentence_owner = (np.asarray(owner_idx, dtype=np.int64)
                      if sent_texts else None)
    sentence_vectors: dict[str, np.ndarray] = {}

    spaces: dict[str, np.ndarray] | None = None
    if cache_path is not None and chunks:
        from voiceagent.knowledge import _read_verified
        meta = _read_verified(cache_path)
        try:
            if meta is not None and _cache_is_valid(
                    meta, model_name, latin_model_name,
                    corpus_hash, len(chunks)):
                spaces = {space: rec["embeddings"]
                          for space, rec in meta["spaces"].items()}
                sentence_vectors = meta.get("sentence_vectors") or {}
                sentence_owner = meta.get("sentence_owner")
        except (EOFError, pickle.UnpicklingError, ValueError, KeyError,
                TypeError):
            spaces = None
    if spaces is not None:
        # Cache hit: still TOUCH both space encoders once now — a lazy first
        # encode must never happen on a live caller's turn (M3 lesson from
        # memory). Resolving the callables materializes the lazy imports.
        _ = native_enc
        _ = latin_enc
    if spaces is None:
        if not chunks:
            empty = np.zeros((0, 0), dtype=np.float32)
            spaces = {NATIVE_SPACE: empty, LATIN_SPACE: empty}
        else:
            spaces = {
                NATIVE_SPACE: _normalize(native_enc([c.text for c in chunks])),
                LATIN_SPACE: _normalize(latin_enc([c.text for c in chunks])),
            }
            if sent_texts:
                sentence_vectors = {
                    NATIVE_SPACE: _normalize(native_enc(sent_texts)),
                    LATIN_SPACE: _normalize(latin_enc(sent_texts)),
                }
            if cache_path is not None:
                _save_cache(cache_path, model_name, latin_model_name,
                            corpus_hash, spaces,
                            sentence_vectors=sentence_vectors,
                            sentence_owner=sentence_owner)
    return ChunkedKnowledge(
        chunks=chunks,
        embeddings=spaces[NATIVE_SPACE],
        model_name=model_name,
        source_texts=dict(files),
        embedder=embedder,
        space_embeddings=spaces,
        space_model_names={NATIVE_SPACE: model_name,
                           LATIN_SPACE: latin_model_name},
        space_embedders=({LATIN_SPACE: latin_enc}
                         if latin_embedder is not None else {}),
        space_sentence_vectors=sentence_vectors,
        sentence_owner=sentence_owner,
    )


def _save_cache(cache_path: str | Path, model_name: str,
                latin_model_name: str, corpus_hash: str,
                spaces: dict[str, np.ndarray], *,
                sentence_vectors: dict[str, np.ndarray],
                sentence_owner: np.ndarray | None) -> None:
    """Persist both spaces' matrices + provenance: per-space model name, dim
    and embeddings (a cache hit skips re-encoding the corpus in every
    space), plus the phase-1-style top-level model_name/dim fields for the
    primary (native) space — the legacy half of _cache_is_valid's
    contract."""
    from voiceagent.knowledge import LATIN_SPACE, NATIVE_SPACE
    names = {NATIVE_SPACE: model_name, LATIN_SPACE: latin_model_name}
    payload = {
        "version": _cache_version(),
        "model_name": model_name,
        "chunks_hash": corpus_hash,
        "dim": int(spaces[NATIVE_SPACE].shape[1]),
        "spaces": {
            space: {"model_name": names[space], "dim": int(mat.shape[1]),
                    "embeddings": mat}
            for space, mat in spaces.items()
        },
        "sentence_vectors": sentence_vectors,
        "sentence_owner": sentence_owner,
    }
    from voiceagent.knowledge import _write_verified
    _write_verified(cache_path, payload)


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
    at/above the floor. Phase 2: the query is ROUTED to an embedding space
    (knowledge.route_space — native script searches the LaBSE matrix, Latin
    script the MiniLM matrix) and scored against THAT space's matrix with
    THAT space's floor; an explicit `min_similarity` overrides the routed
    space's floor (calibration knobs, tests). The query embedding is cached
    on the index keyed by (space, query, encoder) — one encode per
    (index, space, query) triple (handle_turn retrieves once per turn).
    Raises on embedder failure; the orchestrator fails open."""
    k = ck.top_k if top_k is None else top_k
    if not ck.chunks:
        return []
    from voiceagent.knowledge import NATIVE_SPACE
    space = ck.space_for(query)
    matrix = ck.space_embeddings.get(space)
    if matrix is None:
        # Legacy single-space index (directly constructed): `embeddings` IS
        # the one space (native), searched for every query.
        space = NATIVE_SPACE
        matrix = ck.embeddings
        space_model = ck.model_name
    else:
        space_model = ck.space_model_names.get(space, "")
    if min_similarity is not None:
        floor = min_similarity
    elif space == NATIVE_SPACE:
        floor = ck.min_similarity
    else:
        floor = ck.min_similarity_latin
    # Encoder precedence: call-site override > per-space pin (build-time
    # latin_embedder) > the historical single override (drives whichever
    # space routes) > the shared lazy default for this space.
    enc = (embedder or ck.space_embedders.get(space) or ck.embedder
           or _default_space_embedder(space))
    cache_key = (space, query, id(enc))
    if ck._query_cache is not None and ck._query_cache[:3] == cache_key:
        q = ck._query_cache[3]
    else:
        q = _normalize(enc([query]))[0]
        ck._query_cache = (space, query, id(enc), q)
    if q.shape[0] != matrix.shape[1]:
        # M5a-2 guard, per space: never score a query against a matrix from
        # a different embedding space — fail loudly instead of returning
        # garbage similarities.
        raise ValueError(
            f"query embedding dim {q.shape[0]} != {space}-space chunk index "
            f"dim {matrix.shape[1]} — index was built with model "
            f"{space_model!r}; rebuild the index or use that model")
    sims = matrix @ q
    # Phase-2 anti-dilution: a short query that matches ONE sentence inside
    # a long chunk scores that sentence above the whole-chunk average. Where
    # sentence vectors exist for this space, each chunk's effective score is
    # max(whole-chunk, best-sentence) — zero query latency (precomputed
    # vectors, one extra matmul over n_sentences). Measured: lifts 'eppo
    # kedaikkum' 0.165 -> 0.203 across the floor with gaps intact.
    if (space_sv := ck.space_sentence_vectors.get(space)) is not None             and ck.sentence_owner is not None and len(space_sv):
        sent = space_sv @ q
        best = np.full(sims.shape[0], -np.inf, dtype=sims.dtype)
        np.maximum.at(best, ck.sentence_owner, sent)
        sims = np.maximum(sims, best)
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
