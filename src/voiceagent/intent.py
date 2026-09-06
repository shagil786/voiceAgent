# src/voiceagent/intent.py
"""Deterministic intent classifier built on the same multilingual embeddings
used for RAG. The action decision is a nearest-neighbour match against curated
exemplar queries — no LLM involved, so it cannot drift format or reason itself
into the wrong action the way a small generative model does.

Output is constrained to the fixed intent vocabulary by construction.

M5a: each intent carries 3 Hindi (Devanagari), 3 Tamil and 3 Telugu
exemplars so native-script queries match same-language neighbours. M5a-2
swapped the default embedder to LaBSE (768-dim, 109 languages): native-script
resolution jumped (bn 0.300->1.000, te 0.700->1.000, gu 0.800->1.000,
ta 0.533->0.967) mostly via LaBSE cross-lingual transfer onto the existing
en/hinglish/hi/ta/te exemplars — the planned 3 Bengali, 3 Gujarati and 3
Marathi exemplars per intent were NOT added (mr/gu/bn eval rows resolve on
transfer alone). M5b: hybrid routing — the M5a-2 sweep also showed LaBSE
COLLAPSES on Romanized code-mixed Hindi (hinglish 0.993->0.700) while MiniLM
handles it well, so the classifier keeps TWO exemplar matrices over the SAME
exemplar strings: MiniLM-encoded for en/hinglish queries, LaBSE-encoded for
native-script queries, routed by script per classify() call. NOTE: the non-en
exemplars (Hinglish, hi/ta/te) are LLM-authored SYNTHETIC phrasings, not
transcripts of real customers — plausible support language, but real-traffic
validation is still pending (quality caveat).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from voiceagent.knowledge import (DEFAULT_EMBEDDER, LATIN_SPACE, NATIVE_SPACE,
                                  SPACE_EMBEDDERS, route_space)

# The built-in default tenant's intent exemplars are BUNDLE DATA (Task E):
# they load from the COMMITTED default bundle (data/tenants/default/intents/ —
# file NAME = intent label, YAML list = exemplars, the same schema any tenant
# bundle declares). This module ships no demo business vocabulary.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXEMPLARS_DIR = (_REPO_ROOT / "data" / "tenants" / "default"
                         / "intents")


def load_default_exemplars(directory: str | Path = DEFAULT_EXEMPLARS_DIR
                           ) -> dict[str, list[str]]:
    """Exemplars from a bundle's intents/ directory (per-intent YAML lists,
    filename = intent label). The no-bundle Agent path (scripts/chat.py's
    build_agent, memory.py's default classifier, the benchmark) seeds its
    classifier from the committed default bundle through here."""
    import yaml
    d = Path(directory)
    exemplars: dict[str, list[str]] = {}
    if not d.is_dir():
        return exemplars
    for f in sorted(d.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{f}: expected a YAML list of exemplars")
        exemplars[f.stem] = [str(x) for x in data]
    return exemplars


INTENT_EXEMPLARS: dict[str, list[str]] = load_default_exemplars()


class IntentClassifier:
    """M5b hybrid: TWO exemplar matrices over the SAME exemplar strings —
    one LaBSE-encoded (native space) for native-script queries, one
    MiniLM-encoded (latin space) for en/hinglish queries. classify() routes
    by detect_language via knowledge.route_space (same routing rule as
    IndexHandle.search, so RAG and intent always agree on the space).

    Both encoders are constructed EAGERLY at init (~2-4s total, both models
    are small and already cached): no first-query latency cliff for either
    script family, and no lazy-init state to reason about in the voice
    server. The matrices themselves are tiny (~350 exemplars x dim)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER,
                 latin_model_name: str = SPACE_EMBEDDERS[LATIN_SPACE],
                 exemplars: dict[str, list[str]] | None = None):
        # model_name keeps its historical meaning: the native-script-space
        # encoder (primary). The latin-space encoder is latin_model_name.
        # exemplars: per-tenant intent exemplars (tenant bundle, M6b);
        # None -> the built-in INTENT_EXEMPLARS.
        self._exemplars = exemplars if exemplars is not None else INTENT_EXEMPLARS
        self._native_model = SentenceTransformer(model_name)
        self._latin_model = SentenceTransformer(latin_model_name)
        self._intents: list[str] = []
        # space -> (exemplar matrix, labels); both spaces cover every intent
        self._matrices: dict[str, tuple[np.ndarray, list[str]]] = {}
        self._build()

    def _build(self) -> None:
        queries: list[str] = []
        labels: list[str] = []
        for intent, exs in self._exemplars.items():
            for ex in exs:
                if not ex.strip():
                    continue  # empty text embeds as NaN -> poisons the matmul
                queries.append(ex)
                labels.append(intent)
        self._intents = list(self._exemplars.keys())
        for space, model in ((NATIVE_SPACE, self._native_model),
                             (LATIN_SPACE, self._latin_model)):
            emb = np.asarray(model.encode(queries, normalize_embeddings=True),
                             dtype=np.float32)
            self._matrices[space] = (np.nan_to_num(emb, nan=0.0),
                                     list(labels))

    def reseed(self, exemplars: dict[str, list[str]]) -> None:
        """M2 (ADR-002): swap the exemplar set and rebuild BOTH space
        matrices IN PLACE — the live Agent's classifier learns new memory
        prototypes without being rebuilt. Cost is bounded (the matrices are
        tiny exemplar x dim products; the encoders are reused). Callers get
        the exemplar set from runtime.classifier_exemplars (declared floor +
        conflict-guarded prototypes)."""
        self._exemplars = exemplars
        self._build()

    def classify(self, text: str, k: int = 1) -> tuple[str, float]:
        """Return (best_intent, cosine_score), comparing the query against
        the exemplar matrix of the space matched to its script."""
        space = route_space(text)
        embs, labels = self._matrices[space]
        model = (self._native_model if space == NATIVE_SPACE
                 else self._latin_model)
        q = np.asarray(model.encode([text], normalize_embeddings=True),
                       dtype=np.float32)
        q = np.nan_to_num(q, nan=0.0)
        scores = embs @ q.T  # (n_exemplars, 1)
        scores = scores[:, 0]
        order = np.argsort(-scores)[:k]
        best = int(order[0])
        return labels[best], float(scores[best])
