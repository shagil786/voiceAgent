# src/voiceagent/knowledge.py
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


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
    def __init__(self, index, ids, model):
        self._index = index
        self._ids = ids
        self._model = model

    def search(self, query: str, k: int = 3) -> list[dict]:
        emb = self._model.encode([query], normalize_embeddings=True)
        scores, idxs = self._index.search(np.asarray(emb, dtype=np.float32), k)
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
                model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in docs]
    emb = model.encode(texts, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(emb, dtype=np.float32))
    handle = IndexHandle(index, [d["id"] for d in docs], model)
    handle._store = {"texts": texts, "sections": [d["section"] for d in docs]}
    return handle
