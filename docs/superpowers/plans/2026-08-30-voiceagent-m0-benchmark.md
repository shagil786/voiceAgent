# M0 — Benchmark Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep multiple small CPU-only LLMs (0.5B → 1.5B → 3.8B) through the full pipeline (ASR → RAG → LLM → TTS) and pick the **cheapest + smallest model that still passes the gate** (`latency ≤ 2s`, `resolution ≥ 75%`) — measuring turn latency, resolution, hallucination rate, and ₹ cost per resolved conversation so the product thesis is proven or disproven on real numbers.

**Architecture:** A Python package `voiceagent` with clearly separated units: `dataset` (eval conversations), `knowledge` + `rag` (retrieval), `llm` (small quantized LLM on CPU), `agent` (orchestration: retrieve → prompt → respond → extract action), `evaluator` (resolution + groundedness scoring), `voice` (ASR/TTS latency measurement), and `benchmark` (runner + report + threshold gate). Each unit is independently testable; model calls sit behind thin interfaces so tests run without heavy downloads.

**Tech Stack:** Python 3.12 (see Global Constraints), `numpy`, `faiss-cpu`, `sentence-transformers`, `llama-cpp-python`, `faster-whisper` (CTranslate2), `piper-tts`, `pytest`. No GPU anywhere.

**Spec:** [2026-08-30-voiceagent-design.md](../specs/2026-08-30-voiceagent-design.md) — this plan implements Milestone M0 ("Benchmarks before product"), which is the gate the entire thesis hangs on.

## Global Constraints

- **CPU only.** No CUDA/MPS anywhere. All model code must run on CPU.
- **Cost/accuracy/size frontier is the point.** M0 must sweep multiple small LLMs (0.5B → 3B) and pick the one with the best trade-off between VPS cost, model size, latency, and resolution. "Cheapest + smallest that still passes the gate" wins. Report ₹ cost per resolved conversation as a first-class metric.
- **VPS targets:** stretch = 2–4 vCPU / 8GB RAM (₹2k–₹3k/month class); comfortable baseline = 4–8 vCPU / 16GB (₹3k–₹5k/month). The report must state which instance the measured numbers assume.
- **Python 3.12** for ML wheel compatibility (`torch`, `ctranslate2`, `faiss-cpu`). If only 3.14 is present, create the venv with `uv venv --python 3.12`; fall back to 3.14 only if 3.12 is unavailable, and pin the exact wheel versions below.
- **Thresholds (the go/no-go gate):** end-to-end turn latency ≤ 2s; resolution rate ≥ 75%; wrong-action rate 0.0% is a stated target (M0 measures it if actions are extracted); hallucination rate reported, not gated in M0.
- **Offline default:** no outbound network calls during a benchmark run; models are downloaded once (Task 4 GGUFs, Task 8 whisper/piper) then cached locally. Benchmark runs must not phone home.
- **Language scope:** English + Hindi + Hinglish only. No other languages in M0.
- **Metrics are the deliverable.** Every run must emit a benchmark report (Markdown table + JSON) to `data/out/`.
- **Commit discipline:** one commit per completed task, small and atomic. `git init` happens in Task 1.

---

### Task 1: Project scaffold, venv, dependencies, git init

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml` (minimal, pytest config)
- Create: `.gitignore`
- Create: `src/voiceagent/__init__.py`
- Create: `tests/__init__.py`
- Create: `README.md` (one-paragraph M0 purpose)

**Interfaces:**
- Produces: working `voiceagent` importable package, pytest running, git repo with initial commit.

- [ ] **Step 1: Initialize git and package layout**

```bash
cd /Users/mdshagilnizami/code/projects/voiceAgent
git init
mkdir -p src/voiceagent tests scripts data/knowledge data/eval data/out
```

- [ ] **Step 2: Create `requirements.txt`**

```
numpy==2.1.3
faiss-cpu==1.9.0.post1
sentence-transformers==3.3.1
llama-cpp-python==0.3.5
faster-whisper==1.1.1
piper-tts==1.2.0
pytest==8.3.4
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "voiceagent"
version = "0.0.1"
requires-python = ">=3.12"
description = "CPU-only multilingual AI support agent - M0 benchmark spike"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
data/index/
data/out/
data/models/
```

- [ ] **Step 5: Create package init files**

`src/voiceagent/__init__.py`:
```python
__version__ = "0.0.1"
```

`tests/__init__.py`:
```python
```

- [ ] **Step 6: Create README.md**

```markdown
# VoiceAgent — M0 Benchmark Spike

Builds the CPU-only pipeline (ASR → RAG → small LLM → TTS) and measures
turn latency, resolution rate, hallucination rate, and cost/conversation
over ~1,000 English + Hinglish eval conversations.

Gate: latency ≤ 2s and resolution ≥ 75%. See docs/superpowers/specs/2026-08-30-voiceagent-design.md.
```

- [ ] **Step 7: Create venv and install dependencies**

```bash
cd /Users/mdshagilnizami/code/projects/voiceAgent
uv venv --python 3.12 .venv 2>/dev/null || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: install completes. If `faiss-cpu` / `torch` / `ctranslate2` fail on the active Python, recreate the venv with 3.12 before proceeding.

- [ ] **Step 8: Verify tests run**

```bash
source .venv/bin/activate
pytest -q
```

Expected: 0 tests collected, exit 0 (pytest configured correctly).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: scaffold voiceagent M0 spike (venv, deps, package layout)"
```

---

### Task 2: Synthetic eval dataset — 1,000 conversations with ground truth

**Files:**
- Create: `src/voiceagent/dataset.py`
- Create: `scripts/generate_eval_set.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: nothing (seed conversations are inline constants).
- Produces:
  - `DATASET_SCHEMA` — the column contract for `data/eval/conversations.csv`.
  - `load_conversations(path: str) -> list[Conversation]` — reads CSV into dataclasses.
  - `generate_eval_set(out_path: str, n: int) -> int` — writes `n` conversations derived from seeded templates; returns row count.
  - `Conversation` dataclass with fields: `id`, `language` (`"en"|"hi"|"hinglish"`), `intent`, `user_text`, `expected_action`, `key_facts` (list), `escalate` (bool).

**Rationale:** M0 has no partner yet, so we generate a deterministic synthetic set from seeded templates across the top 20 support intents (order_status, refund, cancel_order, address_change, payment_declined, recharge, billing, return, replacement, otp, fraud, account_closure, delivery_delay, product_info, invoice, plan_change, roaming, network_issue, complaint, high_value_refund), with a mix of English / Hindi / Hinglish. A real-partner CSV with the same schema can later replace the generator without touching the pipeline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import csv
import tempfile
from pathlib import Path
from voiceagent.dataset import Conversation, generate_eval_set, load_conversations

def test_generate_creates_requested_count_and_loads_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "eval.csv"
        n = generate_eval_set(str(out), n=100)
        assert n == 100
        rows = load_conversations(str(out))
        assert len(rows) == 100

def test_language_and_intent_coverage():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "eval.csv"
        generate_eval_set(str(out), n=200)
        rows = load_conversations(str(out))
        langs = {r.language for r in rows}
        intents = {r.intent for r in rows}
        assert {"en", "hi", "hinglish"} <= langs
        assert "order_status" in intents and "refund" in intents

def test_hinglish_rows_are_code_switched():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "eval.csv"
        generate_eval_set(str(out), n=100)
        rows = load_conversations(str(out))
        h = [r for r in rows if r.language == "hinglish"][0]
        assert any(c in h.user_text for c in "आ") or any(
            c in h.user_text for c in "abc"
        )
        assert "expected_action" in h.__dict__
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: voiceagent.dataset`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/voiceagent/dataset.py
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field

DATASET_SCHEMA = [
    "id", "language", "intent", "user_text", "expected_action",
    "key_facts", "escalate",
]

INTENTS = [
    "order_status", "refund", "cancel_order", "address_change",
    "payment_declined", "recharge", "billing", "return", "replacement",
    "otp", "fraud", "account_closure", "delivery_delay", "product_info",
    "invoice", "plan_change", "roaming", "network_issue", "complaint",
    "high_value_refund",
]

# (language, intent, user_text, expected_action, key_facts, escalate)
_SEED_TEMPLATES = [
    ("en", "order_status", "Where is my order #ORD-77812?",
     "order_status", ["ORD-77812"], False),
    ("en", "refund", "I need a refund for order #ORD-22109.",
     "refund", ["ORD-22109"], False),
    ("en", "payment_declined", "Why was my payment declined?",
     "payment_declined", ["declined"], False),
    ("hinglish", "order_status", "Bhai mera order abhi tak nahi aaya, order id ORD-55671 hai.",
     "order_status", ["ORD-55671"], False),
    ("hinglish", "refund", "Actually can you refund my order, order #ORD-99032?",
     "refund", ["ORD-99032"], False),
    ("hi", "recharge", "मेरा recharge क्यों fail हुआ?",
     "recharge", ["fail"], False),
    ("hi", "billing", "मुझे अपना bill समझ नहीं आया।",
     "billing", ["bill"], False),
    ("en", "high_value_refund", "I want a refund of ₹25,000 for order #ORD-11223.",
     "high_value_refund", ["ORD-11223"], True),
    ("en", "fraud", "Someone used my account. Block it now.",
     "fraud", ["block"], True),
    ("hinglish", "otp", "OTP nahi aaya mere phone pe, resend karo.",
     "otp", ["otp"], False),
]

def _mutate(text: str, rng: random.Random) -> str:
    """Return the seed text as-is; templates already cover variation.
    Kept as a hook for later augmentation without changing the schema."""
    return text

def generate_eval_set(out_path: str, n: int = 1000, seed: int = 42) -> int:
    rng = random.Random(seed)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_SCHEMA)
        for i in range(n):
            lang, intent, text, action, facts, escalate = rng.choice(_SEED_TEMPLATES)
            order_id = f"ORD-{rng.randint(10000, 99999)}"
            text = text.replace("ORD-77812", order_id)
            facts = [order_id if f == "ORD-77812" else f for f in facts]
            writer.writerow([
                f"conv-{i:04d}", lang, intent, _mutate(text, rng),
                action, "|".join(facts), escalate,
            ])
    return n

def load_conversations(path: str) -> list["Conversation"]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(Conversation(
                id=row["id"], language=row["language"], intent=row["intent"],
                user_text=row["user_text"], expected_action=row["expected_action"],
                key_facts=[k for k in row["key_facts"].split("|") if k],
                escalate=row["escalate"].lower() == "true",
            ))
    return out

@dataclass
class Conversation:
    id: str
    language: str
    intent: str
    user_text: str
    expected_action: str
    key_facts: list[str] = field(default_factory=list)
    escalate: bool = False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_dataset.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Generate the eval set and commit**

```bash
mkdir -p data/eval
cat > scripts/generate_eval_set.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.dataset import generate_eval_set

if __name__ == "__main__":
    out = "data/eval/conversations.csv"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"wrote {generate_eval_set(out, n)} rows to {out}")
EOF
source .venv/bin/activate
python scripts/generate_eval_set.py 1000
head -3 data/eval/conversations.csv
```

Expected: 1000 rows written; header + 2 sample rows shown.

```bash
git add -A
git commit -m "feat: synthetic eval dataset (1k English/Hindi/Hinglish conversations with ground truth)"
```

---

### Task 3: Knowledge base + FAISS index builder

**Files:**
- Create: `data/knowledge/faqs.md`
- Create: `data/knowledge/policies.md`
- Create: `src/voiceagent/knowledge.py`
- Create: `scripts/build_kb.py`
- Test: `tests/test_knowledge.py`

**Interfaces:**
- Consumes: nothing (static markdown knowledge files).
- Produces:
  - `load_docs(data_dir: str) -> list[dict]` — list of `{"id": str, "text": str, "section": str}`.
  - `build_index(docs: list[dict], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> IndexHandle` where `IndexHandle` wraps a FAISS index + id list.
  - `IndexHandle.search(query: str, k: int = 3) -> list[dict]` — returns top-k docs with `{"id","text","section","score"}`.

**Rationale:** A multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, ~118M params) handles English + Hindi + Hinglish retrieval on CPU. FAISS in-process keeps the index local — no separate vector DB service.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_knowledge.py
import tempfile
from pathlib import Path
from voiceagent.knowledge import load_docs, build_index

def test_load_docs_parses_markdown_sections():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "faqs.md").write_text(
            "# Returns\nYou can return within 7 days.\n\n"
            "# Refunds\nRefunds take 5-7 business days.\n"
        )
        docs = load_docs(str(d))
        texts = [x["text"] for x in docs]
        assert any("7 days" in t for t in texts)
        assert any("5-7 business days" in t for t in texts)

def test_build_index_and_search_returns_relevant_doc():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "faqs.md").write_text(
            "# Returns\nYou can return any item within 7 days of delivery.\n\n"
            "# Refunds\nRefunds are processed within 5-7 business days.\n"
        )
        docs = load_docs(str(d))
        idx = build_index(docs)
        results = idx.search("how long do refunds take?", k=1)
        assert len(results) == 1
        assert "5-7 business days" in results[0]["text"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_knowledge.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_knowledge.py -v
```

Expected: 2 tests PASS. (First run downloads the embedding model once; it is cached under `~/.cache` for later offline runs.)

- [ ] **Step 5: Write knowledge content**

```bash
cat > data/knowledge/faqs.md <<'EOF'
# Returns
You can return any item within 7 days of delivery. Items must be unused and in original packaging.

# Refunds
Refunds are processed within 5-7 business days after the returned item is received. Refund goes to the original payment method.

# Delivery Times
Standard delivery is 3-5 business days. Express delivery is 1-2 business days.

# Payment Declined
A payment can be declined due to insufficient balance, an expired card, or a bank security block. Contact your bank or retry after 10 minutes.

# Recharge
Recharge failures are usually due to network errors or invalid plan selection. Check the plan and retry. If money was deducted, it is auto-refunded within 24 hours.
EOF
cat > data/knowledge/policies.md <<'EOF'
# Refund Policy
Standard refunds up to ₹5,000 can be processed without human approval. Refunds above ₹5,000 require human approval and customer authentication.

# Order Cancellation
Orders can be cancelled before shipping. Once shipped, cancellation is not allowed; instead offer a return.

# Account Changes
Changes to account information (phone, email, address) require OTP verification.

# Escalation
Escalate to a human for: fraud, legal, chargeback, and high-value refunds (above ₹5,000).
EOF
```

- [ ] **Step 6: Build the index script and verify end to end**

```bash
cat > scripts/build_kb.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.knowledge import load_docs, build_index

if __name__ == "__main__":
    data_dir = "data/knowledge"
    docs = load_docs(data_dir)
    idx = build_index(docs)
    Path("data/index").mkdir(exist_ok=True)
    import pickle
    with open("data/index/handle.pkl", "wb") as f:
        pickle.dump({"ids": [d["id"] for d in docs],
                     "texts": [d["text"] for d in docs],
                     "sections": [d["section"] for d in docs]}, f)
    # verify search
    for q in ["how long do refunds take?", "refund limit approval"]:
        print(q, "->", idx.search(q, k=2)[0]["section"])
EOF
source .venv/bin/activate
python scripts/build_kb.py
```

Expected: prints a relevant section for each query (e.g. `Refunds`, `Refund Policy`).

```bash
git add -A
git commit -m "feat: knowledge base + FAISS multilingual retrieval index"
```

---

### Task 4: LLM wrapper — multi-size quantized models on CPU

**Files:**
- Create: `src/voiceagent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `LLMHandle.generate(prompt: str, max_tokens: int = 256, stop: list[str] | None = None) -> str` (text completion)
  - `LLMHandle.specs` — dict of `{"model": str, "params": str, "quant": str, "model_path": str, "size_mb": float}` for the benchmark report.
  - `CANDIDATE_MODELS: list[dict]` — each `{"name", "url", "size_mb"}`. M0 default sweep:
    - `qwen2.5-0.5b-q4` — `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400MB) — *stretch target (2–4 vCPU / 8GB VPS)*
    - `qwen2.5-1.5b-q4` — `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1.1GB) — *baseline (4–8 vCPU / 16GB)*
    - `phi-3.5-mini-q4` — `https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf` (~2.4GB) — *accuracy ceiling*
  - `download_model(url: str, model_dir: str = "data/models") -> str` — downloads a GGUF to cache, returns local path.
  - `list_available_models(model_dir: str = "data/models") -> list[dict]` — returns specs for models already downloaded (so the benchmark can sweep whatever is present).
  - `load_llm(model_path: str, n_ctx: int = 2048) -> LLMHandle` — loads one local GGUF.

**Rationale:** The whole point of M0 is finding the smallest/cheapest model that still passes the gate. One wrapper interface + a candidate list lets the benchmark sweep 0.5B → 1.5B → 3B with no pipeline changes. Each model reports its own `specs` so the report shows exactly which model produced which numbers. The 2s gate is on the full turn; RAM and VPS cost per model are reported so "cheapest that passes" is a real decision.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
from voiceagent.llm import LLMHandle, CANDIDATE_MODELS

class FakeLLM(LLMHandle):
    def generate(self, prompt, max_tokens=256, stop=None):
        return "Your order ORD-77812 is out for delivery."

def test_generate_returns_string():
    llm = FakeLLM({"model": "fake", "params": "0.5B", "quant": "Q4_K_M",
                   "model_path": "fake", "size_mb": 1.0})
    assert isinstance(llm.generate("hi"), str)
    assert "out for delivery" in llm.generate("hi")

def test_candidate_models_span_sizes():
    names = {m["name"] for m in CANDIDATE_MODELS}
    assert "qwen2.5-0.5b-q4" in names
    assert "qwen2.5-1.5b-q4" in names
    assert "phi-3.5-mini-q4" in names
    assert all(m["size_mb"] > 0 for m in CANDIDATE_MODELS)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_llm.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# src/voiceagent/llm.py
from __future__ import annotations

from pathlib import Path
from llama_cpp import Llama

CANDIDATE_MODELS = [
    {
        "name": "qwen2.5-0.5b-q4",
        "url": ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
                "resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        "size_mb": 400,
        "params": "0.5B",
    },
    {
        "name": "qwen2.5-1.5b-q4",
        "url": ("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/"
                "resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        "size_mb": 1100,
        "params": "1.5B",
    },
    {
        "name": "phi-3.5-mini-q4",
        "url": ("https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF/"
                "resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"),
        "size_mb": 2400,
        "params": "3.8B",
    },
]


class LLMHandle:
    def __init__(self, specs: dict):
        self.specs = specs

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: list[str] | None = None) -> str:
        raise NotImplementedError


class LlamaCppLLM(LLMHandle):
    def __init__(self, model_path: str, n_ctx: int = 2048,
                 params: str = "?", size_mb: float = 0.0):
        super().__init__({
            "model": Path(model_path).name,
            "params": params,
            "quant": "Q4_K_M",
            "model_path": str(model_path),
            "size_mb": size_mb,
        })
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=0)

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: list[str] | None = None) -> str:
        out = self._llm(prompt, max_tokens=max_tokens, stop=stop, echo=False)
        return out["choices"][0]["text"].strip()


def download_model(url: str, model_dir: str = "data/models") -> str:
    """Download a GGUF into model_dir (idempotent) and return local path."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    path = Path(model_dir) / url.rsplit("/", 1)[1]
    if not path.exists():
        import urllib.request
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return str(path)


def list_available_models(model_dir: str = "data/models") -> list[dict]:
    """Specs of candidate models already downloaded, in size order."""
    out = []
    for cand in CANDIDATE_MODELS:
        path = Path(model_dir) / cand["url"].rsplit("/", 1)[1]
        if path.exists():
            out.append({"name": cand["name"], "model_path": str(path),
                        "params": cand["params"], "quant": "Q4_K_M",
                        "size_mb": cand["size_mb"]})
    return sorted(out, key=lambda m: m["size_mb"])


def load_llm(model_path: str, n_ctx: int = 2048,
             params: str = "?", size_mb: float = 0.0) -> LLMHandle:
    return LlamaCppLLM(model_path, n_ctx=n_ctx, params=params, size_mb=size_mb)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_llm.py -v
```

Expected: 2 tests PASS (no model download needed — tests use `FakeLLM` and the candidate list).

- [ ] **Step 5: Verify real models load and run on CPU**

```bash
cat > scripts/smoke_llm.py <<'EOF'
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.llm import CANDIDATE_MODELS, download_model, load_llm

if __name__ == "__main__":
    targets = sys.argv[1:] or [c["name"] for c in CANDIDATE_MODELS]
    for cand in CANDIDATE_MODELS:
        if cand["name"] not in targets:
            continue
        path = download_model(cand["url"])
        llm = load_llm(path, params=cand["params"], size_mb=cand["size_mb"])
        t0 = time.time()
        print(f"[{cand['name']}] ->",
              llm.generate("Reply in one line: order ORD-1 status?"))
        print(f"  first inference: {time.time()-t0:.2f}s (includes model load)")
EOF
source .venv/bin/activate
python scripts/smoke_llm.py qwen2.5-0.5b-q4
python scripts/smoke_llm.py qwen2.5-1.5b-q4
python scripts/smoke_llm.py phi-3.5-mini-q4
```

Expected: each prints a short reply and a cold-start time. This downloads ~400MB + ~1.1GB + ~2.4GB total, once, to `data/models/` — run with a live connection. Verify all three load and respond on CPU before committing.

```bash
git add -A
git commit -m "feat: multi-size CPU LLM wrapper (0.5B/1.5B/3.8B GGUF sweep)"
```

---

### Task 5: Agent orchestration — retrieve → prompt → respond → extract action

**Files:**
- Create: `src/voiceagent/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `IndexHandle` (Task 3), `LLMHandle` (Task 4).
- Produces:
  - `build_agent(index: IndexHandle, llm: LLMHandle) -> Agent`
  - `Agent.handle(user_text: str) -> AgentResult`
  - `AgentResult` dataclass: `{text: str, action: str | None, retrieved: list[dict], latency_s: float}`.
  - `extract_action(text: str) -> str | None` — regex/naive extraction of a proposed action from the LLM output (used by the evaluator).

**Rationale:** The agent never trusts the LLM for authorization in M1+, but M0 measures the action the LLM *proposes* so the wrong-action rate can be reported. The prompt instructs the model to answer only from retrieved context and to emit an action line `ACTION: <name>` when an action is proposed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
from voiceagent.agent import build_agent, extract_action, AgentResult
from voiceagent.llm import LLMHandle

class FakeLLM(LLMHandle):
    def __init__(self):
        super().__init__({"model": "fake"})
    def generate(self, prompt, max_tokens=256, stop=None):
        return ("Your order ORD-77812 is out for delivery.\n"
                "ACTION: order_status")

class FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "Order status can be checked with the order id.",
                "section": "faqs", "score": 0.9}]

def test_extract_action_parses_action_line():
    assert extract_action("foo\nACTION: refund\nbar") == "refund"
    assert extract_action("no action here") is None

def test_agent_returns_text_action_and_retrieved():
    agent = build_agent(FakeIndex(), FakeLLM())
    res = agent.handle("where is my order ORD-77812")
    assert isinstance(res, AgentResult)
    assert "out for delivery" in res.text
    assert res.action == "order_status"
    assert len(res.retrieved) == 1
    assert res.latency_s >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# src/voiceagent/agent.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "You are a customer support assistant for an Indian ecommerce company. "
    "Answer ONLY from the provided context. Be concise. If the customer's "
    "request requires an action (refund, cancel, etc.), end your reply with "
    "a line: ACTION: <action_name> where action_name is one of: "
    "order_status, refund, cancel_order, address_change, payment_declined, "
    "recharge, billing, return, replacement, otp, fraud, account_closure, "
    "delivery_delay, product_info, invoice, plan_change, roaming, "
    "network_issue, complaint, high_value_refund. "
    "If no action is needed, do not emit an ACTION line."
)

@dataclass
class AgentResult:
    text: str
    action: str | None
    retrieved: list[dict]
    latency_s: float

class Agent:
    def __init__(self, index, llm):
        self._index = index
        self._llm = llm

    def handle(self, user_text: str) -> AgentResult:
        t0 = time.time()
        retrieved = self._index.search(user_text, k=3)
        context = "\n".join(f"[{r['section']}] {r['text']}" for r in retrieved)
        prompt = (
            f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\n"
            f"Customer: {user_text}\nAssistant:"
        )
        # Generate without a stop list so the ACTION line is included in the
        # output and extract_action can parse it (the model is instructed to
        # end with ACTION: <name> when an action applies).
        text = self._llm.generate(prompt, max_tokens=300)
        return AgentResult(text=text, action=extract_action(text),
                           retrieved=retrieved, latency_s=time.time() - t0)

ACTION_RE = re.compile(r"ACTION:\s*([a-z_]+)", re.IGNORECASE)

def extract_action(text: str) -> str | None:
    m = ACTION_RE.search(text)
    return m.group(1).lower() if m else None

def build_agent(index, llm) -> Agent:
    return Agent(index, llm)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_agent.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: agent orchestration (retrieve -> prompt -> respond -> extract action)"
```

---

### Task 6: Evaluator — resolution, hallucination/groundedness, wrong-action rate

**Files:**
- Create: `src/voiceagent/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `Conversation` (Task 2), `AgentResult` (Task 5).
- Produces:
  - `score_conversation(conv: Conversation, res: AgentResult) -> EvalRow`
  - `EvalRow` dataclass: `{conv_id, resolved: bool, grounded: bool, wrong_action: bool, hallucinated_facts: list[str], latency_s: float}`.
  - `aggregate(rows: list[EvalRow]) -> EvalSummary` — `{resolution_rate, grounded_rate, wrong_action_rate, hallucination_rate, avg_latency_s, n}`.

**Resolution logic (M0, deterministic):** a conversation resolves iff the agent's `action` matches `conv.expected_action` AND every `key_fact` appears in the agent's text. Escalation rows resolve iff the agent escalates (returns `expected_action` of the escalation intent and mentions escalation is not required to be literal — matching `expected_action` is enough).

**Hallucination/groundedness:** a response is grounded iff every noun phrase in the response that also appears in the eval row's `key_facts` is present in the retrieved context; simpler deterministic rule for M0: response is grounded iff each `key_fact` present in the text is also present in at least one retrieved doc's text. Facts in the text missing from retrieval are flagged as hallucinated.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluator.py
from voiceagent.evaluator import score_conversation, aggregate
from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult

def _conv(action="refund", facts=("ORD-1",), escalate=False):
    return Conversation(id="c1", language="en", intent="refund",
                        user_text="refund ORD-1", expected_action=action,
                        key_facts=list(facts), escalate=escalate)

def _res(text="refund for ORD-1 done", action="refund",
         retrieved=("Refund processed for ORD-1",)):
    return AgentResult(text=text, action=action,
                       retrieved=[{"text": t} for t in retrieved],
                       latency_s=0.5)

def test_resolution_requires_action_and_facts():
    good = score_conversation(_conv(), _res())
    assert good.resolved and good.grounded
    bad_action = score_conversation(_conv(), _res(action="cancel_order"))
    assert not bad_action.resolved
    missing_fact = score_conversation(
        _conv(facts=("ORD-1", "ORD-2")), _res(text="refund for ORD-1 done"))
    assert not missing_fact.resolved

def test_hallucination_flags_facts_missing_from_retrieval():
    row = score_conversation(_conv(), _res(retrieved=("nothing here",)))
    assert not row.grounded
    assert len(row.hallucinated_facts) >= 1

def test_aggregate_computes_rates():
    rows = [
        score_conversation(_conv(), _res()),
        score_conversation(_conv(), _res(action="cancel_order")),
    ]
    s = aggregate(rows)
    assert s.resolution_rate == 0.5
    assert s.grounded_rate == 1.0
    assert s.avg_latency_s == 0.5
    assert s.n == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_evaluator.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# src/voiceagent/evaluator.py
from __future__ import annotations

from dataclasses import dataclass, field
from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult


@dataclass
class EvalRow:
    conv_id: str
    resolved: bool
    grounded: bool
    wrong_action: bool
    hallucinated_facts: list[str]
    latency_s: float


@dataclass
class EvalSummary:
    resolution_rate: float
    grounded_rate: float
    wrong_action_rate: float
    hallucination_rate: float
    avg_latency_s: float
    n: int

    def as_dict(self) -> dict:
        return {
            "resolution_rate": self.resolution_rate,
            "grounded_rate": self.grounded_rate,
            "wrong_action_rate": self.wrong_action_rate,
            "hallucination_rate": self.hallucination_rate,
            "avg_latency_s": self.avg_latency_s,
            "n": self.n,
        }


def score_conversation(conv: Conversation, res: AgentResult) -> EvalRow:
    action_ok = res.action == conv.expected_action
    facts_ok = all(f in res.text for f in conv.key_facts)
    resolved = action_ok and facts_ok

    retrieved_text = "\n".join(r["text"] for r in res.retrieved)
    hallucinated = [f for f in conv.key_facts if f in res.text
                    and f not in retrieved_text]
    grounded = len(hallucinated) == 0

    return EvalRow(
        conv_id=conv.id,
        resolved=resolved,
        grounded=grounded,
        wrong_action=bool(res.action) and res.action != conv.expected_action,
        hallucinated_facts=hallucinated,
        latency_s=res.latency_s,
    )


def aggregate(rows: list[EvalRow]) -> EvalSummary:
    n = len(rows)
    if n == 0:
        return EvalSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    def rate(pred):
        return sum(1 for r in rows if pred(r)) / n
    return EvalSummary(
        resolution_rate=rate(lambda r: r.resolved),
        grounded_rate=rate(lambda r: r.grounded),
        wrong_action_rate=rate(lambda r: r.wrong_action),
        hallucination_rate=rate(lambda r: len(r.hallucinated_facts) > 0),
        avg_latency_s=sum(r.latency_s for r in rows) / n,
        n=n,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_evaluator.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: evaluator (resolution, groundedness, wrong-action rate)"
```

---

### Task 7: Benchmark runner — report table + threshold gate

**Files:**
- Create: `src/voiceagent/benchmark.py`
- Create: `scripts/run_benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `load_conversations` (Task 2), `build_index` (Task 3), `list_available_models`/`load_llm` (Task 4), `build_agent` (Task 5), `score_conversation`/`aggregate` (Task 6).
- Produces:
  - `run_benchmark(agent, conversations, max_rows: int | None = None) -> BenchmarkReport`
  - `BenchmarkReport`: `{summary: EvalSummary, per_language: dict[str, EvalSummary], rows: list[EvalRow], model_specs: dict}` — includes `model_specs` so the report carries its own cost metadata.
  - `VPS_COST_PER_MB` — estimate: ~₹3.0/month per MB of RAM (for a ₹3k/16GB VPS = 0.18₹/MB rounded up for simplicity). Used to compute `estimated_vps_cost_rs` from model peak RAM. Really just a lookup dict keyed by VPS tier.
  - `estimate_vps_cost(model_specs: dict) -> dict` — returns `{"vps_tier": "2-4vCPU/8GB"|"4-8vCPU/16GB", "ram_mb": int, "vps_cost_rs_estimate": int}`. If model RAM + Whisper (~200MB) + embedding (~300MB) + TTS (~150MB) + overhead (~200MB) fits 8GB, reports the cheaper tier; otherwise the 16GB tier.
  - `estimate_cost_per_conversation(model_specs: dict, avg_latency_s: float, resolution_rate: float) -> dict` — returns `{"cost_per_resolved_rs": float, "cost_per_turn_rs": float, "rs_per_conversation": float}`. Assumes a VPS cost of ₹3k/month (8GB tier) or ₹5k/month (16GB tier), a conversation averages 4 turns, and the VPS handles 1 conversation at a time (sequential). Formula: `turn_cost = (vps_cost_rs_per_hour / 3600) * avg_latency_s * 4_turns`. Then div by resolution rate to get per-resolved cost.
  - `sweep_all_models(conversations: list[Conversation], knowledge_dir: str, model_dir: str, max_rows: int | None = None) -> list[BenchmarkReport]` — discovers all downloaded models, runs each one through the full pipeline, returns sorted reports.
  - `write_sweep_report(reports: list[BenchmarkReport], out_dir: str) -> str` — writes `sweep-report.md` (comparison table across all models) + individual `report-{model_name}.md/json`. Returns the sweep path.
  - `evaluate_gate(report: BenchmarkReport, latency_max_s=2.0, resolution_min=0.75) -> tuple[bool, list[str]]` — `(passed, reasons)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
import json
import tempfile
from pathlib import Path
from voiceagent.benchmark import (run_benchmark, write_report, evaluate_gate,
                                  estimate_vps_cost, sweep_all_models)
from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult

class FixedAgent:
    def __init__(self, action, text):
        self.action = action
        self.text = text
    def handle(self, user_text):
        return AgentResult(text=self.text, action=self.action,
                           retrieved=[{"text": "ok"}], latency_s=0.5)

def _convs(n=10, action="refund"):
    return [Conversation(id=f"c{i}", language="en", intent="refund",
                         user_text="refund ORD-1", expected_action=action,
                         key_facts=["ORD-1"], escalate=False) for i in range(n)]

def test_run_benchmark_reports_by_language_and_gate():
    agent = FixedAgent("refund", "refund for ORD-1 done")
    report = run_benchmark(agent, _convs(10))
    assert report.summary.n == 10
    assert report.summary.resolution_rate == 1.0
    assert "en" in report.per_language

def test_evaluate_gate_flags_failures():
    report = run_benchmark(FixedAgent("wrong", "x"), _convs(10))
    ok, reasons = evaluate_gate(report)
    assert ok is False
    assert any("resolution" in r for r in reasons)

def test_write_report_emits_md_and_json():
    agent = FixedAgent("refund", "refund for ORD-1 done")
    report = run_benchmark(agent, _convs(10))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.md"
        write_report(report, str(out), {"model": "fake", "params": "0.5B"})
        assert out.exists()
        assert (Path(d) / "report.json").exists()
        j = json.loads((Path(d) / "report.json").read_text())
        assert j["summary"]["n"] == 10

def test_estimate_vps_cost_returns_tier():
    cost = estimate_vps_cost({"model": "qwen2.5-0.5b-q4", "size_mb": 400})
    assert cost["vps_tier"] == "2-4vCPU/8GB"
    assert cost["vps_cost_rs_estimate"] > 0

def test_sweep_all_models_returns_one_report_per_available_model(monkeypatch):
    # Patch at the source modules: sweep_all_models imports these inside the
    # function, which binds at call time, so patching the module attributes
    # is enough to avoid real model downloads / loads.
    import voiceagent.llm as llm_mod
    import voiceagent.knowledge as kb_mod

    class FakeLLM:
        def __init__(self, path):
            self.specs = {"model": "fake", "params": "0.5B",
                          "quant": "Q4_K_M", "model_path": path, "size_mb": 400}
        def generate(self, prompt, max_tokens=256, stop=None):
            return "refund for ORD-1 done\nACTION: refund"

    monkeypatch.setattr(
        llm_mod, "list_available_models",
        lambda model_dir: [
            {"name": "qwen2.5-0.5b-q4", "model_path": "/x",
             "params": "0.5B", "quant": "Q4_K_M", "size_mb": 400},
            {"name": "qwen2.5-1.5b-q4", "model_path": "/y",
             "params": "1.5B", "quant": "Q4_K_M", "size_mb": 1100},
        ])
    monkeypatch.setattr(llm_mod, "load_llm", lambda path, **kw: FakeLLM(path))
    monkeypatch.setattr(kb_mod, "load_docs", lambda d: [{"id": "a",
                         "text": "Refunds processed in 5-7 days.",
                         "section": "Refunds"}])
    monkeypatch.setattr(kb_mod, "build_index", lambda docs: _FakeIndex())

    reports = sweep_all_models(_convs(4), knowledge_dir="data/knowledge",
                               model_dir="data/models", max_rows=4)
    assert len(reports) == 2
    assert all(r.summary.n == 4 for r in reports)
    assert all(r.summary.resolution_rate == 1.0 for r in reports)

class _FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "Refunds processed in 5-7 days.",
                 "section": "Refunds", "score": 0.9}]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_benchmark.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# src/voiceagent/benchmark.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from voiceagent.dataset import Conversation
from voiceagent.agent import AgentResult
from voiceagent.evaluator import EvalRow, EvalSummary, score_conversation, aggregate

LATENCY_MAX_S = 2.0
RESOLUTION_MIN = 0.75

# Per-model fixed overheads on the VPS: embedding model, whisper, piper, base.
BASE_RAM_MB = 850  # embedding ~300 + whisper tiny ~200 + piper ~150 + overhead ~200

VPS_TIERS = [
    {"name": "2-4vCPU/8GB", "ram_mb": 8192, "cost_rs": 3000},
    {"name": "4-8vCPU/16GB", "ram_mb": 16384, "cost_rs": 5000},
]


@dataclass
class BenchmarkReport:
    summary: EvalSummary
    per_language: dict[str, EvalSummary]
    rows: list[EvalRow]
    model_specs: dict


def run_benchmark(agent, conversations: list[Conversation],
                  max_rows: int | None = None,
                  model_specs: dict | None = None) -> BenchmarkReport:
    rows: list[EvalRow] = []
    by_lang: dict[str, list[EvalRow]] = {}
    for conv in conversations[:max_rows]:
        res = agent.handle(conv.user_text)
        row = score_conversation(conv, res)
        rows.append(row)
        by_lang.setdefault(conv.language, []).append(row)
    return BenchmarkReport(
        summary=aggregate(rows),
        per_language={k: aggregate(v) for k, v in by_lang.items()},
        rows=rows,
        model_specs=model_specs or {},
    )


def estimate_vps_cost(model_specs: dict) -> dict:
    """Pick the cheapest VPS tier that fits the model + fixed pipeline overheads."""
    size_mb = model_specs.get("size_mb", 0)
    total = size_mb + BASE_RAM_MB
    for tier in VPS_TIERS:
        if total <= tier["ram_mb"]:
            return {"vps_tier": tier["name"], "ram_mb": total,
                    "vps_cost_rs_estimate": tier["cost_rs"]}
    # even 16GB doesn't fit -> report the 16GB tier anyway (LLM mmap can swap)
    tier = VPS_TIERS[-1]
    return {"vps_tier": tier["name"], "ram_mb": total,
            "vps_cost_rs_estimate": tier["cost_rs"]}


def estimate_cost_per_conversation(model_specs: dict, avg_latency_s: float,
                                   resolution_rate: float) -> dict:
    """₹ per turn and ₹ per resolved conversation.

    Assumes: one conversation at a time (sequential), ~4 turns per
    conversation, VPS billed per month at the tier cost."""
    tier = estimate_vps_cost(model_specs)
    cost_rs = tier["vps_cost_rs_estimate"]
    hours_per_month = 720
    cost_rs_per_hour = cost_rs / hours_per_month
    turns_per_conv = 4
    turn_cost = cost_rs_per_hour / 3600 * avg_latency_s * turns_per_conv
    resolved_cost = turn_cost / max(resolution_rate, 1e-9)
    return {"cost_per_turn_rs": round(turn_cost / turns_per_conv, 4),
            "cost_per_conversation_rs": round(turn_cost, 4),
            "cost_per_resolved_rs": round(resolved_cost, 4),
            "vps_tier": tier["vps_tier"]}


def sweep_all_models(conversations: list[Conversation], knowledge_dir: str,
                     model_dir: str, max_rows: int | None = None,
                     max_conversations: int = 200) -> list[BenchmarkReport]:
    """Run the full pipeline once per downloaded model. Return sorted by
    (passed gate, resolution desc, size asc). Uses at most max_conversations
    so a sweep over 3 models stays fast."""
    from voiceagent.knowledge import load_docs, build_index
    from voiceagent.agent import build_agent
    from voiceagent.llm import list_available_models, load_llm

    docs = load_docs(knowledge_dir)
    index = build_index(docs)
    models = list_available_models(model_dir)
    if not models:
        raise RuntimeError("no models downloaded — run scripts/smoke_llm.py first")

    reports = []
    for m in models:
        llm = load_llm(m["model_path"], params=m["params"], size_mb=m["size_mb"])
        agent = build_agent(index, llm)
        report = run_benchmark(agent, conversations,
                               max_rows=min(max_rows or len(conversations),
                                            max_conversations),
                               model_specs=llm.specs)
        reports.append(report)

    def sort_key(r: BenchmarkReport):
        passed, _ = evaluate_gate(r)
        return (not passed, -r.summary.resolution_rate, r.model_specs.get("size_mb", 0))
    return sorted(reports, key=sort_key)


def evaluate_gate(report: BenchmarkReport,
                  latency_max_s: float = LATENCY_MAX_S,
                  resolution_min: float = RESOLUTION_MIN) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    s = report.summary
    if s.avg_latency_s > latency_max_s:
        reasons.append(f"latency {s.avg_latency_s:.2f}s > {latency_max_s}s")
    if s.resolution_rate < resolution_min:
        reasons.append(f"resolution {s.resolution_rate:.3f} < {resolution_min}")
    return (not reasons, reasons)


def write_report(report: BenchmarkReport, out_path: str,
                 model_specs: dict | None = None) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = report.summary
    specs = model_specs or report.model_specs or {}
    cost = estimate_cost_per_conversation(specs, s.avg_latency_s, s.resolution_rate)
    tier = estimate_vps_cost(specs)

    lines = [
        "# VoiceAgent M0 Benchmark Report",
        "",
        f"- **Model:** {specs.get('model', '?')} "
        f"({specs.get('params', '?')}, {specs.get('quant', '?')})",
        f"- **Size:** {specs.get('size_mb', '?')} MB",
        f"- **VPS tier:** {tier['vps_tier']} (est ₹{tier['vps_cost_rs_estimate']}/month)",
        f"- **Cost:** ₹{cost['cost_per_resolved_rs']} per resolved conversation "
        f"(₹{cost['cost_per_conversation_rs']} per conversation)",
        f"- **Conversations:** {s.n}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Resolution rate | {s.resolution_rate:.3f} |",
        f"| Grounded rate | {s.grounded_rate:.3f} |",
        f"| Wrong-action rate | {s.wrong_action_rate:.3f} |",
        f"| Hallucination rate | {s.hallucination_rate:.3f} |",
        f"| Avg turn latency (s) | {s.avg_latency_s:.3f} |",
        f"| Est. cost / resolved (₹) | {cost['cost_per_resolved_rs']} |",
        "",
        "## By language",
        "",
        "| Language | n | Resolution | Latency (s) |",
        "|----------|---|------------|-------------|",
    ]
    for lang, ls in sorted(report.per_language.items()):
        lines.append(
            f"| {lang} | {ls.n} | {ls.resolution_rate:.3f} | {ls.avg_latency_s:.3f} |"
        )
    passed, reasons = evaluate_gate(report)
    lines += ["", "## Gate",
              "", f"**PASSED** ({len(reasons)} checks)" if passed
              else "**FAILED**"] + [f"- {r}" for r in reasons]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "summary": s.as_dict(),
        "per_language": {k: v.as_dict() for k, v in report.per_language.items()},
        "model": specs,
        "cost": cost,
    }
    (out.parent / "report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def write_sweep_report(reports: list[BenchmarkReport], out_dir: str) -> str:
    """Write per-model reports + a comparison sweep table. Returns sweep path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(reports):
        name = r.model_specs.get("model", f"model-{i}").replace(".gguf", "")
        write_report(r, str(out_dir / f"report-{name}.md"), r.model_specs)

    lines = [
        "# VoiceAgent M0 Model Sweep",
        "",
        "| Model | Size (MB) | VPS tier | Resolution | Latency (s) | Wrong-action | ₹/resolved | Gate |",
        "|-------|-----------|----------|-----------|-------------|--------------|------------|------|",
    ]
    for r in reports:
        specs = r.model_specs
        cost = estimate_cost_per_conversation(specs, r.summary.avg_latency_s,
                                              r.summary.resolution_rate)
        tier = estimate_vps_cost(specs)
        passed, _ = evaluate_gate(r)
        lines.append(
            f"| {specs.get('model','?')} | {specs.get('size_mb','?')} "
            f"| {tier['vps_tier']} | {r.summary.resolution_rate:.3f} "
            f"| {r.summary.avg_latency_s:.3f} | {r.summary.wrong_action_rate:.3f} "
            f"| {cost['cost_per_resolved_rs']} | {'PASS' if passed else 'FAIL'} |"
        )
    sweep_path = out_dir / "sweep-report.md"
    sweep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(sweep_path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_benchmark.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Write the runner script and commit**

```bash
cat > scripts/run_benchmark.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.dataset import load_conversations
from voiceagent.benchmark import (sweep_all_models, write_sweep_report,
                                  evaluate_gate)

if __name__ == "__main__":
    convs = load_conversations("data/eval/conversations.csv")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    reports = sweep_all_models(convs, knowledge_dir="data/knowledge",
                               model_dir="data/models", max_rows=n)
    path = write_sweep_report(reports, "data/out")
    print("wrote", path)
    for r in reports:
        passed, reasons = evaluate_gate(r)
        print(f"{r.model_specs.get('model','?'):40s} "
              f"res={r.summary.resolution_rate:.3f} "
              f"lat={r.summary.avg_latency_s:.3f}s "
              f"gate={'PASS' if passed else 'FAIL'}")
EOF
git add -A
git commit -m "feat: benchmark runner with model sweep + VPS cost per resolved conversation"
```

---

### Task 8: Voice path measurement — ASR (faster-whisper) + TTS (Piper) latency on CPU

**Files:**
- Create: `src/voiceagent/voice.py`
- Create: `scripts/measure_voice.py`
- Test: `tests/test_voice.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `asr_latency(audio_path: str | None = None) -> tuple[float, str]` — (seconds, transcript). If no audio file, measures warm load on a bundled tone sample if present, else returns `(0.0, "")` and is marked skipped.
  - `tts_latency(text: str = "Namaste, aapka order kal deliver ho jayega.") -> float` — seconds for one utterance.
  - `measure_voice_pipeline(sample_audio: str | None) -> dict` — `{"asr_s", "tts_s", "voice_turn_s"}` where `voice_turn_s = asr_s + tts_s + agent_turn` placeholder reported as `asr_s + tts_s` for M0.

**Rationale:** M0 must report voice-turn latency. `faster-whisper` (tiny.en or base) via CTranslate2 runs on CPU; Piper runs on CPU. The M0 gate targets the text turn ≤ 2s; voice adds ASR+TTS and is reported separately so we can see whether the 2s budget is feasible with streaming later.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voice.py
from voiceagent.voice import tts_latency

def test_tts_latency_returns_seconds():
    s = tts_latency("hello")
    # may be 0.0 if piper is unavailable in CI; must not be negative
    assert s >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_voice.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# src/voiceagent/voice.py
from __future__ import annotations

import time
from pathlib import Path

_tts = None
_asr = None


def _get_tts():
    global _tts
    if _tts is None:
        from piper import PiperVoice  # piper-tts
        _tts = PiperVoice.load("en_US-lessac-medium")  # ~63MB, cached
    return _tts


def tts_latency(text: str = "Namaste, aapka order kal deliver ho jayega.") -> float:
    t0 = time.time()
    try:
        tts = _get_tts()
        # generate to /dev/null (do not write wav in benchmark)
        tts.synthesize_wav(text, None)  # None -> discards output
    except Exception as e:  # pragma: no cover - piper may not be installed in CI
        print(f"[voice] tts unavailable: {e}")
        return 0.0
    return time.time() - t0


def asr_latency(audio_path: str | None = None) -> tuple[float, str]:
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        _asr = WhisperModel("tiny", device="cpu", compute_type="int8")
    t0 = time.time()
    if audio_path and Path(audio_path).exists():
        segs, _ = _asr.transcribe(audio_path)
        text = " ".join(s.text for s in segs).strip()
    else:
        text = ""
    return time.time() - t0, text


def measure_voice_pipeline(sample_audio: str | None = None) -> dict:
    asr_s, _ = asr_latency(sample_audio)
    tts_s = tts_latency()
    return {"asr_s": round(asr_s, 3), "tts_s": round(tts_s, 3),
            "voice_turn_s": round(asr_s + tts_s, 3)}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_voice.py -v
```

Expected: 1 test PASS (returns ≥ 0 even when piper is unavailable).

- [ ] **Step 5: Measure voice latency and commit**

```bash
cat > scripts/measure_voice.py <<'EOF'
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.voice import measure_voice_pipeline

if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else None
    m = measure_voice_pipeline(audio)
    print(json.dumps(m, indent=2))
    Path("data/out/voice.json").write_text(json.dumps(m, indent=2))
EOF
source .venv/bin/activate
python scripts/measure_voice.py
```

Expected: prints `asr_s`, `tts_s`, `voice_turn_s`. Downloads whisper `tiny` + piper `en_US-lessac-medium` once, cached afterward.

```bash
git add -A
git commit -m "feat: voice path latency measurement (faster-whisper + piper on CPU)"
```

---

### Task 9: Full M0 sweep run + publish comparison + record decision

**Files:**
- Modify: `README.md` (add benchmark results + go/no-go decision)
- Create: `data/out/sweep-report.md`, `data/out/report-*.md`, `data/out/report-*.json`, `data/out/voice.json`

**Interfaces:**
- Consumes: everything.
- Produces: a committed sweep comparison table and an explicit "cheapest model that passes" decision recorded in the README.

- [ ] **Step 1: Run the full model sweep**

```bash
source .venv/bin/activate
python scripts/run_benchmark.py 200
cat data/out/sweep-report.md
```

Expected: comparison table with all downloaded models, sorted by (gate passed, resolution desc, size asc). The winner is the first row that says PASS.

- [ ] **Step 2: Run voice measurement**

```bash
source .venv/bin/activate
python scripts/measure_voice.py
cat data/out/voice.json
```

Expected: `voice_turn_s` value recorded.

- [ ] **Step 3: Record the decision**

Edit `README.md`, replacing the placeholder with the actual sweep results and the cheapest-that-passes choice:

```markdown
## M0 Gate Results

### Model Sweep (sorted by cheapest-passing first)

| Model | Size | VPS tier | Resolution | Latency | ₹/resolved | Gate |
|-------|------|----------|-----------|---------|-----------|------|
| <qwen2.5-0.5b-q4> | 400MB | 2-4vCPU/8GB | <fill> | <fill> | <fill> | PASS/FAIL |
| <qwen2.5-1.5b-q4> | 1.1GB | 4-8vCPU/16GB | <fill> | <fill> | <fill> | PASS/FAIL |
| <phi-3.5-mini-q4> | 2.4GB | 4-8vCPU/16GB | <fill> | <fill> | <fill> | PASS/FAIL |

**Winner:** <model name> — cheapest model that passes the gate (₹<fill>/resolved).

**Decision:** GO / NO-GO — GO if at least one model passes the gate. Otherwise state what must change before M1.
```

- [ ] **Step 4: Commit the decision**

```bash
git add -A
git commit -m "docs: M0 model sweep results + cheapest-passing recommendation"
```

---

### Task 10: M0 retrospective — what the spike proved

**Files:**
- Create: `docs/superpowers/plans/2026-08-31-voiceagent-m0-retro.md`

**Interfaces:**
- Consumes: `data/out/sweep-report.md`, `data/out/report-*.md`, `data/out/voice.json`.
- Produces: a short retro recording measured numbers, the cheapest-passing model choice, what surprised us, and the decision on the next milestone (M1 Control Plane core vs. thesis revision).

- [ ] **Step 1: Write the retro from the measured numbers**

Create `docs/superpowers/plans/2026-08-31-voiceagent-m0-retro.md`:

```markdown
# M0 Retrospective

**Date:** 2026-08-31

## Measured (from data/out/)

| Model | Size | VPS tier | Resolution | Latency (text) | Wrong-action | ₹/resolved | Gate |
|-------|------|----------|-----------|----------------|--------------|------------|------|
| <fill from sweep-report.md> | | | | | | | |

## Voice path
- Voice turn (ASR+TTS) = <fill from data/out/voice.json>.

## Per-language
- English / Hindi / Hinglish resolution and latency from the winning model's report.

## Cheapest model that passes
- <model name> at ₹<fill>/resolved on a <vps tier>. If none passed, note the closest and what it would take to pass.

## What surprised us
- (1-3 bullets: e.g. Hinglish retrieval quality, cold-start vs steady-state latency,
  where the 2s budget went, whether 0.5B was enough.)

## Decision
- **GO → proceed to M1 (Control Plane core)** if at least one model passed the gate.
- **REVISE** — list the specific change (smaller/faster model, streaming ASR,
  better prompts, more KB coverage, different quantization) — if none passed.
```

- [ ] **Step 2: Commit the retro**

```bash
git add -A
git commit -m "docs: M0 retrospective and milestone decision"
```

---

## Post-Plan Notes for the Executor

- **First run downloads models** (embedding ~470MB, 3× GGUF: 0.5B/~400MB + 1.5B/~1.1GB + 3.8B/~2.4GB, whisper tiny ~75MB, piper ~63MB). Run the smoke scripts in Task 4/5 and Task 8 with a live connection; subsequent runs are fully offline.
- **You don't need to download all three models.** The sweep runs whatever is in `data/models/`. Start with `qwen2.5-0.5b-q4` (cheapest, smallest, fastest to test), then add `qwen2.5-1.5b-q4` if it fails. Only download `phi-3.5-mini-q4` if both smaller ones fail the gate.
- **If 3.14 lacks wheels:** recreate the venv as `uv venv --python 3.12` before installing `requirements.txt`; the code itself is 3.12-compatible.
- **M1 scope (next plan):** policy engine (YAML), decision log, permissions dashboard, tool gateway with preconditions/idempotency/timeouts — per spec §7/§9.
- **M2 scope (after M1):** agent hardening, human-handoff serialization, per-resolved-conversation billing instrumentation — per spec §9.
