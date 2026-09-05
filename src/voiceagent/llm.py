# src/voiceagent/llm.py
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from llama_cpp import Llama

# The model registry is DATA (data/models/registry.yaml, one source shared
# with config.default_candidate_models). `models` entries are auto-download
# URLs; `local_models` entries are local-only artifacts (name + filename,
# url=None) produced by the Kaggle workflow — listed when the file exists
# in the models dir, never auto-downloaded. The built-in list below is the
# fallback when the YAML is missing/broken.
_BUILTIN_CANDIDATE_MODELS = [
    {
        "name": "qwen3-0.6b-q4",
        "url": ("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/"
                "resolve/main/Qwen3-0.6B-Q4_K_M.gguf"),
        "size_mb": 397,
        "params": "0.6B",
    },
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
        "name": "qwen2.5-0.5b-hinglish-q4",
        "url": None,
        "filename": "qwen2.5-0.5b-hinglish-q4_k_m.gguf",
    },
]

# Anchored to the repo root so the registry resolves from any CWD (tests,
# scripts) — data/ is repo data, not a cwd-relative lookup.
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "registry.yaml"


def load_model_registry(path: str | Path | None = None) -> list[dict]:
    """Model registry entries from data/models/registry.yaml — `models`
    (name/url/size_mb/params, auto-downloadable) plus `local_models`
    (name/filename, url=None). A missing/broken file falls back to the
    built-in list so model discovery never crashes offline."""
    p = Path(path) if path is not None else REGISTRY_PATH
    try:
        import yaml
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, yaml.YAMLError):
        raw = None
    if isinstance(raw, dict) and (raw.get("models") or raw.get("local_models")):
        entries = [dict(e) for e in (raw.get("models") or [])
                   if isinstance(e, dict) and e.get("name") and e.get("url")]
        entries += [dict(e, url=None)
                    for e in (raw.get("local_models") or [])
                    if isinstance(e, dict) and e.get("name") and e.get("filename")]
        if entries:
            return entries
    return [dict(m) for m in _BUILTIN_CANDIDATE_MODELS]


CANDIDATE_MODELS = load_model_registry()


# ---------------------------------------------------------------------------
# Chat template registry: model family -> renderer(system, context, user).
# Family is inferred from the model filename (infer_family) and can be
# overridden explicitly (LlamaCppLLM(..., family="llama3")).
# ---------------------------------------------------------------------------

def _render_qwen(system: str, context: str, user_text: str) -> str:
    """Qwen ChatML — the format Qwen2.5/Qwen3 instruct models were trained
    on. Raw-completion prompts cause small instruct models to ramble and
    ignore the ACTION instruction."""
    user = f"{context}\n\nCustomer: {user_text}"
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _render_llama3(system: str, context: str, user_text: str) -> str:
    """Llama 3 / 3.1 instruct header format (llama.cpp adds the
    <|begin_of_text|> BOS itself)."""
    user = f"{context}\n\nCustomer: {user_text}"
    return (
        "<|start_header_id|>system<|end_header_id|>\n\n" + system
        + "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n" + user
        + "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _render_gemma(system: str, context: str, user_text: str) -> str:
    """Gemma turn format — Gemma has no system role, so the system text is
    folded into the first user turn."""
    user = f"{system}\n\n{context}\n\nCustomer: {user_text}"
    return (
        "<start_of_turn>user\n" + user + "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _render_generic(system: str, context: str, user_text: str) -> str:
    """Raw-completion prompt for unknown / non-chat families (the legacy
    LLMHandle.chat_template format)."""
    return (
        f"{system}\n\nContext:\n{context}\n\n"
        f"Customer: {user_text}\nAssistant:"
    )


CHAT_TEMPLATES = {
    "qwen": _render_qwen,
    "llama3": _render_llama3,
    "gemma": _render_gemma,
    "generic": _render_generic,
}


def infer_family(model_path: str, explicit: str | None = None) -> str:
    """Chat-template family for a model, guessed from its filename (GGUF
    metadata would be nicer but filenames cover our candidates). An explicit
    family wins; unknown explicit names are a hard error so a typo never
    silently degrades to the wrong template."""
    if explicit is not None:
        if explicit not in CHAT_TEMPLATES:
            raise ValueError(f"unknown family {explicit!r}; "
                             f"known: {sorted(CHAT_TEMPLATES)}")
        return explicit
    name = Path(model_path).name.lower()
    if "qwen" in name:
        return "qwen"
    if "llama-3" in name or "llama3" in name:
        return "llama3"
    if "gemma" in name:
        return "gemma"
    return "generic"


# Qwen3 emits a "thinking" phase before the answer. llama.cpp renders it
# either as " thinking\n...\n response" or as "<think>...</think>"
# depending on version. Words inside the thinking block (e.g. "cancel_order"
# considered then rejected) would corrupt action extraction, so stop at the
# marker and strip the block afterwards. (Moved here from agent.py — output
# cleanup is an adapter concern, not an agent one.)
THINKING_RE = re.compile(
    r"(?:<think>.*?</think>|^\s*thinking\s*\n.*?(?=\s*response|\Z))",
    re.DOTALL | re.MULTILINE,
)
RESPONSE_PREFIX_RE = re.compile(r"^\s*response\s*\n?", re.MULTILINE)

# Families that emit a reasoning phase before the answer.
THINKING_FAMILIES = {"qwen"}
FAMILY_STOP_TOKENS: dict[str, list[str]] = {"qwen": [" thinking"]}


def clean_thinking_text(text: str) -> str:
    """Remove a thinking phase + 'response' prefix from raw model output."""
    return RESPONSE_PREFIX_RE.sub("", THINKING_RE.sub("", text)).strip()


def family_stop_tokens(family: str) -> list[str] | None:
    """Generation stop strings for a family (None = no stops)."""
    return list(FAMILY_STOP_TOKENS[family]) if family in FAMILY_STOP_TOKENS else None


class LLMHandle:
    # Adapter attributes Agent reads defensively (getattr): per-family stop
    # strings and output cleanup. Base = no stops, no-op cleanup.
    family = "generic"
    stop_tokens: list[str] | None = None

    def __init__(self, specs: dict):
        self.specs = specs

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: list[str] | None = None) -> str:
        raise NotImplementedError

    def chat_template(self, system: str, context: str, user_text: str) -> str:
        """Wrap a raw prompt in the model's chat format. Base returns the
        plain concatenation so non-chat (e.g. test) handles behave like a
        raw completion prompt."""
        return _render_generic(system, context, user_text)

    def postprocess(self, text: str) -> str:
        """Clean raw generation output. Default no-op; adapters override
        (e.g. the llama.cpp adapter strips the Qwen3 thinking phase)."""
        return text


class FamilyLLM(LLMHandle):
    """LLMHandle bound to a chat-template family: registry-driven
    chat_template, per-family stop tokens and thinking-phase cleanup.
    Shared by the local (llama.cpp) and remote (OpenAI-compatible) adapters."""

    def __init__(self, specs: dict, family: str):
        super().__init__(specs)
        self.family = family
        self.stop_tokens = family_stop_tokens(family)

    def chat_template(self, system: str, context: str, user_text: str) -> str:
        return CHAT_TEMPLATES[self.family](system, context, user_text)

    def postprocess(self, text: str) -> str:
        """Thinking-style families get the reasoning phase stripped; other
        families pass text through untouched."""
        return clean_thinking_text(text) if self.family in THINKING_FAMILIES else text


class LlamaCppLLM(FamilyLLM):
    def __init__(self, model_path: str, n_ctx: int = 2048,
                 params: str = "?", size_mb: float = 0.0,
                 family: str | None = None):
        super().__init__({
            "model": Path(model_path).name,
            "params": params,
            "quant": "Q4_K_M",
            "model_path": str(model_path),
            "size_mb": size_mb,
        }, family=infer_family(model_path, family))
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=0)

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: list[str] | None = None) -> str:
        out = self._llm(prompt, max_tokens=max_tokens, stop=stop, echo=False)
        return out["choices"][0]["text"].strip()


class OpenAICompatLLM(FamilyLLM):
    """Any OpenAI-compatible /chat/completions endpoint (vLLM, Ollama, LM
    Studio, ...) over stdlib urllib — no new dependencies. chat_template()
    builds the system/user messages; generate() performs the HTTP round
    trip through the same interface the agent already uses."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 30.0):
        super().__init__({
            "model": model, "params": "remote", "quant": "-",
            "model_path": base_url, "size_mb": 0.0,
        }, family=infer_family(model))
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._pending_messages: list[dict] | None = None

    def chat_template(self, system: str, context: str, user_text: str) -> str:
        """Build the system/user chat messages for /chat/completions. The
        agent calls chat_template immediately before generate, so the
        messages are stashed for that call; the returned string is a
        human-readable rendering of them."""
        user = f"{context}\n\nCustomer: {user_text}"
        self._pending_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return f"system: {system}\n\nuser: {user}"

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: list[str] | None = None) -> str:
        messages = self._pending_messages or [{"role": "user", "content": prompt}]
        self._pending_messages = None
        payload: dict = {"model": self.model, "messages": messages,
                         "max_tokens": max_tokens}
        if stop:
            payload["stop"] = stop
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"}
                   if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible endpoint returned "
                               f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach OpenAI-compatible endpoint "
                               f"{self.base_url}: {e.reason}") from e
        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise RuntimeError(f"unexpected response from "
                               f"{self.base_url}: {body!r}") from e


def build_llm_from_env() -> OpenAICompatLLM | None:
    """Remote LLM from the environment: VOICEAGENT_LLM_BASE_URL +
    VOICEAGENT_LLM_MODEL (both required), VOICEAGENT_LLM_API_KEY (optional).
    Returns None when not configured so callers fall back to a local GGUF
    via load_llm()."""
    base = os.environ.get("VOICEAGENT_LLM_BASE_URL")
    model = os.environ.get("VOICEAGENT_LLM_MODEL")
    if not base or not model:
        return None
    return OpenAICompatLLM(base, model,
                           api_key=os.environ.get("VOICEAGENT_LLM_API_KEY"))


def download_model(url: str, model_dir: str = "data/models") -> str:
    """Download a GGUF into model_dir (idempotent) and return local path."""
    if not url:
        raise ValueError(
            "download_model needs a URL — local_models registry entries "
            "carry a filename only and are never auto-downloaded")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    path = Path(model_dir) / url.rsplit("/", 1)[1]
    if not path.exists():
        import urllib.request
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return str(path)


def list_available_models(model_dir: str = "data/models") -> list[dict]:
    """Specs of candidate models already present in model_dir, in size
    order. Auto-download entries match by URL basename; local_models
    entries (url=None) match by their declared filename."""
    out = []
    for cand in CANDIDATE_MODELS:
        if cand.get("url"):
            path = Path(model_dir) / cand["url"].rsplit("/", 1)[1]
            size_mb = cand["size_mb"]
        else:
            path = Path(model_dir) / cand["filename"]
            size_mb = (round(path.stat().st_size / (1024 * 1024))
                       if path.exists() else 0)
        if path.exists():
            out.append({"name": cand["name"], "model_path": str(path),
                        "params": cand.get("params", "?"), "quant": "Q4_K_M",
                        "size_mb": size_mb})
    return sorted(out, key=lambda m: m["size_mb"])


def load_llm(model_path: str, n_ctx: int = 2048,
             params: str = "?", size_mb: float = 0.0) -> LLMHandle:
    return LlamaCppLLM(model_path, n_ctx=n_ctx, params=params, size_mb=size_mb)
