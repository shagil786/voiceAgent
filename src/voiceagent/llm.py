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
