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
    assert "qwen3-0.6b-q4" in names
    assert "qwen2.5-0.5b-q4" in names
    assert "qwen2.5-1.5b-q4" in names
    # size_mb is an auto-download property; local_models entries (url=None)
    # get theirs from disk at listing time.
    assert all(m["size_mb"] > 0 for m in CANDIDATE_MODELS if m.get("url"))
