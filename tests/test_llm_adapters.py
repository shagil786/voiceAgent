# tests/test_llm_adapters.py
"""Adapter layer: chat-template registry, thinking-phase cleanup, the
OpenAI-compatible backend (against a stub HTTP server — no network), env
selection, and Agent's adapter-driven stop/postprocess plumbing."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from voiceagent.agent import Agent, build_agent, SYSTEM_PROMPT
from voiceagent.llm import (CHAT_TEMPLATES, FAMILY_STOP_TOKENS, LLMHandle,
                            LlamaCppLLM, OpenAICompatLLM,
                            THINKING_FAMILIES, build_llm_from_env,
                            clean_thinking_text, family_stop_tokens,
                            infer_family)
from voiceagent.policy import PolicyEngine

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "x", "section": "faqs", "score": 0.9}]


def _family_stub(cls, family):
    """Instance of an adapter class without running __init__ (which would
    load a real GGUF); postprocess/chat_template only need family attrs."""
    obj = cls.__new__(cls)
    obj.family = family
    obj.stop_tokens = family_stop_tokens(family)
    return obj


class _StubHandler(BaseHTTPRequestHandler):
    def _reply(self, code, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.server.last_request = {
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(n) or b"{}"),
        }
        if getattr(self.server, "fail", False):
            self._reply(500, b'{"error": "internal boom"}')
            return
        self._reply(200, json.dumps(
            {"choices": [{"message": {"content": "  hi there\n"}}]}).encode())

    def log_message(self, *args):
        pass


@pytest.fixture()
def stub_server():
    srv = HTTPServer(("127.0.0.1", 0), _StubHandler)
    srv.last_request = None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _stub_url(srv) -> str:
    return f"http://127.0.0.1:{srv.server_port}"


# ---------------------------------------------------------------------------
# family inference / template registry
# ---------------------------------------------------------------------------

def test_infer_family_from_filename():
    assert infer_family("data/models/Qwen3-0.6B-Q4_K_M.gguf") == "qwen"
    assert infer_family("data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf") == "qwen"
    assert infer_family("x/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf") == "llama3"
    assert infer_family("x/gemma-2-2b-it-q4_k_m.gguf") == "gemma"
    assert infer_family("x/mistral-7b-q4.gguf") == "generic"

def test_infer_family_explicit_override():
    assert infer_family("x/qwen.gguf", explicit="llama3") == "llama3"
    with pytest.raises(ValueError):
        infer_family("x/qwen.gguf", explicit="chatml")

def test_qwen_template_byte_identical():
    # Exact legacy ChatML string (llm.py pre-refactor behaviour) — must not
    # drift or the existing Qwen models' format-following changes.
    out = CHAT_TEMPLATES["qwen"]("SYS", "CTX", "hello")
    assert out == (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nCTX\n\nCustomer: hello<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def test_llama3_template():
    out = CHAT_TEMPLATES["llama3"]("SYS", "CTX", "hi")
    assert out == (
        "<|start_header_id|>system<|end_header_id|>\n\nSYS<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\nCTX\n\nCustomer: hi<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def test_gemma_template_folds_system_into_user_turn():
    out = CHAT_TEMPLATES["gemma"]("SYS", "CTX", "hi")
    assert out == (
        "<start_of_turn>user\nSYS\n\nCTX\n\nCustomer: hi<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

def test_generic_template_matches_legacy_base():
    assert CHAT_TEMPLATES["generic"]("S", "C", "U") == (
        "S\n\nContext:\nC\n\nCustomer: U\nAssistant:")

# ---------------------------------------------------------------------------
# thinking-phase cleanup + per-family stops
# ---------------------------------------------------------------------------

def test_family_stop_and_thinking_tables():
    assert THINKING_FAMILIES == {"qwen"}
    assert FAMILY_STOP_TOKENS["qwen"] == [" thinking"]
    assert family_stop_tokens("qwen") == [" thinking"]
    assert family_stop_tokens("llama3") is None

def test_clean_thinking_text_word_marker_form():
    raw = (" thinking\nmaybe cancel_order? no, refund\n response\n"
           "Refund started for ORD-1")
    assert clean_thinking_text(raw) == "Refund started for ORD-1"

def test_clean_thinking_text_think_tag_form():
    assert clean_thinking_text("<think>hmm, cancel_order?</think>Done") == "Done"

def test_clean_thinking_text_no_thinking_is_noop():
    assert clean_thinking_text("Plain answer\nACTION: refund") == \
        "Plain answer\nACTION: refund"

def test_llamacpp_postprocess_by_family():
    qwen = _family_stub(LlamaCppLLM, "qwen")
    assert qwen.postprocess(" thinking\nx\n response\nAnswer") == "Answer"
    assert qwen.stop_tokens == [" thinking"]
    for fam in ("llama3", "gemma", "generic"):
        h = _family_stub(LlamaCppLLM, fam)
        assert h.postprocess(" raw text ") == " raw text "  # no-op, untouched
        assert h.stop_tokens is None

def test_llamacpp_chat_template_dispatch():
    assert _family_stub(LlamaCppLLM, "qwen").chat_template("S", "C", "U") == \
        CHAT_TEMPLATES["qwen"]("S", "C", "U")
    assert "start_header_id" in \
        _family_stub(LlamaCppLLM, "llama3").chat_template("S", "C", "U")

# ---------------------------------------------------------------------------
# OpenAICompatLLM against a stub HTTP server
# ---------------------------------------------------------------------------

def test_openai_compat_generate_roundtrip(stub_server):
    llm = OpenAICompatLLM(_stub_url(stub_server), "qwen2.5-0.5b-instruct",
                          api_key="test-key")
    assert llm.generate("hello", max_tokens=32) == "hi there"
    req = stub_server.last_request
    assert req["path"] == "/chat/completions"
    assert req["auth"] == "Bearer test-key"
    assert req["body"]["model"] == "qwen2.5-0.5b-instruct"
    assert req["body"]["max_tokens"] == 32
    assert req["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert "stop" not in req["body"]  # None stop must be omitted

def test_openai_compat_chat_template_messages(stub_server):
    llm = OpenAICompatLLM(_stub_url(stub_server), "m")
    llm.chat_template("SYS", "CTX", "hi")
    llm.generate("ignored", max_tokens=5, stop=["x"])
    body = stub_server.last_request["body"]
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "CTX\n\nCustomer: hi"},
    ]
    assert body["stop"] == ["x"]

def test_openai_compat_no_auth_header_when_no_key(stub_server):
    llm = OpenAICompatLLM(_stub_url(stub_server), "m")
    llm.generate("hi")
    assert stub_server.last_request["auth"] is None

def test_openai_compat_http_error_raises_runtimeerror_with_body(stub_server):
    llm = OpenAICompatLLM(_stub_url(stub_server), "m")
    stub_server.fail = True
    with pytest.raises(RuntimeError, match="internal boom"):
        llm.generate("hi")

def test_openai_compat_unreachable_raises_runtimeerror():
    srv = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = srv.server_port
    srv.server_close()  # free the port so the connection is refused
    llm = OpenAICompatLLM(f"http://127.0.0.1:{port}", "m")
    with pytest.raises(RuntimeError, match="cannot reach"):
        llm.generate("hi")

def test_openai_compat_family_from_model_name(stub_server):
    # Thinking-family models (Qwen3 via any backend) get the same stops and
    # cleanup as the local llama.cpp adapter.
    qwen = OpenAICompatLLM(_stub_url(stub_server), "Qwen3-0.6B")
    assert qwen.family == "qwen"
    assert qwen.stop_tokens == [" thinking"]
    assert qwen.postprocess(" thinking\nx\n response\nA") == "A"
    llama3 = OpenAICompatLLM(_stub_url(stub_server), "meta-llama-3-8b-instruct")
    assert llama3.family == "llama3"
    assert llama3.stop_tokens is None

# ---------------------------------------------------------------------------
# build_llm_from_env
# ---------------------------------------------------------------------------

def test_build_llm_from_env_unset(monkeypatch):
    for k in ("VOICEAGENT_LLM_BASE_URL", "VOICEAGENT_LLM_MODEL",
              "VOICEAGENT_LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert build_llm_from_env() is None

def test_build_llm_from_env_partial(monkeypatch):
    monkeypatch.setenv("VOICEAGENT_LLM_BASE_URL", "http://x/v1")
    monkeypatch.delenv("VOICEAGENT_LLM_MODEL", raising=False)
    assert build_llm_from_env() is None

def test_build_llm_from_env_full(monkeypatch):
    monkeypatch.setenv("VOICEAGENT_LLM_BASE_URL", "http://x/v1/")
    monkeypatch.setenv("VOICEAGENT_LLM_MODEL", "qwen2.5-0.5b-instruct")
    monkeypatch.setenv("VOICEAGENT_LLM_API_KEY", "sk-test")
    llm = build_llm_from_env()
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.base_url == "http://x/v1"
    assert llm.model == "qwen2.5-0.5b-instruct"
    assert llm.api_key == "sk-test"

# ---------------------------------------------------------------------------
# Agent + adapter plumbing (stop/postprocess/system prompt)
# ---------------------------------------------------------------------------

class BareLLM(LLMHandle):
    """A bare custom handle implementing only generate() (everything else is
    inherited from LLMHandle: stop_tokens None, postprocess no-op)."""
    def __init__(self, stop_tokens=None):
        super().__init__({"model": "bare"})
        self.stop_tokens = stop_tokens
        self.calls = []
    def generate(self, prompt, max_tokens=256, stop=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens,
                           "stop": stop})
        return "Your order ORD-77812 is out for delivery.\nACTION: order_status"

def test_agent_with_bare_custom_llmhandle():
    llm = BareLLM()
    res = build_agent(FakeIndex(), llm).handle("where is my order ORD-77812")
    assert res.action == "order_status"
    assert "out for delivery" in res.text
    assert llm.calls[0]["stop"] is None  # stop derived from adapter (default)
    assert llm.calls[0]["max_tokens"] == 300

def test_agent_derives_stop_from_adapter():
    llm = BareLLM(stop_tokens=[" thinking"])
    build_agent(FakeIndex(), llm).handle("hi")
    assert llm.calls[0]["stop"] == [" thinking"]

def test_agent_calls_adapter_postprocess():
    class ThinkingLLM(BareLLM):
        def generate(self, prompt, max_tokens=256, stop=None):
            self.calls.append({"prompt": prompt, "max_tokens": max_tokens,
                               "stop": stop})
            return " thinking\nmaybe refund?\n response\nAnswer\nACTION: refund"
        def postprocess(self, text):
            return clean_thinking_text(text)
    llm = ThinkingLLM(stop_tokens=[" thinking"])
    res = build_agent(FakeIndex(), llm).handle("hi")
    # M5c: the ACTION line is decision scaffolding, scrubbed from the
    # customer-visible text after extract_action() captured "refund".
    assert res.text == "Answer"
    assert res.action == "refund"

def test_policy_declared_actions_drive_system_prompt():
    llm = BareLLM()
    agent = build_agent(FakeIndex(), llm,
                        policy={"actions": ["refund", "order_status"],
                                "refund": {"require_auth": True}})
    agent.handle("hi")
    assert "one of: refund, order_status" in llm.calls[0]["prompt"]
    assert agent._system_prompt.startswith(SYSTEM_PROMPT[:30])

def test_static_prompt_kept_when_policy_has_no_declared_actions():
    llm = BareLLM()
    agent = build_agent(FakeIndex(), llm,
                        policy={"refund": {"require_auth": True}})
    agent.handle("hi")
    assert "one of: order_status, refund, cancel_order" in llm.calls[0]["prompt"]
    assert agent._system_prompt == SYSTEM_PROMPT

def test_policy_engine_known_actions_accessor():
    assert PolicyEngine({"refund": {"allow": True}}).known_actions() == []
    assert PolicyEngine({"actions": ["a", "b"]}).known_actions() == ["a", "b"]
    assert PolicyEngine({"actions": "nope"}).known_actions() == []

def test_system_prompt_byte_identical_to_legacy_text():
    # Pin the exact pre-refactor SYSTEM_PROMPT so prompt changes are always
    # a conscious, reviewed decision. Consciously changed: the default
    # persona is the neutral "customer support assistant" (was "...for an
    # Indian ecommerce company"); the M5c informational actions
    # (refund_info, delivery_eta) were added to the vocabulary earlier.
    assert SYSTEM_PROMPT == (
        "You are a customer support assistant. "
        "Answer directly and concisely — do NOT use a thinking or "
        "reasoning phase. Answer ONLY from the provided context. "
        "Always address the customer's specific reference (order id, phone, "
        "plan, account) from their message in your reply — echo it verbatim. "
        "If the customer's request requires an action (refund, cancel, etc.), "
        "end your reply with a line: ACTION: <action_name> where action_name "
        "is one of: order_status, refund, cancel_order, address_change, "
        "payment_declined, recharge, billing, return, replacement, otp, "
        "fraud, account_closure, delivery_delay, product_info, invoice, "
        "plan_change, roaming, network_issue, complaint, high_value_refund, "
        "refund_info, delivery_eta. "
        "If no action is needed, do not emit an ACTION line."
    )
