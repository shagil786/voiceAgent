# LLM Adapter Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LLM a fully swappable component in VoiceAgent — any small model, any backend (local GGUF via llama.cpp, or any OpenAI-compatible `/chat/completions` endpoint), with per-model-family chat templates, stop tokens, and output cleanup.

**Architecture:** A family registry in `llm.py` maps model family → chat template renderer + stop tokens + postprocessing. `LlamaCppLLM` and the new `OpenAICompatLLM` share a `FamilyLLM` base that resolves family from the model filename (explicit override allowed). `agent.py` stops hard-coding Qwen3 thinking-phase logic — it derives stop tokens and calls `postprocess()` on the adapter (defensive `getattr`, since `test_benchmark.py` uses a duck-typed FakeLLM). SYSTEM_PROMPT's action list becomes single-sourced from `PolicyEngine.known_actions()` when the policy declares it, else the unchanged static list. `chat_server.py` picks the backend via env vars, then `VOICEAGENT_MODEL` name/filename, then the legacy default.

**Tech Stack:** Python 3.12, stdlib only additions (`urllib`, `http.server` for tests), `llama-cpp-python` (existing dep, local path unchanged), pytest.

**Spec:** User's refactor brief (session message) — hard constraints:
- Preserve `CANDIDATE_MODELS` entries verbatim (esp. `qwen2.5-0.5b-hinglish-q4`); do NOT touch `data/models`, `data/eval` contents.
- Do NOT change signatures of `list_available_models()`, `download_model()`, `load_llm()`; `run_benchmark.py` must work unchanged.
- No new third-party deps. Python 3.12. Keep agent.py deterministic paths (classifier, guardrail, policy) untouched.
- All 49 existing tests pass unchanged. Byte-identical Qwen output (same cleaned text) for local qwen models.

## Global Constraints

- `load_llm()` signature: `(model_path, n_ctx=2048, params="?", size_mb=0.0)` — unchanged.
- Existing Qwen family behavior must stay byte-identical: stop `[" thinking"]`, thinking-strip + `.strip()`, ChatML template.
- No commits (user did not request them; harness rule).
- Tests must not load real GGUF files.

---

### Task 1: Family registry + thinking cleanup + `FamilyLLM` in `llm.py`

**Files:**
- Modify: `src/voiceagent/llm.py`
- Test: `tests/test_llm_adapters.py` (new file, tests written first)

**Interfaces:**
- Consumes: existing `LLMHandle` (specs dict, `generate()`, raw `chat_template()`).
- Produces: `CHAT_TEMPLATES` dict, `infer_family(model_path, explicit=None) -> str`, `THINKING_RE`/`RESPONSE_PREFIX_RE`/`clean_thinking_text()`, `THINKING_FAMILIES`, `family_stop_tokens(family)`, `FamilyLLM` base (attrs `family`, `stop_tokens`; methods `chat_template`, `postprocess`).

- [ ] **Step 1: Write failing tests** — `tests/test_llm_adapters.py` (template byte-identity, family inference, cleanup literals, stop tables; helpers `_family_stub`).
- [ ] **Step 2: Run to confirm they fail** — `python -m pytest tests/test_llm_adapters.py -q` (ImportError: no CHAT_TEMPLATES).
- [ ] **Step 3: Implement** in `llm.py` (see final code below; moved regexes, `_render_*` fns, registry, `infer_family`, `clean_thinking_text`, `family_stop_tokens`, `FamilyLLM`).
- [ ] **Step 4: Run to confirm pass.**

### Task 2: `LlamaCppLLM` on `FamilyLLM` + `OpenAICompatLLM` + `build_llm_from_env`

- [ ] **Step 1: Write failing tests** — OpenAICompatLLM against a stdlib stub HTTP server (round-trip, message shaping incl. chat_template→generate, auth header, error→RuntimeError w/ body, unreachable), `build_llm_from_env` (monkeypatch), `LlamaCppLLM.postprocess`/`chat_template` via `_family_stub`.
- [ ] **Step 2: Run to confirm they fail.**
- [ ] **Step 3: Implement** — `LlamaCppLLM(FamilyLLM)` (adds `family=` kwarg, `family_stop_tokens`), `OpenAICompatLLM(FamilyLLM)` (urllib POST `/chat/completions`, graceful `RuntimeError`), `build_llm_from_env()`.
- [ ] **Step 4: Run to confirm pass.**

### Task 3: `agent.py` adapter-driven stops/postprocess + policy-sourced action list; `policy.py` accessor

- [ ] **Step 1: Write failing tests** — agent with bare custom `LLMHandle` (generate only), agent derives stop from adapter, agent calls adapter postprocess, policy-declared `actions:` drives prompt, static prompt pinned byte-identical.
- [ ] **Step 2: Run to confirm they fail.**
- [ ] **Step 3: Implement** — agent: `DEFAULT_ACTIONS` + `_SYSTEM_PROMPT_TMPL` + `SYSTEM_PROMPT` (byte-identical), `system_prompt_with_actions()`, `Agent.__init__` builds `self._system_prompt` from `policy.known_actions()` if non-empty, `handle()` uses `getattr(llm, "stop_tokens", None)` + `getattr(llm, "postprocess")`; move `THINKING_RE`/`RESPONSE_PREFIX_RE` out. policy: `PolicyEngine.known_actions()`.
- [ ] **Step 4: Run full suite** — all 49 old tests still pass.

### Task 4: `scripts/chat_server.py` backend/model selection

- [ ] **Step 1: Implement** — env backend via `build_llm_from_env()` (skip local entirely, print serving model); else `VOICEAGENT_MODEL` (name or exact filename) > legacy `qwen2.5-0.5b-q4` > first available; print serving model at startup.
- [ ] **Step 2: Verify** — `python -m py_compile scripts/chat_server.py`; manual env-var smoke via `VOICEAGENT_LLM_BASE_URL=... python -c "import scripts... "` check.

### Task 5: Verification + report

- [ ] `python -m pytest -q` (all pass, count)
- [ ] `python3 -m py_compile src/voiceagent/llm.py src/voiceagent/agent.py`
- [ ] `python scripts/run_benchmark.py 5` (per-model lines)
- [ ] Demo snippets: OpenAICompatLLM round-trip against stub; chat_template qwen vs llama3
- [ ] Report: changed files + one-line reasons, flags.

---

## Final code (Task 1-3)

### `src/voiceagent/llm.py` — key additions

```python
# --- chat template registry -------------------------------------------------
def _render_qwen(system, context, user_text):  # byte-identical to old LlamaCppLLM.chat_template
    user = f"{context}\n\nCustomer: {user_text}"
    return ("<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def _render_llama3(system, context, user_text):
    user = f"{context}\n\nCustomer: {user_text}"
    return ("<|start_header_id|>system<|end_header_id|>\n\n" + system + "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")

def _render_gemma(system, context, user_text):  # no system role; folded into user turn
    user = f"{system}\n\n{context}\n\nCustomer: {user_text}"
    return ("<start_of_turn>user\n" + user + "<end_of_turn>\n"
            "<start_of_turn>model\n")

def _render_generic(system, context, user_text):  # raw completion (legacy LLMHandle.chat_template)
    return (f"{system}\n\nContext:\n{context}\n\n"
            f"Customer: {user_text}\nAssistant:")

CHAT_TEMPLATES = {"qwen": _render_qwen, "llama3": _render_llama3,
                  "gemma": _render_gemma, "generic": _render_generic}

def infer_family(model_path: str, explicit: str | None = None) -> str:
    if explicit is not None:
        if explicit not in CHAT_TEMPLATES:
            raise ValueError(f"unknown family {explicit!r}; known: {sorted(CHAT_TEMPLATES)}")
        return explicit
    name = Path(model_path).name.lower()
    if "qwen" in name: return "qwen"
    if "llama-3" in name or "llama3" in name: return "llama3"
    if "gemma" in name: return "gemma"
    return "generic"

# (moved from agent.py, unchanged) THINKING_RE / RESPONSE_PREFIX_RE
THINKING_RE = re.compile(r"(?:<think>.*?</think>|^\s*thinking\s*\n.*?(?=\s*response|\Z))",
                         re.DOTALL | re.MULTILINE)
RESPONSE_PREFIX_RE = re.compile(r"^\s*response\s*\n?", re.MULTILINE)
THINKING_FAMILIES = {"qwen"}
FAMILY_STOP_TOKENS = {"qwen": [" thinking"]}

def clean_thinking_text(text: str) -> str:
    return RESPONSE_PREFIX_RE.sub("", THINKING_RE.sub("", text)).strip()

def family_stop_tokens(family: str) -> list[str] | None:
    return list(FAMILY_STOP_TOKENS[family]) if family in FAMILY_STOP_TOKENS else None

class FamilyLLM(LLMHandle):
    """LLMHandle with a chat-template family: registry chat_template,
    per-family stop tokens and thinking-phase cleanup. Subclassed by the
    local (llama.cpp) and remote (OpenAI-compatible) adapters."""
    def __init__(self, specs: dict, family: str):
        super().__init__(specs)
        self.family = family
        self.stop_tokens = family_stop_tokens(family)
    def chat_template(self, system, context, user_text) -> str:
        return CHAT_TEMPLATES[self.family](system, context, user_text)
    def postprocess(self, text: str) -> str:
        return clean_thinking_text(text) if self.family in THINKING_FAMILIES else text
```

`LlamaCppLLM(FamilyLLM)` — `__init__` gains trailing `family: str | None = None`, sets `self.family = infer_family(model_path, family)` and keeps everything else; `chat_template`/`postprocess` inherited.

### `OpenAICompatLLM(FamilyLLM)`

```python
class OpenAICompatLLM(FamilyLLM):
    """Any OpenAI-compatible /chat/completions endpoint (vLLM, Ollama, LM
    Studio) over stdlib urllib — no new deps."""
    def __init__(self, base_url, model, api_key=None, timeout=30.0):
        super().__init__({"model": model, "params": "remote", "quant": "-",
                          "model_path": base_url, "size_mb": 0.0},
                         family=infer_family(model))
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self._pending_messages: list[dict] | None = None
    def chat_template(self, system, context, user_text) -> str:
        user = f"{context}\n\nCustomer: {user_text}"
        self._pending_messages = [{"role": "system", "content": system},
                                  {"role": "user", "content": user}]
        return f"system: {system}\n\nuser: {user}"
    def generate(self, prompt, max_tokens=256, stop=None) -> str:
        messages = self._pending_messages or [{"role": "user", "content": prompt}]
        self._pending_messages = None
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens}
        if stop: payload["stop"] = stop
        req = urllib.request.Request(self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenAI-compatible endpoint returned HTTP {e.code}: "
                               f"{e.read().decode('utf-8', errors='replace')}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach OpenAI-compatible endpoint "
                               f"{self.base_url}: {e.reason}") from e
        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise RuntimeError(f"unexpected response from {self.base_url}: {body!r}") from e

def build_llm_from_env() -> OpenAICompatLLM | None:
    base = os.environ.get("VOICEAGENT_LLM_BASE_URL")
    model = os.environ.get("VOICEAGENT_LLM_MODEL")
    if not base or not model:
        return None
    return OpenAICompatLLM(base, model,
                           api_key=os.environ.get("VOICEAGENT_LLM_API_KEY"))
```

### `src/voiceagent/agent.py` — key changes

```python
DEFAULT_ACTIONS = ["order_status", "refund", "cancel_order", "address_change",
                   "payment_declined", "recharge", "billing", "return",
                   "replacement", "otp", "fraud", "account_closure",
                   "delivery_delay", "product_info", "invoice", "plan_change",
                   "roaming", "network_issue", "complaint", "high_value_refund"]
_SYSTEM_PROMPT_TMPL = ( ...same text, "one of: {actions}. " ... )
SYSTEM_PROMPT = _SYSTEM_PROMPT_TMPL.format(actions=", ".join(DEFAULT_ACTIONS))
def system_prompt_with_actions(actions: list[str]) -> str:
    return _SYSTEM_PROMPT_TMPL.format(actions=", ".join(actions))
```
`Agent.__init__`: after policy wrap — `self._system_prompt = SYSTEM_PROMPT; declared = self._policy.known_actions() if self._policy else []; if declared: self._system_prompt = system_prompt_with_actions(declared)`.
`handle()`: use `self._system_prompt`; `stop=getattr(self._llm, "stop_tokens", None)`; `post = getattr(self._llm, "postprocess", None); clean = post(text) if callable(post) else text`.

### `src/voiceagent/policy.py`

```python
def known_actions(self) -> list[str]:
    """Action vocabulary declared via an optional top-level `actions:` list
    in the policy file. Rule keys are NOT the vocabulary (partial coverage,
    differing names like order_cancellation vs cancel_order), so empty means
    'not declared' and callers keep their own list."""
    acts = self.policies.get("actions")
    if not isinstance(acts, list):
        return []
    return [a for a in acts if isinstance(a, str)]
```
