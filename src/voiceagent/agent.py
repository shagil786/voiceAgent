# src/voiceagent/agent.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# Qwen3 emits a "thinking" phase before the answer. llama.cpp renders it
# either as " thinking\n...\n response" or as "<think>...</think>"
# depending on version. Words inside the thinking block (e.g. "cancel_order"
# considered then rejected) would corrupt action extraction, so strip it.
THINKING_RE = re.compile(
    r"(?:<think>.*?</think>|^\s*thinking\s*\n.*?(?=\s*response|\Z))",
    re.DOTALL | re.MULTILINE,
)
RESPONSE_PREFIX_RE = re.compile(r"^\s*response\s*\n?", re.MULTILINE)

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
        # Default to raw-completion prompt (tests use FakeLLM which has no
        # chat template). Real LlamaCppLLM opts in via build_agent below.
        self._use_template = False

    def handle(self, user_text: str) -> AgentResult:
        t0 = time.time()
        retrieved = self._index.search(user_text, k=3)
        context = "\n".join(f"[{r['section']}] {r['text']}" for r in retrieved)
        if self._use_template:
            # ChatML for Qwen3/Qwen2.5 instruct models — vastly better
            # format-following than a raw completion prompt on small models.
            prompt = self._llm.chat_template(SYSTEM_PROMPT, context, user_text)
        else:
            prompt = (
                f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\n"
                f"Customer: {user_text}\nAssistant:"
            )
        # Qwen3 wraps its deliberation in "thinking ... response". Strip it
        # so words hallucinated in reasoning don't corrupt action extraction.
        text = self._llm.generate(prompt, max_tokens=300)
        clean = RESPONSE_PREFIX_RE.sub("", THINKING_RE.sub("", text)).strip()
        return AgentResult(text=clean, action=extract_action(clean),
                           retrieved=retrieved, latency_s=time.time() - t0)

ACTION_RE = re.compile(r"ACTION:\s*([a-z_]+)", re.IGNORECASE)

def extract_action(text: str) -> str | None:
    m = ACTION_RE.search(text)
    return m.group(1).lower() if m else None

def build_agent(index, llm) -> Agent:
    agent = Agent(index, llm)
    # Real LlamaCppLLM has chat_template; FakeLLM (tests) does not.
    agent._use_template = hasattr(llm, "chat_template")
    return agent