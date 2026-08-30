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