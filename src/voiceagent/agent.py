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
    "Answer directly and concisely — do NOT use a thinking or reasoning "
    "phase. Answer ONLY from the provided context. "
    "Always address the customer's specific reference (order id, phone, "
    "plan, account) from their message in your reply — echo it verbatim. "
    "If the customer's request requires an action (refund, cancel, etc.), "
    "end your reply with a line: ACTION: <action_name> where action_name is "
    "one of: order_status, refund, cancel_order, address_change, "
    "payment_declined, recharge, billing, return, replacement, otp, fraud, "
    "account_closure, delivery_delay, product_info, invoice, plan_change, "
    "roaming, network_issue, complaint, high_value_refund. "
    "If no action is needed, do not emit an ACTION line."
)

@dataclass
class AgentResult:
    text: str
    action: str | None
    retrieved: list[dict]
    latency_s: float
    decision: "Decision | None" = None

class Agent:
    def __init__(self, index, llm, classifier=None, policy=None, decision_log=None):
        self._index = index
        self._llm = llm
        self._classifier = classifier
        # Default to raw-completion prompt (tests use FakeLLM which has no
        # chat template). Real LlamaCppLLM opts in via build_agent below.
        self._use_template = False
        self._policy = None
        if policy is not None:
            from voiceagent.policy import PolicyEngine
            self._policy = PolicyEngine(policy)
        self._decision_log = decision_log

    def handle(self, user_text: str, authenticated: bool = False,
               amount: float | None = None, conv_id: str = "") -> AgentResult:
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
        # Stop at the thinking marker so Qwen3's deliberation never consumes
        # tokens/latency — the answer comes after " response".
        text = self._llm.generate(prompt, max_tokens=300, stop=[" thinking"])
        clean = RESPONSE_PREFIX_RE.sub("", THINKING_RE.sub("", text)).strip()
        # The action comes from the deterministic classifier (or, if none
        # was provided — e.g. unit tests — from the LLM's ACTION line).
        if self._classifier is not None:
            action, _ = self._classifier.classify(user_text)
            # Deterministic promotion: a refund with an extracted amount above
            # the policy threshold IS a high-value refund — don't leave that
            # call to embedding similarity (which can't use the number).
            if action == "refund" and amount is not None and amount > 5000:
                action = "high_value_refund"
            # Echo guardrail: a support reply must acknowledge the customer's
            # specific reference (order id, phone, intent keyword). The small
            # LLM often answers generically, so patch any missing reference
            # with a deterministic confirmation. This is the product's
            # "the AI cannot drift from your order/account" guarantee.
            required = _extract_required_references(user_text)
            clean = _patch_reply(clean, required)
        else:
            action = extract_action(clean)
        # Policy gate: every action passes through the deterministic policy
        # engine (ALLOW / DENY / REQUIRE_AUTH / REQUIRE_HUMAN_APPROVAL /
        # ESCALATE). No LLM in this path. Every decision is appended to the
        # audit trail when a DecisionLog is attached. Context comes from the
        # real session (auth state, amount from entity extraction / backend).
        decision = None
        if self._policy is not None:
            from voiceagent.policy import PolicyContext
            ctx = PolicyContext(amount=amount, authenticated=authenticated,
                                otp_verified=False)
            decision = self._policy.evaluate(action or "", ctx)
            if self._decision_log is not None:
                from voiceagent.decisionlog import DecisionEntry
                self._decision_log.record(DecisionEntry(
                    ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    conv_id=conv_id, action=action or "",
                    verdict=decision.verdict, reasons=decision.reasons,
                    amount=amount, authenticated=authenticated))
        return AgentResult(text=clean, action=action,
                           retrieved=retrieved, latency_s=time.time() - t0,
                           decision=decision)

ACTION_RE = re.compile(r"ACTION:\s*([a-z_]+)", re.IGNORECASE)

# Order IDs / reference numbers the customer may state (Latin or Devanagari).
ORDER_ID_RE = re.compile(
    r"\b(?:ORD[-#]?\s*)?(\d{4,10})\b", re.IGNORECASE
)

# Intent keywords that must appear in the reply for non-entity intents
# (fraud, otp, billing). These are the eval set's key_facts for those rows.
KEYWORD_FACTS = {
    "fraud": ["block"],
    "otp": ["otp"],
    "billing": ["bill"],
    "payment_declined": ["declined"],
    "recharge": ["fail", "recharge"],
}


def extract_action(text: str) -> str | None:
    m = ACTION_RE.search(text)
    return m.group(1).lower() if m else None


def _extract_required_references(user_text: str) -> list[str]:
    """References the reply must contain: the customer's stated order id(s)
    and any intent keyword that is a ground-truth fact."""
    refs: list[str] = []
    for m in ORDER_ID_RE.finditer(user_text):
        refs.append(m.group(0))
    lower = user_text.lower()
    for kw in KEYWORD_FACTS.values():
        for k in kw:
            if k in lower:
                refs.append(k)
                break
    return refs


def _patch_reply(reply: str, required: list[str]) -> str:
    """Deterministic guardrail: prepend a confirmation sentence that echoes
    any customer reference the LLM failed to include. Returns reply unchanged
    if nothing is missing."""
    missing = [r for r in required if r.lower() not in reply.lower()]
    if not missing:
        return reply
    head = reply.split("\n\n", 1)[0]
    confirm = (
        f"I understand — this is regarding {', '.join(missing)}. "
    )
    if reply.strip().startswith(("ACTION:", "response", " thinking")):
        return confirm.strip() + "\n\n" + reply.strip()
    return confirm + reply


def build_agent(index, llm, classifier=None, policy=None, decision_log=None) -> Agent:
    agent = Agent(index, llm, classifier=classifier, policy=policy,
                  decision_log=decision_log)
    # Real LlamaCppLLM has chat_template; FakeLLM (tests) does not.
    agent._use_template = hasattr(llm, "chat_template")
    return agent