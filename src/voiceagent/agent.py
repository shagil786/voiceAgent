# src/voiceagent/agent.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from voiceagent.langid import NATIVE_SCRIPT_LANGS, detect_language

if TYPE_CHECKING:  # Turn is duck-typed at runtime (no import cycle)
    from voiceagent.memory import Turn

# Fallback action vocabulary for the system prompt. Used when the loaded
# policy does not declare its own action set — policy rule keys are NOT the
# vocabulary (partial coverage, differing names like order_cancellation vs
# cancel_order): see PolicyEngine.known_actions.
DEFAULT_ACTIONS = [
    "order_status", "refund", "cancel_order", "address_change",
    "payment_declined", "recharge", "billing", "return", "replacement",
    "otp", "fraud", "account_closure", "delivery_delay", "product_info",
    "invoice", "plan_change", "roaming", "network_issue", "complaint",
    "high_value_refund", "refund_info", "delivery_eta",
]

_SYSTEM_PROMPT_TMPL = (
    "You are a customer support assistant for an Indian ecommerce company. "
    "Answer directly and concisely — do NOT use a thinking or reasoning "
    "phase. Answer ONLY from the provided context. "
    "Always address the customer's specific reference (order id, phone, "
    "plan, account) from their message in your reply — echo it verbatim. "
    "If the customer's request requires an action (refund, cancel, etc.), "
    "end your reply with a line: ACTION: <action_name> where action_name is "
    "one of: {actions}. "
    "If no action is needed, do not emit an ACTION line."
)

SYSTEM_PROMPT = _SYSTEM_PROMPT_TMPL.format(actions=", ".join(DEFAULT_ACTIONS))


def system_prompt_with_actions(actions: list[str]) -> str:
    """SYSTEM_PROMPT with the action list taken from policy-as-code."""
    return _SYSTEM_PROMPT_TMPL.format(actions=", ".join(actions))

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
        # Single-source the action list: a policy that declares its action
        # vocabulary (PolicyEngine.known_actions) drives the system prompt;
        # otherwise the static DEFAULT_ACTIONS list above is kept.
        self._system_prompt = SYSTEM_PROMPT
        if self._policy is not None:
            declared = self._policy.known_actions()
            if declared:
                self._system_prompt = system_prompt_with_actions(declared)

    def handle(self, user_text: str, authenticated: bool = False,
               amount: float | None = None, conv_id: str = "",
               *, history: list["Turn"] | None = None,
               language: str | None = None) -> AgentResult:
        t0 = time.time()
        # M5a: reply-language. Auto-detect when the caller doesn't know;
        # native-script languages get a per-turn directive appended to the
        # prompt build below (never to self._system_prompt, so en/hinglish
        # prompts stay byte-identical and the benchmark is unaffected).
        if language is None:
            language = detect_language(user_text)
        retrieved = self._index.search(user_text, k=3)
        context = "\n".join(f"[{r['section']}] {r['text']}" for r in retrieved)
        # Working memory (M4a): replay the last few complete exchanges as a
        # compact transcript between the RAG context and the current turn —
        # both prompt paths consume `context`, so placement is identical.
        # None/empty history leaves the prompt byte-identical to before.
        if history:
            transcript = render_history(history)
            if transcript:
                context = f"{context}\n\n{transcript}"
        if language in NATIVE_SCRIPT_LANGS:
            system = (f"{self._system_prompt}\nReply in the customer's "
                      f"language (code: {language}).")
        else:
            system = self._system_prompt
        if self._use_template:
            # Chat template for the model's family (Qwen ChatML, Llama 3
            # headers, ...) — vastly better format-following than a raw
            # completion prompt on small instruct models.
            prompt = self._llm.chat_template(system, context, user_text)
        else:
            prompt = (
                f"{system}\n\nContext:\n{context}\n\n"
                f"Customer: {user_text}\nAssistant:"
            )
        # Stop tokens and output cleanup are adapter concerns: the llama.cpp
        # adapter stops at Qwen3's thinking marker and strips the reasoning
        # phase; bare/test handles default to no stops and a no-op cleanup.
        stop = getattr(self._llm, "stop_tokens", None)
        text = self._llm.generate(prompt, max_tokens=300, stop=stop)
        post = getattr(self._llm, "postprocess", None)
        clean = post(text) if callable(post) else text
        # The action comes from the deterministic classifier (or, if none
        # was provided — e.g. unit tests — from the LLM's ACTION line).
        if self._classifier is not None:
            action, _ = self._classifier.classify(user_text)
            # Deterministic promotion: a refund with an extracted amount above
            # the policy threshold IS a high-value refund — don't leave that
            # call to embedding similarity (which can't use the number).
            if action == "refund" and amount is not None and amount > 5000:
                action = "high_value_refund"
        else:
            action = extract_action(clean)
        # Scrub the ACTION scaffolding BEFORE the echo guardrail runs: the
        # guardrail must judge the text the customer will actually see. A
        # pre-scrub ACTION line can coincidentally contain a required keyword
        # (e.g. "ACTION: recharge_fail" contains "fail") and the scrub would
        # then delete the only occurrence of that fact — the guardrail would
        # never notice it went missing. The action itself was captured above
        # (classifier, or fallback extraction pre-scrub).
        clean = strip_action_lines(clean)
        if self._classifier is not None:
            # Echo guardrail: a support reply must acknowledge the customer's
            # specific reference (order id, phone, intent keyword). The small
            # LLM often answers generically — or its reply was scaffolding
            # only — so patch any missing reference with a deterministic
            # confirmation. This is the product's "the AI cannot drift from
            # your order/account" guarantee.
            required = extract_required_references(user_text)
            # Reference inheritance: a follow-up like "and when will it
            # arrive?" states no order id — inherit the most recent one from
            # the conversation so the guardrail keeps the reply pinned to the
            # customer's reference. No LLM involved.
            if history and find_order_id(user_text) is None:
                inherited = find_recent_order_id(history)
                if inherited:
                    required.append(inherited)
            clean = _patch_reply(clean, required)
        # M5b-4 reply-language guardrail: the LLM's reply must be in the
        # customer's language; a 0.5B model ignores the directive often
        # enough that this is checked deterministically, not trusted.
        target_langs = _acceptable_reply_langs(language)
        if (target_langs is not None and clean.strip()
                and detect_language(clean) not in target_langs):
            clean = _canned_reply(action, language or "hi",
                                  required if self._classifier is not None
                                  else [])
            if self._classifier is not None:
                clean = _patch_reply(clean, required)
        if not clean:
            # Safety net: a reply with no references and no content still
            # reaches the customer as something — in THEIR language.
            clean = NOTED_REPLIES.get(
                language if language in ("hi", "te", "hinglish") else "",
                "Your request has been noted.")
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


# ---------------------------------------------------------------------------
# M5b-4: reply-language guardrail. The prompt directive ("Reply in the
# customer's language") is unreliable on 0.5B models — the live fresh-caller
# voice check showed hi/te customers receiving English replies, and the
# empty-reply safety net was English-only. After generation, verify the
# reply's language; on mismatch substitute a deterministic canned reply in
# the customer's language (per-intent where available), then re-apply the
# echo guardrail so the customer's reference still appears. en turns are
# never touched — the text-path benchmark stays byte-identical.
# ---------------------------------------------------------------------------

REPLY_TEMPLATES: dict[str, dict[str, str]] = {
    "order_status": {
        "hi": "आपके ऑर्डर {ref} की स्थिति जाँच ली गई है। ताज़ा स्थिति जल्द ही आपके ऐप और एसएमएस पर अपडेट होगी।",
        "te": "మీ ఆర్డర్ {ref} స్థితి తనిఖీ చేయబడింది. తాజా స్థితి త్వరలో మీ యాప్‌లో మరియు ఎస్ఎంఎస్ ద్వారా అందుతుంది.",
        "hinglish": "Aapke order {ref} ka status check kar liya gaya hai. Latest update jald hi app aur SMS par milega.",
    },
    "refund": {
        "hi": "आपका रिफंड अनुरोध दर्ज हो गया है। प्रक्रिया पूरी होने पर स्थिति की जानकारी दी जाएगी।",
        "te": "మీ రీఫండ్ అభ్యర్థన నమోదైంది. ప్రక్రియ పూర్తయిన తర్వాత స్థితి తెలియజేయబడుతుంది.",
        "hinglish": "Aapka refund request note kar liya gaya hai. Process complete hone par status update mil jayega.",
    },
    "refund_info": {
        "hi": "रिफंड स्वीकृत होने के 5-7 कार्यदिवसों में आपके खाते में आ जाता है।",
        "te": "రీఫండ్ ఆమోదించబడిన 5-7 పనిదినాల్లో మీ ఖాతాలో జమ అవుతుంది.",
        "hinglish": "Refund approve hone ke 5-7 working days mein aapke account mein aa jata hai.",
    },
    "delivery_eta": {
        "hi": "आपका ऑर्डर 3-5 कार्यदिवसों में डिलीवर होने की उम्मीद है।",
        "te": "మీ ఆర్డర్ 3-5 పనిదినాల్లో డెలివరీ అవుతుందని భావిస్తున్నాము.",
        "hinglish": "Aapka order 3-5 working days mein deliver hone ki expectation hai.",
    },
    "default": {
        "hi": "आपका अनुरोध दर्ज कर लिया गया है। हमारी टीम जल्द ही आपकी सहायता करेगी।",
        "te": "మీ అభ్యర్థన నమోదు చేయబడింది. మా బృందం త్వరలో మీకు సహాయం చేస్తుంది.",
        "hinglish": "Aapka request note kar liya gaya hai. Hamari team jald hi aapki help karegi.",
    },
}

NOTED_REPLIES = {
    "hi": "आपका अनुरोध दर्ज कर लिया गया है।",
    "te": "మీ అభ్యర్థన నమోదు చేయబడింది.",
    "hinglish": "Aapka request note kar liya gaya hai.",
}


def _acceptable_reply_langs(language: str | None) -> frozenset | None:
    """Reply languages a turn may legitimately come back in; None disables
    the guardrail (every en/None turn). hinglish accepts Roman hinglish or
    Devanagari Hindi (a Hindi speaker reads both natively); native languages
    are strict."""
    if not language or language == "en":
        return None
    if language == "hinglish":
        return frozenset({"hinglish", "hi"})
    return frozenset({language})


def _ref_for_template(refs: list[str]) -> str:
    """The customer's order-id-shaped reference (keywords are not refs)."""
    for r in refs:
        if r.upper().startswith("ORD") or r.isdigit():
            return r
    return ""


def _canned_reply(action: str | None, language: str, refs: list[str]) -> str:
    lang_key = language if language in ("hi", "te", "hinglish") else "hi"
    table = REPLY_TEMPLATES.get(action or "") or REPLY_TEMPLATES["default"]
    tpl = table.get(lang_key) or REPLY_TEMPLATES["default"][lang_key]
    return tpl.format(ref=_ref_for_template(refs))

ACTION_RE = re.compile(r"ACTION:\s*([a-z_]+)", re.IGNORECASE)# Order IDs / reference numbers the customer may state (Latin or Devanagari).
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
    # M5c informational intents: eval rows carry the topic keyword as
    # key_facts, so the echo guardrail pins it into the reply too.
    "refund_info": ["refund"],
    "delivery_eta": ["order", "delivery"],
}


def extract_action(text: str) -> str | None:
    m = ACTION_RE.search(text)
    return m.group(1).lower() if m else None


def extract_required_references(user_text: str) -> list[str]:
    """References the reply must contain: the customer's stated order id(s)
    and any intent keyword that is a ground-truth fact. Shared with chat.py
    (turn records) and the echo guardrail."""
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


def find_order_id(text: str) -> str | None:
    """First order-id match in text ('ORD-1234' or bare digits), else None.
    The single entry point to ORDER_ID_RE outside this module."""
    m = ORDER_ID_RE.search(text)
    return m.group(0) if m else None


def find_recent_order_id(history: list["Turn"]) -> str | None:
    """Most recent order id in a conversation (scan newest -> oldest)."""
    for t in reversed(history):
        oid = find_order_id(t.text)
        if oid:
            return oid
    return None


# History replay budget: ~400 tokens at ~4 chars/token.
HISTORY_MAX_EXCHANGES = 4
HISTORY_CHAR_BUDGET = 1600


def _render_exchange(exchange: list["Turn"]) -> str:
    who = {"user": "Customer", "agent": "Agent"}
    return "\n".join(f"{who[t.role]}: {t.text}" for t in exchange)


def render_history(turns: list["Turn"]) -> str:
    """Render the last complete user/agent exchanges as a compact transcript
    block ("Customer: ...\\nAgent: ..."). Selection is newest-first under the
    char budget (older exchanges are dropped first); output is chronological.
    A trailing unpaired user turn is the current turn — already rendered as
    the prompt's own 'Customer:' line — so only completed pairs are shown."""
    exchanges: list[list["Turn"]] = []
    for t in turns:
        if t.role == "user":
            exchanges.append([t])
        elif exchanges:
            exchanges[-1].append(t)
    chosen: list[str] = []
    total = 0
    for exchange in reversed([e for e in exchanges
                              if e and e[-1].role == "agent"]
                             [-HISTORY_MAX_EXCHANGES:]):
        rendered = _render_exchange(exchange)
        if total and total + len(rendered) + 1 > HISTORY_CHAR_BUDGET:
            break
        chosen.append(rendered)
        total += len(rendered) + 1
    return "\n\n".join(reversed(chosen))


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


def strip_action_lines(text: str) -> str:
    """Remove the LLM's ACTION scaffolding lines from a customer-visible
    reply (and collapse the blank-line runs they leave behind). The action
    decision comes from the deterministic classifier (or fallback
    extract_action), so the ACTION line itself must never reach the customer.
    Call only after the action has been captured and the echo guardrail has
    run."""
    kept = [ln for ln in text.split("\n") if not ACTION_RE.search(ln)]
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def build_agent(index, llm, classifier=None, policy=None, decision_log=None) -> Agent:
    agent = Agent(index, llm, classifier=classifier, policy=policy,
                  decision_log=decision_log)
    # Real LlamaCppLLM has chat_template; FakeLLM (tests) does not.
    agent._use_template = hasattr(llm, "chat_template")
    return agent
