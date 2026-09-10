# src/voiceagent/agent.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from voiceagent.langid import NATIVE_SCRIPT_LANGS, detect_language
from voiceagent.memory import (CAPTURE_CONFIDENCE_THRESHOLD,
                               classifier_exemplars)
from voiceagent.sentiment import (candidate_phrases_from,
                                  detect_frustration)
from voiceagent.security import detect_injection, sanitize_for_prompt
from voiceagent.tenant import DEFAULT_CURRENCY, Tenant

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # Turn is duck-typed at runtime (no import cycle)
    from voiceagent.memory import Turn

# The demo deployment's action vocabulary is DEMO TENANT DATA
# (voiceagent.demo_data), not core: orchestration core ships no e-commerce
# list. Precedence for the system prompt's vocabulary: a policy that declares
# its action set (PolicyEngine.known_actions) wins, then the action vocabulary
# the assembly seam resolved from the tenant bundle (build_agent(actions=...)),
# then the demo tenant data — so the no-tenant demo path keeps working
# byte-identically. Policy rule keys are NOT the vocabulary (partial coverage,
# differing names like order_cancellation vs cancel_order): see
# PolicyEngine.known_actions.
from voiceagent.demo_data import (DEMO_TENANT_ACTIONS,  # noqa: F401
                                  EMPATHY_PREFIXES, NOTED_REPLIES,  # noqa: F401
                                  REPLY_TEMPLATES)  # noqa: F401

# Task D1 (architecture debt): the deterministic reply-guard pipeline lives in
# voiceagent.reply_guards (moved VERBATIM from this module — no behavior
# change) and the demo reply TEXT TABLES live in voiceagent.demo_data. The
# imports below re-export both surfaces, so `from voiceagent.agent import X`
# keeps working unchanged for every existing caller.
from voiceagent.reply_guards import (ACTION_RE,  # noqa: F401
                                     _APOLOGY_MARKERS,  # noqa: F401
                                     _SERVED_REPLY_LANGS,  # noqa: F401
                                     _acceptable_reply_langs,  # noqa: F401
                                     _already_apologetic,  # noqa: F401
                                     _canned_reply,  # noqa: F401
                                     _patch_reply,  # noqa: F401
                                     _ref_for_template,  # noqa: F401
                                     _repair_allowed_langs,  # noqa: F401
                                     echo_spec_registry,  # noqa: F401
                                     extract_action,  # noqa: F401
                                     extract_required_references,  # noqa: F401
                                     find_order_id,  # noqa: F401
                                     find_recent_order_id,  # noqa: F401
                                     repair_reply,  # noqa: F401
                                     strip_action_lines)  # noqa: F401

# Neutral by default: a persona is tenant data (M6a). The legacy default
# ("...for an Indian ecommerce company") claimed a false identity for every
# tenant bundle without a persona and every no-tenant legacy path.
_DEFAULT_PERSONA = "customer support assistant"
_INSTRUCTION_TAIL = (
    "Answer directly and concisely — do NOT use a thinking or reasoning "
    "phase. Answer ONLY from the provided context. "
    "Always address the customer's specific reference (order id, phone, "
    "plan, account) from their message in your reply — echo it verbatim. "
    "If the customer's request requires an action (refund, cancel, etc.), "
    "end your reply with a line: ACTION: <action_name> where action_name is "
    "one of: {actions}. "
    "If no action is needed, do not emit an ACTION line."
)

SYSTEM_PROMPT = ("You are a " + _DEFAULT_PERSONA + ". " + _INSTRUCTION_TAIL
                 .format(actions=", ".join(DEMO_TENANT_ACTIONS)))


def system_prompt_with_actions(actions: list[str], persona=None) -> str:
    """System prompt from the policy's action vocabulary and the tenant's
    structured persona (M6b): the prompt is COMPILED from reviewable fields
    — role, tone, promise permissions, forbidden claims — so compliance can
    assert things like 'the agent never promises guaranteed refunds' in CI.
    persona=None keeps the historical default, byte-identical."""
    if persona is None or isinstance(persona, str):
        head = "You are a " + (persona or _DEFAULT_PERSONA) + "."
    else:
        p = persona
        lines = [f"You are a {p.role}."]
        if p.tone:
            lines.append(f"Tone: {p.tone}.")
        if p.may_promise:
            lines.append("You may promise exactly: "
                         + "; ".join(p.may_promise)
                         + ". Never promise anything else.")
        if p.never_say:
            lines.append("Never say or imply: " + "; ".join(p.never_say) + ".")
        head = " ".join(lines)
    return head + " " + _INSTRUCTION_TAIL.format(actions=", ".join(actions))

@dataclass
class AgentResult:
    text: str
    action: str | None
    retrieved: list[dict]
    latency_s: float
    decision: "Decision | None" = None
    tool_outcome: "GovernedOutcome | None" = None
    # Task B / Task E (guardrails guide, not replace): 1 when the ONE
    # governed frontier re-render was attempted this turn, 0 otherwise. The
    # single repair covers BOTH guardrails (language + echo) in one call —
    # the count is the per-turn repair budget spent, so it is 1 even when
    # two constraints violated (BASE tier, or no guardrail violation: 0).
    # Lightweight counter on the turn result — the Agent path has no
    # per-turn DecisionLog record of its own to carry it.
    repair_attempts: int = 0


class Agent:
    def __init__(self, index, llm, classifier=None, policy=None,
                 decision_log=None, tenant=None, sentiment_store=None,
                 tool_runner=None, erp=None,
                 actions: list[str] | None = None,
                 intent_memory=None,
                 capture_threshold: float = CAPTURE_CONFIDENCE_THRESHOLD):
        self._index = index
        self._llm = llm
        self._classifier = classifier
        # ADR-002: the learned intent memory (IntentMemoryStore) — episodic
        # capture of low-confidence / unknown turns during live calls.
        # None = the whole memory layer is inert (opt-in via
        # VOICEAGENT_MEMORY_DB; zero behavior change by default).
        self._intent_memory = intent_memory
        self._capture_threshold = capture_threshold
        # M2 (ADR-002): live reseed state. The DECLARED exemplars are snapshotted
        # once (they are the floor); when the memory store's version changes the
        # classifier is reseeded in place with declared + conflict-guarded
        # prototypes — checked at most once per turn, fail-open.
        self._declared_exemplars = None
        if classifier is not None:
            declared = getattr(classifier, "_exemplars", None)
            if isinstance(declared, dict):
                self._declared_exemplars = {
                    k: list(v) for k, v in declared.items()}
        self._memory_version: int | None = None
        if intent_memory is not None and self._declared_exemplars is not None:
            try:
                self._memory_version = intent_memory.version()
            except Exception:
                self._memory_version = None
        self._tenant = tenant
        # M6b: the learnable frustration lexicon (None = static lexicon).
        self._sentiment = sentiment_store
        # Default to raw-completion prompt (tests use FakeLLM which has no
        # chat template). Real LlamaCppLLM opts in via build_agent below.
        self._use_template = False
        self._policy = None
        if policy is not None:
            from voiceagent.policy import PolicyEngine
            # Currency is tenant data: the policy reason strings use the
            # tenant's symbol; no tenant -> the platform default.
            currency = getattr(tenant, "currency", None) or DEFAULT_CURRENCY
            self._policy = PolicyEngine(policy, currency=currency)
        # The high-value-refund threshold (Sprint A2) is policy data: the
        # promotion decision reads the VALUE through a PolicyEngine — the
        # wired one when present, else the platform-default policy config —
        # never an inline literal. (self._policy stays None without a wired
        # policy: the governance gate keeps its historical semantics.)
        if self._policy is not None:
            self._threshold_policy = self._policy
        else:
            from voiceagent.policy import PolicyEngine as _DefaultPolicy
            self._threshold_policy = _DefaultPolicy()
        self._decision_log = decision_log
        self._tool_runner = tool_runner
        self._erp = erp
        # Single-source the action list: a policy that declares its action
        # vocabulary (PolicyEngine.known_actions) drives the system prompt;
        # then the vocabulary the assembly seam resolved from the tenant
        # bundle; otherwise the demo tenant data keeps the no-tenant path
        # byte-identical. The persona comes from the tenant config (M6a).
        persona = getattr(tenant, "persona", None) or _DEFAULT_PERSONA
        declared = self._policy.known_actions() if self._policy else []
        self._system_prompt = system_prompt_with_actions(
            declared or actions or DEMO_TENANT_ACTIONS, persona)
        # Echo-guardrail facts (Sprint A3) are TOOL-CONTRACT data. ONE shared
        # resolution (echo_spec_registry, below) for both here and the
        # extract_required_references default: when a governed tool surface is
        # wired, its specs are the base — and in DEMO mode (no real tenant
        # bundle declared) the demo tenant contracts are merged in, so the
        # historical keyword guarantees survive a bare code-default
        # ToolGateway. With a real tenant bundle declared, ONLY the bundle's
        # declared specs apply (no demo leakage).
        gateway = getattr(self._tool_runner, "gateway", None)
        specs = getattr(gateway, "specs", None)
        bundle_declared = isinstance(tenant, Tenant) and tenant.exists
        self._echo_specs = specs
        self._echo_demo = not bundle_declared
        # Task B (guardrails guide, not replace): frontier detection reuses
        # the SAME condition that routes the frontier today — the adapter
        # identity llm.build_llm_from_env() wires as the remote brain
        # (OpenAICompatLLM.frontier = True). Local GGUF handles and unmarked
        # test stubs are the BASE tier: deterministic templates stay first
        # choice there, byte-identically.
        self._frontier = bool(getattr(llm, "frontier", False))

    @property
    def record_id_shapes(self) -> "list[dict] | None":
        """This deployment's declared ID shapes (tenant bundle entities.yaml),
        or None — downstream resolves None to the default bundle's
        declaration. chat.py reads this so turn records use the same shapes
        as the echo guardrail."""
        shapes_attr = getattr(self._tenant, "record_id_shapes", None)
        resolved = shapes_attr() if callable(shapes_attr) else shapes_attr
        assert resolved is None or isinstance(resolved, list)
        return resolved

    def handle(self, user_text: str, authenticated: bool = False,
               amount: float | None = None, conv_id: str = "",
               *, history: list["Turn"] | None = None,
               language: str | None = None,
               customer_id: str | None = None) -> AgentResult:
        t0 = time.time()
        # M2 (ADR-002): live reseed — if the memory store consolidated new
        # prototypes since the last turn, swap them into the classifier in
        # place (declared floor + conflict-guarded prototypes). At most one
        # version() check per turn; any failure leaves the current exemplars.
        self._maybe_reseed_classifier()
        # M5a: reply-language. Auto-detect when the caller doesn't know;
        # native-script languages get a per-turn directive appended to the
        # prompt build below (never to self._system_prompt, so en/hinglish
        # prompts stay byte-identical and the benchmark is unaffected).
        if language is None:
            language = detect_language(user_text)
        # M6a: deterministic frustration detection — the 'Sentiment Agent'
        # without an LLM pass. The level becomes a policy signal (whether
        # frustration escalates is data: escalate_when in policies.yaml) and
        # shapes the reply with an empathy line in the customer's language.
        # M6b: the lexicon LEARNS — known phrases come from the store, and
        # novel intensity-only expressions are captured as candidates for
        # review, so detection coverage grows with every conversation.
        learned = self._sentiment.learned_phrases(language) \
            if self._sentiment is not None else None
        fr = detect_frustration(user_text, language, extra_phrases=learned)
        if self._sentiment is not None and fr.level == "none" and fr.intensity:
            self._sentiment.capture_candidates(
                candidate_phrases_from(user_text), language)
        # M6b: prompt-injection guard — detect, strip forged meta-turns from
        # what reaches the LLM prompt, and surface a policy signal. The
        # ACTION is decided by the classifier either way; injection cannot
        # hijack it, only the reply text — which is sanitized and audited.
        inj = detect_injection(user_text)
        prompt_text = sanitize_for_prompt(user_text)
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
            # completion prompt on small instruct models. The SANITIZED
            # customer text enters the prompt (M6b): forged meta-turns
            # cannot reach the model as instructions.
            prompt = self._llm.chat_template(system, context, prompt_text)
        else:
            prompt = (
                f"{system}\n\nContext:\n{context}\n\n"
                f"Customer: {prompt_text}\nAssistant:"
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
        classify_confidence: float | None = None
        classify_label: str | None = None
        if self._classifier is not None:
            classify_label, classify_confidence = \
                self._classifier.classify(user_text)
            action = classify_label
            # Deterministic promotion: a refund with an extracted amount at or
            # above the policy threshold IS a high-value refund — don't leave
            # that call to embedding similarity (which can't use the number).
            # The threshold is POLICY DATA (policies.yaml
            # `high_value_refund_threshold`, evaluated through the PolicyEngine
            # with the tenant's currency wired), never an inline literal.
            if (action == "refund" and amount is not None
                    and amount >= self._threshold_policy
                    .high_value_refund_threshold()):
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
        # The ECHO guardrail and the reply-language guardrail below share ONE
        # repair budget per turn (latency discipline): at most ONE governed
        # re-render TOTAL across both guards. Design decision: when both
        # violate, the single repair prompt fixes both constraints in one
        # call — reply_guards.repair_reply always carries the allowed
        # language(s) AND the required references, so a second frontier round
        # for the second guard would be pure waste on a live phone call.
        # Echo guardrail: a support reply must acknowledge the customer's
        # specific reference (order id, phone, intent keyword). The small
        # LLM often answers generically — or its reply was scaffolding
        # only — so a missing reference is repaired through the frontier
        # (same .frontier marker rule as the language repair) or, on the
        # BASE tier / after a failed repair, patched deterministically.
        # This is the product's "the AI cannot drift from your order/
        # account" guarantee. The keyword facts are TOOL CONTRACT data (the
        # registry resolved in __init__ via echo_spec_registry: wired gateway
        # specs + demo contracts in demo mode) — never a hardcoded dict.
        required: list[str] = []
        if self._classifier is not None:
            required = extract_required_references(
                user_text, specs=self._echo_specs, demo=self._echo_demo,
                id_shapes=self.record_id_shapes)
            # Reference inheritance: a follow-up like "and when will it
            # arrive?" states no record id — inherit the most recent one from
            # the conversation so the guardrail keeps the reply pinned to the
            # customer's reference. No LLM involved.
            if history and find_order_id(
                    user_text, self.record_id_shapes) is None:
                inherited = find_recent_order_id(history,
                                                 self.record_id_shapes)
                if inherited:
                    required.append(inherited)
        missing = [r for r in required if r.lower() not in clean.lower()]
        # M5b-4 reply-language guardrail: the LLM's reply must be in the
        # customer's language; a 0.5B model ignores the directive often
        # enough that this is checked deterministically, not trusted.
        # Task B / Task E: with a frontier configured the guards GUIDE, not
        # replace — ONE governed re-render asks the brain to restate its own
        # reply within the constraints (allowed languages, required facts,
        # persona never-say / may-promise via the compiled system prompt).
        # Only a still-violating re-render (or any repair failure — fail-open
        # at the surface) falls back to the deterministic path, and the
        # no-frontier BASE tier keeps it immediately, unchanged.
        target_langs = _acceptable_reply_langs(language)
        lang_violation = bool(target_langs is not None and clean.strip()
                              and detect_language(clean) not in target_langs)
        repair_attempts = 0
        repaired_ok = False
        if self._frontier and (missing or lang_violation):
            repair_attempts = 1  # ONE extra frontier round, max, BOTH guards
            try:
                repaired = self._repair_reply(
                    clean, prompt_text, language,
                    _repair_allowed_langs(language), required)
            except Exception:
                repaired = None
            if (repaired is not None and repaired.strip()
                    and detect_language(repaired)
                    in _repair_allowed_langs(language)
                    and not [r for r in required
                             if r.lower() not in repaired.lower()]):
                # The re-render satisfies BOTH constraints: use it as-is —
                # _patch_reply must not bolt a keyword sentence onto a
                # compliant reply.
                clean = repaired
                repaired_ok = True
        if not repaired_ok:
            if lang_violation:
                clean = _canned_reply(action, language or "en",
                                      required if self._classifier is not None
                                      else [])
            if self._classifier is not None:
                clean = _patch_reply(clean, required)
        if not clean:
            # Safety net: a reply with no references and no content still
            # reaches the customer as something — in THEIR language when the
            # language is covered, neutral English otherwise (never an
            # unrelated language). The English default is demo tenant data
            # (NOTED_REPLIES["en"]) — core ships no demo text inline.
            clean = NOTED_REPLIES.get(language or "", NOTED_REPLIES["en"])
        # Policy gate: every action passes through the deterministic policy
        # engine (ALLOW / DENY / REQUIRE_AUTH / REQUIRE_HUMAN_APPROVAL /
        # ESCALATE). No LLM in this path. Every decision is appended to the
        # audit trail when a DecisionLog is attached. Context comes from the
        # real session (auth state, amount from entity extraction / backend).
        decision = None
        if self._policy is not None:
            from voiceagent.policy import PolicyContext
            ctx = PolicyContext(amount=amount, authenticated=authenticated,
                                otp_verified=False,
                                signals={"frustrated": fr.frustrated,
                                         "frustration_level": fr.level,
                                         "injection_suspected": inj.detected})
            decision = self._policy.evaluate(action or "", ctx)
            if self._decision_log is not None:
                from voiceagent.decisionlog import DecisionEntry
                self._decision_log.record(DecisionEntry(
                    ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    conv_id=conv_id, action=action or "",
                    verdict=decision.verdict, reasons=decision.reasons,
                    amount=amount, authenticated=authenticated))
        # M6a: acknowledge detected frustration in the customer's language
        # before the substantive reply — but never double-apologize if the
        # reply already carries an apology.
        if fr.level == "high" and not _already_apologetic(clean):
            clean = EMPATHY_PREFIXES.get(language, "") + clean
        # ADR-002 episodic capture: low-confidence or unknown-intent turns
        # feed the learned intent memory (the classifier produces the only
        # (label, confidence) pair in the live path, so this is THE capture
        # site). The episode records the classifier's OWN label/confidence
        # (pre-promotion: a refund promoted to high_value_refund by amount is
        # still a `refund` understanding fact); the outcome is the turn's
        # resulting action, or 'unmatched' when there was no usable label.
        # FAIL-OPEN: any memory error (corrupt db, missing table, embedder
        # failure) is logged and swallowed — the turn the customer is on must
        # never break because learning did.
        if self._intent_memory is not None and classify_confidence is not None:
            try:
                if (classify_confidence < self._capture_threshold
                        or not action):
                    self._intent_memory.capture(
                        getattr(self._tenant, "name", None) or "default",
                        user_text, classify_label or "", classify_confidence,
                        outcome=action or "unmatched")
            except Exception:
                logger.warning("intent memory: capture failed (fail-open)",
                               exc_info=True)
        return AgentResult(text=clean, action=action,
                           retrieved=retrieved, latency_s=time.time() - t0,
                           decision=decision,
                           repair_attempts=repair_attempts)

    def _maybe_reseed_classifier(self) -> None:
        """M2 (ADR-002): reseed the live classifier in place when the memory
        store's prototype version changed (cheap counter check — the
        retrieval snapshot from build time goes stale otherwise). Fail-open:
        any error keeps the current exemplars, the turn proceeds unchanged."""
        if (self._intent_memory is None or self._classifier is None
                or self._declared_exemplars is None):
            return
        try:
            version = self._intent_memory.version()
        except Exception:
            return
        if version == self._memory_version:
            return
        self._memory_version = version
        try:
            merged = classifier_exemplars(
                self._declared_exemplars, self._intent_memory,
                getattr(self._tenant, "name", None) or "default")
            self._classifier.reseed(merged)
            logger.debug("intent memory: classifier reseeded (version %s)",
                         version)
        except Exception:
            logger.warning("intent memory: reseed failed (fail-open)",
                           exc_info=True)

    def _repair_reply(self, violating_reply: str, user_text: str,
                      language: str, allowed_langs: frozenset,
                      required_refs: list[str]) -> str:
        """Task B: ONE governed re-render of a guardrail-violating frontier
        reply (signature kept). Task D1 moved the implementation VERBATIM to
        voiceagent.reply_guards.repair_reply — the guard pipeline module."""
        return repair_reply(self._llm, self._system_prompt,
                            self._use_template, violating_reply, user_text,
                            language, allowed_langs, required_refs)


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


def build_agent(index, llm, classifier=None, policy=None, decision_log=None,
                tenant=None, sentiment_store=None, tool_runner=None,
                erp=None, actions: list[str] | None = None,
                intent_memory=None) -> Agent:
    agent = Agent(index, llm, classifier=classifier, policy=policy,
                  decision_log=decision_log, tenant=tenant,
                  sentiment_store=sentiment_store, tool_runner=tool_runner,
                  erp=erp, actions=actions, intent_memory=intent_memory)
    # Real LlamaCppLLM has chat_template; FakeLLM (tests) does not.
    agent._use_template = hasattr(llm, "chat_template")
    return agent
