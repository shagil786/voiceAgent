# tests/test_guardrail_repair.py — Task B Part 1: guardrails guide, not
# replace.
"""The reply-language guardrail used to REPLACE the brain's reply with a
canned template on every violation. With a frontier brain configured, the
guardrail now guides: ONE governed re-render whose repair prompt carries the
original violating reply, the original user turn, the allowed language(s) and
the required references; a still-violating re-render (or any repair failure)
falls back to today's canned path. No frontier configured (BASE tier) ->
byte-identical behavior: canned immediately, no repair call."""
from voiceagent.agent import build_agent
from voiceagent.llm import LLMHandle
from voiceagent.langid import detect_language


class FakeClassifier:
    def classify(self, text):
        return ("order_status", 1.0)


class FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "Order status can be checked with the order id.",
                 "section": "faqs", "score": 0.9}]


VIOLATING = "Your order ORD-77812 is out for delivery."      # English
COMPLIANT_HI = "आपका ऑर्डर ORD-77812 रास्ते में है।"            # Hindi
HI_TURN = "मेरा ऑर्डर ORD-77812 कब आएगा"


class ScriptedFrontier(LLMHandle):
    """A frontier-configured brain: `frontier = True` is the same adapter
    identity llm.build_llm_from_env routes as the frontier (the remote
    OpenAI-compatible adapter class carries it; local GGUF/stubs do not)."""
    frontier = True

    def __init__(self, replies):
        super().__init__({"model": "fake-frontier"})
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt, max_tokens=256, stop=None):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


class ExplodingRepairFrontier(ScriptedFrontier):
    """First generation works; the repair re-render raises."""

    def generate(self, prompt, max_tokens=256, stop=None):
        self.prompts.append(prompt)
        if len(self.prompts) >= 2:
            raise RuntimeError("frontier exploded during repair")
        return self.replies.pop(0)


# --- frontier configured: repair once, use the repaired reply -----------------

def test_frontier_language_violation_repaired_once():
    llm = ScriptedFrontier([VIOLATING, COMPLIANT_HI])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(HI_TURN)
    # the repaired reply IS the final reply, untouched by templates
    assert res.text == COMPLIANT_HI
    assert detect_language(res.text) == "hi"
    # exactly one repair call: original + one re-render
    assert len(llm.prompts) == 2
    assert res.repair_attempts == 1


def test_repair_prompt_carries_original_reply_user_turn_and_constraints():
    llm = ScriptedFrontier([VIOLATING, COMPLIANT_HI])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    agent.handle(HI_TURN)
    repair_prompt = llm.prompts[1]
    assert VIOLATING in repair_prompt               # original frontier reply
    assert HI_TURN in repair_prompt                 # original user turn
    assert "hi" in repair_prompt                    # allowed language(s)
    assert "ORD-77812" in repair_prompt             # required reference
    # persona / never-say constraints travel through the compiled system
    # prompt (the per-turn language directive is intentionally NOT re-added)
    assert repair_prompt.startswith(agent._system_prompt)


def test_compliant_reply_never_triggers_repair():
    llm = ScriptedFrontier([COMPLIANT_HI])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(HI_TURN)
    assert res.text == COMPLIANT_HI
    assert len(llm.prompts) == 1                    # no repair round
    assert res.repair_attempts == 0


# --- frontier configured but repair still violates: canned fallback -----------

def test_repair_still_violating_falls_back_to_canned():
    llm = ScriptedFrontier([VIOLATING, "Another English reply for ORD-77812."])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(HI_TURN)
    assert detect_language(res.text) == "hi"        # canned path serves hi
    assert "ORD-77812" in res.text                  # echo guardrail re-applied
    assert len(llm.prompts) == 2                    # exactly one repair, no more
    assert res.repair_attempts == 1


# --- no frontier configured: BASE tier unchanged (canned immediately) ---------

class BaseTierLLM(LLMHandle):
    """A BASE-tier handle (local GGUF stand-in): no `frontier` marker."""

    def __init__(self, reply=VIOLATING):
        super().__init__({"model": "fake-base"})
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, max_tokens=256, stop=None):
        self.calls += 1
        return self.reply


def test_no_frontier_canned_immediately_unchanged():
    llm = BaseTierLLM()
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(HI_TURN)
    assert detect_language(res.text) == "hi"
    assert "ORD-77812" in res.text
    assert llm.calls == 1                           # canned immediately
    assert res.repair_attempts == 0


# --- repair raising: fail-open at the surface, canned fallback ----------------

def test_repair_raise_falls_back_to_canned_no_exception():
    llm = ExplodingRepairFrontier([VIOLATING])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(HI_TURN)                     # must not raise
    assert detect_language(res.text) == "hi"
    assert "ORD-77812" in res.text
    assert len(llm.prompts) == 2
    assert res.repair_attempts == 1


# --- Task E: the ECHO guard shares the same ONE governed repair ---------------
"""The echo guardrail (required facts missing) used to be deterministic-only
(_patch_reply inserts the keywords). With a frontier configured it now uses
the SAME governed re-render as the language guard: one repair call per turn
TOTAL across BOTH guards — the repair prompt carries the allowed language(s)
AND the required references, so two simultaneous violations never cost two
frontier rounds. Compliant repair -> used as-is (_patch_reply NOT applied);
still-missing facts (or any repair failure / no frontier) -> the existing
deterministic patch, unchanged."""

GENERIC = "Let me check that for you right away."          # en, no reference
COMPLIANT_REF = "I found your order ORD-77812 — it is out for delivery."
EN_TURN = "where is my order ORD-77812"


def test_echo_violation_compliant_repair_used_patch_not_applied():
    llm = ScriptedFrontier([GENERIC, COMPLIANT_REF])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(EN_TURN)
    # the repaired reply IS the final reply — no deterministic patch prefix
    assert res.text == COMPLIANT_REF
    assert "I understand — this is regarding" not in res.text
    assert len(llm.prompts) == 2                    # original + one re-render
    assert res.repair_attempts == 1


def test_echo_repair_prompt_carries_reply_turn_and_missing_refs():
    llm = ScriptedFrontier([GENERIC, COMPLIANT_REF])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    agent.handle(EN_TURN)
    repair_prompt = llm.prompts[1]
    assert GENERIC in repair_prompt                 # original frontier reply
    assert EN_TURN in repair_prompt                 # original user turn
    assert "ORD-77812" in repair_prompt             # required reference
    assert "MISSING" in repair_prompt               # called out for verbatim fix
    assert repair_prompt.startswith(agent._system_prompt)  # persona constraints


def test_echo_repair_still_missing_falls_back_to_patch():
    llm = ScriptedFrontier([GENERIC, "Still nothing about your order."])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(EN_TURN)
    # the reply is the ORIGINAL, deterministically patched (as today) with
    # every missing reference ("order" is the declared tool-contract fact
    # for a stated order id, and _patch_reply echoes all of them)
    assert res.text.startswith(
        "I understand — this is regarding ORD-77812, order.")
    assert GENERIC in res.text
    assert len(llm.prompts) == 2                    # exactly one repair, no more
    assert res.repair_attempts == 1


def test_echo_no_frontier_patched_immediately_unchanged():
    llm = BaseTierLLM(reply=GENERIC)
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(EN_TURN)
    assert res.text.startswith(
        "I understand — this is regarding ORD-77812, order.")
    assert llm.calls == 1                           # no repair round at all
    assert res.repair_attempts == 0


def test_both_guards_violating_single_repair_fixes_both():
    # hi turn whose reply is (a) English AND (b) missing the order reference:
    # ONE repair call must fix both constraints in one prompt.
    llm = ScriptedFrontier([GENERIC, COMPLIANT_HI])
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(HI_TURN)                     # HI_TURN states ORD-77812
    assert res.text == COMPLIANT_HI                 # hi AND carries the ref
    assert detect_language(res.text) == "hi"
    assert len(llm.prompts) == 2                    # exactly ONE repair call
    assert res.repair_attempts == 1


def test_both_guards_violating_repair_lang_only_still_patched():
    # The re-render fixes the language but drops the reference: BOTH
    # constraints are re-checked, so the repair is rejected and the
    # deterministic path (canned hi reply + echo patch) applies.
    llm = ScriptedFrontier([GENERIC, COMPLIANT_HI])
    llm.replies[1] = "आपकी आवेदन दर्ज कर ली गई है।"   # hi, but no ORD-77812
    agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
    res = agent.handle(HI_TURN)
    assert detect_language(res.text) == "hi"
    assert "ORD-77812" in res.text                  # echo patch re-applied
    assert len(llm.prompts) == 2


# --- English turns are never touched (both tiers) -----------------------------

def test_english_turns_never_repaired_or_replaced():
    for llm in (BaseTierLLM(), ScriptedFrontier([VIOLATING])):
        agent = build_agent(FakeIndex(), llm, classifier=FakeClassifier())
        res = agent.handle("where is my order ORD-77812")
        assert res.text == VIOLATING
        assert res.repair_attempts == 0
        if hasattr(llm, "prompts"):
            assert len(llm.prompts) == 1
        else:
            assert llm.calls == 1
