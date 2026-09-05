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
