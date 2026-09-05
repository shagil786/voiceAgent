# tests/test_agent.py
from voiceagent.agent import build_agent, extract_action, AgentResult
from voiceagent.llm import LLMHandle

class FakeLLM(LLMHandle):
    def __init__(self, reply=("Your order ORD-77812 is out for delivery.\n"
                              "ACTION: order_status")):
        super().__init__({"model": "fake"})
        self.reply = reply
    def generate(self, prompt, max_tokens=256, stop=None):
        return self.reply

class FakeClassifier:
    def classify(self, text):
        return ("order_status", 1.0)

class FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "Order status can be checked with the order id.",
                "section": "faqs", "score": 0.9}]

def test_extract_action_parses_action_line():
    assert extract_action("foo\nACTION: refund\nbar") == "refund"
    assert extract_action("no action here") is None

def test_agent_returns_text_action_and_retrieved():
    agent = build_agent(FakeIndex(), FakeLLM())
    res = agent.handle("where is my order ORD-77812")
    assert isinstance(res, AgentResult)
    assert "out for delivery" in res.text
    assert res.action == "order_status"
    assert len(res.retrieved) == 1
    assert res.latency_s >= 0

# ---------------------------------------------------------------------------
# M5c Fix 1: the ACTION scaffolding line never reaches the customer. The
# action decision comes from the classifier / fallback extract_action(), so
# scrubbing must happen after extraction and after the echo guardrail.
# ---------------------------------------------------------------------------

def test_action_line_stripped_from_reply_end():
    # Live-observed shape: reply ends with an ACTION line.
    agent = build_agent(FakeIndex(), FakeLLM())
    res = agent.handle("where is my order ORD-77812")
    assert "ACTION" not in res.text
    assert res.text == "Your order ORD-77812 is out for delivery."
    assert res.action == "order_status"  # fallback extraction keeps working

def test_action_line_mid_text_stripped_and_blank_runs_collapsed():
    agent = build_agent(FakeIndex(), FakeLLM(
        reply="Answer one.\n\nACTION: refund\n\nAnswer two."))
    res = agent.handle("I need a refund")
    assert "ACTION" not in res.text
    assert res.text == "Answer one.\n\nAnswer two."
    assert res.action == "refund"

def test_action_first_reply_scrubbed():
    agent = build_agent(FakeIndex(), FakeLLM(
        reply="ACTION: refund\nYour refund for ORD-77812 is done."))
    res = agent.handle("I need a refund for ORD-77812")
    assert res.text == "Your refund for ORD-77812 is done."
    assert res.action == "refund"

def test_scrub_keeps_guardrail_confirm_sentence():
    # Classifier path: the echo guardrail runs first (prepends the confirm
    # for the order id the LLM dropped), THEN the ACTION line is scrubbed.
    from voiceagent.agent import Agent
    agent = Agent(FakeIndex(),
                  FakeLLM("Your request is being handled.\nACTION: order_status"),
                  classifier=FakeClassifier())
    res = agent.handle("where is my order ORD-1234")
    assert "ACTION" not in res.text
    assert "I understand" in res.text          # confirm still present
    assert "ORD-1234" in res.text              # guardrail echo still present
    assert "being handled" in res.text         # original reply kept

def test_scrub_when_reply_starts_with_action_line():
    from voiceagent.agent import Agent
    agent = Agent(FakeIndex(), FakeLLM(
        reply="ACTION: order_status\nYour order is on the way."),
        classifier=FakeClassifier())
    res = agent.handle("where is my order ORD-1234")
    assert "ACTION" not in res.text
    assert "I understand" in res.text
    assert "on the way" in res.text

# ---------------------------------------------------------------------------
# M5c Fix 2: informational intents reach the system prompt's action list.
# ---------------------------------------------------------------------------

def test_informational_actions_in_default_action_list():
    # Sprint A1: the vocabulary is demo tenant data now; the informational
    # actions stay in it so the demo path keeps serving them.
    from voiceagent.demo_data import DEMO_TENANT_ACTIONS
    assert "refund_info" in DEMO_TENANT_ACTIONS
    assert "delivery_eta" in DEMO_TENANT_ACTIONS


# ---------------------------------------------------------------------------
# M5a: per-turn reply-language directive for native-script queries. The
# directive is appended to THIS turn's prompt only — self._system_prompt and
# SYSTEM_PROMPT stay untouched, and en/hinglish prompts stay byte-identical.
# ---------------------------------------------------------------------------

class PromptCaptureLLM(LLMHandle):
    def __init__(self, reply="Your request has been noted."):
        super().__init__({"model": "fake"})
        self.prompts: list[str] = []
        self.reply = reply

    def generate(self, prompt, max_tokens=256, stop=None):
        self.prompts.append(prompt)
        return self.reply


DIRECTIVE = "Reply in the customer's language (code: {lang})."


def test_native_script_query_gets_language_directive():
    llm = PromptCaptureLLM()
    agent = build_agent(FakeIndex(), llm)
    agent.handle("मेरा recharge क्यों fail हुआ?")
    assert DIRECTIVE.format(lang="hi") in llm.prompts[0]


def test_explicit_language_param_adds_directive():
    llm = PromptCaptureLLM()
    agent = build_agent(FakeIndex(), llm)
    agent.handle("where is my order ORD-77812", language="ta")
    assert DIRECTIVE.format(lang="ta") in llm.prompts[0]


def test_english_prompt_byte_identical_with_language_param():
    # Auto-detect (language=None) and explicit language="en" must both
    # produce a prompt byte-identical to the pre-M5a prompt.
    llm = PromptCaptureLLM()
    agent = build_agent(FakeIndex(), llm)
    agent.handle("where is my order ORD-77812")               # auto -> en
    agent.handle("where is my order ORD-77812", language="en")
    agent.handle("where is my order ORD-77812", language=None)
    assert llm.prompts[0] == llm.prompts[1]
    assert llm.prompts[0] == llm.prompts[2]


def test_hinglish_prompt_byte_identical_no_directive():
    llm = PromptCaptureLLM()
    agent = build_agent(FakeIndex(), llm)
    agent.handle("mera order ORD-55671 abhi tak nahi aaya")
    assert llm.prompts[0].count("Customer:") == 1
    assert "Reply in the customer's language" not in llm.prompts[0]


def test_directive_does_not_mutate_stored_system_prompt():
    from voiceagent.agent import Agent
    llm = PromptCaptureLLM()
    agent = Agent(FakeIndex(), llm)
    before = agent._system_prompt
    agent.handle("मेरा recharge क्यों fail हुआ?")     # native -> directive
    agent.handle("where is my order")                # english -> no directive
    assert agent._system_prompt == before
    assert "Reply in the customer's language" not in llm.prompts[1]


def test_template_path_receives_directive_in_system_arg():
    class TemplateCaptureLLM(PromptCaptureLLM):
        def chat_template(self, system, context, user_text):
            self.prompts.append(f"SYS<<{system}>>{context}{user_text}")
            return self.reply

    llm = TemplateCaptureLLM()
    build_agent(FakeIndex(), llm).handle("मेरा ऑर्डर कहाँ है")
    assert DIRECTIVE.format(lang="hi") in llm.prompts[0]


# ---------------------------------------------------------------------------
# M5b-4: reply-language guardrail — a 0.5B model ignores the "reply in the
# customer's language" directive often enough that the reply's language is
# checked deterministically; mismatches get a canned reply in the customer's
# language with the echo guardrail's reference still applied.
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, "src")  # keep imports working when run standalone

from voiceagent.langid import detect_language  # noqa: E402

class EnglishOnlyLLM(LLMHandle):
    """Simulates the observed 0.5B failure: ignores the language directive,
    always answers in English."""
    def __init__(self, reply="Your order ORD-77812 is out for delivery."):
        super().__init__({"model": "fake"})
        self.reply = reply
    def generate(self, prompt, max_tokens=256, stop=None):
        return self.reply

def test_hi_customer_gets_hindi_reply_when_llm_answers_english():
    agent = build_agent(FakeIndex(), EnglishOnlyLLM(), classifier=FakeClassifier())
    res = agent.handle("मेरा ऑर्डर ORD-77812 कब आएगा")
    assert detect_language(res.text) == "hi"
    assert "ORD-77812" in res.text  # echo guardrail still applies

def test_te_customer_gets_telugu_reply():
    agent = build_agent(FakeIndex(), EnglishOnlyLLM(), classifier=FakeClassifier())
    res = agent.handle("నా ఆర్డర్ ORD-77812 ఎప్పుడు వస్తుంది")
    assert detect_language(res.text) == "te"
    assert "ORD-77812" in res.text

def test_hinglish_customer_gets_hinglish_reply():
    agent = build_agent(FakeIndex(), EnglishOnlyLLM(), classifier=FakeClassifier())
    res = agent.handle("mera order ORD-77812 kab aayega")
    assert "Aapke order" in res.text  # Roman hinglish template served
    assert "ORD-77812" in res.text

def test_reply_already_in_customer_language_untouched():
    hindi_reply = "आपका ऑर्डर ORD-77812 रास्ते में है।"
    agent = build_agent(FakeIndex(), EnglishOnlyLLM(reply=hindi_reply),
                        classifier=FakeClassifier())
    res = agent.handle("मेरा ऑर्डर ORD-77812 कब आएगा")
    assert res.text == hindi_reply  # no template substitution

def test_english_turns_never_touched():
    reply = "Your order ORD-77812 is out for delivery."
    agent = build_agent(FakeIndex(), EnglishOnlyLLM(reply=reply),
                        classifier=FakeClassifier())
    res = agent.handle("where is my order ORD-77812")
    assert res.text == reply  # en turns bypass the guardrail entirely


# ---------------------------------------------------------------------------
# Global languages (es/fr/de/pt) in the canned-reply layer. The old fallback
# (`lang_key = ... else "hi"`) served HINDI text to every es/fr/de/pt
# customer whose LLM reply failed the guardrail — langid could not detect
# their language, so every turn failed it.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from voiceagent.agent import (EMPATHY_PREFIXES, NOTED_REPLIES,  # noqa: E402
                              REPLY_TEMPLATES, _acceptable_reply_langs,
                              _canned_reply)

GLOBAL_LANGS = ("es", "fr", "de", "pt", "en")


def test_canned_reply_serves_each_global_language():
    for lang in GLOBAL_LANGS:
        out = _canned_reply("order_status", lang, ["ORD-77812"])
        assert "ORD-77812" in out
        assert out == REPLY_TEMPLATES["order_status"][lang].format(
            ref="ORD-77812")
        assert "आपके" not in out  # never the old Hindi fallback


def test_canned_reply_covers_all_registered_intents_per_language():
    for lang in GLOBAL_LANGS:
        for intent in ("order_status", "refund", "refund_info",
                       "delivery_eta", "default"):
            assert REPLY_TEMPLATES[intent][lang], (intent, lang)


def test_canned_reply_uncovered_language_gets_english_not_hindi():
    # A language we cannot serve gets neutral English — never an unrelated
    # language.
    out = _canned_reply("order_status", "ta", ["ORD-77812"])
    assert out == REPLY_TEMPLATES["order_status"]["en"].format(
        ref="ORD-77812")


def test_existing_hi_te_hinglish_templates_byte_identical():
    # Pre-existing customer-visible strings are frozen; only the NEW languages
    # may add entries.
    assert REPLY_TEMPLATES == {
        "order_status": {
            "hi": "आपके ऑर्डर {ref} की स्थिति जाँच ली गई है। ताज़ा स्थिति जल्द ही आपके ऐप और एसएमएस पर अपडेट होगी।",
            "te": "మీ ఆర్డర్ {ref} స్థితి తనిఖీ చేయబడింది. తాజా స్థితి త్వరలో మీ యాప్‌లో మరియు ఎస్ఎంఎస్ ద్వారా అందుతుంది.",
            "hinglish": "Aapke order {ref} ka status check kar liya gaya hai. Latest update jald hi app aur SMS par milega.",
            "en": REPLY_TEMPLATES["order_status"]["en"],
            "es": REPLY_TEMPLATES["order_status"]["es"],
            "fr": REPLY_TEMPLATES["order_status"]["fr"],
            "de": REPLY_TEMPLATES["order_status"]["de"],
            "pt": REPLY_TEMPLATES["order_status"]["pt"],
        },
        "refund": {
            "hi": "आपका रिफंड अनुरोध दर्ज हो गया है। प्रक्रिया पूरी होने पर स्थिति की जानकारी दी जाएगी।",
            "te": "మీ రీఫండ్ అభ్యర్థన నమోదైంది. ప్రక్రియ పూర్తయిన తర్వాత స్థితి తెలియజేయబడుతుంది.",
            "hinglish": "Aapka refund request note kar liya gaya hai. Process complete hone par status update mil jayega.",
            "en": REPLY_TEMPLATES["refund"]["en"],
            "es": REPLY_TEMPLATES["refund"]["es"],
            "fr": REPLY_TEMPLATES["refund"]["fr"],
            "de": REPLY_TEMPLATES["refund"]["de"],
            "pt": REPLY_TEMPLATES["refund"]["pt"],
        },
        "refund_info": {
            "hi": "रिफंड स्वीकृत होने के 5-7 कार्यदिवसों में आपके खाते में आ जाता है।",
            "te": "రీఫండ్ ఆమోదించబడిన 5-7 పనిదినాల్లో మీ ఖాతాలో జమ అవుతుంది.",
            "hinglish": "Refund approve hone ke 5-7 working days mein aapke account mein aa jata hai.",
            "en": REPLY_TEMPLATES["refund_info"]["en"],
            "es": REPLY_TEMPLATES["refund_info"]["es"],
            "fr": REPLY_TEMPLATES["refund_info"]["fr"],
            "de": REPLY_TEMPLATES["refund_info"]["de"],
            "pt": REPLY_TEMPLATES["refund_info"]["pt"],
        },
        "delivery_eta": {
            "hi": "आपका ऑर्डर 3-5 कार्यदिवसों में डिलीवर होने की उम्मीद है।",
            "te": "మీ ఆర్డర్ 3-5 పనిదినాల్లో డెలివరీ అవుతుందని భావిస్తున్నాము.",
            "hinglish": "Aapka order 3-5 working days mein deliver hone ki expectation hai.",
            "en": REPLY_TEMPLATES["delivery_eta"]["en"],
            "es": REPLY_TEMPLATES["delivery_eta"]["es"],
            "fr": REPLY_TEMPLATES["delivery_eta"]["fr"],
            "de": REPLY_TEMPLATES["delivery_eta"]["de"],
            "pt": REPLY_TEMPLATES["delivery_eta"]["pt"],
        },
        "default": {
            "hi": "आपका अनुरोध दर्ज कर लिया गया है। हमारी टीम जल्द ही आपकी सहायता करेगी।",
            "te": "మీ అభ్యర్థన నమోదు చేయబడింది. మా బృందం త్వరలో మీకు సహాయం చేస్తుంది.",
            "hinglish": "Aapka request note kar liya gaya hai. Hamari team jald hi aapki help karegi.",
            "en": REPLY_TEMPLATES["default"]["en"],
            "es": REPLY_TEMPLATES["default"]["es"],
            "fr": REPLY_TEMPLATES["default"]["fr"],
            "de": REPLY_TEMPLATES["default"]["de"],
            "pt": REPLY_TEMPLATES["default"]["pt"],
        },
    }


def test_existing_noted_replies_byte_identical():
    assert NOTED_REPLIES["hi"] == "आपका अनुरोध दर्ज कर लिया गया है।"
    assert NOTED_REPLIES["te"] == "మీ అభ్యర్థన నమోదు చేయబడింది."
    assert NOTED_REPLIES["hinglish"] == "Aapka request note kar liya gaya hai."


def test_noted_replies_cover_english_and_global_languages():
    # The empty-reply safety net serves the CALLER's language when covered;
    # English is the explicit neutral entry (never Hindi for an uncovered
    # language).
    assert NOTED_REPLIES["en"] == "Your request has been noted."
    for lang in ("es", "fr", "de", "pt"):
        assert NOTED_REPLIES[lang]


def test_existing_empathy_prefixes_byte_identical():
    assert EMPATHY_PREFIXES["en"] == "I'm really sorry about the trouble. "
    assert EMPATHY_PREFIXES["hinglish"] == "Mujhe khed hai ki aapko pareshani hui. "
    assert EMPATHY_PREFIXES["hi"] == "मुझे खेद है कि आपको परेशानी हुई। "
    assert EMPATHY_PREFIXES["te"] == "ఇబ్బంది కోసం క్షమించండి. "
    assert EMPATHY_PREFIXES["es"] == "Lamento mucho las molestias. "
    assert EMPATHY_PREFIXES["fr"] == "Je suis vraiment désolé pour ce désagrément. "
    assert EMPATHY_PREFIXES["de"] == "Es tut mir wirklich leid für die Umstände. "


def test_guardrail_strict_for_global_languages():
    for lang in ("es", "fr", "de", "pt"):
        assert _acceptable_reply_langs(lang) == frozenset({lang})


@pytest.mark.parametrize("lang,text", [
    ("es", "¿Dónde está mi pedido ORD-77812?"),
    ("fr", "Où est ma commande ORD-77812 ?"),
    ("de", "Wo ist meine Bestellung ORD-77812?"),
    ("pt", "Onde está meu pedido ORD-77812?"),
])
def test_global_customer_never_receives_hindi_or_english(lang, text):
    # The customer-facing bug: an es/fr/de/pt caller got the Hindi canned
    # fallback on every turn (detection + fallback both India-locked).
    agent = build_agent(FakeIndex(), EnglishOnlyLLM(),
                        classifier=FakeClassifier())
    res = agent.handle(text, language=lang)
    assert res.text == REPLY_TEMPLATES["order_status"][lang].format(
        ref="ORD-77812")
    assert "ORD-77812" in res.text


@pytest.mark.parametrize("lang", ["es", "fr", "de", "pt", "hi", "te",
                                  "hinglish"])
def test_empty_reply_safety_net_serves_caller_language(lang):
    agent = build_agent(FakeIndex(), FakeLLM(reply="  "))
    res = agent.handle("hello", language=lang)
    assert res.text == NOTED_REPLIES[lang]


def test_empty_reply_safety_net_uncovered_language_gets_english():
    agent = build_agent(FakeIndex(), FakeLLM(reply="  "))
    res = agent.handle("hello", language="ta")
    assert res.text == "Your request has been noted."


def test_empathy_prefix_for_pt():
    # Intensity-only frustration (SHOUTING + ???) is language-independent.
    agent = build_agent(FakeIndex(), EnglishOnlyLLM(),
                        classifier=FakeClassifier())
    res = agent.handle("ONDE ESTÁ MEU PEDIDO ORD-77812???", language="pt")
    assert res.text.startswith(EMPATHY_PREFIXES["pt"])
    assert "ORD-77812" in res.text


# ---------------------------------------------------------------------------
# Sprint A1: the action vocabulary is TENANT DATA, not core code. The
# e-commerce list moves to the demo tenant data module (voiceagent.demo_data);
# a tenant bundle's declared vocabulary flows into the Agent prompt via the
# runtime assembly; the no-tenant path keeps serving the demo vocabulary.
# ---------------------------------------------------------------------------

def test_core_no_longer_ships_the_demo_action_list():
    import voiceagent.agent as agent_mod
    from voiceagent.demo_data import DEMO_TENANT_ACTIONS
    # The demo list exists, clearly labeled as demo tenant data...
    assert "order_status" in DEMO_TENANT_ACTIONS and "refund" in DEMO_TENANT_ACTIONS
    # ...and agent.py no longer carries it as core code.
    assert not hasattr(agent_mod, "DEFAULT_ACTIONS")


def test_no_tenant_agent_prompt_keeps_demo_vocabulary():
    # The built-in demo path is unchanged: the fallback vocabulary is the
    # demo tenant data, byte-identical prompts for existing tests.
    from voiceagent.demo_data import DEMO_TENANT_ACTIONS
    llm = PromptCaptureLLM()
    build_agent(FakeIndex(), llm).handle("hello")
    assert ", ".join(DEMO_TENANT_ACTIONS) in llm.prompts[0]


def test_declared_actions_drive_the_agent_prompt():
    # The assembly seam (runtime/deployment resolves the bundle vocabulary)
    # passes the declared actions into the Agent; demo actions disappear.
    agent = build_agent(FakeIndex(), PromptCaptureLLM(),
                        actions=["check_balance", "block_card"])
    prompt = agent._system_prompt
    assert "one of: check_balance, block_card" in prompt
    assert "cancel_order" not in prompt  # a demo-only action is gone


def test_policy_declared_actions_still_win_over_passed_actions():
    agent = build_agent(FakeIndex(), PromptCaptureLLM(),
                        actions=["check_balance"],
                        policy={"actions": ["kyc_reset"],
                                "kyc_reset": {"allow": True}})
    assert "one of: kyc_reset" in agent._system_prompt


# ---------------------------------------------------------------------------
# Sprint A2: the high_value_refund THRESHOLD is policy data (policies.yaml),
# read through the PolicyEngine (tenant currency wired), never an inline
# literal in the decision. Undeclared -> the platform default (5000).
# ---------------------------------------------------------------------------

class RefundClassifier:
    def classify(self, text):
        return ("refund", 1.0)


def test_high_value_refund_threshold_comes_from_policy():
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=RefundClassifier(),
                        policy={"high_value_refund_threshold": 200,
                                "refund": {"allow": True}})
    assert agent.handle("refund please", amount=200).action == \
        "high_value_refund"          # the threshold itself is high-value
    assert agent.handle("small refund", amount=199).action == "refund"


def test_high_value_refund_threshold_default_preserved():
    # No policy declares a threshold -> the historical 5000 value, but the
    # value comes from the policy config object, not an inline literal.
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=RefundClassifier())
    assert agent.handle("refund please", amount=4999.99).action == "refund"
    assert agent.handle("refund please", amount=5000).action == \
        "high_value_refund"


def test_high_value_refund_threshold_currency_is_tenant_data():
    # The threshold is evaluated through the PolicyEngine, which carries the
    # tenant's currency — the policy reason strings use it.
    from voiceagent.tenant import TenantConfig
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=RefundClassifier(),
                        policy={"high_value_refund_threshold": 200,
                                "refund": {"allow": True}},
                        tenant=TenantConfig(currency="₹"))
    assert agent._policy.currency == "₹"
    assert agent._policy.high_value_refund_threshold() == 200
    res = agent.handle("refund please", amount=250)
    assert res.action == "high_value_refund"


def test_policy_engine_threshold_accessor_validates():
    from voiceagent.policy import (DEFAULT_HIGH_VALUE_REFUND_THRESHOLD,
                                   PolicyEngine)
    assert PolicyEngine({}).high_value_refund_threshold() == \
        DEFAULT_HIGH_VALUE_REFUND_THRESHOLD == 5000
    assert PolicyEngine(
        {"high_value_refund_threshold": 200}).high_value_refund_threshold() \
        == 200
    # Bad declarations fall back to the platform default, never crash.
    assert PolicyEngine(
        {"high_value_refund_threshold": "lots"}).high_value_refund_threshold() \
        == 5000
    assert PolicyEngine(
        {"high_value_refund_threshold": -1}).high_value_refund_threshold() \
        == 5000


# ---------------------------------------------------------------------------
# Sprint A3: the echo guardrail's required facts are TOOL-CONTRACT data
# (ToolSpec.facts), derived from the deployment's declared specs — not a
# hardcoded keyword dict in agent.py.
# ---------------------------------------------------------------------------

def test_echo_guard_facts_come_from_tool_spec():
    from voiceagent.policy import PolicyEngine
    from voiceagent.tools import (GovernedToolRunner, ToolGateway, ToolSpec)
    specs = {"check_balance": ToolSpec(params=(), facts=("balance",))}
    runner = GovernedToolRunner(ToolGateway(specs=specs), PolicyEngine({}))
    llm = PromptCaptureLLM(reply="All done.")
    agent = build_agent(FakeIndex(), llm, classifier=RefundClassifier(),
                        tool_runner=runner)

    class BalanceClassifier:
        def classify(self, text):
            return ("check_balance", 1.0)

    agent._classifier = BalanceClassifier()
    res = agent.handle("my balance is wrong", amount=None)
    assert "balance" in res.text  # spec fact forced into the reply


def test_spec_facts_change_reflected_by_guard():
    from voiceagent.policy import PolicyEngine
    from voiceagent.tools import (GovernedToolRunner, ToolGateway, ToolSpec)

    class BalanceClassifier:
        def classify(self, text):
            return ("check_balance", 1.0)

    def agent_with(spec):
        runner = GovernedToolRunner(
            ToolGateway(specs={"check_balance": spec}), PolicyEngine({}))
        agent = build_agent(FakeIndex(), PromptCaptureLLM(reply="All done."),
                            classifier=BalanceClassifier(),
                            tool_runner=runner)
        return agent

    with_facts = agent_with(
        ToolSpec(params=(), facts=("ledger",))).handle("fix my ledger")
    assert "ledger" in with_facts.text
    # Same spec without the fact -> the guard forces nothing.
    bare = agent_with(ToolSpec(params=())).handle("fix my ledger")
    assert bare.text == "All done."


def test_demo_echo_guard_facts_unchanged():
    # The no-spec (demo) path preserves the historical keyword behavior.
    from voiceagent.agent import extract_required_references
    refs = extract_required_references("my otp never came and recharge fail")
    assert "otp" in refs and "fail" in refs
    assert extract_required_references("hello") == []


def test_keyword_facts_dict_gone_from_core():
    import voiceagent.agent as agent_mod
    assert not hasattr(agent_mod, "KEYWORD_FACTS")


# ---------------------------------------------------------------------------
# Review fixes: (MAJOR 2) the echo guard's scan is FIRST-MATCH-PER-GROUP —
# exact-equality pins against the historical KEYWORD_FACTS behavior;
# (MAJOR 1) a wired bare code-default ToolGateway (scripts/chat.py wiring)
# must still enforce the demo keyword facts, while a declared tenant bundle
# suppresses them.
# ---------------------------------------------------------------------------

def test_echo_guard_first_match_per_group_pinned():
    # Exact equality with the pre-refactor KEYWORD_FACTS scan: at most ONE
    # keyword per fact group reaches the required refs.
    from voiceagent.agent import extract_required_references
    assert extract_required_references("my recharge failed") == ["fail"]
    assert extract_required_references(
        "when will my order delivery happen") == ["order"]
    assert extract_required_references("block my card, fraud") == ["block"]
    assert extract_required_references(
        "my otp never came and recharge fail") == ["otp", "fail"]


def test_chat_style_wired_gateway_enforces_demo_facts():
    # scripts/chat.py wiring: classifier + ToolGateway(erp=...) with bare
    # DEFAULT_TOOL_SPECS (facts = order/refund only). Demo mode (no tenant
    # bundle) must MERGE the demo contracts back in — the historical
    # otp/bill/block/declined/fail/recharge guarantees survive.
    from voiceagent.policy import PolicyEngine
    from voiceagent.tools import GovernedToolRunner, MockERP, ToolGateway

    runner = GovernedToolRunner(ToolGateway(erp=MockERP()),
                                PolicyEngine({}))
    cases = [("my otp never came", "otp"), ("check my bill", "bill"),
             ("block my card", "block"),
             ("my payment was declined", "declined"),
             ("my recharge failed", "fail"),
             ("recharge did not go through", "recharge"),
             ("where is my refund", "refund")]
    for text, keyword in cases:
        class _Cls:
            def classify(self, t):
                return ("whatever", 1.0)

        agent = build_agent(FakeIndex(), PromptCaptureLLM(reply="Noted."),
                            classifier=_Cls(), tool_runner=runner)
        res = agent.handle(text)
        assert keyword in res.text, (text, keyword, res.text)


def test_declared_tenant_bundle_suppresses_demo_facts(tmp_path):
    # With a REAL tenant bundle declared, only the bundle's declared specs
    # apply — the demo otp/bill/block contracts must NOT leak in.
    import json as _json
    from voiceagent.policy import PolicyEngine
    from voiceagent.tenant import Tenant
    from voiceagent.tools import GovernedToolRunner, ToolGateway

    root = tmp_path / "acme-live"
    root.mkdir()
    (root / "tenant.json").write_text(_json.dumps({"name": "acme-live"}))
    bundle = Tenant.load(root)
    assert bundle.exists

    class _Cls:
        def classify(self, t):
            return ("whatever", 1.0)

    runner = GovernedToolRunner(ToolGateway(), PolicyEngine({}))
    agent = build_agent(FakeIndex(), PromptCaptureLLM(reply="Noted."),
                        classifier=_Cls(), tool_runner=runner, tenant=bundle)
    res = agent.handle("my otp never came")
    assert res.text == "Noted."  # no demo fact forced into the reply
