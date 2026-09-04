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
    from voiceagent.agent import DEFAULT_ACTIONS
    assert "refund_info" in DEFAULT_ACTIONS
    assert "delivery_eta" in DEFAULT_ACTIONS


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
