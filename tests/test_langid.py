# tests/test_langid.py
"""M5a: stdlib-only language detection for text turns. Native Unicode-script
detection wins over Latin heuristics; Latin-script text is hinglish when it
carries >=2 distinct tokens from a small Hinglish lexicon, else en.

Global target set (README: en/es/fr/de/pt): Latin-script languages are
detected from GLOBAL_LEXICONS with the same >=2-distinct-token contract as
the Hinglish lexicon — checked after native scripts, before hinglish."""
import csv
from pathlib import Path

import pytest

from voiceagent.langid import (GLOBAL_LEXICONS, NATIVE_SCRIPT_LANGS,
                               detect_language)

# One natural support-customer sentence per native script range.
SCRIPT_SAMPLES = {
    "hi": "मेरा ऑर्डर अभी तक नहीं आया",
    "ta": "என் ஆர்டர் இன்னும் வரவில்லை",
    "te": "నా ఆర్డర్ ఇంకా రాలేదు",
    "bn": "আমার অর্ডার এখনো আসেনি",
    "kn": "ನನ್ನ ಆರ್ಡರ್ ಇನ್ನೂ ಬಂದಿಲ್ಲ",
    "ml": "എന്റെ ഓർഡർ ഇപ്പോഴും വന്നിട്ടില്ല",
    "pa": "ਮੇਰਾ ਆਰਡਰ ਹਾਲੇ ਨਹੀਂ ਆਇਆ",
    "gu": "મારો ઓર્ડર હજી આવ્યો નથી",
}

ALL_CODES = {"en", "hinglish", "hi", "ta", "te", "bn", "mr", "gu", "kn",
             "ml", "pa"}


@pytest.mark.parametrize("lang,text", sorted(SCRIPT_SAMPLES.items()))
def test_native_script_detection(lang, text):
    assert detect_language(text) == lang


def test_mixed_devanagari_latin_native_script_wins():
    # Spec example: Latin-heavy code-switched text still detects the native
    # script — the native language drives the reply directive.
    assert detect_language("मेरा recharge क्यों fail हुआ?") == "hi"


def test_mostly_latin_with_native_phrase_is_native():
    assert detect_language("please help me, मेरा order") == "hi"


def test_single_native_char_is_not_a_language():
    # Majority-script rule: one stray native character in an otherwise
    # English sentence does not flip the detection.
    assert detect_language("I want a refund ॐ now") == "en"


def test_hinglish_lexicon_detection():
    assert detect_language("mera order kab aayega bhai") == "hinglish"
    assert detect_language("recharge fail ho gaya, kya karu") == "hinglish"


def test_hinglish_with_order_id_still_hinglish():
    assert detect_language("mera order ORD-55671 abhi tak nahi aaya") == \
        "hinglish"


def test_plain_english_is_not_hinglish():
    assert detect_language("Where is my order ORD-12345?") == "en"
    # A single 'do' is not enough: lexicon hits are counted as DISTINCT tokens.
    assert detect_language("What do you want to do today?") == "en"


@pytest.mark.parametrize("text", ["", "   ", "12345 678", "ORD-12345", "?!"])
def test_edge_cases_default_to_english(text):
    assert detect_language(text) == "en"


@pytest.mark.parametrize("text", ["hello", "kya haal hai", "मेरा ऑर्डर",
                                  "வணக்கம்", "ధన్యవాదాలు"])
def test_only_known_codes_are_returned(text):
    assert detect_language(text) in ALL_CODES


def test_native_script_langs_vocabulary():
    assert NATIVE_SCRIPT_LANGS == {"hi", "ta", "te", "bn", "mr", "gu",
                                   "kn", "ml", "pa"}


# ---------------------------------------------------------------------------
# Global Latin-script languages (es/fr/de/pt). Same contract as the Hinglish
# lexicon: exact-token match, >= 2 DISTINCT hits. Without detection the
# reply-language guardrail fails every es/fr/de/pt turn and the canned-reply
# fallback served Hindi to those customers.
# ---------------------------------------------------------------------------

GLOBAL_SAMPLES = {
    "es": [
        "¿Dónde está mi pedido?",
        "Quiero un reembolso de mi pedido",
        "Gracias, pero mi pedido no ha llegado",
        "¿Cuándo llega mi pedido y mi factura?",
        "Por favor, quiero una factura para mi pedido",
        # ASR output often strips accents — unaccented forms must detect too.
        "donde esta mi pedido",
        "cuando llega mi pedido",
    ],
    "fr": [
        "Où est ma commande ?",
        "Je veux un remboursement de ma commande",
        "Merci, mais ma commande n'est pas arrivée",
        "Quand est la livraison de ma commande ?",
        "Vous devez me rembourser ma commande",
        "ou est ma commande et ma livraison",
    ],
    "de": [
        "Wo ist meine Bestellung?",
        "Ich möchte eine Rückerstattung für meine Bestellung",
        "Danke, aber meine Bestellung ist nicht angekommen",
        "Wann kommt die Lieferung?",
        "Bitte prüfen Sie meine Bestellung",
        "bitte pruefen sie meine bestellung",
    ],
    "pt": [
        "Onde está meu pedido?",
        "Quero um reembolso do meu pedido",
        "Obrigado, mas meu pedido não chegou",
        "Quando chega a entrega do meu pedido?",
        "Você pode verificar o meu pedido?",
        "obrigado, mas a minha fatura esta errada",
    ],
}


@pytest.mark.parametrize("lang,text", sorted((l, t) for l, ts in
                                              GLOBAL_SAMPLES.items()
                                              for t in ts))
def test_global_language_detection(lang, text):
    assert detect_language(text) == lang


def test_global_lexicons_cover_exactly_the_target_set():
    assert set(GLOBAL_LEXICONS) == {"es", "fr", "de", "pt"}


def test_native_script_still_wins_over_latin_lexicons():
    # Check order: native Unicode script first, Latin lexicons after.
    assert detect_language("मेरा pedido क्यों fail हुआ?") == "hi"


def test_hinglish_still_detected_after_global_lexicons():
    # Real hinglish sentences carry no es/fr/de/pt lexicon tokens, so the
    # global check must not swallow them.
    assert detect_language("mera order kab aayega bhai") == "hinglish"
    assert detect_language("recharge fail ho gaya, kya karu") == "hinglish"


@pytest.mark.parametrize("text", [
    "Los Angeles order status",               # es: los
    "I ordered from the Los Angeles store",   # es: los
    "My laptop battery will die within an hour",        # de: die
    "The brewery est. 1990 offers tours in LA",         # fr: la only
    "Los Angeles to Le Mans freight status",  # es: los + fr: le -> no lexicon reaches 2
])
def test_english_with_one_overlap_token_stays_english(text):
    assert detect_language(text) == "en"


# Multi-token adversarials: English sentences reaching 2 hits on one lexicon
# must STILL stay English (the wrong-language failure mode is symmetric —
# a German false positive serves German text to an English caller).
@pytest.mark.parametrize("text", [
    "The die is cast, das is fine",       # de: das (die deliberately excluded)
    "I will die without das feedback",    # de: das
])
def test_english_with_two_overlap_tokens_stays_english(text):
    assert detect_language(text) == "en"


# Recall guard: a realistic German support sentence with NO domain noun and
# a simple ü->u transliteration must still detect as German.
def test_german_support_sentence_without_domain_noun_is_german():
    assert detect_language("Ich mochte eine Ruckerstattung bitte") == "de"


def test_eval_corpus_english_rows_stay_english():
    # The global lexicons change detection for some previously-"en" inputs;
    # the only acceptable change is text that is genuinely es/fr/de/pt. The
    # text-path benchmark corpus is English — not one row may flip.
    path = Path(__file__).resolve().parents[1] / "data/eval/conversations.csv"
    if not path.exists():
        pytest.skip("eval corpus not present")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Devanagari rows report "hi" for mr too (script identity, documented).
    allowed = {"en": {"en"}, "hinglish": {"hinglish", "en"}, "hi": {"hi"},
               "mr": {"hi"}, "te": {"te"}, "ta": {"ta"}, "bn": {"bn"},
               "gu": {"gu"}}
    assert {r["language"] for r in rows} <= set(allowed)
    for r in rows:
        got = detect_language(r["user_text"])
        assert got in allowed[r["language"]], (r["id"], r["user_text"], got)
