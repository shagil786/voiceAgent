# tests/test_langid.py
"""M5a: stdlib-only language detection for text turns. Native Unicode-script
detection wins over Latin heuristics; Latin-script text is hinglish when it
carries >=2 distinct tokens from a small Hinglish lexicon, else en."""
import pytest

from voiceagent.langid import NATIVE_SCRIPT_LANGS, detect_language

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
