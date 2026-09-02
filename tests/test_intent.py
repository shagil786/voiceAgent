# tests/test_intent.py
"""M5c Fix 2: informational intents (refund_info, delivery_eta) so
policy-relevant questions ("aur refund kitne din me aata hai?") stop
misrouting to high_value_refund -> ESCALATE. Uses the real multilingual
embedding model (same convention as test_knowledge.py)."""
import pytest
from voiceagent.intent import IntentClassifier
from voiceagent.policy import PolicyContext, PolicyEngine, load_policies

# English + Hinglish phrasings that are INFORMATIONAL questions about refund
# timing — not refund requests, not high-value refunds.
REFUND_INFO_PHRASES = [
    "refund kitne din me aata hai",
    "aur refund kitne din me aata hai?",  # live-observed misroute
    "when will I get my refund back",
    "paise kab wapas milenge",
    "how long does a refund take",
    "how many days for the refund",
    "refund kab tak aayega",
    "mera refund kab milega",
]

# English + Hinglish ETA questions about a pending delivery.
DELIVERY_ETA_PHRASES = [
    "when will my order be delivered",
    "order kab tak aayega",
    "delivery kab hogi",
    "how many days will delivery take",
    "when will my order arrive",
    "mera order kab milega",
    "delivery date kya hai",
    "when can I expect my delivery",
]


@pytest.fixture(scope="module")
def classifier():
    return IntentClassifier()


def test_refund_info_phrases_classify_as_refund_info(classifier):
    for p in REFUND_INFO_PHRASES:
        action, score = classifier.classify(p)
        assert action == "refund_info", f"{p!r} -> {action} ({score:.3f})"


def test_delivery_eta_phrases_classify_as_delivery_eta(classifier):
    for p in DELIVERY_ETA_PHRASES:
        action, score = classifier.classify(p)
        assert action == "delivery_eta", f"{p!r} -> {action} ({score:.3f})"


def test_live_misroute_no_longer_escalates(classifier):
    # Before M5c: "aur refund kitne din me aata hai?" -> high_value_refund
    # -> ESCALATE. It is an informational question: it must classify as
    # refund_info and the policy must ALLOW it (read-only, no auth).
    action, score = classifier.classify("aur refund kitne din me aata hai?")
    assert action == "refund_info"
    eng = PolicyEngine(load_policies("data/policies/policies.yaml"))
    assert eng.evaluate(action, PolicyContext()).verdict == "ALLOW"


# ---------------------------------------------------------------------------
# M5a: hi (Devanagari) / ta / te exemplars per intent. Classification checks
# below use HELD-OUT phrasings (not the exemplar strings themselves) so the
# tests prove generalization, not memorization. Exemplars are LLM-authored
# synthetic support-customer phrasings (quality caveat documented in
# intent.py) — the multilingual embedding model does the cross-lingual work.
# ---------------------------------------------------------------------------

def _has_char_in_block(text: str, lo: int, hi: int) -> bool:
    return any(lo <= ord(c) <= hi for c in text)


def _is_devanagari(t):  return _has_char_in_block(t, 0x0900, 0x097F)
def _is_tamil(t):       return _has_char_in_block(t, 0x0B80, 0x0BFF)
def _is_telugu(t):      return _has_char_in_block(t, 0x0C00, 0x0C7F)


def test_every_intent_has_hi_ta_te_exemplars():
    from voiceagent.intent import INTENT_EXEMPLARS
    for intent, exemplars in INTENT_EXEMPLARS.items():
        for name, pred in (("hi", _is_devanagari), ("ta", _is_tamil),
                           ("te", _is_telugu)):
            n = sum(1 for e in exemplars if pred(e))
            assert n >= 3, f"{intent}: only {n} {name} exemplars"


def test_tamil_order_status_text_classifies(classifier):
    action, score = classifier.classify("என் ஆர்டர் எங்கே இருக்கிறது?")
    assert action == "order_status", f"score={score:.3f}"


def test_telugu_refund_text_classifies(classifier):
    action, score = classifier.classify("నా ఆర్డర్ డబ్బు రీఫండ్ చేయండి")
    assert action == "refund", f"score={score:.3f}"


def test_tamil_otp_text_classifies(classifier):
    action, score = classifier.classify("ஓடிபி எனக்கு கிடைக்கவில்லை")
    assert action == "otp", f"score={score:.3f}"


def test_tamil_delivery_eta_text_classifies(classifier):
    # Tamil ETA generalizes with a clear margin. (Telugu ETA phrasings sit
    # on a ~0.003 cosine margin vs replacement/otp — too fragile to pin in
    # a test; noted as a known boundary in the M5a report.)
    action, score = classifier.classify("என் ஆர்டர் எப்போது டெலிவரி ஆகும்")
    assert action == "delivery_eta", f"score={score:.3f}"


def test_telugu_account_closure_text_classifies(classifier):
    action, score = classifier.classify("నా ఖాతా మూసేయాలని అనుకుంటున్నాను")
    assert action == "account_closure", f"score={score:.3f}"


def test_tamil_cancel_order_text_classifies(classifier):
    action, score = classifier.classify("என் ஆர்டரை இப்போது ரத்து செய்யுங்கள்")
    assert action == "cancel_order", f"score={score:.3f}"
