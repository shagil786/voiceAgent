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
