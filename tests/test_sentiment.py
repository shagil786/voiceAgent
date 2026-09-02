# tests/test_sentiment.py — M6a: deterministic frustration detection.
from voiceagent.sentiment import detect_frustration

def test_en_high_frustration_multiple_hits():
    fr = detect_frustration("This is ridiculous, worst service ever. I demand a manager.")
    assert fr.level == "high"
    assert "ridiculous" in fr.hits and "worst" in fr.hits

def test_en_intensity_shouting_and_bangs():
    fr = detect_frustration("WHAT IS THIS???")
    assert fr.level == "high"
    assert "shouting caps" in fr.intensity and "repeated !/?" in fr.intensity

def test_en_mild_single_hit():
    fr = detect_frustration("I am a bit angry about the delay.")
    assert fr.level == "mild"
    assert fr.hits == ["angry"]

def test_en_none_for_calm_text():
    fr = detect_frustration("Where is my order ORD-4821?")
    assert fr.level == "none"
    assert not fr.frustrated

def test_hinglish_frustration():
    fr = detect_frustration("Ye toh bakwas hai, bar bar nahi ho raha")
    assert fr.level == "high"
    assert "bakwas" in fr.hits

def test_hindi_devanagari_frustration():
    fr = detect_frustration("यह बहुत बेकार सर्विस है, मुझे गुस्सा आ रहा है")
    assert fr.level == "high"

def test_hinglish_sees_devanagari_and_vice_versa():
    assert detect_frustration("ye service bekaar hai", "hinglish").frustrated
    assert detect_frustration("angry हूँ मैं, बकवास", "hi").frustrated

def test_spanish_frustration():
    fr = detect_frustration("Esto es inaceptable, quiero hablar con alguien")
    assert fr.frustrated


# ---------------------------------------------------------------------------
# M6b: the lexicon LEARNS — SQLite-backed store, candidate capture from
# novel intensity-only expressions, human-approved promotion.
# ---------------------------------------------------------------------------

import pytest
from voiceagent.sentiment import SentimentStore, candidate_phrases_from

@pytest.fixture
def store(tmp_path):
    return SentimentStore(str(tmp_path / "sentiment.db"))

def test_learned_phrase_extends_detection(store):
    # Unknown slang isn't detected yet
    text = "this app is total malarkey, fix it"
    assert detect_frustration(text, "en").level == "none"
    store.add_phrase("malarkey", "en", source="manual")
    fr = detect_frustration(text, "en",
                            extra_phrases=store.learned_phrases("en"))
    assert fr.level == "mild"

def test_novel_intensity_captured_as_candidate_then_promoted(store):
    text = "YOUR SERVICE IS ABSOLUTELY FLIBBERTIGIBBET!!!"
    fr = detect_frustration(text, "en")
    # intensity present, but no known phrase — the novel expression
    assert fr.level == "high" and not fr.hits
    store.capture_candidates(candidate_phrases_from(text), "en")
    cands = store.candidates()
    assert any(c["candidate"] == "flibbertigibbet" for c in cands)
    # repeat turns bump the counter, not the row count
    store.capture_candidates(candidate_phrases_from(text), "en")
    assert len(store.candidates()) == len(cands)
    # human review promotes it -> detection now knows the word
    assert store.promote("flibbertigibbet", "en")
    fr2 = detect_frustration(text, "en",
                             extra_phrases=store.learned_phrases("en"))
    assert fr2.hits and "flibbertigibbet" in fr2.hits

def test_promote_missing_candidate_returns_false(store):
    assert not store.promote("nonexistent", "en")
