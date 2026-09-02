# tests/test_entities_snapping.py — Sprint A WS1: phonetic & contextual
# entity snapping for ASR-garbled order references.
from voiceagent.entities import extract_order_id

CANDIDATES = ["ORD-4821", "ORD-7734"]

def test_clean_ord_id_exact_no_candidates_needed():
    assert extract_order_id("Where is my order ORD-55671?") == "ORD-55671"

def test_punctuation_split_garble_snaps():
    assert extract_order_id("I need to change address for order or D7734",
                            CANDIDATES) == "ORD-7734"

def test_spaced_digits_snap():
    assert extract_order_id("my order ORD 7 7 3 4 please", CANDIDATES) == "ORD-7734"

def test_loose_punctuation_with_ord_marker():
    assert extract_order_id("ord - 7734 status") == "ORD-7734"

def test_devanagari_digits_snap():
    assert extract_order_id("मेरा ऑर्डर ४८२१ का स्टेटस", CANDIDATES) == "ORD-4821"

def test_low_similarity_returns_none_not_a_guess():
    # 'वाली 4808' vs ORD-4821 is only ~0.5 similar — a wrong snap is worse
    # than asking the customer again.
    assert extract_order_id("order वाली 4808", CANDIDATES) is None

def test_phone_number_never_snaps_to_order():
    assert extract_order_id("call me back at 9876543210", CANDIDATES) is None

def test_no_candidates_no_exact_no_match():
    assert extract_order_id("or D7734 please") is None

def test_no_digit_material():
    assert extract_order_id("where is my order", CANDIDATES) is None

def test_best_candidate_wins():
    cands = ["ORD-1234", "ORD-7734"]
    assert extract_order_id("status of order or D7734", cands) == "ORD-7734"
