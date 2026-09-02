# tests/test_entities.py
from voiceagent.entities import extract_entities

def test_extracts_rupee_amount():
    e = extract_entities("I want a refund of ₹25,000 for my order")
    assert e.amount == 25000.0

def test_extracts_plain_amount_words():
    e = extract_entities("refund of 20000 rupees please")
    assert e.amount == 20000.0

def test_extracts_order_id():
    e = extract_entities("Where is my order ORD-55671?")
    assert e.order_id == "ORD-55671"

def test_no_entities():
    e = extract_entities("Someone used my account, block it")
    assert e.amount is None
    assert e.order_id is None

def test_hindi_amount():
    e = extract_entities("मुझे 25000 का रिफंड चाहिए")
    assert e.amount == 25000.0


# --------------------------------------------------------------------------
# M5b-3: number-words normalization (Qwen3-ASR writes numbers as words)
# --------------------------------------------------------------------------

def test_order_id_from_english_number_words():
    # Qwen3-ASR transcript style: "ORD four thousand eight hundred twenty one"
    e = extract_entities("Hello, I want to check the status of my order "
                         "ORD four thousand eight hundred twenty one.")
    assert e.order_id == "ORD-4821"
    assert e.amount is None  # the order id must not double as an amount


def test_order_id_from_digit_list_words():
    e = extract_entities("mera order ORD five five six seven one hai")
    assert e.order_id == "ORD-55671"


def test_order_id_words_span_removed_for_amount_scan():
    e = extract_entities("refund rupees five hundred for ORD four eight two one")
    assert e.order_id == "ORD-4821"
    assert e.amount == 500.0


def test_digit_form_order_id_still_not_an_amount():
    e = extract_entities("my order ORD-4821 pe ₹500 refund do")
    assert e.order_id == "ORD-4821"
    assert e.amount == 500.0


def test_short_number_words_are_not_order_ids():
    e = extract_entities("order ORD five five please")
    assert e.order_id is None  # 55: wrong shape for an order id


def test_amount_from_currency_number_words():
    e = extract_entities("I want a refund of rupees five thousand and two hundred")
    assert e.amount == 5200.0  # "and" swallowed: 5*1000 + 2*100


def test_devanagari_digits_normalized():
    e = extract_entities("मेरा ऑर्डर ४८२१ का स्टेटस चेक करो ORD-4821")
    assert e.order_id == "ORD-4821"
