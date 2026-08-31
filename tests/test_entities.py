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
