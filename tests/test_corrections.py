# tests/test_corrections.py
from voiceagent.learn.corrections import classify_correction

def test_owner_fact_correction_is_global():
    c = classify_correction("No, that fee is 499 not 299", "Fee is 299", is_owner=True)
    assert (c.is_correction, c.patch_type, c.scope) == (True, "fact", "global")
    assert "499" in c.quote and len(c.quote) <= 280

def test_owner_policy_and_tone_types():
    assert classify_correction("Never promise loan approval", is_owner=True).patch_type == "policy"
    assert classify_correction("No, keep it shorter and polite", is_owner=True).patch_type == "tone"
    assert classify_correction("No, say 'site visits 10-6' like this", is_owner=True).patch_type == "exemplar"

def test_customer_correction_never_global():
    c = classify_correction("No, my flat is 3BHK not 2BHK", "Noted 2BHK", is_owner=False)
    assert c.is_correction is True and c.scope == "candidate"

def test_plain_chat_is_not_correction():
    c = classify_correction("What are the timings?", "10am to 6pm")
    assert (c.is_correction, c.patch_type, c.scope) == (False, "none", "none")
