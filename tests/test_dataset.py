import csv
import tempfile
from pathlib import Path
from voiceagent.dataset import Conversation, generate_eval_set, load_conversations

def test_generate_creates_requested_count_and_loads_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "eval.csv"
        n = generate_eval_set(str(out), n=100)
        assert n == 100
        rows = load_conversations(str(out))
        assert len(rows) == 100

def test_language_and_intent_coverage():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "eval.csv"
        generate_eval_set(str(out), n=200)
        rows = load_conversations(str(out))
        langs = {r.language for r in rows}
        intents = {r.intent for r in rows}
        assert {"en", "hi", "hinglish"} <= langs
        assert "order_status" in intents and "refund" in intents

def test_hinglish_rows_are_code_switched():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "eval.csv"
        generate_eval_set(str(out), n=100)
        rows = load_conversations(str(out))
        h = [r for r in rows if r.language == "hinglish"][0]
        assert any(c in h.user_text for c in "आ") or any(
            c in h.user_text for c in "abc"
        )
        assert "expected_action" in h.__dict__

def test_eval_set_has_appended_informational_rows():
    # M5c Fix 2: NEW eval rows (id sequence continues at conv-1000; no
    # history rewrite) covering refund-timing / ETA questions that used to
    # misroute to high_value_refund -> ESCALATE.
    rows = load_conversations("data/eval/conversations.csv")
    info = [r for r in rows
            if r.expected_action in ("refund_info", "delivery_eta")]
    assert len(info) >= 8
    assert {r.language for r in info} == {"en", "hinglish"}
    assert all(not r.escalate and not r.authenticated for r in info)
    assert all(int(r.id.split("-")[1]) >= 1000 for r in info)