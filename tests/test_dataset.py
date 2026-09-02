import csv
import re
import tempfile
from pathlib import Path
from voiceagent.dataset import (Conversation, append_multilingual_eval_set,
                                generate_eval_set, load_conversations)

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


# ---------------------------------------------------------------------------
# M5a: appended native-script rows for ta/te/mr/bn/gu (ids conv-2000..,
# append-only — existing rows untouched). Same intent mix as the seed
# templates; order ids stay as digits.
# ---------------------------------------------------------------------------

# (language, Unicode block) for the script check below.
_LANG_BLOCK = {
    "ta": (0x0B80, 0x0BFF), "te": (0x0C00, 0x0C7F),
    "mr": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "gu": (0x0A80, 0x0AFF),
}


def _test_csv(tmp: str) -> str:
    out = Path(tmp) / "eval.csv"
    generate_eval_set(str(out), n=20)
    return str(out)


def test_multilingual_append_adds_rows_without_touching_existing():
    with tempfile.TemporaryDirectory() as d:
        path = _test_csv(d)
        before = load_conversations(path)
        added = append_multilingual_eval_set(
            path, languages=("ta", "te"), per_language=5, seed=2000)
        assert added == 10
        rows = load_conversations(path)
        assert rows[:len(before)] == before          # existing rows intact
        assert len(rows) == len(before) + 10
        assert len({r.id for r in rows}) == len(rows)  # no duplicate ids


def test_multilingual_rows_are_native_script_with_digit_order_ids():
    with tempfile.TemporaryDirectory() as d:
        path = _test_csv(d)
        append_multilingual_eval_set(
            path, languages=("ta", "te", "mr", "bn", "gu"),
            per_language=3, seed=2000)
        rows = [r for r in load_conversations(path)
                if r.language in _LANG_BLOCK]
        assert len(rows) == 15
        for r in rows:
            lo, hi = _LANG_BLOCK[r.language]
            assert any(lo <= ord(c) <= hi for c in r.user_text), r
            if r.expected_action in ("order_status", "refund",
                                     "high_value_refund"):
                assert re.search(r"ORD-\d+", r.user_text), r
                assert all(f.startswith("ORD-") for f in r.key_facts), r
                assert r.authenticated
            else:
                assert r.key_facts == []
            assert r.id.startswith("conv-2")


def test_multilingual_default_languages_and_seed():
    with tempfile.TemporaryDirectory() as d:
        path = _test_csv(d)
        added = append_multilingual_eval_set(path)  # defaults: 5 langs x 30
        assert added == 150
        rows = [r for r in load_conversations(path)
                if r.language in _LANG_BLOCK]
        ids = [int(r.id.split("-")[1]) for r in rows]
        assert min(ids) == 2000 and max(ids) == 2149
        from collections import Counter
        counts = Counter(r.language for r in rows)
        assert set(counts) == {"ta", "te", "mr", "bn", "gu"}
        assert all(c == 30 for c in counts.values())


def test_real_eval_csv_has_multilingual_rows():
    # The shipped dataset must carry the appended M5a rows.
    rows = load_conversations("data/eval/conversations.csv")
    new = [r for r in rows if 2000 <= int(r.id.split("-")[1]) <= 2149]
    assert len(new) == 150
    from collections import Counter
    counts = Counter(r.language for r in new)
    assert set(counts) == {"ta", "te", "mr", "bn", "gu"}
    assert all(c == 30 for c in counts.values())
    assert len({r.id for r in rows}) == len(rows)  # no duplicates overall
    # Same intent mix vocabulary as the base set.
    assert {r.intent for r in new} <= {
        "order_status", "refund", "payment_declined", "recharge", "billing",
        "high_value_refund", "fraud", "otp"}