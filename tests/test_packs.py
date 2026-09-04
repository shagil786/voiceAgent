# tests/test_packs.py
import pytest
from voiceagent.packs import detect_patterns, load_pack, load_vertical

def test_packs_load_strict():
    assert load_pack("answer")["pattern"] == "answer"
    assert len(load_pack("qualify")["tools"]) >= 1
    with pytest.raises(FileNotFoundError):
        load_pack("nope")
    with pytest.raises(ValueError, match="unknown field"):
        __import__("voiceagent.packs", fromlist=["_validate"])._validate(
            {"pattern": "x", "bogus": 1}, {"pattern"}, "t")

def test_verticals_ported_and_detect():
    auto = load_vertical("luxury_automotive")
    assert auto.catalog[0]["id"] == "EV-SUV-01"
    assert "IRDAI" in " ".join(load_vertical("insurance").statutory_disclosures)
    assert load_vertical("cards").name
    assert detect_patterns({"offering": "sell flats", "top_asks": ["site visit slot?"]}) == ["answer", "qualify", "follow_up", "draft_action"] or "qualify" in detect_patterns({"offering": "sell flats", "top_asks": ["site visit slot?"]})
    assert detect_patterns({"offering": "", "top_asks": []}) == ["answer"]
