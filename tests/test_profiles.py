# tests/test_profiles.py
import pytest
from voiceagent.learn.profiles import (
    PROFILE_TTL_DAYS, InMemoryProfiles, Profile, SQLiteProfiles,
    contact_key, normalize_phone,
)
from voiceagent.swarm.blackboard import CallerProfile

def test_e164_key_and_fallback():
    assert PROFILE_TTL_DAYS == 365
    assert normalize_phone("+91 98765 43210") == "+919876543210"
    assert normalize_phone("(020) 7946-0018") == "+02079460018"
    p = CallerProfile(customer_id="C-9", phone="+91 98765 43210")
    assert contact_key(p) == "+919876543210"
    assert contact_key(CallerProfile(customer_id="C-9")) == "cid:C-9"

def test_alias_resolve_and_session_links():
    s = InMemoryProfiles()
    s.put(Profile(key="+911", alias="", prefs=[], corrections=[],
              open_items=[], pending_global=[], consent={},
              updated_at="2026-01-01T00:00:00"))
    s.set_alias("Sharma-family", "+911")
    assert s.resolve("Sharma-family") == "+911"
    s.link_session("+911", "sess-1"); s.link_session("+911", "sess-1")
    assert s.sessions_for("+911") == ["sess-1"]

def test_delete_returns_sessions_and_export_roundtrip(tmp_path):
    for cls, arg in ((InMemoryProfiles, ()), (SQLiteProfiles, (str(tmp_path / "p.db"),))):
        s = cls(*arg) if arg else cls()
        s.put(Profile(key="k1", alias="Fam", prefs=["3BHK only"], corrections=[],
                      open_items=["callback"], pending_global=[], consent={"recording": True},
                      updated_at="2026-09-01T00:00:00"))
        s.link_session("k1", "s1")
        assert s.export_contact("k1")["prefs"] == ["3BHK only"]
        assert s.delete_contact("k1") == {"sessions": ["s1"]}
        with pytest.raises(KeyError):
            s.export_contact("k1")

def test_ttl_prune(tmp_path):
    s = InMemoryProfiles()
    s.put(Profile(key="old", alias="", prefs=[], corrections=[], open_items=[],
                  pending_global=[], consent={}, updated_at="2020-01-01T00:00:00"))
    assert s.prune_expired(now="2026-09-04T00:00:00") == 1
    assert s.get("old") is None
