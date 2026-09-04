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

def test_expired_get_drops_alias_and_links():
    s = InMemoryProfiles()
    s.put(Profile(key="stale", alias="", prefs=[], corrections=[],
                  open_items=[], pending_global=[], consent={},
                  updated_at="2020-01-01T00:00:00"))
    s.set_alias("Stale-fam", "stale")
    s.link_session("stale", "sess-9")
    assert s.get("stale") is None
    assert s.resolve("Stale-fam") == "Stale-fam"
    assert s.sessions_for("stale") == []

def test_get_put_copy_semantics():
    s = InMemoryProfiles()
    s.put(Profile(key="c1", alias="", prefs=["2BHK"], corrections=[],
                  open_items=[], pending_global=[], consent={},
                  updated_at="2026-09-01T00:00:00"))
    p = s.get("c1")
    assert p is not None
    p.prefs.append("X")
    assert s.get("c1").prefs == ["2BHK"]

def test_sqlite_cascade_and_expiry(tmp_path):
    from voiceagent.learn.profiles import SQLiteProfiles
    s = SQLiteProfiles(str(tmp_path / "c.db"))
    s.put(Profile(key="k9", alias="Fam9", prefs=["x"], corrections=[],
                  open_items=[], pending_global=[], consent={},
                  updated_at="2020-01-01T00:00:00"))
    s.link_session("k9", "sess-9")
    assert s.get("k9") is None  # TTL prunes on read (2020 vs 365-day TTL)
    assert s.sessions_for("k9") == []  # links die with the profile
    with pytest.raises(KeyError):
        s.export_contact("k9")

def test_put_store_copy_semantics():
    s = InMemoryProfiles()
    p2 = Profile(key="k2", alias="", prefs=["2BHK"], corrections=[],
                 open_items=[], pending_global=[], consent={},
                 updated_at="2026-09-01T00:00:00")
    s.put(p2)
    p2.prefs.append("X")
    assert s.get("k2").prefs == ["2BHK"]

def test_instant_correct_timing_on_realistic_bundle(tmp_path):
    import shutil, time
    from voiceagent.deploy.bundle import EvalCheck, load_bundle, save_bundle
    from voiceagent.learn.instant import instant_correct
    shutil.copytree("data/deployments/_example/v1", tmp_path / "v1")
    b = load_bundle(tmp_path / "v1")
    # realistic bundle: golden spec/tools/policies + 5 knowledge chunks;
    # 10 contains-only evals so the real go-live path is exercised (R3).
    b.knowledge = [{"text": f"Chunk {i}: support hours 9am to 7pm",
                    "source": "owner_paste",
                    "crawled_at": "2026-09-04T00:00:00"}
                   for i in range(5)]
    b.evals = [EvalCheck(name=f"live-{i:02d}", turns=[{"user": "Hello"}],
                         assert_={"contains": "Hello"}) for i in range(10)]
    save_bundle(b, tmp_path / "v1")
    t0 = time.monotonic()
    out = instant_correct(str(tmp_path), "No, support hours are 9am to 7pm")
    assert out["live"] is True and (time.monotonic() - t0) < 60
