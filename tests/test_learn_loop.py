from voiceagent.memory import InMemoryMemory
from voiceagent.orchestrator import Deployment, Orchestrator
from voiceagent.swarm.blackboard import CallerProfile
from voiceagent.swarm.frontier import FrontierAgentBridge
from voiceagent.learn.profiles import InMemoryProfiles, Profile
from tests.test_orchestrator import ScriptedBrain, reply

def _dep():
    return Deployment(name="acme", system_prompt="You are Acme.",
                      knowledge={"hours": "10am to 6pm"})

def _orch(**kw):
    kw.setdefault("memory", InMemoryMemory())
    orch = Orchestrator(brain=FrontierAgentBridge(ScriptedBrain([reply("Visits 10-6.")])), **kw)
    orch.deploy(_dep())
    return orch

def test_contact_memory_reaches_brain_and_candidates_captured():
    profs = InMemoryProfiles()
    profs.put(Profile(key="+911", alias="", prefs=["3BHK only"], corrections=[],
                      open_items=["callback Tue"], pending_global=[], consent={},
                      updated_at="2026-09-04T00:00:00"))
    orch = _orch(profiles=profs)
    orch.handle_turn("s1", "Any 2BHK?", profile=CallerProfile(phone="+911"))
    sent_first = orch.brain.client.calls[-1]["messages"][0]["content"]
    assert "3BHK only" in sent_first and "callback Tue" in sent_first
    assert profs.sessions_for("+911") == ["s1"]
    # candidate path: customer correction lands in pending_global, never bundle
    orch.handle_turn("s1", "No, mine is 3BHK not 2BHK",
                     profile=CallerProfile(phone="+911"))
    assert any("3BHK" in c["quote"] for c in profs.get("+911").pending_global)

def test_profiles_none_is_legacy_behavior():
    orch = _orch()
    r = orch.handle_turn("s9", "Hi", profile=CallerProfile(phone="+911"))
    assert r.reply == "Visits 10-6."

def test_delete_contact_cascades_sessions():
    profs = InMemoryProfiles()
    orch = _orch(profiles=profs)
    orch.handle_turn("sD", "Hi", profile=CallerProfile(phone="+911"))
    out = orch.delete_contact("+911")
    assert out == {"sessions": ["sD"]}
    assert orch.memory.history("sD") == []


def _anon():
    return CallerProfile(phone="", customer_id="")


def test_anonymous_turns_share_nothing():
    profs = InMemoryProfiles()
    orch = _orch(profiles=profs)
    orch.handle_turn("anon1", "No, mine is 3BHK not 2BHK", profile=_anon())
    orch.handle_turn("anon2", "Hi there", profile=_anon())
    sent = orch.brain.client.calls[-1]["messages"][0]["content"]
    assert "## Contact memory" not in sent
    assert profs.sessions_for("cid:unknown") == []
    assert profs._links == {}


def test_alias_to_existing_profile_still_works():
    profs = InMemoryProfiles()
    profs.put(Profile(key="+911", alias="", prefs=["3BHK only"], corrections=[],
                      open_items=[], pending_global=[], consent={},
                      updated_at="2026-09-04T00:00:00"))
    profs.set_alias("fam", "+911")
    orch = _orch(profiles=profs)
    orch.handle_turn("sA", "Hi", profile=_anon(), contact_alias="fam")
    sent = orch.brain.client.calls[-1]["messages"][0]["content"]
    assert "## Contact memory" in sent and "3BHK only" in sent


def test_blank_contact_lifecycle_guards():
    import pytest
    profs = InMemoryProfiles()
    orch = _orch(profiles=profs)
    assert orch.delete_contact("") == {"sessions": []}
    with pytest.raises(KeyError):
        orch.export_contact("")


def test_instant_correct_rejects_non_correction(tmp_path):
    import shutil
    from voiceagent.deploy.bundle import read_live
    from voiceagent.learn.instant import instant_correct
    shutil.copytree("data/deployments/_example/v1", tmp_path / "v1")
    out = instant_correct(str(tmp_path), "hello there")
    assert out["live"] is False and out["reason"] == "not a correction"
    assert out["version"] is None and out["passed"] is False
    assert out["checks"] == [] and "hello there" in out["changelog"]["quote"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v1"]
    assert read_live(str(tmp_path)) is None


def test_instant_correct_owner_only(tmp_path):
    import pytest
    from voiceagent.learn.instant import instant_correct
    with pytest.raises(ValueError, match="owner-only"):
        instant_correct(str(tmp_path), "No, visits run 10-6", actor="customer")


def test_pending_global_capped_at_50():
    from voiceagent.orchestrator import MAX_PENDING_GLOBAL
    assert MAX_PENDING_GLOBAL == 50
    profs = InMemoryProfiles()
    orch = Orchestrator(
        brain=FrontierAgentBridge(ScriptedBrain([reply("ok")] * 55)),
        memory=InMemoryMemory(), profiles=profs)
    orch.deploy(_dep())
    for i in range(55):
        orch.handle_turn(f"cap{i}", f"No, mine is 3BHK not 2BHK #{i}",
                         profile=CallerProfile(phone="+911"))
    assert len(profs.get("+911").pending_global) == 50


def test_inmemory_profiles_thread_safe_links():
    import threading
    profs = InMemoryProfiles()
    threads = [threading.Thread(
        target=lambda base: [profs.link_session("k", f"s{base}-{i}")
                             for i in range(25)], args=(t,))
        for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(profs.sessions_for("k")) == 100
