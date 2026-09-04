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
