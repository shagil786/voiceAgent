# tests/test_dialogue.py — Task B Part 2: the clarify-and-dig not-found ladder.
"""A not-found lookup used to leave the brain to improvise on the first miss
(instant human-escalation demos, invented orders). With the top-level
`not_found_ladder:` declared in policies.yaml, the DialogueTracker runs a
BOUNDED per-session, per-slot probe ladder: miss 1 -> ask the caller to
re-confirm the id; miss 2 -> offer the declared alternate lookups; only after
the declared retries are exhausted does escalation remain the mandatory
terminal. Absent (or malformed) ladder config -> the pre-ladder behavior is
preserved byte-for-byte."""
import json

from tests.test_orchestrator import ScriptedBrain, reply, tc

from voiceagent.decisionlog import DecisionLog
from voiceagent.dialogue import DialogueTracker, render_directive
from voiceagent.memory import InMemoryMemory
from voiceagent.orchestrator import Deployment, Orchestrator
from voiceagent.policy import PolicyEngine, load_policies
from voiceagent.swarm.blackboard import CallerProfile
from voiceagent.swarm.frontier import FrontierAgentBridge
from voiceagent.tools import GovernedToolRunner, MockERP, ToolGateway

ALTERNATE = "check the recent orders placed on this phone number"

LADDER_POLICY = {
    "order_status": {"allow": True},
    "escalate_to_human": {"allow": True},
    "not_found_ladder": {"max_retries": 2, "offer_alternates": True,
                         "alternates": [ALTERNATE]},
}

NO_LADDER_POLICY = {
    "order_status": {"allow": True},
    "escalate_to_human": {"allow": True},
}

UNKNOWN = "ORD-9999"


def make_ladder_orch(brain, policy_cfg, log=None, erp=None) -> Orchestrator:
    erp = erp or MockERP()
    runner = GovernedToolRunner(ToolGateway(erp=erp),
                                PolicyEngine(policy_cfg), decision_log=log)
    orch = Orchestrator(brain=FrontierAgentBridge(brain), runner=runner,
                        memory=InMemoryMemory(), decision_log=log)
    orch.deploy(Deployment(name="ladder-test",
                           system_prompt="You are Acme's voice agent.",
                           gateway_tools={
                               "fetch_order_status":
                                   {"action": "order_status"},
                               # the safety valve must stay proposeable so
                               # the exhausted rung can reach the governed
                               # handoff
                               "escalate_to_human":
                                   {"action": "escalate_to_human"}}))
    return orch


def miss_call(call_id="t1"):
    return reply(calls=[tc(call_id, "fetch_order_status", order_id=UNKNOWN)])


# --- DialogueTracker unit semantics -------------------------------------------

def test_tracker_ladder_progression_is_bounded():
    t = DialogueTracker()
    d1 = t.not_found("s1", "order_id", value=UNKNOWN, max_retries=2,
                     alternates=[ALTERNATE])
    assert d1.kind == "ask_reconfirm"
    d2 = t.not_found("s1", "order_id", value=UNKNOWN, max_retries=2,
                     alternates=[ALTERNATE])
    assert d2.kind == "offer_alternates" and d2.alternates == (ALTERNATE,)
    d3 = t.not_found("s1", "order_id", value=UNKNOWN, max_retries=2,
                     alternates=[ALTERNATE])
    assert d3.kind == "escalate"                    # mandatory terminal
    # bounded: the episode ended, the counter reset — a new caller miss starts
    # the ladder from the top, it can never run away
    assert t.probes("s1", "order_id") == 0
    assert t.not_found("s1", "order_id", value=UNKNOWN,
                       max_retries=2).kind == "ask_reconfirm"


def test_tracker_resets_when_slot_filled():
    t = DialogueTracker()
    t.not_found("s1", "order_id", value=UNKNOWN, max_retries=2)
    assert t.probes("s1", "order_id") == 1
    t.found("s1", "order_id")                       # slot FILLED -> reset
    assert t.probes("s1", "order_id") == 0
    d = t.not_found("s1", "order_id", value=UNKNOWN, max_retries=2)
    assert d.kind == "ask_reconfirm"                # fresh ladder, not step 2


def test_tracker_slots_are_independent_per_session():
    t = DialogueTracker()
    t.not_found("s1", "order_id", max_retries=2)
    assert t.probes("s2", "order_id") == 0          # other session untouched
    assert t.probes("s1", "new_date") == 0          # other slot untouched


def test_offer_alternates_false_repeats_reconfirm_ask():
    t = DialogueTracker()
    d1 = t.not_found("s1", "order_id", value=UNKNOWN, max_retries=2,
                     alternates=[])
    d2 = t.not_found("s1", "order_id", value=UNKNOWN, max_retries=2,
                     alternates=[])
    assert d1.kind == d2.kind == "ask_reconfirm"
    d3 = t.not_found("s1", "order_id", value=UNKNOWN, max_retries=2,
                     alternates=[])
    assert d3.kind == "escalate"


def test_render_directive_echoes_reference_and_alternates():
    t = DialogueTracker()
    ask = render_directive(t.not_found("s", "order_id", value=UNKNOWN,
                                       max_retries=2))
    assert UNKNOWN in ask and "confirm" in ask.lower()
    offer = render_directive(t.not_found("s", "order_id", value=UNKNOWN,
                                         max_retries=2,
                                         alternates=[ALTERNATE]))
    assert UNKNOWN in offer and ALTERNATE in offer


# --- PolicyEngine surface ------------------------------------------------------

def test_policyengine_exposes_declared_ladder():
    eng = PolicyEngine(dict(LADDER_POLICY))
    assert eng.not_found_ladder() == {"max_retries": 2,
                                      "offer_alternates": True,
                                      "alternates": [ALTERNATE]}


def test_policyengine_absent_or_malformed_ladder_is_none():
    assert PolicyEngine(dict(NO_LADDER_POLICY)).not_found_ladder() is None
    assert PolicyEngine({}).not_found_ladder() is None
    assert PolicyEngine({"not_found_ladder": {"max_retries": "two"}}) \
        .not_found_ladder() is None
    assert PolicyEngine({"not_found_ladder": "yes"}).not_found_ladder() is None


def test_example_acme_bundle_declares_the_ladder():
    eng = PolicyEngine(load_policies("data/tenants/example-acme/policies.yaml"))
    ladder = eng.not_found_ladder()
    assert ladder is not None
    assert ladder["max_retries"] == 2
    assert ladder["offer_alternates"] is True
    assert ladder["alternates"] == [ALTERNATE]


# --- end-to-end through the governed turn loop ---------------------------------

def test_ladder_turn1_asks_to_reconfirm_not_escalate():
    erp = MockERP()
    brain = ScriptedBrain([miss_call()])
    orch = make_ladder_orch(brain, LADDER_POLICY, erp=erp)
    res = orch.handle_turn("s-lad-1", f"Where is my order {UNKNOWN}?",
                           profile=CallerProfile(authenticated=True))
    assert UNKNOWN in res.reply and "confirm" in res.reply.lower()
    assert "recent orders" not in res.reply         # that's step 2's offer
    assert not res.escalated
    assert erp.handoffs == []
    # the not-found fact is still audited on the turn's actions
    assert res.actions[0]["error"].startswith("order_not_found")
    # deterministic directive: no extra brain round was spent improvising
    assert len(brain.calls) == 1


def test_ladder_turn2_offers_alternate_lookup():
    erp = MockERP()
    brain = ScriptedBrain([miss_call(), miss_call("t2")])
    orch = make_ladder_orch(brain, LADDER_POLICY, erp=erp)
    orch.handle_turn("s-lad-2", f"Where is my order {UNKNOWN}?",
                     profile=CallerProfile(authenticated=True))
    res2 = orch.handle_turn("s-lad-2", f"Still nothing about {UNKNOWN}?",
                            profile=CallerProfile(authenticated=True))
    assert ALTERNATE in res2.reply
    assert not res2.escalated
    assert erp.handoffs == []


def test_ladder_exhausted_escalates_as_mandatory_terminal():
    erp, log = MockERP(), DecisionLog()
    brain = ScriptedBrain([
        miss_call(),                                   # turn 1 miss -> ask
        miss_call("t2"),                               # turn 2 miss -> offer
        miss_call("t3"),                               # turn 3 miss -> exhausted
        reply(calls=[tc("t4", "escalate_to_human",
                        reason="order id could not be resolved")]),
        reply("I'm connecting you to a human specialist now."),
    ])
    orch = make_ladder_orch(brain, LADDER_POLICY, erp=erp, log=log)
    orch.handle_turn("s-lad-3", f"Where is my order {UNKNOWN}?",
                     profile=CallerProfile(authenticated=True))
    orch.handle_turn("s-lad-3", f"I typed it again: {UNKNOWN}.",
                     profile=CallerProfile(authenticated=True))
    res3 = orch.handle_turn("s-lad-3", f"It really is {UNKNOWN}!",
                            profile=CallerProfile(authenticated=True))
    # the governed handoff really happened and is the turn outcome
    assert res3.escalated is True
    assert len(erp.handoffs) == 1
    # policy verdict path intact: DecisionLog records the escalate
    entry = log.entries()[-1]
    assert entry.action == "escalate_to_human" and entry.verdict == "ALLOW"
    assert entry.conv_id == "s-lad-3"
    # the brain was told the ladder was exhausted before it proposed the handoff
    tool_payloads = [json.loads(m["content"])
                     for m in brain.calls[3]["messages"]
                     if m.get("role") == "tool"]
    assert any(p.get("not_found_ladder_exhausted") for p in tool_payloads)


def test_ladder_resets_once_a_valid_id_is_provided():
    erp = MockERP()
    brain = ScriptedBrain([
        miss_call(),                                   # turn 1 miss -> ask
        reply(calls=[tc("t2", "fetch_order_status",
                        order_id="ORD-4821")]),        # turn 2: valid id
        reply("Your order ORD-4821 is confirmed and on its way."),
        miss_call("t4"),                               # turn 3: new miss
    ])
    orch = make_ladder_orch(brain, LADDER_POLICY, erp=erp)
    orch.handle_turn("s-lad-4", f"Where is my order {UNKNOWN}?",
                     profile=CallerProfile(authenticated=True))
    ok = orch.handle_turn("s-lad-4", "Sorry, it is ORD-4821.",
                          profile=CallerProfile(authenticated=True))
    assert ok.actions[0]["ok"] is True                 # known order: success
    # the ladder RESET on the successful lookup: the next miss starts at step 1
    res3 = orch.handle_turn("s-lad-4", f"And what about {UNKNOWN}?",
                            profile=CallerProfile(authenticated=True))
    assert "confirm" in res3.reply.lower()
    assert "recent orders" not in res3.reply
    assert not res3.escalated


def test_known_order_immediate_success_unchanged():
    erp = MockERP()
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "fetch_order_status", order_id="ORD-4821")]),
        reply("Your order ORD-4821 is confirmed and on its way."),
    ])
    orch = make_ladder_orch(brain, LADDER_POLICY, erp=erp)
    res = orch.handle_turn("s-lad-5", "Where is my order ORD-4821?",
                           profile=CallerProfile(authenticated=True))
    assert res.actions[0]["ok"] is True
    assert res.actions[0]["value"]["order_id"] == "ORD-4821"
    assert res.reply == "Your order ORD-4821 is confirmed and on its way."
    assert not res.escalated


def test_no_ladder_declared_preserves_current_single_miss_behavior():
    erp = MockERP()
    brain = ScriptedBrain([
        miss_call(),
        reply("I could not find that order — let me transfer you."),
    ])
    orch = make_ladder_orch(brain, NO_LADDER_POLICY, erp=erp)
    res = orch.handle_turn("s-lad-6", f"Where is my order {UNKNOWN}?",
                           profile=CallerProfile(authenticated=True))
    # today's behavior: the not-found tool result is fed back and the brain
    # replies; no directive is injected, no bounded probe state exists
    assert res.reply == "I could not find that order — let me transfer you."
    assert res.actions[0]["error"].startswith("order_not_found")
    assert not res.escalated
    assert len(brain.calls) == 2
