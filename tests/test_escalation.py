# tests/test_escalation.py — governed escalation + expanded action surface.
"""The live-brain probe found three gaps: (1) the brain SAID "connecting you
to a human" but nothing executed; (2) it promised returns/replacements with
no tool deployed; (3) it hallucinated a tracking URL. Pins the fixes: real
governed escalate_to_human + initiate_return tools (auditable handoffs,
precondition-gated returns), deterministic frustration signals on EVERY
governed evaluation, and the live runner's hardened prompt (no invented
URLs; only promise actions that exist in the tool surface)."""
import importlib.util
import json
from pathlib import Path

from tests.test_orchestrator import ScriptedBrain, reply, tc

from voiceagent.decisionlog import DecisionLog
from voiceagent.memory import InMemoryMemory
from voiceagent.orchestrator import Deployment, Orchestrator
from voiceagent.policy import PolicyContext, PolicyEngine, load_policies
from voiceagent.sentiment import detect_frustration
from voiceagent.swarm.blackboard import CallerProfile
from voiceagent.swarm.frontier import FrontierAgentBridge
from voiceagent.tools import GovernedToolRunner, MockERP, ToolGateway


def make_orch(brain, policy, gateway_tools, erp=None, log=None) -> Orchestrator:
    """Orchestrator whose governed runner sits over the given policy engine."""
    erp = erp or MockERP()
    runner = GovernedToolRunner(ToolGateway(erp=erp), policy, decision_log=log)
    orch = Orchestrator(brain=FrontierAgentBridge(brain), runner=runner,
                        memory=InMemoryMemory(), decision_log=log)
    orch.deploy(Deployment(name="esc-test",
                           system_prompt="You are Acme's voice agent.",
                           gateway_tools=gateway_tools))
    return orch


# --- MockERP backend methods -------------------------------------------------

def test_mockerp_mark_return_sets_status_and_reason():
    erp = MockERP()
    out = erp.mark_return("ORD-7734", "arrived damaged")
    assert out["status"] == "RETURN_REQUESTED"
    assert out["return_reason"] == "arrived damaged"
    assert erp.get_order("ORD-7734")["status"] == "RETURN_REQUESTED"


def test_mockerp_record_handoff_appends_audit_entry():
    erp = MockERP()
    out = erp.record_handoff("customer demands a human")
    assert out == {"handed_off": True, "reason": "customer demands a human"}
    assert len(erp.handoffs) == 1
    assert erp.handoffs[0]["reason"] == "customer demands a human"
    assert erp.handoffs[0]["ts"]  # timestamped for auditability


# --- the escalation tool: a real, auditable handoff --------------------------

def test_escalate_to_human_executes_and_sets_turn_escalated():
    erp, log = MockERP(), DecisionLog()
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "escalate_to_human",
                        reason="customer demands a human")]),
        reply("I'm connecting you to a human specialist now."),
    ])
    orch = make_orch(brain,
                     PolicyEngine({"escalate_to_human": {"allow": True}}),
                     {"escalate_to_human": {"action": "escalate_to_human"}},
                     erp=erp, log=log)
    result = orch.handle_turn("s-esc-1", "Give me a real person.",
                              profile=CallerProfile(authenticated=True))
    # the handoff really happened and is auditable
    assert len(erp.handoffs) == 1
    assert erp.handoffs[0]["reason"] == "customer demands a human"
    # a successful governed handoff flags the turn as escalated
    assert result.escalated is True
    assert result.actions[0]["tool"] == "escalate_to_human"
    assert result.actions[0]["verdict"] == "ALLOW"
    assert result.actions[0]["ok"] is True
    assert result.actions[0]["value"]["handed_off"] is True
    # the brain received the tool result payload with handed_off True
    msgs = [m for m in brain.calls[1]["messages"] if m.get("role") == "tool"]
    payload = json.loads(msgs[0]["content"])
    assert payload["ok"] is True and payload["value"]["handed_off"] is True
    # audited under the session id
    entry = log.entries()[-1]
    assert entry.verdict == "ALLOW" and entry.action == "escalate_to_human"
    assert entry.conv_id == "s-esc-1"


def test_escalate_to_human_blocked_does_not_set_escalated():
    erp = MockERP()
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "escalate_to_human", reason="handover please")]),
        reply("I can't transfer you on this line right now."),
    ])
    # no rule for escalate_to_human -> least-privilege DENY, nothing paged
    orch = make_orch(brain, PolicyEngine({"reschedule": {"allow": True}}),
                     {"escalate_to_human": {"action": "escalate_to_human"}},
                     erp=erp)
    result = orch.handle_turn("s-esc-2", "Transfer me to a human.",
                              profile=CallerProfile(authenticated=True))
    assert result.escalated is False
    assert result.actions[0]["verdict"] == "DENY"
    assert result.actions[0]["ok"] is False
    assert erp.handoffs == []


# --- returns: only shipped/delivered orders can be returned ------------------

def test_initiate_return_on_shipped_order_is_allowed_and_executed():
    erp = MockERP()
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "initiate_return",
                        order_id="ORD-7734", reason="arrived damaged")]),
        reply("Your return is requested — pickup will be scheduled shortly."),
    ])
    orch = make_orch(brain,
                     PolicyEngine({"initiate_return": {"require_auth": True}}),
                     {"initiate_return": {"action": "initiate_return"}},
                     erp=erp)
    result = orch.handle_turn("s-ret-1", "I need to return my smart watch",
                              profile=CallerProfile(authenticated=True))
    assert result.actions[0]["verdict"] == "ALLOW"
    assert result.actions[0]["ok"] is True
    assert result.actions[0]["value"]["status"] == "RETURN_REQUESTED"
    assert erp.get_order("ORD-7734")["status"] == "RETURN_REQUESTED"
    assert erp.get_order("ORD-7734")["return_reason"] == "arrived damaged"
    assert not result.escalated  # a return is not an escalation


def test_initiate_return_on_confirmed_order_is_precondition_blocked():
    gw = ToolGateway()
    r = gw.execute("initiate_return", {"order_id": "ORD-4821",
                                       "reason": "changed my mind"})
    assert not r.ok
    assert "precondition_failed" in r.error and "CONFIRMED" in r.error
    assert gw.erp.get_order("ORD-4821")["status"] == "CONFIRMED"  # untouched


# --- frustration signal wiring ------------------------------------------------

class RecordingPolicy(PolicyEngine):
    """Spy engine: captures the PolicyContext each evaluation received."""

    def __init__(self, policies):
        super().__init__(policies)
        self.contexts: list[PolicyContext] = []

    def evaluate(self, action, ctx=None):
        self.contexts.append(ctx)
        return super().evaluate(action, ctx)


def test_frustration_detector_baseline():
    calm = detect_frustration("Where is my order?")
    assert calm.frustrated is False and calm.level == "none"
    angry = detect_frustration("This is ridiculous!! Worst service ever.")
    assert angry.frustrated is True and angry.level == "high"


def test_frustration_signals_reach_every_governed_evaluation():
    erp = MockERP()
    policy = RecordingPolicy({"reschedule_delivery": {"require_auth": True}})
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "reschedule_delivery",
                        order_id="ORD-4821", new_date="2026-09-10")]),
        reply("Done — moved to the 10th."),
    ])
    orch = make_orch(brain, policy,
                     {"reschedule_delivery": {"action": "reschedule_delivery"}},
                     erp=erp)
    orch.handle_turn("s-frac-1", "This is ridiculous!! Move my delivery, "
                                 "WORST service ever.",
                     profile=CallerProfile(authenticated=True))
    ctx = policy.contexts[0]
    assert ctx.signals["frustrated"] is True
    assert ctx.signals["frustration_level"] == "high"
    assert ctx.signals["session_id"] == "s-frac-1"  # existing signals kept


def test_frustrated_customer_escalates_complaint_action():
    erp = MockERP()
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "file_complaint", order_id="ORD-4821",
                        note="delivery is late again")]),
        reply("Understood — let me note that down."),
        reply(calls=[tc("t2", "file_complaint", order_id="ORD-4821",
                        note="delivery is late again")]),
        reply("I'm so sorry — connecting you to a human specialist now."),
    ])
    orch = make_orch(brain,
                     PolicyEngine({"complaint": {"allow": True,
                                                 "escalate_when": {
                                                     "frustrated": True}}}),
                     {"file_complaint": {"action": "complaint"}},
                     erp=erp)
    # calm customer: the complaint action is ALLOWED, no escalation
    calm = orch.handle_turn("s-frac-2", "My delivery is a day late, can you check?",
                            profile=CallerProfile(authenticated=True))
    assert calm.actions[0]["verdict"] == "ALLOW" and not calm.escalated
    # frustrated customer: same action escalates straight to a human
    upset = orch.handle_turn("s-frac-3",
                             "This is ridiculous!! I have called four times — "
                             "worst service, I demand a solution.",
                             profile=CallerProfile(authenticated=True))
    assert upset.actions[0]["verdict"] == "ESCALATE"
    assert upset.escalated is True
    assert erp.get_order("ORD-4821")["status"] == "CONFIRMED"  # nothing executed


# --- policy rules -------------------------------------------------------------

def test_policy_complaint_escalates_on_frustrated_signal():
    eng = PolicyEngine({"complaint": {"allow": True,
                                      "escalate_when": {"frustrated": True}}})
    d = eng.evaluate("complaint", PolicyContext(
        signals={"frustrated": True, "frustration_level": "high"}))
    assert d.verdict == "ESCALATE"


def test_real_yaml_escalation_and_return_rules():
    eng = PolicyEngine(load_policies("data/policies/policies.yaml"))
    assert eng.evaluate("escalate_to_human").verdict == "ALLOW"
    assert eng.evaluate("initiate_return",
                        PolicyContext(authenticated=False)).verdict == "REQUIRE_AUTH"
    assert eng.evaluate("initiate_return",
                        PolicyContext(authenticated=True)).verdict == "ALLOW"


# --- regression pins: existing governed flows unchanged -----------------------

def test_gateway_unknown_tool_still_errors():
    gw = ToolGateway()
    r = gw.execute("nope", {})
    assert not r.ok and "unknown_tool" in r.error


def test_reschedule_confirmed_order_still_allowed_with_real_policies():
    erp = MockERP()
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "reschedule_delivery",
                        order_id="ORD-4821", new_date="2026-09-10")]),
        reply("Your delivery is now set for September 10."),
    ])
    orch = make_orch(brain,
                     PolicyEngine(load_policies("data/policies/policies.yaml")),
                     {"reschedule_delivery": {"action": "reschedule_delivery"}},
                     erp=erp)
    result = orch.handle_turn("s-reg-1", "Please move my delivery to the 10th",
                              profile=CallerProfile(authenticated=True))
    assert result.actions[0]["verdict"] == "ALLOW"
    assert result.actions[0]["ok"] is True
    assert erp.get_order("ORD-4821")["delivery_date"] == "2026-09-10"
    assert not result.escalated


# --- live runner deployment smoke --------------------------------------------

def test_live_runner_deployment_imports_and_compiles(monkeypatch):
    path = (Path(__file__).resolve().parents[1] / "scripts"
            / "live_conversation.py")
    spec = importlib.util.spec_from_file_location(
        "live_conversation_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # compiles + imports the runner module
    monkeypatch.setenv("VOICEAGENT_FRONTIER_URL", "http://stub.local/v1")
    monkeypatch.setenv("VOICEAGENT_FRONTIER_MODEL", "stub-model")
    # no network: FrontierClient only stores its config at construction
    orch = mod.build_orchestrator()
    names = {s["function"]["name"] for s in orch.brain.tool_schemas()}
    assert {"fetch_order_status", "reschedule_delivery", "cancel_order",
            "escalate_to_human", "initiate_return"} <= names
    # hardened prompt: no invented URLs; only real actions promised; upset
    # customers are routed to escalate_to_human
    system = orch.brain._system_prompt
    assert "never invent" in system.lower()
    assert "WhatsApp" in system
    assert "escalate_to_human" in system
    # the runner is governed by the real policy file (new rules present)
    rules = orch.runner.policy.policies
    assert rules.get("escalate_to_human") == {"allow": True}
    assert rules.get("initiate_return") == {"require_auth": True}
