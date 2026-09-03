# tests/test_orchestrator.py — the one-turn agent runtime loop.
"""Orchestrator: binds the frontier brain, governed tool runner, policy,
and memory into a single handle_turn loop. The brain is a stub client with
scripted FrontierReply sequences (round 1 tool calls, round 2 read-only,
round 3 final content) — no network anywhere in this suite."""
import json

from voiceagent.decisionlog import DecisionLog
from voiceagent.memory import InMemoryMemory
from voiceagent.orchestrator import Deployment, Orchestrator
from voiceagent.policy import PolicyEngine
from voiceagent.swarm.blackboard import BlackboardState, CallerProfile
from voiceagent.swarm.frontier import (
    FrontierAgentBridge,
    FrontierReply,
    FrontierToolCall,
)
from voiceagent.swarm.specialist import SpecialistSpec, SpecialistTool
from voiceagent.tools import GovernedToolRunner, MockERP, ToolGateway


# --- stubs & fixtures --------------------------------------------------------

def tc(call_id: str, name: str, **args) -> FrontierToolCall:
    return FrontierToolCall(id=call_id, name=name, arguments=args)


def reply(content: str | None = None, calls: list[FrontierToolCall] | None = None) -> FrontierReply:
    return FrontierReply(content=content, tool_calls=calls or [],
                         model="stub", latency_s=0.001, raw={})


class ScriptedBrain:
    """Stub FrontierClient: pops scripted FrontierReplies, captures chat args."""

    def __init__(self, replies: list[FrontierReply]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, tool_choice="auto",
             temperature=0.4, max_tokens=512) -> FrontierReply:
        self.calls.append({"messages": messages, "tools": tools,
                           "tool_choice": tool_choice})
        if self._replies:
            return self._replies.pop(0)
        return reply("(script exhausted)")


def tool_messages(captured_call: dict) -> list[dict]:
    return [m for m in captured_call["messages"] if m.get("role") == "tool"]


def sample_spec() -> SpecialistSpec:
    return SpecialistSpec(
        domain_id="real_estate",
        name="Property Closer",
        role_description="Books site visits for premium listings.",
        system_prompt="You close premium property deals.",
        tools=[SpecialistTool(
            name="book_site_visit",
            description="Book a site visit slot (read-only, safe to execute).",
            parameters={"type": "object",
                        "properties": {"listing_id": {"type": "string"}}},
            handler=lambda args: {"booked": True, **args},
        )],
    )


def make_deployment() -> Deployment:
    return Deployment(
        name="acme",
        system_prompt="You are Acme's voice agent.",
        specs=[sample_spec()],
        gateway_tools={
            "reschedule_delivery": {
                "action": "reschedule",
                "side_effects": True,
                "description": "Reschedule the delivery date for an order.",
                "parameters": {"type": "object",
                               "properties": {"order_id": {"type": "string"},
                                              "new_date": {"type": "string"}},
                               "required": ["order_id", "new_date"]},
            },
            "cancel_order": {
                "action": "order_cancellation",
                "side_effects": True,
                "description": "Cancel an order.",
                "parameters": {"type": "object",
                               "properties": {"order_id": {"type": "string"},
                                              "reason": {"type": "string"}},
                               "required": ["order_id", "reason"]},
            },
            "initiate_refund": {
                "action": "high_value_refund",
                "side_effects": True,
                "description": "Initiate a refund for an order.",
                "parameters": {"type": "object",
                               "properties": {"order_id": {"type": "string"},
                                              "amount": {"type": "number"},
                                              "reason": {"type": "string"}}},
            },
        },
        knowledge={"returns": "Returns are accepted within 7 days of delivery."},
    )


def make_orchestrator(brain: ScriptedBrain, replies_runner: GovernedToolRunner | None = None,
                      memory=None, max_tool_rounds: int = 3) -> Orchestrator:
    orch = Orchestrator(brain=FrontierAgentBridge(brain), runner=replies_runner,
                        memory=memory if memory is not None else InMemoryMemory(),
                        max_tool_rounds=max_tool_rounds)
    orch.deploy(make_deployment())
    return orch


# --- governed execution ------------------------------------------------------

def test_governed_tool_executes_on_allow_and_mutates_erp():
    erp, log = MockERP(), DecisionLog()
    runner = GovernedToolRunner(ToolGateway(erp=erp),
                                PolicyEngine({"reschedule": {"allow": True}}),
                                decision_log=log)
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "reschedule_delivery",
                        order_id="ORD-4821", new_date="2026-09-10")]),
        reply("Your delivery is now set for September 10."),
    ])
    orch = make_orchestrator(brain, runner)
    result = orch.handle_turn("s-1", "Move my delivery to the 10th",
                              profile=CallerProfile(customer_id="CUST-001",
                                                    authenticated=True))
    # ERP state actually mutated through the governed path
    assert erp.get_order("ORD-4821")["delivery_date"] == "2026-09-10"
    assert len(result.actions) == 1
    assert result.actions[0]["action"] == "reschedule"
    assert result.actions[0]["tool"] == "reschedule_delivery"
    assert result.actions[0]["verdict"] == "ALLOW" and result.actions[0]["ok"] is True
    assert result.actions[0]["value"]["delivery_date"] == "2026-09-10"
    assert result.reply == "Your delivery is now set for September 10."
    assert result.raw_tool_calls == 1 and not result.escalated
    assert result.brain_latency_s > 0.0 and result.session_id == "s-1"
    # decision logged under the session id
    entry = log.entries()[-1]
    assert entry.verdict == "ALLOW" and entry.action == "reschedule"
    assert entry.conv_id == "s-1" and entry.authenticated is True
    # result fed back to the brain as an OpenAI tool message
    msgs = tool_messages(brain.calls[1])
    assert [m["tool_call_id"] for m in msgs] == ["t1"]
    payload = json.loads(msgs[0]["content"])
    assert payload["ok"] is True and payload["value"]["delivery_date"] == "2026-09-10"


def test_unauthenticated_cancel_is_blocked_and_verdict_fed_back():
    erp, log = MockERP(), DecisionLog()
    runner = GovernedToolRunner(ToolGateway(erp=erp),
                                PolicyEngine({"order_cancellation":
                                              {"require_auth": True}}),
                                decision_log=log)
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "cancel_order",
                        order_id="ORD-4821", reason="changed my mind")]),
        reply("I can help with that, but first I need to verify your identity."),
    ])
    orch = make_orchestrator(brain, runner)
    result = orch.handle_turn("s-2", "Cancel my order",
                              profile=CallerProfile(authenticated=False))
    # nothing executed
    assert erp.get_order("ORD-4821")["status"] == "CONFIRMED"
    assert result.actions[0]["verdict"] == "REQUIRE_AUTH"
    assert result.actions[0]["ok"] is False and "value" not in result.actions[0]
    assert not result.escalated  # blocked, but NOT an escalation
    assert result.reply.startswith("I can help with that")
    # verdict + reasons fed back so the brain can explain / re-plan
    payload = json.loads(tool_messages(brain.calls[1])[0]["content"])
    assert payload["verdict"] == "REQUIRE_AUTH" and payload["ok"] is False
    assert payload["reasons"]
    # audited with the unauthenticated context
    entry = log.entries()[-1]
    assert entry.verdict == "REQUIRE_AUTH" and entry.authenticated is False
    assert orch._sessions["s-2"].profile.authenticated is False


def test_least_privilege_deny_when_no_policy_defined():
    erp = MockERP()
    # PolicyEngine({}) would fall back to DEFAULT_POLICIES (its ctor does
    # `policies or ...`), so pass a real rule set that omits cancellation.
    runner = GovernedToolRunner(ToolGateway(erp=erp),
                                PolicyEngine({"reschedule": {"allow": True}}))
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "cancel_order", order_id="ORD-4821", reason="x")]),
        reply("That action isn't available on this line."),
    ])
    orch = make_orchestrator(brain, runner)
    result = orch.handle_turn("s-3", "cancel it",
                              profile=CallerProfile(authenticated=True))
    assert result.actions[0]["verdict"] == "DENY"
    assert result.actions[0]["ok"] is False and not result.escalated
    assert erp.get_order("ORD-4821")["status"] == "CONFIRMED"


def test_escalate_sets_flag_and_refunds_stay_empty():
    erp, log = MockERP(), DecisionLog()
    runner = GovernedToolRunner(ToolGateway(erp=erp),
                                PolicyEngine({"high_value_refund":
                                              {"escalate": True}}),
                                decision_log=log)
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "initiate_refund", order_id="ORD-7734",
                        amount=6500.0, reason="damaged")]),
        reply("This needs a human specialist — I'm transferring you now."),
    ])
    orch = make_orchestrator(brain, runner)
    result = orch.handle_turn("s-4", "I want my refund",
                              profile=CallerProfile(authenticated=True))
    assert result.escalated is True
    assert result.actions[0]["verdict"] == "ESCALATE" and result.actions[0]["ok"] is False
    assert erp.refunds == []
    payload = json.loads(tool_messages(brain.calls[1])[0]["content"])
    assert payload["verdict"] == "ESCALATE" and payload["reasons"]


def test_governed_tool_without_runner_is_never_executed():
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "cancel_order", order_id="ORD-4821", reason="x")]),
        reply("I can't perform that right now."),
    ])
    orch = make_orchestrator(brain, None)  # no runner wired
    result = orch.handle_turn("s-5", "cancel it",
                              profile=CallerProfile(authenticated=True))
    assert result.actions[0]["verdict"] == "DENY" and result.actions[0]["ok"] is False
    payload = json.loads(tool_messages(brain.calls[1])[0]["content"])
    assert payload["ok"] is False


# --- read-only brain tools ---------------------------------------------------

def test_readonly_tool_uses_execute_call_and_feeds_result_back():
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "book_site_visit", listing_id="L1")]),
        reply("Your site visit is booked."),
    ])
    orch = make_orchestrator(brain)
    result = orch.handle_turn("s-6", "Can I visit the site?")
    assert result.actions == []  # read-only tools are not governed actions
    assert result.raw_tool_calls == 1 and not result.escalated
    payload = json.loads(tool_messages(brain.calls[1])[0]["content"])
    assert payload == {"ok": True, "value": {"booked": True, "listing_id": "L1"}}


def test_unknown_tool_call_is_surfaced_back_not_dropped():
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "does_not_exist", x=1)]),
        reply("Sorry, let me rephrase."),
    ])
    orch = make_orchestrator(brain)
    result = orch.handle_turn("s-7", "do the thing")
    payload = json.loads(tool_messages(brain.calls[1])[0]["content"])
    assert payload["ok"] is False and "does_not_exist" in payload["error"]
    assert result.raw_tool_calls == 1 and result.actions == []
    assert result.reply == "Sorry, let me rephrase."


# --- the multi-round loop ----------------------------------------------------

def test_multi_round_governed_then_readonly_then_final():
    erp = MockERP()
    runner = GovernedToolRunner(ToolGateway(erp=erp),
                                PolicyEngine({"reschedule": {"allow": True}}))
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "reschedule_delivery",
                        order_id="ORD-4821", new_date="2026-09-10")]),
        reply(calls=[tc("t2", "book_site_visit", listing_id="L1")]),
        reply("All set — delivery moved and visit booked."),
    ])
    orch = make_orchestrator(brain, runner)
    result = orch.handle_turn("s-8", "Move my delivery and book a visit",
                              profile=CallerProfile(authenticated=True))
    assert len(brain.calls) == 3  # round1 governed, round2 read-only, round3 final
    assert result.reply == "All set — delivery moved and visit booked."
    assert result.raw_tool_calls == 2 and not result.escalated
    # both results were fed back before the final call
    tool_ids = [m["tool_call_id"] for m in tool_messages(brain.calls[2])]
    assert tool_ids == ["t1", "t2"]
    assert erp.get_order("ORD-4821")["delivery_date"] == "2026-09-10"


def test_tool_rounds_bounded_and_final_reply_still_produced():
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "book_site_visit", listing_id="L1")]),
        reply(calls=[tc("t2", "book_site_visit", listing_id="L2")]),
        reply(calls=[tc("t3", "book_site_visit", listing_id="L3")]),  # over the bound
        reply("Here is your answer."),
    ])
    orch = make_orchestrator(brain, max_tool_rounds=2)
    result = orch.handle_turn("s-9", "go")
    assert len(brain.calls) == 4  # 2 tool rounds + 1 over-bound round + forced close
    assert brain.calls[3]["tools"] is None  # forced close strips the tool surface
    assert result.reply == "Here is your answer."
    assert result.raw_tool_calls == 3  # 2 executed + 1 unexecuted stray


def test_final_reply_always_produced_even_when_brain_keeps_calling_tools():
    brain = ScriptedBrain([
        reply(calls=[tc("t1", "book_site_visit", listing_id="L1")]),
        reply(calls=[tc("t2", "book_site_visit", listing_id="L2")]),
        reply(None, calls=[tc("t3", "book_site_visit", listing_id="L3")]),
    ])
    orch = make_orchestrator(brain, max_tool_rounds=1)
    result = orch.handle_turn("s-10", "go")
    assert result.reply.strip()  # deterministic fallback, never empty
    assert len(brain.calls) == 3


# --- memory & session state --------------------------------------------------

def test_memory_records_turns_and_history_grows_across_turns():
    memory = InMemoryMemory()
    brain = ScriptedBrain([reply("Hi! How can I help?"),
                           reply("Anything else I can do?")])
    orch = make_orchestrator(brain, memory=memory)
    orch.handle_turn("s-11", "hello")
    orch.handle_turn("s-11", "tell me more")
    turns = memory.history("s-11")
    assert [(t.role, t.text) for t in turns] == [
        ("user", "hello"), ("agent", "Hi! How can I help?"),
        ("user", "tell me more"), ("agent", "Anything else I can do?")]
    # the second turn's prompt carried the first turn's history
    assert [m["role"] for m in brain.calls[1]["messages"]] == [
        "system", "user", "assistant", "user"]
    assert orch._sessions["s-11"].history[-1]["text"] == "Anything else I can do?"


def test_sessions_are_isolated_per_session_id():
    brain = ScriptedBrain([reply("A"), reply("B")])
    orch = make_orchestrator(brain)
    orch.handle_turn("sess-a", "one")
    orch.handle_turn("sess-b", "two")
    assert set(orch._sessions) == {"sess-a", "sess-b"}
    assert [m["role"] for m in brain.calls[1]["messages"]] == ["system", "user"]
    # memory kept the streams apart
    assert len(orch.memory.history("sess-a")) == 2
    assert len(orch.memory.history("sess-b")) == 2


# --- deploy ------------------------------------------------------------------

def test_deploy_registers_spec_tools_gateway_tools_and_knowledge():
    brain = FrontierAgentBridge(ScriptedBrain([]))
    orch = Orchestrator(brain=brain, memory=InMemoryMemory())
    deployment = make_deployment()
    orch.deploy(deployment)
    names = {s["function"]["name"] for s in brain.tool_schemas()}
    assert {"book_site_visit", "reschedule_delivery",
            "cancel_order", "initiate_refund"} <= names
    # governed tools are visible to the brain but NEVER brain-executed
    assert "reschedule_delivery" not in brain._handlers
    assert "cancel_order" not in brain._handlers
    assert "book_site_visit" in brain._handlers  # spec handler kept for read-only
    system = brain.build_messages(BlackboardState(session_id="x"), "hi")[0]["content"]
    assert deployment.system_prompt in system
    assert "Returns are accepted within 7 days of delivery." in system
    assert "Property Closer (real_estate)" in system  # specialist role block


# --- outbound campaign placement ---------------------------------------------

def test_campaign_turn_injects_goal_and_lead_context():
    brain = ScriptedBrain([
        reply("Hi Rohan, this is Acme — a quick one about your loan offer."),
    ])
    orch = make_orchestrator(brain)
    result = orch.campaign_turn(
        "camp-1",
        lead={"name": "Rohan", "city": "Pune", "offer": "pre-approved 5L"},
        script_goal="Book a callback for the pre-approved loan offer")
    system = brain.calls[0]["messages"][0]["content"]
    assert "Book a callback for the pre-approved loan offer" in system
    assert "Rohan" in system and "pre-approved 5L" in system
    assert "You are Acme's voice agent." in system  # base prompt preserved
    assert result.reply.startswith("Hi Rohan")
    assert result.session_id == "camp-1"
    assert len(orch.memory.history("camp-1")) == 2
