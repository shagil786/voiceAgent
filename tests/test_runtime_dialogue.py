# tests/test_runtime_dialogue.py
"""End-to-end integration tests for Sprint A runtime wiring:
DialogueTracker + GovernedToolRunner + MockERP running through run_turn().
"""
import datetime
import pytest

from voiceagent.agent import Agent, build_agent
from voiceagent.chat import run_turn
from voiceagent.decisionlog import DecisionLog
from voiceagent.intent import IntentClassifier
from voiceagent.memory import InMemoryMemory
from voiceagent.policy import PolicyEngine, load_policies
from voiceagent.tools import MockERP, ToolGateway, GovernedToolRunner


class FakeIndex:
    def search(self, query: str, k: int = 3):
        return [{"id": "faq-1", "text": "Deliveries occur between 9am and 7pm.",
                 "section": "Delivery", "score": 0.9}]


class FakeLLM:
    def __init__(self):
        self.specs = {"model": "fake", "params": "0.5B"}

    def generate(self, prompt: str, max_tokens: int = 256, stop=None) -> str:
        return "I can assist you with your request."


@pytest.fixture
def runtime_agent():
    """Build agent wired with MockERP, ToolGateway, and GovernedToolRunner."""
    erp = MockERP()
    policy = PolicyEngine(load_policies("data/policies/policies.yaml"))
    log = DecisionLog()
    gateway = ToolGateway(erp=erp)
    runner = GovernedToolRunner(gateway, policy, decision_log=log)
    classifier = IntentClassifier()
    agent = build_agent(
        index=FakeIndex(),
        llm=FakeLLM(),
        classifier=classifier,
        policy=policy.policies,
        decision_log=log,
        tool_runner=runner,
        erp=erp,
    )
    return agent, erp, log


def test_flagship_reschedule_flow_with_garbled_order_snapping(runtime_agent):
    """Flagship 4-turn reschedule flow:
    Turn 1: Request reschedule -> Asks for order ID
    Turn 2: Garbled order ID ('or D7734') snaps against CUST-001's known orders -> Asks for date
    Turn 3: Relative date 'tomorrow' -> Asks for confirmation
    Turn 4: 'yes please' -> Executes reschedule on MockERP, updates delivery date!
    """
    agent, erp, log = runtime_agent
    mem = InMemoryMemory()
    conv = "reschedule-flagship-1"

    # Turn 1: Initiate reschedule
    t1 = run_turn(agent, "I want to reschedule my delivery", conv_id=conv, memory=mem)
    assert t1["action"] == "reschedule_delivery"
    assert t1["directive"] == "ASK_SLOT"
    assert "order ID" in t1["reply"]

    # Turn 2: Provide order ID with ASR garble ("or D7734")
    t2 = run_turn(agent, "for order or D7734", conv_id=conv, memory=mem)
    assert t2["directive"] == "ASK_SLOT"
    assert "date" in t2["reply"].lower()

    # Turn 3: Provide relative date
    t3 = run_turn(agent, "tomorrow", conv_id=conv, memory=mem)
    assert t3["directive"] == "CONFIRM_ACTION"
    assert "ORD-7734" in t3["reply"]
    expected_date = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    assert expected_date in t3["reply"]

    # Turn 4: Confirmation gate -> Execution
    t4 = run_turn(agent, "yes please", conv_id=conv, memory=mem, authenticated=True)
    assert t4["directive"] == "EXECUTE_READY"
    assert t4["executed"] is True
    assert t4["tool_result"]["delivery_date"] == expected_date
    assert "rescheduled" in t4["reply"].lower()

    # Verify MockERP was actually mutated!
    updated_order = erp.get_order("ORD-7734")
    assert updated_order["delivery_date"] == expected_date


def test_cancel_order_precondition_blocks_shipped_order(runtime_agent):
    """Attempting to cancel ORD-7734 (which is SHIPPED) must be blocked
    by the ToolGateway precondition, explaining order has shipped."""
    agent, erp, log = runtime_agent
    mem = InMemoryMemory()
    conv = "cancel-shipped-1"

    # Turn 1: Request cancellation with order ID
    t1 = run_turn(agent, "cancel my order ORD-7734", conv_id=conv, memory=mem)
    assert t1["action"] == "cancel_order"
    # Still needs reason
    assert t1["directive"] == "ASK_SLOT"
    assert "why" in t1["reply"].lower() or "reason" in t1["reply"].lower()

    # Turn 2: Give reason, authenticated session
    t2 = run_turn(agent, "it was delayed", conv_id=conv, memory=mem, authenticated=True)
    assert t2["directive"] == "CONFIRM_ACTION"
    assert "ORD-7734" in t2["reply"]

    # Turn 3: Confirm
    t3 = run_turn(agent, "yes", conv_id=conv, memory=mem, authenticated=True)
    assert t3["directive"] == "EXECUTE_READY"
    assert t3["executed"] is False
    assert "shipped" in t3["reply"].lower()

    # Order must NOT be cancelled
    assert erp.get_order("ORD-7734")["status"] == "SHIPPED"


def test_cancel_order_succeeds_on_confirmed_order(runtime_agent):
    """Cancelling a CONFIRMED order (ORD-4821) succeeds and updates ERP."""
    agent, erp, log = runtime_agent
    mem = InMemoryMemory()
    conv = "cancel-confirmed-1"

    t1 = run_turn(agent, "cancel my order ORD-4821", conv_id=conv, memory=mem)
    t2 = run_turn(agent, "damaged item", conv_id=conv, memory=mem, authenticated=True)
    t3 = run_turn(agent, "yes", conv_id=conv, memory=mem, authenticated=True)

    assert t3["executed"] is True
    assert "cancelled successfully" in t3["reply"].lower()
    assert erp.get_order("ORD-4821")["status"] == "CANCELLED"


def test_cancel_order_blocked_when_unauthenticated(runtime_agent):
    """Policy requires authentication for cancellation."""
    agent, erp, log = runtime_agent
    mem = InMemoryMemory()
    conv = "cancel-no-auth"

    t1 = run_turn(agent, "cancel my order ORD-4821", conv_id=conv, memory=mem)
    t2 = run_turn(agent, "defective", conv_id=conv, memory=mem, authenticated=False)
    # Auth is a required slot for cancel_order; since authenticated=False, it asks for auth
    assert t2["directive"] == "ASK_SLOT"
    assert "otp" in t2["reply"].lower() or "identity" in t2["reply"].lower()


def test_customer_declines_confirmation_escalates(runtime_agent):
    """If customer declines confirmation ('no stop'), system escalates to human."""
    agent, erp, log = runtime_agent
    mem = InMemoryMemory()
    conv = "reschedule-decline"

    run_turn(agent, "reschedule delivery for order ORD-4821", conv_id=conv, memory=mem)
    run_turn(agent, "tomorrow", conv_id=conv, memory=mem)
    t3 = run_turn(agent, "no stop cancel that", conv_id=conv, memory=mem)

    assert t3["directive"] == "ESCALATE_TO_HUMAN"
    assert t3["decision"] == "ESCALATE"
    assert "human" in t3["reply"].lower()


def test_hinglish_multi_turn_reschedule(runtime_agent):
    """Multi-turn reschedule in Hinglish with localized responses."""
    agent, erp, log = runtime_agent
    mem = InMemoryMemory()
    conv = "reschedule-hinglish"

    t1 = run_turn(agent, "mera order reschedule karna hai", conv_id=conv, memory=mem)
    assert "order ID" in t1["reply"]

    t2 = run_turn(agent, "order ORD-4821 hai", conv_id=conv, memory=mem)
    assert "date" in t2["reply"].lower() or "tarikh" in t2["reply"].lower()

    t3 = run_turn(agent, "kal deliver karna", conv_id=conv, memory=mem)
    assert "confirm" in t3["reply"].lower()

    t4 = run_turn(agent, "haanji", conv_id=conv, memory=mem, authenticated=True)
    assert t4["executed"] is True
    assert "reschedule ho gayi" in t4["reply"] or "rescheduled" in t4["reply"]
