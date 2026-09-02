# tests/test_dialogue.py — Sprint A WS3: deterministic slot tracking.
from voiceagent.dialogue import DialogueTracker
from voiceagent.entities import extract_entities
from voiceagent.policy import PolicyContext, PolicyEngine
from voiceagent.tools import GovernedToolRunner, MockERP, ToolGateway

CANDIDATES = ["ORD-4821", "ORD-7734"]


def test_order_status_single_slot_flow():
    tr = DialogueTracker("order_status")
    tr.update("I want to check my delivery status", extract_entities("x"))
    d = tr.next_step()
    assert d.kind == "ASK_SLOT" and d.slot == "order_id" and d.hint

    tr.update("it is ORD-4821", extract_entities("it is ORD-4821"))
    d = tr.next_step()
    assert d.kind == "EXECUTE_READY"
    assert d.payload == {"order_id": "ORD-4821"}


def test_reschedule_multi_turn_flagship_flow():
    """The task's verification scenario: garbled ASR reference snaps, slots
    fill deterministically, confirmation gates execution."""
    tr = DialogueTracker("reschedule_delivery", candidate_orders=CANDIDATES)

    # Turn 1
    tr.update("I want to reschedule my delivery", extract_entities("x"))
    d = tr.next_step()
    assert d.kind == "ASK_SLOT" and d.slot == "order_id"

    # Turn 2: ASR garble 'or D7734' snaps to ORD-7734 via known candidates
    text2 = "For order or D7734"
    tr.update(text2, extract_entities(text2))
    d = tr.next_step()
    assert d.kind == "ASK_SLOT" and d.slot == "new_date"
    assert tr.slots["order_id"].value == "ORD-7734"
    assert tr.slots["order_id"].status == "FILLED"

    # Turn 3: spoken date phrase fills the slot, then confirmation gate
    text3 = "Tomorrow afternoon"
    tr.update(text3, extract_entities(text3))
    d = tr.next_step()
    assert d.kind == "CONFIRM_ACTION"
    assert d.action == "reschedule_delivery"
    assert "ORD-7734" in d.details

    # Turn 4: explicit reconfirmation -> EXECUTE_READY
    tr.update("Yes please", extract_entities("yes"))
    d = tr.next_step()
    assert d.kind == "EXECUTE_READY"
    assert d.payload == {"order_id": "ORD-7734",
                         "new_date": tr.slots["new_date"].value}

    # And the governed execution actually mutates the ERP.
    import datetime
    expected = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    assert tr.slots["new_date"].value == expected
    runner = GovernedToolRunner(
        ToolGateway(erp=MockERP()),
        PolicyEngine({"reschedule_delivery": {"allow": True}}))
    out = runner.run("reschedule_delivery", PolicyContext(),
                     "reschedule_delivery", d.payload, idempotency_key="r-1")
    assert out.executed and out.result.ok
    assert out.result.value["delivery_date"] == expected


def test_declined_confirmation_escalates():
    tr = DialogueTracker("reschedule_delivery", candidate_orders=CANDIDATES)
    tr.update("reschedule order ORD-4821 to tomorrow", extract_entities("ORD-4821"))
    assert tr.next_step().kind == "CONFIRM_ACTION"
    tr.update("no, forget it", extract_entities("x"))
    d = tr.next_step()
    assert d.kind == "ESCALATE_TO_HUMAN"
    assert "declined" in d.reason


def test_refund_workflow_needs_auth_before_execution():
    tr = DialogueTracker("refund")
    tr.update("I want a refund of rupees five thousand for ORD-4821, it arrived damaged",
              extract_entities("refund of ₹5000 ORD-4821 damaged"))
    d = tr.next_step()
    assert d.kind == "ASK_SLOT" and d.slot == "auth_verified"
    tr.update("ok", extract_entities("x"), auth_state=True)
    d = tr.next_step()
    assert d.kind == "CONFIRM_ACTION"
    tr.update("yes", extract_entities("x"))
    d = tr.next_step()
    assert d.kind == "EXECUTE_READY"
    assert d.payload == {"order_id": "ORD-4821", "amount": 5000.0,
                         "reason": "damaged", "auth_verified": True}


def test_hindi_number_amount_and_garbled_id_fill_slots():
    tr = DialogueTracker("refund", candidate_orders=CANDIDATES)
    tr.update("मुझे 6 हजार का रिफंड चाहिए order or D7734 के लिए",
              extract_entities("मुझे 6 हजार का रिफंड चाहिए"))
    assert tr.slots["amount"].value == 6000.0
    assert tr.slots["order_id"].value == "ORD-7734"  # snapped


def test_unknown_workflow_rejected():
    import pytest
    with pytest.raises(ValueError):
        DialogueTracker("sell kidneys")
