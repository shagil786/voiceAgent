# tests/test_chat.py — run_turn now drives the governed Orchestrator.
"""run_turn passes a turn through an object with `handle_turn(session_id,
user_text, *, authenticated)`; the governed Orchestrator is the production brain.
These tests use a fake that implements that exact contract so we exercise the
run_turn glue offline (no network, no frontier)."""
from voiceagent.chat import run_turn


class FakeOrch:
    def __init__(self):
        self.calls = 0
    def handle_turn(self, session_id, user_text, *, authenticated=False,
                    amount=None, contact_alias=None):
        self.calls += 1
        return type("R", (), {
            "reply": "Your order ORD-5 is on the way.",
            "actions": [{
                "tool": "fetch_order_status", "verdict": "ALLOW",
                "ok": True, "value": {"status": "shipped"},
                "reasons": ["order_status allowed by policy"],
            }],
            "brain_latency_s": 0.01, "session_id": session_id,
            "raw_tool_calls": 1, "escalated": False,
        })()


def test_run_turn_returns_reply_action_decision():
    orch = FakeOrch()
    out = run_turn(orch, "where is ORD-5", authenticated=True)
    assert out["reply"] == "Your order ORD-5 is on the way."
    assert out["action"] == "fetch_order_status"
    assert out["decision"] == "ALLOW"
    assert out["reasons"] == ["order_status allowed by policy"]
    assert out["executed"] is True
    assert orch.calls == 1


def test_run_turn_cap_returns_escalate_without_brain():
    orch = FakeOrch()
    mem = type("M", (), {"history": lambda c, n=None: ["x"] * 50})()
    out = run_turn(orch, "hi", memory=mem)
    assert out["decision"] == "ESCALATE"
    assert orch.calls == 0


class StrictOrch:
    """The REAL Orchestrator contract: handle_turn has NO amount kwarg
    (policy amounts come from tool params, not the turn)."""

    def __init__(self):
        self.seen_kwargs = None

    def handle_turn(self, session_id, user_text, *, authenticated=None,
                    contact_alias=None):
        self.seen_kwargs = {"authenticated": authenticated,
                            "contact_alias": contact_alias}
        return type("R", (), {"reply": "ok", "actions": [],
                              "brain_latency_s": 0.0,
                              "session_id": session_id,
                              "raw_tool_calls": 0,
                              "escalated": False})()


def test_amount_is_never_forwarded_to_the_governed_brain():
    # Regression: forwarding amount to handle_turn raised TypeError on the
    # real Orchestrator (masked by a permissive test fake before).
    orch = StrictOrch()
    out = run_turn(orch, "refund my order", amount=100.0)
    assert out["reply"] == "ok"
    assert orch.seen_kwargs == {"authenticated": False, "contact_alias": None}


class RecordingMemory:
    """History store that records appends (the /api/history + cap seam)."""

    def __init__(self):
        self.turns: list = []

    def history(self, conv_id, last_n=None):
        return self.turns

    def append(self, conv_id, turn):
        self.turns.append(turn)


def test_memory_records_user_and_agent_turns_for_the_governed_path():
    orch = FakeOrch()
    mem = RecordingMemory()
    out = run_turn(orch, "where is ORD-5", authenticated=True, memory=mem,
                   conv_id="c1")
    roles = [t.role for t in mem.turns]
    assert roles == ["user", "agent"]
    assert mem.turns[0].text == "where is ORD-5"
    assert mem.turns[1].text == out["reply"]
    assert mem.turns[1].verdict == "ALLOW"
