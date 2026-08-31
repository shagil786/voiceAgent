# tests/test_chat.py
from voiceagent.chat import run_turn

class FakeAgent:
    def __init__(self):
        self.calls = 0
    def handle(self, user_text, authenticated=False, amount=None, conv_id=""):
        self.calls += 1
        return type("R", (), {
            "text": "Your order ORD-5 is on the way.",
            "action": "order_status",
            "decision": type("D", (), {"verdict": "ALLOW", "reasons": ["ok"]})(),
        })()

def test_run_turn_returns_reply_action_decision():
    agent = FakeAgent()
    out = run_turn(agent, "where is ORD-5", authenticated=True)
    assert out["reply"] == "Your order ORD-5 is on the way."
    assert out["action"] == "order_status"
    assert out["decision"] == "ALLOW"
    assert out["reasons"] == ["ok"]
    assert agent.calls == 1