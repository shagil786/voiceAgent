# tests/test_agent.py
from voiceagent.agent import build_agent, extract_action, AgentResult
from voiceagent.llm import LLMHandle

class FakeLLM(LLMHandle):
    def __init__(self):
        super().__init__({"model": "fake"})
    def generate(self, prompt, max_tokens=256, stop=None):
        return ("Your order ORD-77812 is out for delivery.\n"
                "ACTION: order_status")

class FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "Order status can be checked with the order id.",
                "section": "faqs", "score": 0.9}]

def test_extract_action_parses_action_line():
    assert extract_action("foo\nACTION: refund\nbar") == "refund"
    assert extract_action("no action here") is None

def test_agent_returns_text_action_and_retrieved():
    agent = build_agent(FakeIndex(), FakeLLM())
    res = agent.handle("where is my order ORD-77812")
    assert isinstance(res, AgentResult)
    assert "out for delivery" in res.text
    assert res.action == "order_status"
    assert len(res.retrieved) == 1
    assert res.latency_s >= 0
