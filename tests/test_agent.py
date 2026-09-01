# tests/test_agent.py
from voiceagent.agent import build_agent, extract_action, AgentResult
from voiceagent.llm import LLMHandle

class FakeLLM(LLMHandle):
    def __init__(self, reply=("Your order ORD-77812 is out for delivery.\n"
                              "ACTION: order_status")):
        super().__init__({"model": "fake"})
        self.reply = reply
    def generate(self, prompt, max_tokens=256, stop=None):
        return self.reply

class FakeClassifier:
    def classify(self, text):
        return ("order_status", 1.0)

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

# ---------------------------------------------------------------------------
# M5c Fix 1: the ACTION scaffolding line never reaches the customer. The
# action decision comes from the classifier / fallback extract_action(), so
# scrubbing must happen after extraction and after the echo guardrail.
# ---------------------------------------------------------------------------

def test_action_line_stripped_from_reply_end():
    # Live-observed shape: reply ends with an ACTION line.
    agent = build_agent(FakeIndex(), FakeLLM())
    res = agent.handle("where is my order ORD-77812")
    assert "ACTION" not in res.text
    assert res.text == "Your order ORD-77812 is out for delivery."
    assert res.action == "order_status"  # fallback extraction keeps working

def test_action_line_mid_text_stripped_and_blank_runs_collapsed():
    agent = build_agent(FakeIndex(), FakeLLM(
        reply="Answer one.\n\nACTION: refund\n\nAnswer two."))
    res = agent.handle("I need a refund")
    assert "ACTION" not in res.text
    assert res.text == "Answer one.\n\nAnswer two."
    assert res.action == "refund"

def test_action_first_reply_scrubbed():
    agent = build_agent(FakeIndex(), FakeLLM(
        reply="ACTION: refund\nYour refund for ORD-77812 is done."))
    res = agent.handle("I need a refund for ORD-77812")
    assert res.text == "Your refund for ORD-77812 is done."
    assert res.action == "refund"

def test_scrub_keeps_guardrail_confirm_sentence():
    # Classifier path: the echo guardrail runs first (prepends the confirm
    # for the order id the LLM dropped), THEN the ACTION line is scrubbed.
    from voiceagent.agent import Agent
    agent = Agent(FakeIndex(),
                  FakeLLM("Your request is being handled.\nACTION: order_status"),
                  classifier=FakeClassifier())
    res = agent.handle("where is my order ORD-1234")
    assert "ACTION" not in res.text
    assert "I understand" in res.text          # confirm still present
    assert "ORD-1234" in res.text              # guardrail echo still present
    assert "being handled" in res.text         # original reply kept

def test_scrub_when_reply_starts_with_action_line():
    from voiceagent.agent import Agent
    agent = Agent(FakeIndex(), FakeLLM(
        reply="ACTION: order_status\nYour order is on the way."),
        classifier=FakeClassifier())
    res = agent.handle("where is my order ORD-1234")
    assert "ACTION" not in res.text
    assert "I understand" in res.text
    assert "on the way" in res.text

# ---------------------------------------------------------------------------
# M5c Fix 2: informational intents reach the system prompt's action list.
# ---------------------------------------------------------------------------

def test_informational_actions_in_default_action_list():
    from voiceagent.agent import DEFAULT_ACTIONS
    assert "refund_info" in DEFAULT_ACTIONS
    assert "delivery_eta" in DEFAULT_ACTIONS
