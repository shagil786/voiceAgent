# tests/test_memory.py — M4a short-term working memory: stores, agent
# history replay, and run_turn integration.
import sqlite3
import threading

import pytest

from voiceagent.memory import (ConversationMemory, InMemoryMemory, SQLiteMemory,
                               Turn, public_dict)


def test_turn_defaults():
    t = Turn(ts="2026-09-02T10:00:00", role="user", text="hi")
    assert t.action is None
    assert t.verdict is None
    assert t.refs == []


def test_memory_satisfies_protocol():
    assert isinstance(InMemoryMemory(), ConversationMemory)


def test_inmemory_roundtrip_preserves_fields():
    m = InMemoryMemory()
    m.append("c1", Turn(ts="t1", role="user", text="where is ORD-1234",
                        refs=["ORD-1234"]))
    m.append("c1", Turn(ts="t2", role="agent", text="on the way",
                        action="order_status", verdict="ALLOW"))
    turns = m.history("c1")
    assert [t.role for t in turns] == ["user", "agent"]
    assert turns[0].refs == ["ORD-1234"]
    assert turns[1].action == "order_status"
    assert turns[1].verdict == "ALLOW"


def test_inmemory_last_n_returns_newest():
    m = InMemoryMemory()
    for i in range(3):
        m.append("c1", Turn(ts=f"t{i}", role="user", text=f"m{i}"))
    assert [t.text for t in m.history("c1", last_n=2)] == ["m1", "m2"]
    assert [t.text for t in m.history("c1", last_n=99)] == ["m0", "m1", "m2"]


def test_inmemory_unknown_conv_is_empty():
    assert InMemoryMemory().history("nope") == []


def test_inmemory_maxlen_evicts_oldest():
    m = InMemoryMemory(maxlen_per_conv=3)
    for i in range(5):
        m.append("c1", Turn(ts=f"t{i}", role="user", text=f"m{i}"))
    turns = m.history("c1")
    assert len(turns) == 3
    assert [t.text for t in turns] == ["m2", "m3", "m4"]


def test_inmemory_clear():
    m = InMemoryMemory()
    m.append("c1", Turn(ts="t", role="user", text="m"))
    m.clear("c1")
    assert m.history("c1") == []


def test_sqlite_roundtrip_and_persistence(tmp_path):
    db = str(tmp_path / "memory.db")
    m = SQLiteMemory(db)
    m.append("c1", Turn(ts="t1", role="user", text="where is ORD-1234",
                        refs=["ORD-1234"]))
    m.append("c1", Turn(ts="t2", role="agent", text="on the way",
                        action="order_status", verdict="REQUIRE_AUTH"))
    # A new instance on the same file sees the stored turns (refs survive
    # the JSON round-trip).
    turns = SQLiteMemory(db).history("c1")
    assert [t.role for t in turns] == ["user", "agent"]
    assert turns[0].refs == ["ORD-1234"]
    assert turns[1].verdict == "REQUIRE_AUTH"
    assert turns[1].action == "order_status"


def test_sqlite_last_n_and_clear(tmp_path):
    m = SQLiteMemory(str(tmp_path / "memory.db"))
    for i in range(3):
        m.append("c1", Turn(ts=f"t{i}", role="user", text=f"m{i}"))
    assert [t.text for t in m.history("c1", last_n=2)] == ["m1", "m2"]
    m.clear("c1")
    assert m.history("c1") == []
    assert m.history("never-existed") == []


def test_sqlite_wal_mode_and_conv_index(tmp_path):
    db = str(tmp_path / "memory.db")
    SQLiteMemory(db)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        indexes = conn.execute("PRAGMA index_list('turns')").fetchall()
        cols = [r[2] for r in conn.execute("PRAGMA index_info(idx_turns_conv)")]
        assert any(ix[1] == "idx_turns_conv" for ix in indexes)
        assert cols == ["conv_id"]
    finally:
        conn.close()


def test_sqlite_thread_safe_appends(tmp_path):
    m = SQLiteMemory(str(tmp_path / "memory.db"))

    def worker(n):
        for i in range(n):
            m.append("c1", Turn(ts="t", role="user", text=f"w{i}"))

    threads = [threading.Thread(target=worker, args=(25,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(m.history("c1")) == 200


def test_public_dict_is_the_api_shape():
    d = public_dict(Turn(ts="t1", role="agent", text="hi",
                         action="order_status", verdict="ALLOW",
                         refs=["ORD-1"]))
    assert d == {"ts": "t1", "role": "agent", "text": "hi",
                 "action": "order_status", "verdict": "ALLOW"}


# ---------------------------------------------------------------------------
# Agent integration: history replay into the prompt + reference inheritance
# ---------------------------------------------------------------------------

from voiceagent.agent import Agent, build_agent, find_order_id  # noqa: E402
from voiceagent.llm import LLMHandle  # noqa: E402


class CapturingLLM(LLMHandle):
    def __init__(self, reply="It is out for delivery.\nACTION: order_status"):
        super().__init__({"model": "fake"})
        self.prompts: list[str] = []
        self.reply = reply

    def generate(self, prompt, max_tokens=256, stop=None):
        self.prompts.append(prompt)
        return self.reply


class TemplateCapturingLLM(CapturingLLM):
    def chat_template(self, system, context, user_text):
        self.last_template_args = (system, context, user_text)
        return f"SYS<<{system}>>CTX<<{context}>>USER<<{user_text}>>"


class FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "Order status needs the order id.",
                 "section": "faqs", "score": 0.9}]


class FakeClassifier:
    def classify(self, text):
        return ("order_status", 1.0)


def _exchange(i):
    return [Turn(ts=f"t{i}u", role="user", text=f"message {i}"),
            Turn(ts=f"t{i}a", role="agent", text=f"reply {i}")]


def test_find_order_id_helper():
    assert find_order_id("track ORD-1234 please") == "ORD-1234"
    assert find_order_id("order 77812 status") == "77812"
    assert find_order_id("no refs here") is None


def test_handle_without_history_prompt_is_byte_identical():
    agent = Agent(FakeIndex(), CapturingLLM(), classifier=FakeClassifier())
    agent.handle("where is my order")
    agent.handle("where is my order", history=None)
    agent.handle("where is my order", history=[])
    assert agent._llm.prompts[0] == agent._llm.prompts[1]
    assert agent._llm.prompts[1] == agent._llm.prompts[2]


def test_handle_renders_history_between_context_and_customer():
    agent = Agent(FakeIndex(), CapturingLLM(), classifier=FakeClassifier())
    agent.handle("and when will it arrive",
                 history=[Turn("t1", "user", "where is ORD-1234"),
                          Turn("t2", "agent", "Your order ORD-1234 shipped")])
    prompt = agent._llm.prompts[-1]
    transcript = "Customer: where is ORD-1234\nAgent: Your order ORD-1234 shipped"
    assert transcript in prompt
    assert (prompt.index("[faqs] Order status") < prompt.index(transcript)
            < prompt.index("Customer: and when will it arrive"))


def test_handle_template_path_receives_transcript_in_context():
    llm = TemplateCapturingLLM()
    build_agent(FakeIndex(), llm).handle(
        "and now?",
        history=[Turn("t1", "user", "earlier question"),
                 Turn("t2", "agent", "earlier reply")])
    system, context, user_text = llm.last_template_args
    assert "Customer: earlier question\nAgent: earlier reply" in context
    assert user_text == "and now?"


def test_handle_renders_at_most_four_exchanges():
    agent = Agent(FakeIndex(), CapturingLLM(), classifier=FakeClassifier())
    history = [t for i in range(6) for t in _exchange(i)]
    agent.handle("current question", history=history)
    prompt = agent._llm.prompts[-1]
    assert "message 1" not in prompt and "reply 1" not in prompt
    assert "message 2" in prompt and "reply 5" in prompt


def test_handle_stops_at_char_budget():
    agent = Agent(FakeIndex(), CapturingLLM(), classifier=FakeClassifier())
    history = []
    for i in range(3):
        history += [Turn(ts=f"t{i}u", role="user", text=f"m{i} " + "x" * 900),
                    Turn(ts=f"t{i}a", role="agent", text=f"r{i}")]
    agent.handle("current question", history=history)
    prompt = agent._llm.prompts[-1]
    assert "m0" not in prompt and "m1" not in prompt
    assert "m2" in prompt  # newest exchange always fits
    assert len(prompt) < len(agent._llm.prompts[0]) + 1700


def test_handle_skips_trailing_unpaired_user_turn():
    agent = Agent(FakeIndex(), CapturingLLM(), classifier=FakeClassifier())
    agent.handle("current question")
    without = agent._llm.prompts[0]
    agent.handle("current question", history=[Turn("t1", "user", "current question")])
    assert agent._llm.prompts[1] == without


def test_ref_inheritance_from_history():
    agent = Agent(FakeIndex(), CapturingLLM(), classifier=FakeClassifier())
    history = [Turn("t1", "user", "where is my order ORD-1234"),
               Turn("t2", "agent", "Shipped yesterday.")]
    res = agent.handle("and when will it arrive", history=history)
    assert "ORD-1234" in res.text  # echo guardrail inherits the reference


def test_no_inheritance_when_user_states_the_ref():
    agent = Agent(FakeIndex(), CapturingLLM(), classifier=FakeClassifier())
    history = [Turn("t1", "user", "where is my order ORD-1234"),
               Turn("t2", "agent", "Shipped yesterday.")]
    res = agent.handle("what about ORD-9999", history=history)
    assert "ORD-9999" in res.text
    assert "ORD-1234" not in res.text


def test_no_inheritance_when_history_has_no_ref():
    agent = Agent(FakeIndex(), CapturingLLM(), classifier=FakeClassifier())
    history = [Turn("t1", "user", "hello"), Turn("t2", "agent", "hi there")]
    res = agent.handle("thanks", history=history)
    # M5c: ACTION scaffolding scrubbed from the customer-visible text.
    assert res.text == "It is out for delivery."


# ---------------------------------------------------------------------------
# run_turn integration: memory recording, history passing, conversation cap
# ---------------------------------------------------------------------------

from voiceagent.chat import run_turn  # noqa: E402


class RecordingAgent:
    """Agent double that accepts (and records) the history kwarg."""

    def __init__(self):
        self.calls = []

    def handle(self, user_text, authenticated=False, amount=None,
               conv_id="", history=None):
        self.calls.append({"text": user_text, "conv_id": conv_id,
                           "history": history})
        return type("R", (), {
            "text": f"reply to {user_text}",
            "action": "order_status",
            "decision": type("D", (), {"verdict": "ALLOW",
                                       "reasons": ["ok"]})(),
        })()


def test_run_turn_records_user_and_agent_turns():
    agent, mem = RecordingAgent(), InMemoryMemory()
    run_turn(agent, "where is ORD-7777", authenticated=True, conv_id="c1",
             memory=mem)
    run_turn(agent, "and when will it arrive", conv_id="c1", memory=mem)
    turns = mem.history("c1")
    assert [t.role for t in turns] == ["user", "agent", "user", "agent"]
    assert turns[0].refs == ["ORD-7777"]
    assert turns[1].text == "reply to where is ORD-7777"
    assert turns[1].action == "order_status"
    assert turns[1].verdict == "ALLOW"
    assert all(t.ts for t in turns)


def test_run_turn_passes_recent_history_to_the_agent():
    agent, mem = RecordingAgent(), InMemoryMemory()
    run_turn(agent, "where is ORD-7777", conv_id="c1", memory=mem)
    run_turn(agent, "and when will it arrive", conv_id="c1", memory=mem)
    first, second = agent.calls[0]["history"], agent.calls[1]["history"]
    assert [t.role for t in first] == ["user"]  # the just-appended turn
    assert [t.role for t in second] == ["user", "agent", "user"]
    assert agent.calls[1]["text"] == "and when will it arrive"


def test_run_turn_caps_conversation_length():
    agent, mem = RecordingAgent(), InMemoryMemory()
    for i in range(2):  # 2 turns -> 4 recorded entries
        run_turn(agent, f"msg {i}", conv_id="c1", memory=mem,
                 max_turns_per_conv=4)
    out = run_turn(agent, "one more", conv_id="c1", memory=mem,
                   max_turns_per_conv=4)
    assert out == {
        "reply": "This conversation has reached its length limit — "
                 "connecting you to a human agent.",
        "action": None, "decision": "ESCALATE",
        "reasons": ["conversation length cap reached"],
        "executed": False, "tool_result": None, "directive": None}
    assert len(agent.calls) == 2  # agent NOT called on the capped turn
    assert len(mem.history("c1")) == 4  # capped turn is not recorded


def test_run_turn_without_memory_keeps_legacy_path():
    class LegacyAgent:  # pre-M4a duck type: no history kwarg
        def handle(self, user_text, authenticated=False, amount=None,
                   conv_id=""):
            return type("R", (), {"text": "ok", "action": None,
                                  "decision": None})()
    out = run_turn(LegacyAgent(), "hi")
    # Sprint A: the turn dict gained stable executed/tool_result/directive
    # keys (always present, None/False on the legacy one-shot path).
    assert out == {"reply": "ok", "action": None, "decision": None,
                   "reasons": [], "executed": False, "tool_result": None,
                   "directive": None}
