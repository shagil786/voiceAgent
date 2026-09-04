# tests/test_runtime.py — the governed Orchestrator assembly seam.
"""Guards the "ONE brain" contract: every entry point (LiveKit worker, REPL,
tests) must build the SAME governed Orchestrator via voiceagent.runtime, and it
must fail closed (return None / refuse to start) when no frontier brain is set
rather than serving a crashing `orchestrator=None`."""
from __future__ import annotations

import pytest

from voiceagent.runtime import build_orchestrator, make_deployment


class _FakeLLM:
    """Minimal frontier stand-in: returns a governed-tool call that the runner
    can execute, so we exercise the real handle_turn path offline."""
    def __init__(self):
        self.calls = []
        self._n = 0

    def chat(self, messages, tools=None, tool_choice="auto",
              temperature=0.4, max_tokens=512):
        from voiceagent.swarm.frontier import FrontierReply, FrontierToolCall
        self.calls.append(messages)
        self._n += 1
        if tools and self._n == 1:
            return FrontierReply(
                content=None,
                tool_calls=[FrontierToolCall(
                    id="c1", name="fetch_order_status", arguments={})],
                model="fake", latency_s=0.001, raw={})
        return FrontierReply(content="Here is your order status.",
                            tool_calls=[], model="fake", latency_s=0.001, raw={})


def test_no_frontier_returns_none():
    # No VOICEAGENT_FRONTIER_URL -> the worker must refuse, not serve None.
    orch = build_orchestrator(env={"VOICEAGENT_FRONTIER_URL": ""})
    assert orch is None


def test_builds_governed_orchestrator_with_stub_frontier():
    fake = _FakeLLM()
    orch = build_orchestrator(
        env={"VOICEAGENT_FRONTIER_URL": "https://fake/v1"},
        erp=_FakeERP(), memory=_FakeMem(), decision_log=_FakeLog())
    # Inject a stub brain so no network call happens.
    orch.brain.client = fake
    res = orch.handle_turn("s1", "where is my order ORD-1", authenticated=True)
    assert res.reply
    # The governed tool call was routed + executed (MockERP returns a status).
    assert res.actions and res.actions[0]["tool"] == "fetch_order_status"
    assert res.actions[0]["verdict"] in ("ALLOW", "DENY")


def test_make_deployment_is_governed_data_only():
    dep = make_deployment()
    assert dep.gateway_tools["fetch_order_status"]["action"] == "order_status"
    assert "escalate_to_human" in dep.gateway_tools
    assert dep.system_prompt and dep.knowledge


class _FakeERP:
    def get_order(self, order_id):
        return {"order_id": order_id, "status": "CONFIRMED", "amount": 1299.0}
    def fetch_order_status(self, **kw):
        return type("R", (), {"ok": True, "value": {"status": "shipped"}})()


class _FakeMem:
    def append(self, *a, **k):
        pass
    def history(self, *a, **k):
        return []


class _FakeLog:
    def record(self, *a, **k):
        pass
