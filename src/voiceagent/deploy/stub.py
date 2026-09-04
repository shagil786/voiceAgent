"""Deterministic scripted brain client for deploy self-checks.

Ships with src (unlike the test-only ScriptedBrain in tests/test_orchestrator)
so production self-checks never depend on the tests package. Behavior is
identical: constructed with a list of FrontierReply, pops one per chat call,
falls back to a "(script exhausted)" reply when the script runs out.
"""
from __future__ import annotations

from voiceagent.swarm.frontier import FrontierReply, FrontierToolCall


class ScriptedClient:
    """Stub frontier client: pops scripted FrontierReplies, captures chat args."""

    def __init__(self, replies: list[FrontierReply]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, tool_choice="auto",
             temperature=0.4, max_tokens=512) -> FrontierReply:
        self.calls.append({"messages": messages, "tools": tools,
                           "tool_choice": tool_choice})
        if self._replies:
            return self._replies.pop(0)
        return FrontierReply(content="(script exhausted)", tool_calls=[],
                             model="stub", latency_s=0.001, raw={})


def reply(content: str | None = None,
          calls: list[FrontierToolCall] | None = None) -> FrontierReply:
    return FrontierReply(content=content, tool_calls=calls or [],
                         model="stub", latency_s=0.001, raw={})


def make_default_client(texts: list[str]) -> ScriptedClient:
    """One scripted reply per text — the default self-check brain."""
    return ScriptedClient([reply(t) for t in texts])
