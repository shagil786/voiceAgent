# tests/test_frontier.py
"""Frontier LLM adapter: tool-surface generation, OpenAI-compatible transport
(injectable), tool-call parsing, and blackboard->proposal mapping. Real
endpoint tests are gated behind VOICEAGENT_FRONTIER_INTEGRATION=1."""
import io
import json
import os
import urllib.error

import pytest

from voiceagent.swarm.blackboard import Blackboard, CallerProfile
from voiceagent.swarm.frontier import (
    FrontierAgentBridge,
    FrontierClient,
    FrontierConfig,
    FrontierError,
    config_from_env,
    tool_schema,
    tools_from_spec,
)
from voiceagent.swarm.specialist import DomainSpecialist, SpecialistSpec, SpecialistTool


# --- fixtures ---------------------------------------------------------------

def fake_transport(response: dict, capture: dict | None = None):
    def _t(url, payload, headers, timeout_s):
        if capture is not None:
            capture.update({"url": url, "payload": payload,
                            "headers": headers, "timeout": timeout_s})
        return response
    return _t


def canned_response(content="Here is my pitch.", calls=None, model="test-model"):
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"model": model,
            "choices": [{"message": message}]}


def sample_spec() -> SpecialistSpec:
    return SpecialistSpec(
        domain_id="real_estate",
        name="Property Closer",
        role_description="Sells premium listings, books site visits.",
        system_prompt="You close premium property deals.",
        statutory_disclosures=["MahaRERA Reg No. P51800012345"],
        tools=[
            SpecialistTool(
                name="book_site_visit",
                description="Book a site visit slot for the caller.",
                parameters={"type": "object",
                            "properties": {"listing_id": {"type": "string"},
                                           "date": {"type": "string"}},
                            "required": ["listing_id"]},
                handler=lambda args: {"booked": True, **args},
            ),
            SpecialistTool(
                name="share_floor_plan",
                description="Push the floor plan PDF to WhatsApp.",
                parameters=None,  # no-arg tool -> default empty schema
                handler=lambda args: {"sent": True},
            ),
        ],
    )


# --- schema surface ---------------------------------------------------------

def test_tool_schema_envelope_shape():
    schema = tool_schema("ping", "A ping tool.", {"type": "object"})
    assert schema == {"type": "function", "function": {
        "name": "ping", "description": "A ping tool.",
        "parameters": {"type": "object"}}}


def test_tool_schema_defaults_empty_parameters():
    schema = tool_schema("noop", "Does nothing.", None)
    assert schema["function"]["parameters"] == {"type": "object", "properties": {}}


def test_tools_from_spec_passthrough_and_defaults():
    schemas = tools_from_spec(sample_spec())
    assert [s["function"]["name"] for s in schemas] == [
        "book_site_visit", "share_floor_plan"]
    assert schemas[0]["function"]["parameters"]["required"] == ["listing_id"]
    assert schemas[1]["function"]["parameters"] == {
        "type": "object", "properties": {}}


# --- configuration ----------------------------------------------------------

def test_config_from_env_unset_returns_none():
    assert config_from_env({}) is None


def test_config_from_env_builds_config():
    cfg = config_from_env({
        "VOICEAGENT_FRONTIER_URL": "https://api.groq.com/openai/v1/",
        "VOICEAGENT_FRONTIER_MODEL": "llama-3.3-70b",
        "VOICEAGENT_FRONTIER_KEY": "gsk_test"})
    assert cfg is not None
    assert cfg.base_url == "https://api.groq.com/openai/v1"  # trailing / stripped
    assert cfg.model == "llama-3.3-70b"
    assert cfg.api_key == "gsk_test"


# --- client -----------------------------------------------------------------

def test_client_sends_tools_and_parses_calls():
    capture: dict = {}
    response = canned_response(calls=[
        {"id": "c1", "function": {"name": "book_site_visit",
                                  "arguments": "{\"listing_id\": \"L1\", \"date\": \"fri\"}"}},
        {"id": "c2", "function": {"name": "share_floor_plan",
                                  "arguments": {"auto": True}}},  # dict-encoded args
    ])
    client = FrontierClient(
        FrontierConfig(base_url="https://x/v1", model="test-model",
                       api_key="k123"),
        transport=fake_transport(response, capture))
    reply = client.chat(
        [{"role": "user", "content": "hi"}],
        tools=[tool_schema("book_site_visit", "book", None)])
    # payload assertions
    assert capture["url"] == "https://x/v1/chat/completions"
    assert capture["payload"]["model"] == "test-model"
    assert capture["payload"]["tools"][0]["function"]["name"] == "book_site_visit"
    assert capture["payload"]["tool_choice"] == "auto"
    assert capture["headers"]["Authorization"] == "Bearer k123"
    # parse assertions
    assert reply.content == "Here is my pitch."
    assert [c.name for c in reply.tool_calls] == ["book_site_visit", "share_floor_plan"]
    assert reply.tool_calls[0].arguments == {"listing_id": "L1", "date": "fri"}
    assert reply.tool_calls[1].arguments == {"auto": True}
    assert reply.latency_s >= 0.0


def test_client_tolerates_malformed_arguments():
    response = canned_response(calls=[
        {"id": "c3", "function": {"name": "broken", "arguments": "{not json"}},
    ])
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=fake_transport(response))
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.tool_calls[0].arguments == {}


def test_client_without_tools_omits_tools_key():
    capture: dict = {}
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=fake_transport(canned_response(), capture))
    client.chat([{"role": "user", "content": "hi"}])
    assert "tools" not in capture["payload"]


def test_client_http_error_raises_frontier_error():
    # 503 is retryable: the client must exhaust its bounded retry budget,
    # then surface the historical error contract unchanged.
    attempts = {"n": 0}

    def failing_transport(url, payload, headers, timeout_s):
        attempts["n"] += 1
        raise urllib.error.HTTPError(url, 503, "overloaded", {},
                                     io.BytesIO(b"try later"))

    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=failing_transport,
                            sleep_fn=lambda _s: None)
    with pytest.raises(FrontierError, match="HTTP 503"):
        client.chat([{"role": "user", "content": "hi"}])
    assert attempts["n"] == 3  # 1 attempt + max_retries=2


def test_client_retries_on_5xx_then_succeeds():
    # Transient server errors must not kill the turn: back off, then recover.
    attempts = {"n": 0}

    def flaky_transport(url, payload, headers, timeout_s):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise urllib.error.HTTPError(url, 503, "overloaded", {},
                                         io.BytesIO(b"try later"))
        return canned_response("recovered")

    sleeps: list[float] = []
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=flaky_transport, sleep_fn=sleeps.append)
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "recovered"
    assert attempts["n"] == 3
    # bounded exponential backoff: base 0.4 * 2**attempt + sub-second jitter
    assert len(sleeps) == 2
    assert 0.4 <= sleeps[0] <= 0.45
    assert 0.8 <= sleeps[1] <= 0.85


def test_client_client_error_is_not_retried():
    # 4xx (save 429) cannot be fixed by retrying: fail on the first attempt.
    attempts = {"n": 0}

    def failing_transport(url, payload, headers, timeout_s):
        attempts["n"] += 1
        raise urllib.error.HTTPError(url, 400, "bad request", {},
                                     io.BytesIO(b"nope"))

    sleeps: list[float] = []
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=failing_transport, sleep_fn=sleeps.append)
    with pytest.raises(FrontierError, match="HTTP 400"):
        client.chat([{"role": "user", "content": "hi"}])
    assert attempts["n"] == 1
    assert sleeps == []


def test_client_connection_error_retries_then_raises_frontier_error():
    attempts = {"n": 0}

    def failing_transport(url, payload, headers, timeout_s):
        attempts["n"] += 1
        raise urllib.error.URLError("connection refused")

    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=failing_transport,
                            sleep_fn=lambda _s: None)
    with pytest.raises(FrontierError, match="unreachable"):
        client.chat([{"role": "user", "content": "hi"}])
    assert attempts["n"] == 3


def test_client_malformed_response_raises_frontier_error():
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=fake_transport({"oops": True}))
    with pytest.raises(FrontierError, match="malformed"):
        client.chat([{"role": "user", "content": "hi"}])


# --- bridge -----------------------------------------------------------------

def test_bridge_maps_known_tool_calls_to_proposals():
    capture: dict = {}
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=fake_transport(canned_response(calls=[
        {"id": "c1", "function": {"name": "book_site_visit",
                                  "arguments": "{\"listing_id\": \"L9\"}"}},
        {"id": "c2", "function": {"name": "send_sms", "arguments": "{}"}},
    ]), capture))
    bridge = FrontierAgentBridge(client)
    bridge.register_tool("book_site_visit", "Book a visit.",
                         {"type": "object"})
    state = Blackboard(session_id="s1", profile=CallerProfile(name="Rohan")).state
    state.append_turn("agent", "Hello!")
    state.append_turn("customer", "Can I visit Friday?")
    turn = bridge.propose(state, "Book me in for Friday please")
    # mapping
    assert len(turn.proposals) == 1
    p = turn.proposals[0]
    assert p.action == "book_site_visit"
    assert p.params == {"listing_id": "L9"}
    assert p.source_agent == "frontier"
    assert p.metadata["tool_call_id"] == "c1"
    assert [c.name for c in turn.unmapped_calls] == ["send_sms"]
    # prompt assembly: system + mapped history + user turn
    roles = [m["role"] for m in capture["payload"]["messages"]]
    assert roles == ["system", "assistant", "user", "user"]
    assert "Book me in for Friday please" in capture["payload"]["messages"][-1]["content"]


def test_bridge_system_prompt_includes_specialist_role_and_disclosures():
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=fake_transport(canned_response("ok")))
    bridge = FrontierAgentBridge(client)
    bridge.register_specialist(DomainSpecialist(spec=sample_spec()))
    state = Blackboard(session_id="s1").state
    msgs = bridge.build_messages(state, "hi")
    system = msgs[0]["content"]
    assert "Property Closer (real_estate)" in system
    assert "MahaRERA Reg No. P51800012345" in system
    names = [s["function"]["name"] for s in bridge.tool_schemas()]
    assert names == ["book_site_visit", "share_floor_plan"]


def test_bridge_proposals_only_when_tools_registered():
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=fake_transport(canned_response("sure")))
    bridge = FrontierAgentBridge(client)
    state = Blackboard(session_id="s1").state
    turn = bridge.propose(state, "hello")
    assert turn.reply_text == "sure"
    assert turn.proposals == [] and turn.tool_calls == []


def test_bridge_execute_call_runs_handler_only_when_marked_safe():
    client = FrontierClient(FrontierConfig(base_url="https://x/v1", model="m"),
                            transport=fake_transport(canned_response()))
    bridge = FrontierAgentBridge(client)
    seen = {}
    bridge.register_tool("get_floor_plan", "read-only lookup",
                         handler=lambda args: (seen.update(args), {"url": "plan.pdf"})[1])
    from voiceagent.swarm.frontier import FrontierToolCall
    out = bridge.execute_call(FrontierToolCall(id="x", name="get_floor_plan",
                                               arguments={"listing": "L1"}))
    assert out == {"url": "plan.pdf"} and seen == {"listing": "L1"}
    with pytest.raises(FrontierError, match="no handler"):
        bridge.execute_call(FrontierToolCall(id="y", name="unknown_tool",
                                             arguments={}))


# --- gated real-endpoint test ------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("VOICEAGENT_FRONTIER_INTEGRATION") != "1"
    or not os.environ.get("VOICEAGENT_FRONTIER_URL"),
    reason="real frontier endpoint integration — set VOICEAGENT_FRONTIER_INTEGRATION=1 "
           "+ VOICEAGENT_FRONTIER_URL/KEY")
def test_real_frontier_endpoint_replies():
    cfg = config_from_env()
    assert cfg is not None
    client = FrontierClient(cfg)
    bridge = FrontierAgentBridge(client)
    bridge.register_tool("get_time", "Returns the current UTC time string.")
    turn = bridge.propose(Blackboard(session_id="it").state, "Say OK.")
    assert (turn.reply_text or "").strip(), turn.raw


# --- provider fallback ------------------------------------------------------

def test_configs_from_env_builds_primary_plus_fallback():
    env = {
        "VOICEAGENT_FRONTIER_URL": "https://primary.example/v1",
        "VOICEAGENT_FRONTIER_MODEL": "m1",
        "VOICEAGENT_FRONTIER_KEY": "k1",
        "VOICEAGENT_FRONTIER_FALLBACK_URL": "https://fallback.example/v1",
        "VOICEAGENT_FRONTIER_FALLBACK_MODEL": "m2",
        "VOICEAGENT_FRONTIER_FALLBACK_KEY": "k2",
    }
    from voiceagent.swarm.frontier import configs_from_env
    cfgs = configs_from_env(env)
    assert [c.base_url for c in cfgs] == [
        "https://primary.example/v1", "https://fallback.example/v1"]
    assert [c.model for c in cfgs] == ["m1", "m2"]
    # fallback model defaults to primary's when unset
    env.pop("VOICEAGENT_FRONTIER_FALLBACK_MODEL")
    cfgs = configs_from_env(env)
    assert cfgs[1].model == "m1"


def test_client_fails_over_to_fallback_on_primary_failure():
    primary = FrontierConfig(base_url="https://primary.example/v1",
                             model="m1", api_key="k1", max_retries=0)
    fallback = FrontierConfig(base_url="https://fallback.example/v1",
                              model="m2", api_key="k2")

    def _t(url, payload, headers, timeout_s):
        if "primary.example" in url:
            raise urllib.error.URLError("primary down")
        return {"choices": [{"message": {"content": "hi from fallback",
                                         "tool_calls": []}}],
                "model": "m2"}

    client = FrontierClient(primary, transport=_t, fallbacks=[fallback])
    reply = client.chat([{"role": "user", "content": "hello"}])
    assert reply.content == "hi from fallback"
    assert reply.model == "m2"


def test_client_raises_last_error_when_all_providers_fail():
    primary = FrontierConfig(base_url="https://primary.example/v1",
                             model="m1", api_key="k1", max_retries=0)
    fallback = FrontierConfig(base_url="https://fallback.example/v1",
                              model="m2", api_key="k2")

    def _t(url, payload, headers, timeout_s):
        raise urllib.error.URLError("nope")

    client = FrontierClient(primary, transport=_t, fallbacks=[fallback])
    with pytest.raises(FrontierError):
        client.chat([{"role": "user", "content": "hello"}])


def test_failover_cools_flapping_primary_then_recovers():
    """A failed primary is skipped for the cooldown (no per-turn retry
    storm); a recovered primary serves again after it lapses."""
    from voiceagent.swarm.frontier import FailoverClient, FrontierError
    calls = {"primary": 0, "backup": 0}
    clock = [1000.0]

    class Primary:
        def chat(self, *a, **k):
            calls["primary"] += 1
            if calls["primary"] == 1:
                raise FrontierError("flap")
            from voiceagent.swarm.frontier import FrontierReply
            return FrontierReply(content="primary back", tool_calls=[],
                                 model="p", latency_s=0.0, raw={})

    class Backup:
        def chat(self, *a, **k):
            calls["backup"] += 1
            from voiceagent.swarm.frontier import FrontierReply
            return FrontierReply(content="backup", tool_calls=[],
                                 model="b", latency_s=0.0, raw={})

    chain = FailoverClient([Primary(), Backup()], cooldown_s=60.0,
                           now=lambda: clock[0])
    assert chain.chat([]).content == "backup"   # primary flaps -> backup
    assert chain.chat([]).content == "backup"   # primary cooling: no retry
    assert calls == {"primary": 1, "backup": 2}
    clock[0] += 61.0                            # cooldown lapses
    assert chain.chat([]).content == "primary back"
    assert calls["primary"] == 2
