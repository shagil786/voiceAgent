# src/voiceagent/swarm/frontier.py
"""Frontier LLM adapter — the swarm's cloud brain.

Connects the agent swarm to ANY OpenAI-compatible /chat/completions endpoint
(Groq, OpenAI, Gemini's compat layer, vLLM, ...) using NATIVE JSON-Schema
function calling instead of regex-parsed text. stdlib urllib only — no new
dependencies.

Design contract (global-first agent):
- The frontier model PROPOSES (tool calls / reply text); deterministic
  governance (policy DSL, compliance watchdog, arbiter) DISPOSES. The bridge
  never executes a side-effectful handler implicitly — use execute_call()
  only for tools the deployment deems read-only.
- Tools are generic: register any callable via register_tool(), or plug a
  whole DomainSpecialist — its declared SpecialistTools become callable
  automatically. A new business vertical is a new SpecialistSpec, not new
  adapter code.
- A deployment "drops in" by configuring an endpoint + its domain specs.
  Deep adaptation (fine-tuning on that deployment's data) changes the
  weights behind the same tool surface; the swarm code is untouched.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from voiceagent.swarm.blackboard import (
    PRIORITY_SALES,
    BlackboardState,
    Proposal,
)
from voiceagent.swarm.specialist import DomainSpecialist, SpecialistSpec

DEFAULT_PRIORITY = PRIORITY_SALES

# --- configuration ---------------------------------------------------------

@dataclass
class FrontierConfig:
    """Endpoint configuration. base_url excludes /chat/completions, e.g.
    https://api.groq.com/openai/v1."""
    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 20.0


def config_from_env(env: Mapping[str, str] | None = None) -> FrontierConfig | None:
    """Build config from VOICEAGENT_FRONTIER_URL / _MODEL / _KEY. Returns None
    when the URL is unset (deployment runs the local/scripted path)."""
    e = os.environ if env is None else env
    base_url = e.get("VOICEAGENT_FRONTIER_URL")
    if not base_url:
        return None
    return FrontierConfig(
        base_url=base_url.rstrip("/"),
        model=e.get("VOICEAGENT_FRONTIER_MODEL", "gpt-4o-mini"),
        api_key=e.get("VOICEAGENT_FRONTIER_KEY") or None,
    )


class FrontierError(RuntimeError):
    """Frontier endpoint failure (transport, HTTP, or malformed protocol)."""


# --- transport -------------------------------------------------------------

Transport = Callable[[str, dict, dict, float], dict]


def _urllib_transport(url: str, payload: dict, headers: dict,
                      timeout_s: float) -> dict:
    # A browser-like User-Agent: some frontiers (Groq behind Cloudflare,
    # error 1010) reject requests from default urllib agents.
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "voiceagent-swarm/1.0",
                 **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --- replies ---------------------------------------------------------------

@dataclass
class FrontierToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class FrontierReply:
    content: str | None
    tool_calls: list[FrontierToolCall]
    model: str
    latency_s: float
    raw: dict


def _parse_tool_calls(message: dict) -> list[FrontierToolCall]:
    """Parse OpenAI-format tool_calls; tolerate dict- or string-encoded
    arguments and malformed JSON (degrades to empty args, never raises)."""
    calls: list[FrontierToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = (json.loads(raw_args) if isinstance(raw_args, str)
                    else dict(raw_args))
        except json.JSONDecodeError:
            args = {}
        calls.append(FrontierToolCall(
            id=tc.get("id") or "",
            name=fn.get("name") or "",
            arguments=args if isinstance(args, dict) else {},
        ))
    return calls


class FrontierClient:
    """OpenAI-compatible chat-completions client with native tools support.
    The HTTP transport is injectable for tests."""

    def __init__(self, config: FrontierConfig, transport: Transport | None = None):
        self.config = config
        self._transport = transport or _urllib_transport

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | dict = "auto", temperature: float = 0.4,
             max_tokens: int = 512) -> FrontierReply:
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        headers = ({"Authorization": f"Bearer {self.config.api_key}"}
                   if self.config.api_key else {})
        t0 = time.perf_counter()
        try:
            raw = self._transport(
                self.config.base_url + "/chat/completions",
                payload, headers, self.config.timeout_s)
        except FrontierError:
            raise
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise FrontierError(
                f"HTTP {exc.code} from frontier endpoint: {body}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise FrontierError(
                f"frontier endpoint unreachable: {exc}") from exc
        latency = time.perf_counter() - t0
        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise FrontierError(
                "malformed frontier response: "
                + json.dumps(raw)[:300]) from exc
        return FrontierReply(
            content=message.get("content"),
            tool_calls=_parse_tool_calls(message),
            model=raw.get("model", self.config.model),
            latency_s=latency,
            raw=raw,
        )


# --- tool schemas ----------------------------------------------------------

def tool_schema(name: str, description: str,
                parameters: dict[str, Any] | None) -> dict:
    """OpenAI function-tool envelope; parameters is a JSON-Schema object."""
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": parameters or {"type": "object", "properties": {}},
    }}


def tools_from_spec(spec: SpecialistSpec) -> list[dict]:
    """Auto-generate the tool surface from a domain spec's declared tools.
    A new vertical plugged in as a SpecialistSpec becomes callable by the
    frontier model with zero adapter code."""
    return [tool_schema(t.name, t.description, t.parameters)
            for t in spec.tools]


# --- the bridge ------------------------------------------------------------

@dataclass
class FrontierTurn:
    """One frontier turn projected onto the swarm: spoken/text reply, raw
    tool calls, the subset that mapped onto registered tools as Proposals,
    and any calls naming unregistered tools (surfaced, never dropped)."""
    reply_text: str | None
    tool_calls: list[FrontierToolCall]
    proposals: list[Proposal]
    unmapped_calls: list[FrontierToolCall]
    latency_s: float
    raw: dict


class FrontierAgentBridge:
    """Registers ANY callables as frontier tools, builds blackboard-aware
    prompts, and maps model tool calls onto swarm Proposals.

    Handlers are NOT invoked by propose() — execution stays with the
    governed layers (arbiter / policy / compliance). execute_call() exists
    for tools a deployment explicitly marks safe (read-only).
    """

    def __init__(self, client: FrontierClient,
                 system_prompt: str | None = None):
        self.client = client
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable[[dict], Any]] = {}
        self._role_blocks: list[str] = []
        self._system_prompt = system_prompt or (
            "You are the reasoning core of a voice-agent swarm. You receive "
            "the conversation and shared state, and you may call tools to "
            "propose actions. You propose; the governed layers decide and "
            "execute. Keep spoken replies concise and natural for voice.")

    # -- registration -------------------------------------------------------

    def register_tool(self, name: str, description: str,
                      parameters: dict[str, Any] | None = None,
                      handler: Callable[[dict], Any] | None = None) -> None:
        """Register any callable as a frontier tool. `parameters` is a
        JSON-Schema object describing the arguments. Registering a handler
        does NOT auto-execute it — see execute_call()."""
        self._tools[name] = tool_schema(name, description, parameters)
        if handler is not None:
            self._handlers[name] = handler

    def register_specialist(self, specialist: DomainSpecialist) -> None:
        """Plug a whole domain: its SpecialistTools join the tool surface and
        its role/prompt/disclosures join the system prompt."""
        spec: SpecialistSpec = specialist.spec
        for t in spec.tools:
            self._tools[t.name] = tool_schema(t.name, t.description,
                                              t.parameters)
            if t.handler is not None:
                self._handlers[t.name] = t.handler
        block = (f"### {spec.name} ({spec.domain_id})\n{spec.role_description}\n"
                 f"{spec.system_prompt}")
        if spec.statutory_disclosures:
            block += ("\nStatutory disclosures that must reach the customer "
                      "verbatim when relevant: "
                      + " | ".join(spec.statutory_disclosures))
        self._role_blocks.append(block)

    def tool_schemas(self) -> list[dict]:
        return list(self._tools.values())

    # -- prompting ----------------------------------------------------------

    def build_messages(self, state: BlackboardState, user_text: str,
                       history_limit: int = 8) -> list[dict]:
        parts = [self._system_prompt]
        if self._role_blocks:
            parts.append("## Deployed specialists\n"
                         + "\n".join(self._role_blocks))
        messages = [{"role": "system", "content": "\n\n".join(parts)}]
        for turn in state.history[-history_limit:]:
            role = ("assistant"
                    if turn.get("role") in ("agent", "assistant") else "user")
            messages.append({"role": role,
                             "content": turn.get("text", "")})
        messages.append({"role": "user", "content": user_text})
        return messages

    # -- the turn -----------------------------------------------------------

    def propose(self, state: BlackboardState, user_text: str,
                history_limit: int = 8,
                tools: list[dict] | None = None) -> FrontierTurn:
        """One frontier turn: prompt from blackboard + history, model reply,
        tool calls mapped to Proposals. Sync by design (leaf call); async
        swarm layers can wrap in asyncio.to_thread."""
        schemas = self.tool_schemas() if tools is None else tools
        reply = self.client.chat(
            self.build_messages(state, user_text, history_limit),
            tools=schemas or None)
        proposals: list[Proposal] = []
        unmapped: list[FrontierToolCall] = []
        for call in reply.tool_calls:
            if call.name in self._tools:
                proposals.append(Proposal(
                    source_agent="frontier",
                    priority=DEFAULT_PRIORITY,
                    action=call.name,
                    params=call.arguments,
                    content=reply.content or "",
                    metadata={"tool_call_id": call.id}))
            else:
                unmapped.append(call)
        return FrontierTurn(
            reply_text=reply.content,
            tool_calls=reply.tool_calls,
            proposals=proposals,
            unmapped_calls=unmapped,
            latency_s=reply.latency_s,
            raw=reply.raw,
        )

    def execute_call(self, call: FrontierToolCall) -> Any:
        """Explicit handler execution — ONLY for tools the deployment marks
        safe (read-only/informational). Side-effectful tools must route
        through the governed ToolGateway instead."""
        handler = self._handlers.get(call.name)
        if handler is None:
            raise FrontierError(f"no handler registered for tool '{call.name}'")
        return handler(call.arguments)
