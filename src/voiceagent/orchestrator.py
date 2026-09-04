# src/voiceagent/orchestrator.py — the cognitive runtime core.
"""ONE turn loop for any placement: inbound chat/voice, outbound campaign
call. The orchestrator binds the swarm's organs into a single agent:

    frontier brain (proposes) -> deterministic layers (dispose) -> reply

Governance contract (non-negotiable): the model may only PROPOSE. Governed
gateway tools run exclusively through GovernedToolRunner (policy verdict
first, execution only on ALLOW, decision log always); DENY/REQUIRE_AUTH/
ESCALATE verdicts are facts fed back to the brain as tool results so it can
explain or re-plan — never bypassed. Only tools a deployment explicitly
gave a handler (read-only lookups) execute via the bridge. Multi-round
tool calling is bounded by max_tool_rounds; a final spoken reply is ALWAYS
produced (a forced text-only close if the brain keeps asking for tools).

Sync by design — voice layers wrap handle_turn in asyncio.to_thread.
A business "drops in" via Deployment: system prompt, SpecialistSpec plugs,
governed gateway actions, and a small inline knowledge block. No new pip
deps; stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from voiceagent.decisionlog import DecisionEntry, DecisionLog
from voiceagent.learn.corrections import classify_correction
from voiceagent.learn.profiles import Profile, ProfileStore, contact_key
from voiceagent.memory import ConversationMemory, InMemoryMemory, Turn, now_ts
from voiceagent.policy import PolicyContext
from voiceagent.sentiment import Frustration, detect_frustration
from voiceagent.swarm.blackboard import BlackboardState, CallerProfile
from voiceagent.swarm.frontier import (
    FrontierAgentBridge,
    FrontierReply,
    FrontierToolCall,
)
from voiceagent.swarm.specialist import DomainSpecialist, SpecialistSpec
from voiceagent.tools import GovernedToolRunner

_FALLBACK_REPLY = (
    "I'm sorry, I wasn't able to complete that here. Let me connect you "
    "with a colleague who can help."
)


# --- deployment descriptor ---------------------------------------------------

@dataclass
class Deployment:
    """Drop the agent into any business: prompt, domain plugs, governed
    actions, inline knowledge. Real RAG / tenant bundles plug in later."""
    name: str
    system_prompt: str
    specs: list[SpecialistSpec] = field(default_factory=list)      # domain plugs
    # governed actions: tool name -> {"action": policy_action, "side_effects":
    # True, optional "description"/"parameters" (JSON-Schema)}. Governed tools
    # are NEVER handler-executed by the brain; the orchestrator routes them
    # through GovernedToolRunner.
    gateway_tools: dict[str, dict] = field(default_factory=dict)
    knowledge: dict[str, str] = field(default_factory=dict)  # id -> text
    metadata: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    reply: str
    actions: list[dict]        # executed governed actions: {action, tool, verdict, ok, value/error}
    brain_latency_s: float
    session_id: str
    raw_tool_calls: int
    escalated: bool = False


# --- contact memory block ----------------------------------------------------

def _contact_memory_block(prof: Profile) -> str:
    """Render a profile's prefs/corrections/open items as a system block
    (empty sections omitted; capped at 1500 chars). Empty string when the
    profile carries nothing worth telling the brain."""
    lines = [f"- {p}" for p in prof.prefs]
    lines += [f"- Correction (use instead): "
              f"{c.get('quote', str(c)) if isinstance(c, dict) else c}"
              for c in prof.corrections]
    lines += [f"- Open: {o}" for o in prof.open_items]
    if not lines:
        return ""
    return ("## Contact memory\n" + "\n".join(lines))[:1500]


# --- the runtime -------------------------------------------------------------

class Orchestrator:
    """Full agent runtime: brain <-> governed-tool ping-pong bounded by
    max_tool_rounds, per-session blackboard state, durable memory."""

    def __init__(self, brain: FrontierAgentBridge,
                 runner: GovernedToolRunner | None = None,
                 memory: ConversationMemory | None = None,
                 decision_log: DecisionLog | None = None,
                 max_tool_rounds: int = 3,
                 profiles: ProfileStore | None = None):
        self.brain = brain
        self.runner = runner
        self.memory: ConversationMemory = memory or InMemoryMemory()
        # Audit seam for the no-runner case (the runner logs its own verdicts).
        self.decision_log = decision_log
        self.max_tool_rounds = max_tool_rounds
        # Instant-Learn seam: None = pre-learn behavior (byte-identical replies).
        self.profiles = profiles
        self._profile_links: dict[str, str] = {}  # session_id -> contact key
        self._gateway_tools: dict[str, dict] = {}
        self._deployment: Deployment | None = None
        self._sessions: dict[str, BlackboardState] = {}

    # -- deployment ---------------------------------------------------------

    def deploy(self, deployment: Deployment) -> None:
        """Register a deployment: specialist plugs + governed tools join the
        brain's tool surface; prompt + knowledge become the system message."""
        self._deployment = deployment
        self._gateway_tools = dict(deployment.gateway_tools)
        for spec in deployment.specs:
            self.brain.register_specialist(DomainSpecialist(spec=spec))
        for tool_name, meta in deployment.gateway_tools.items():
            self.brain.register_tool(
                name=tool_name,
                description=meta.get(
                    "description",
                    f"Governed action '{meta.get('action', tool_name)}' — "
                    "proposals only; executed through the policy-governed "
                    "runner."),
                parameters=meta.get("parameters"),
                handler=None,  # governed tools are NEVER brain-executed
            )
        # The bridge keeps its system prompt private by design (frontier.py is
        # frozen), so deployment writes through this seam.
        parts = [deployment.system_prompt]
        if deployment.knowledge:
            parts.append("## Knowledge\n" + "\n".join(
                f"- [{kid}] {text}"
                for kid, text in deployment.knowledge.items()))
        self.brain._system_prompt = "\n\n".join(parts)

    # -- the turn loop ------------------------------------------------------

    def handle_turn(self, session_id: str, user_text: str, *,
                    profile: CallerProfile | None = None,
                    authenticated: bool | None = None,
                    _system_prefix: str | None = None,
                    contact_alias: str | None = None) -> TurnResult:
        """One full agent turn: prompt -> brain -> governed tools -> reply.

        `authenticated` is a per-turn OVERRIDE (e.g. OTP verified mid-call);
        None defers to the profile's own flag. governs require_auth policies.
        `contact_alias` resolves through the ProfileStore alias map; None
        derives the contact key from the caller profile.
        """
        state = self._session_state(session_id, profile, authenticated)
        # Instant-Learn contact memory in: resolve the contact, inject its
        # prefs/corrections/open items ahead of any placement prefix, and
        # link this session to the contact. profiles=None skips all of this.
        contact: str | None = None
        if self.profiles is not None:
            contact = (self.profiles.resolve(contact_alias)
                       if contact_alias else contact_key(state.profile))
            prof = self.profiles.get(contact)
            if prof is not None:
                block = _contact_memory_block(prof)
                if block:
                    _system_prefix = ((block + "\n\n" + (_system_prefix or ""))
                                      or None)
            self.profiles.link_session(contact, session_id)
            self._profile_links[session_id] = contact
        # Deterministic sentiment: every governed evaluation this turn sees
        # the caller's frustration level (policies route on it via
        # escalate_when — conditions are data, not code).
        frustration = detect_frustration(user_text)
        messages = self.brain.build_messages(state, user_text)
        if _system_prefix:  # campaign placement prepends its block
            messages[0] = {"role": "system",
                           "content": _system_prefix + "\n\n"
                                      + messages[0]["content"]}
        tools = self.brain.tool_schemas() or None

        actions: list[dict] = []
        latency = 0.0
        raw_tool_calls = 0
        escalated = False

        reply = self._chat(messages, tools)
        latency += reply.latency_s
        rounds = 0
        while rounds < self.max_tool_rounds and reply.tool_calls:
            rounds += 1
            messages.append(self._assistant_message(reply))
            for call in reply.tool_calls:
                raw_tool_calls += 1
                payload, entry, is_escalation = self._dispatch_tool_call(
                    call, state, session_id, frustration)
                if entry is not None:
                    actions.append(entry)
                escalated = escalated or is_escalation
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(payload, default=str)})
            reply = self._chat(messages, tools)
            latency += reply.latency_s

        if reply.tool_calls:
            # Round budget exhausted mid-ping-pong: force a text-only close
            # (no tool surface, so the model must speak). Stray calls are
            # surfaced in raw_tool_calls but never executed.
            raw_tool_calls += len(reply.tool_calls)
            reply = self._chat(messages, None)
            latency += reply.latency_s
            raw_tool_calls += len(reply.tool_calls)

        final_text = (reply.content or "").strip() or _FALLBACK_REPLY

        # record the turn on the blackboard and in durable memory
        state.append_turn("user", user_text)
        state.append_turn("agent", final_text)
        primary = actions[0] if actions else None
        self.memory.append(session_id, Turn(ts=now_ts(), role="user",
                                            text=user_text))
        self.memory.append(session_id, Turn(
            ts=now_ts(), role="agent", text=final_text,
            action=primary["action"] if primary else None,
            verdict=primary["verdict"] if primary else None))

        # Instant-Learn candidates out: a customer correction never mutates
        # global state — it lands in pending_global for owner review.
        if self.profiles is not None and contact is not None:
            corr = classify_correction(user_text, final_text, is_owner=False)
            if corr.is_correction:
                prof = (self.profiles.get(contact)
                        or Profile(key=contact, alias="", prefs=[],
                                   corrections=[], open_items=[],
                                   pending_global=[], consent={},
                                   updated_at=now_ts()))
                prof.pending_global.append(
                    {"quote": corr.quote, "patch_type": corr.patch_type,
                     "session_id": session_id, "ts": now_ts()})
                prof.updated_at = now_ts()
                self.profiles.put(prof)

        return TurnResult(reply=final_text, actions=actions,
                          brain_latency_s=latency, session_id=session_id,
                          raw_tool_calls=raw_tool_calls, escalated=escalated)

    def campaign_turn(self, session_id: str, lead: dict, script_goal: str, *,
                      profile: CallerProfile | None = None) -> TurnResult:
        """Outbound placement: the dialer/AMD layer calls this per connected
        call. Wraps handle_turn with a campaign system block (goal + lead)."""
        lead_json = json.dumps(lead, ensure_ascii=False, sort_keys=True,
                               default=str)
        prefix = (
            "## Outbound campaign call\n"
            f"Campaign goal: {script_goal}\n"
            f"Lead context (JSON): {lead_json}\n"
            "You placed this outbound call. Open with a short, natural intro "
            "tied to the goal; honor DNC/opt-out requests immediately.")
        user_text = str(lead.get("utterance")
                        or "(Call connected — open per the campaign goal.)")
        return self.handle_turn(session_id, user_text, profile=profile,
                                _system_prefix=prefix)

    # -- contact lifecycle (Instant-Learn) ----------------------------------

    def delete_contact(self, contact_or_alias: str) -> dict:
        """Delete a contact's profile and cascade to its linked sessions:
        durable memory cleared and live blackboard state dropped."""
        if self.profiles is None:
            raise RuntimeError("no ProfileStore configured")
        out = self.profiles.delete_contact(
            self.profiles.resolve(contact_or_alias))
        for sid in out["sessions"]:
            self.memory.clear(sid)
            self._sessions.pop(sid, None)
            self._profile_links.pop(sid, None)
        return out

    def export_contact(self, contact_or_alias: str) -> dict:
        """Export one contact's profile dict (KeyError when unknown)."""
        if self.profiles is None:
            raise RuntimeError("no ProfileStore configured")
        return self.profiles.export_contact(
            self.profiles.resolve(contact_or_alias))

    # -- internals ----------------------------------------------------------

    def _chat(self, messages: list[dict], tools: list[dict] | None) -> FrontierReply:
        return self.brain.client.chat(messages, tools=tools)

    def _session_state(self, session_id: str,
                       profile: CallerProfile | None,
                       authenticated: bool) -> BlackboardState:
        """Per-session blackboard; new sessions are seeded from durable
        memory so restarts keep context. The per-turn `authenticated` flag
        always lands on the live profile."""
        state = self._sessions.get(session_id)
        if state is None:
            prof = profile or CallerProfile()
            if authenticated is not None:
                prof.authenticated = authenticated
            state = BlackboardState(session_id=session_id, profile=prof)
            for t in self.memory.history(session_id):
                state.append_turn(t.role, t.text)
            self._sessions[session_id] = state
        else:
            if profile is not None:
                state.profile = profile
            if authenticated is not None:
                state.profile.authenticated = authenticated
        return state

    @staticmethod
    def _assistant_message(reply: FrontierReply) -> dict:
        """OpenAI-shape assistant message carrying the brain's tool calls."""
        return {"role": "assistant", "content": reply.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name,
                                             "arguments": json.dumps(
                                                 c.arguments or {},
                                                 default=str)}}
                               for c in reply.tool_calls]}

    def _dispatch_tool_call(self, call: FrontierToolCall, state: BlackboardState,
                            session_id: str,
                            frustration: Frustration) -> tuple[dict, dict | None, bool]:
        """Route one tool call. Returns (tool-result payload fed back to the
        brain, TurnResult.actions entry or None, escalated flag)."""
        gmeta = self._gateway_tools.get(call.name)
        if gmeta is not None:
            return self._run_governed(call, gmeta, state, session_id,
                                      frustration)
        # read-only brain tool: explicit handler execution via the bridge
        try:
            value = self.brain.execute_call(call)
            return {"ok": True, "value": value}, None, False
        except Exception as exc:  # never crash the voice turn on a bad tool
            return ({"ok": False,
                     "error": f"{type(exc).__name__}: {exc}"}, None, False)

    def _run_governed(self, call: FrontierToolCall, gmeta: dict,
                      state: BlackboardState, session_id: str,
                      frustration: Frustration) -> tuple[dict, dict | None, bool]:
        """Governed gateway tool: policy verdict first, execution ONLY on
        ALLOW (via the runner), verdict + reasons always fed back."""
        action = gmeta.get("action", call.name)
        params = dict(call.arguments or {})

        if self.runner is None:
            # Governance is unavailable -> nothing may execute. Logged as a
            # least-privilege DENY when an audit log is wired.
            reasons = ["no GovernedToolRunner configured; action not executed"]
            if self.decision_log is not None:
                self.decision_log.record(DecisionEntry(
                    ts=now_ts(), conv_id=session_id, action=action,
                    verdict="DENY", reasons=list(reasons),
                    authenticated=state.profile.authenticated))
            entry = {"action": action, "tool": call.name, "verdict": "DENY",
                     "ok": False, "error": reasons[0], "reasons": reasons}
            return {"ok": False, "verdict": "DENY",
                    "reasons": reasons, "error": reasons[0]}, entry, False

        ctx = PolicyContext(
            authenticated=state.profile.authenticated,
            amount=(params["amount"]
                    if isinstance(params.get("amount"), (int, float))
                    else None),
            signals={"risk_tier": state.profile.risk_tier,
                     "session_id": session_id,
                     "frustrated": frustration.frustrated,
                     "frustration_level": frustration.level},
        )
        outcome = self.runner.run(action, ctx, call.name, params,
                                  conv_id=session_id)
        entry: dict[str, Any] = {"action": action, "tool": call.name,
                                 "verdict": outcome.decision_verdict,
                                 "ok": bool(outcome.executed)}
        payload: dict[str, Any] = {"verdict": outcome.decision_verdict,
                                   "reasons": list(outcome.reasons),
                                   "ok": bool(outcome.executed)}
        result = outcome.result
        if result is not None and result.ok:
            entry["value"] = result.value
            payload["value"] = result.value
        elif result is not None and result.error:
            entry["error"] = result.error
            payload["error"] = result.error
        if outcome.decision_verdict != "ALLOW":
            entry["reasons"] = list(outcome.reasons)
        # Escalation semantics: an ESCALATE verdict is escalation; so is a
        # SUCCESSFUL governed handoff (the brain proposed escalate_to_human
        # and policy ALLOWed it — the "I'm connecting you to a human" line
        # must always have a real, auditable action behind it). A blocked
        # handoff is not an escalation.
        escalated = (outcome.decision_verdict == "ESCALATE"
                     or (action == "escalate_to_human"
                         and outcome.decision_verdict == "ALLOW"
                         and bool(outcome.executed)))
        return payload, entry, escalated
