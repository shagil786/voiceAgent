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
from voiceagent.dialogue import DialogueTracker, render_directive
from voiceagent.learn.corrections import classify_correction
from voiceagent.learn.profiles import Profile, ProfileStore, contact_key
from voiceagent.memory import ConversationMemory, InMemoryMemory, Turn, now_ts
from voiceagent.metrics import Metrics
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


MAX_PENDING_GLOBAL = 50


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
    # The deployment's declared action vocabulary (Sprint A1): resolved from
    # the tenant bundle (intents/ + tools.yaml `action:` + tenant.json
    # extras) by the runtime assembly. None = nothing declared — consumers
    # keep their existing (demo-fallback) behavior. Core ships no business
    # action list.
    actions: list[str] | None = None
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
                 profiles: ProfileStore | None = None,
                 metrics: Metrics | None = None,
                 actions: list[str] | None = None):
        self.brain = brain
        self.runner = runner
        self.memory: ConversationMemory = memory or InMemoryMemory()
        # Audit seam for the no-runner case (the runner logs its own verdicts).
        self.decision_log = decision_log
        self.max_tool_rounds = max_tool_rounds
        # Instant-Learn seam: None = pre-learn behavior (byte-identical replies).
        self.profiles = profiles
        # Runtime metrics sink: None = no recording (zero behavior change).
        self.metrics = metrics
        # The resolved action vocabulary (Sprint A1): declared tenant data
        # passed in by the assembly seam (build_orchestrator / deploy());
        # None = nothing declared, existing behavior. The frontier brain's
        # proposal surface stays tool-schema-driven — this seam exists so the
        # vocabulary is available to placements/prompt builders without
        # re-deriving it from the bundle.
        self.actions: list[str] | None = list(actions) if actions else None
        # Dialogue state (Task B): the bounded not-found clarify-and-dig
        # ladder. Inert unless the wired policy declares not_found_ladder —
        # absent config keeps the pre-ladder single-miss behavior.
        self._dialogue = DialogueTracker()
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
        if deployment.actions:  # declared vocabulary wins; None keeps existing
            self.actions = list(deployment.actions)
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
        # Cross-contact separation: anonymous turns (no alias resolving to
        # an existing profile AND no usable phone) skip the entire profile
        # seam — never share one `cid:unknown` fallback across callers.
        contact: str | None = None
        if self.profiles is not None:
            key: str | None = None
            if contact_alias:
                r = self.profiles.resolve(contact_alias)
                if self.profiles.get(r) is not None:
                    key = r
            if key is None:
                if (state.profile.phone or "").strip():
                    key = contact_key(state.profile)
            if key is not None:
                contact = key
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
        directive_text: str | None = None

        reply = self._chat(messages, tools)
        latency += reply.latency_s
        rounds = 0
        while rounds < self.max_tool_rounds and reply.tool_calls:
            rounds += 1
            messages.append(self._assistant_message(reply))
            stop = False
            for call in reply.tool_calls:
                raw_tool_calls += 1
                payload, entry, is_escalation = self._dispatch_tool_call(
                    call, state, session_id, frustration)
                if entry is not None:
                    actions.append(entry)
                escalated = escalated or is_escalation
                # Task B clarify-and-dig ladder: a not-found slot lookup may
                # emit a bounded clarify directive (re-confirm the id, offer
                # the declared alternate lookups) instead of leaving the
                # first miss to the brain. No-op unless the policy declares
                # not_found_ladder. Runs BEFORE the tool message is appended
                # so an exhausted ladder can annotate the fed-back payload.
                clarify = self._not_found_ladder(payload, call, session_id)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(payload, default=str)})
                if clarify is not None:
                    directive_text = clarify
                    stop = True
                    break
            if stop:
                # The tracker's directive IS the reply: deterministic, no
                # extra frontier round spent improvising on a known miss.
                break
            reply = self._chat(messages, tools)
            latency += reply.latency_s

        if directive_text is None and reply.tool_calls:
            # Round budget exhausted mid-ping-pong: force a text-only close
            # (no tool surface, so the model must speak). Stray calls are
            # surfaced in raw_tool_calls but never executed.
            raw_tool_calls += len(reply.tool_calls)
            reply = self._chat(messages, None)
            latency += reply.latency_s
            raw_tool_calls += len(reply.tool_calls)

        final_text = (directive_text if directive_text is not None
                      else (reply.content or "").strip() or _FALLBACK_REPLY)

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
                del prof.pending_global[:-MAX_PENDING_GLOBAL]
                prof.updated_at = now_ts()
                self.profiles.put(prof)

        # Metrics hook (single site): one sample per turn — latency plus
        # the primary governed verdict, or "none" for plain chat turns.
        if self.metrics is not None:
            self.metrics.record(
                latency, primary["verdict"] if primary else "none")

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
        durable memory cleared and live blackboard state dropped.
        Owner-only: callers must authenticate + audit-log; never expose as a brain tool without an auth check."""
        if self.profiles is None:
            raise RuntimeError("no ProfileStore configured")
        resolved = self.profiles.resolve(contact_or_alias)
        if not (resolved or "").strip():
            return {"sessions": []}
        out = self.profiles.delete_contact(resolved)
        for sid in out["sessions"]:
            self.memory.clear(sid)
            self._sessions.pop(sid, None)
            self._profile_links.pop(sid, None)
        return out

    def export_contact(self, contact_or_alias: str) -> dict:
        """Export one contact's profile dict (KeyError when unknown).
        Owner-only: callers must authenticate + audit-log; never expose as a brain tool without an auth check."""
        if self.profiles is None:
            raise RuntimeError("no ProfileStore configured")
        resolved = self.profiles.resolve(contact_or_alias)
        if not (resolved or "").strip():
            raise KeyError("unknown contact")
        return self.profiles.export_contact(resolved)

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

    def _not_found_ladder(self, payload: dict, call: FrontierToolCall,
                          session_id: str) -> str | None:
        """Task B clarify-and-dig ladder for not-found slot lookups. Returns
        the directive text to serve as the turn's reply (re-confirm ask /
        alternate-lookup offer), or None to keep the existing flow.

        Only active when the wired policy declares the top-level
        `not_found_ladder:` key — absent config leaves the raw not-found tool
        result fed back to the brain exactly as before. The escalation rung
        is NOT rendered here: exhaustion annotates the fed-back payload so
        the brain proposes the governed escalate_to_human (policy verdict,
        DecisionLog audit) as today — escalation stays the mandatory
        terminal, never a bot loop, never an invented order."""
        engine = (getattr(self.runner, "policy", None)
                  if self.runner is not None else None)
        ladder = engine.not_found_ladder() if engine is not None else None
        if ladder is None:
            return None
        args = call.arguments or {}
        if "order_id" not in args:
            return None                       # only slot-bearing lookups
        slot = "order_id"
        if payload.get("ok"):
            # Slot FILLED: a successful lookup resets the probe counter.
            self._dialogue.found(session_id, slot)
            return None
        error = payload.get("error")
        if not (isinstance(error, str) and error.startswith("order_not_found")):
            return None
        directive = self._dialogue.not_found(
            session_id, slot, value=str(args.get("order_id") or ""),
            max_retries=ladder["max_retries"],
            alternates=(ladder["alternates"]
                        if ladder["offer_alternates"] else []))
        if directive.kind == "escalate":
            payload["not_found_ladder_exhausted"] = True
            payload["instruction"] = (
                "The order id could not be resolved after repeated "
                "clarify attempts — propose escalate_to_human with a short "
                "reason now; do not invent an order.")
            return None
        return render_directive(directive)

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
