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
import logging
from dataclasses import dataclass, field
from typing import Any

from voiceagent.decisionlog import DecisionEntry, DecisionLog
from voiceagent.dialogue import DialogueTracker, render_directive
from voiceagent.knowledge_rag import (
    KNOWLEDGE_BUDGET_CHARS,
    cap_knowledge,
    retrieve_chunks,
)
from voiceagent.learn.corrections import classify_correction
from voiceagent.learn.profiles import Profile, ProfileStore, contact_key
from voiceagent.memory import (CAPTURE_CONFIDENCE_THRESHOLD,
                               ConversationMemory, InMemoryMemory, Turn,
                               _sidecar_classifier, now_ts)
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

logger = logging.getLogger(__name__)

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
    # RAG phase 1: the chunk index for KBs that outgrow the whole-file
    # budget (built by runtime's retrieval switch). When set, deploy() renders
    # NO static knowledge block — handle_turn retrieves the top-K chunks per
    # turn and injects them (with fail-open to the whole-file cap on error).
    chunked_knowledge: Any | None = None
    # Declared greeting (tenant data): spoken instantly on pickup, no brain
    # roundtrip. Empty -> greeting is one governed brain turn (legacy path).
    greeting: str = ""
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
    # Task D4 (knowledge provenance): WHICH knowledge ids from the deployed
    # knowledge block were in this turn's system prompt context — so the
    # telephony turn logger and tests can trace which KB documents informed a
    # reply. (Spoken citations stay out of scope; this is observability.)
    knowledge_ids: list[str] = field(default_factory=list)
    # RAG phase 1 (chunked deployments): the chunk ids actually injected into
    # THIS turn's system prompt, similarity-ordered — per-claim provenance.
    # Empty for whole-file (non-chunked) deployments; [] with a non-empty
    # knowledge_gaps also marks the "nothing matched" turn.
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    # RAG phase 1 gap detection: when NO chunk cleared the similarity floor,
    # the user text (truncated) is recorded here so the telephony turn
    # logger (actions/knowledge fields) can surface KB coverage holes.
    knowledge_gaps: list[str] = field(default_factory=list)


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
                 actions: list[str] | None = None,
                 intent_memory: Any | None = None,
                 intent_classifier: Any | None = None):
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
        # ADR-002: the learned intent memory (IntentMemoryStore) — episodic
        # capture of low-confidence turns via the SIDECAR classifier and
        # prototype retrieval for classifier seeding. None = the memory layer
        # is inert (opt-in via VOICEAGENT_MEMORY_DB). Retrieval on the
        # frontier path is pending (see ADR-002 "Current deviations"): the
        # frontier brain classifies nothing, so this seam today feeds capture
        # only; retrieval via classifier_exemplars applies to the Agent path.
        self.intent_memory = intent_memory
        # Sidecar classifier (M4): built lazily on first capture so an
        # orchestrator without wired memory never loads the encoders;
        # injectable for tests.
        self._intent_classifier = intent_classifier
        # Dialogue state (Task B): the bounded not-found clarify-and-dig
        # ladder. Inert unless the wired policy declares not_found_ladder —
        # absent config keeps the pre-ladder single-miss behavior.
        self._dialogue = DialogueTracker()
        self._profile_links: dict[str, str] = {}  # session_id -> contact key
        self._gateway_tools: dict[str, dict] = {}
        self._deployment: Deployment | None = None
        self._sessions: dict[str, BlackboardState] = {}

    @property
    def greeting(self) -> str:
        """The deployment's declared greeting (tenant data); '' when none."""
        return self._deployment.greeting if self._deployment else ""

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
            # description precedence: tenant tools.yaml > the binding's own
            # ToolSpec.description > generic wording.
            from voiceagent.tools import DEFAULT_TOOL_SPECS
            fallback = DEFAULT_TOOL_SPECS.get(tool_name)
            generic = (f"Governed action '{meta.get('action', tool_name)}' — "
                       "proposals only; executed through the policy-governed "
                       "runner.")
            self.brain.register_tool(
                name=tool_name,
                description=meta.get(
                    "description",
                    (fallback.description if fallback and fallback.description
                     else generic)),
                parameters=meta.get("parameters"),
                handler=None,  # governed tools are NEVER brain-executed
            )
        # The bridge keeps its system prompt private by design (frontier.py is
        # frozen), so deployment writes through this seam.
        parts = [deployment.system_prompt]
        # RAG phase 1: chunked deployments render NO static knowledge block —
        # handle_turn injects the per-turn top-K retrieval instead.
        if deployment.chunked_knowledge is None and deployment.knowledge:
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
        # RAG phase 1: chunked deployments retrieve the top-K knowledge
        # chunks for THIS turn and append them to the system prompt (the
        # whole-file deployments' static block was rendered by deploy()).
        turn_chunk_ids: list[str] = []
        turn_gaps: list[str] = []
        turn_chunk_file_ids: list[str] = []
        if self._deployment is not None \
                and self._deployment.chunked_knowledge is not None:
            block, turn_chunk_ids, turn_gaps, turn_chunk_file_ids = \
                self._turn_knowledge_block(user_text)
            if block:
                messages[0] = {"role": "system",
                               "content": messages[0]["content"]
                                          + "\n\n" + block}
        tools = self.brain.tool_schemas() or None

        actions: list[dict] = []
        latency = 0.0
        raw_tool_calls = 0
        escalated = False
        directive_text: str | None = None

        reply = self._chat(messages, tools)
        latency += reply.latency_s
        rounds = 0
        # DENY-repeat short-circuit: a brain that re-proposes an action the
        # policy already DENYed this turn (same action + same args) gets the
        # cached DENY fed back and the loop closes text-only — no policy
        # re-run, no execution, no burned rounds. Different args still get
        # full evaluation (the caller may have fixed the amount).
        denied: set[str] = set()
        while rounds < self.max_tool_rounds and reply.tool_calls:
            rounds += 1
            messages.append(self._assistant_message(reply))
            stop = False
            for call in reply.tool_calls:
                raw_tool_calls += 1
                deny_key = self._deny_key(call)
                if deny_key in denied:
                    reason = ("already denied this turn — do not re-propose; "
                              "answer in text or propose escalate_to_human")
                    payload = {"ok": False, "verdict": "DENY",
                               "reasons": [reason],
                               "error": f"deny_repeat: {deny_key}"}
                    entry = {"action": deny_key.split("|", 1)[0],
                             "tool": call.name, "verdict": "DENY",
                             "ok": False, "error": f"deny_repeat: {deny_key}",
                             "reasons": [reason]}
                    is_escalation = False
                else:
                    payload, entry, is_escalation = self._dispatch_tool_call(
                        call, state, session_id, frustration)
                    if entry is not None and entry.get("verdict") == "DENY":
                        denied.add(self._deny_key(call, entry))
                if entry is not None:
                    actions.append(entry)
                    if (entry.get("action") == "record_feedback"
                            and entry.get("ok") and self.intent_memory
                            is not None):
                        try:
                            tenant = ((self._deployment.metadata or {})
                                      .get("tenant") or "default")
                            self.intent_memory.record_rating(
                                tenant, session_id,
                                float((call.arguments or {})
                                      .get("rating", 0)),
                                str((call.arguments or {})
                                    .get("comment", "")))
                        except Exception:
                            logger.warning(
                                "intent memory: rating record failed "
                                "(fail-open)", exc_info=True)
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

        # M4 (ADR-002): episodic capture on the frontier path — the brain
        # produces no (label, confidence) pair, so the SIDECAR classifier
        # runs purely for memory (never for decisions). Same capture policy
        # and fail-open wrapper as the Agent path.
        if self.intent_memory is not None:
            self._capture_intent_episode(
                user_text, primary["action"] if primary else None,
                session_id=session_id)

        # Task D4: the knowledge ids that entered this turn's system prompt
        # (deploy() renders the whole block) — recorded for provenance even
        # when the reply needs none of it. Declaration order == prompt order.
        # RAG phase 1 chunked deployments instead carry the source files of
        # the chunks actually injected this turn (or the fail-open fallback
        # files); knowledge_ids for whole-file deployments is UNCHANGED.
        if self._deployment is not None \
                and self._deployment.chunked_knowledge is not None:
            knowledge_ids = turn_chunk_file_ids
        else:
            knowledge_ids = (list(self._deployment.knowledge)
                             if (self._deployment is not None
                                 and self._deployment.knowledge) else [])

        return TurnResult(reply=final_text, actions=actions,
                          brain_latency_s=latency, session_id=session_id,
                          raw_tool_calls=raw_tool_calls, escalated=escalated,
                          knowledge_ids=knowledge_ids,
                          retrieved_chunk_ids=turn_chunk_ids,
                          knowledge_gaps=turn_gaps)

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

    def _turn_knowledge_block(self, user_text: str) -> tuple[str, list[str], list[str], list[str]]:
        """RAG phase 1 per-turn knowledge for chunked deployments: retrieve
        the top-K chunks matching the user text by cosine and render them in
        the historical knowledge-block pattern ("- [chunk_id] text").

        Returns (system-prompt block or "", retrieved_chunk_ids,
        knowledge_gaps, knowledge_file_ids). Gap detection: when NO chunk
        clears the deployment's similarity floor, the block notes the miss
        (the brain must not invent) and the user text (truncated) is recorded
        in knowledge_gaps. FAIL-OPEN: any retrieval error falls back to the
        highest-priority whole files that fit under the budget — exactly
        today's _cap_knowledge behavior (whole-file ids become the turn's
        knowledge_ids; no gap is claimed, the failure is not evidence of one)."""
        ck = self._deployment.chunked_knowledge
        try:
            hits = retrieve_chunks(ck, user_text)
        except Exception:
            logger.warning("knowledge retrieval failed; falling back to "
                           "the whole-file cap (fail-open)", exc_info=True)
            if self.metrics is not None:
                self.metrics.note("rag_fallback")
            capped = cap_knowledge(ck.source_texts, KNOWLEDGE_BUDGET_CHARS)
            if not capped:
                return "", [], [], []
            block = "## Knowledge\n" + "\n".join(
                f"- [{kid}] {text}" for kid, text in capped.items())
            return block, [], [], list(capped)
        if hits:
            # Grounding instruction: retrieved context outranks the model's
            # priors — answer from these chunks, never beyond them.
            block = (
                "## Knowledge (retrieved for this question — answer ONLY "
                "from these entries; if they do not cover it, say so and "
                "offer a human)\n" + "\n".join(
                    f"- [{chunk.chunk_id}] {chunk.text}"
                    for chunk, _ in hits))
            chunk_ids = [chunk.chunk_id for chunk, _ in hits]
            # per-turn file provenance: source files of the injected chunks,
            # first-appearance order (== prompt order)
            file_ids = list(dict.fromkeys(
                chunk.source_file_id for chunk, _ in hits))
            return block, chunk_ids, [], file_ids
        gap_block = (
            "## Knowledge\n"
            "(No knowledge base entry matched this question — do not invent "
            "an answer; say you will check and offer to connect the caller "
            "with a human colleague.)")
        return gap_block, [], [user_text[:200]], []

    def _capture_intent_episode(self, user_text: str,
                                outcome_action: str | None,
                                session_id: str = "") -> None:
        """M4 (ADR-002): capture one episodic fragment on the frontier-brain
        path. The local (sidecar) classifier labels the utterance ONLY to
        feed the memory store — its output never touches the decision path.
        Same policy as the Agent capture site: low-confidence or unknown
        turns are the learning candidates. FAIL-OPEN: any error (classifier
        load, store write) is logged and swallowed — never breaks the turn."""
        try:
            clf = self._intent_classifier
            if clf is None:
                clf = _sidecar_classifier()
                self._intent_classifier = clf
            label, confidence = clf.classify(user_text)
            if confidence < CAPTURE_CONFIDENCE_THRESHOLD or not label:
                tenant = "default"
                if self._deployment is not None:
                    tenant = ((self._deployment.metadata or {})
                              .get("tenant") or "default")
                self.intent_memory.capture(
                    tenant, user_text, label or "", float(confidence),
                    outcome=outcome_action or "unmatched",
                    session_id=session_id)
        except Exception:
            logger.warning("intent memory: sidecar capture failed "
                           "(fail-open)", exc_info=True)

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
        terminal, never a bot loop, never an invented resource. The slot is
        generic: classic order tools resolve the legacy `order_id` slot with
        byte-identical behavior; ADR-004 domain tools (ToolSpec.resource =
        (type, id_param, getter), e.g. a booking lookup) resolve their own
        id slot and `{type}_not_found` prefix — a new tenant's resources get
        the ladder with zero platform changes."""
        engine = (getattr(self.runner, "policy", None)
                  if self.runner is not None else None)
        ladder = engine.not_found_ladder() if engine is not None else None
        if ladder is None:
            return None
        args = call.arguments or {}
        slot, prefix, instruction = self._ladder_slot(call.name, args)
        if slot is None or prefix is None:
            return None                       # only slot-bearing lookups
        if payload.get("ok"):
            # Slot FILLED: a successful lookup resets the probe counter.
            self._dialogue.found(session_id, slot)
            return None
        error = payload.get("error")
        if not (isinstance(error, str) and error.startswith(prefix)):
            return None
        directive = self._dialogue.not_found(
            session_id, slot, value=str(args.get(slot) or ""),
            max_retries=ladder["max_retries"],
            alternates=(ladder["alternates"]
                        if ladder["offer_alternates"] else []))
        if directive.kind == "escalate":
            payload["not_found_ladder_exhausted"] = True
            payload["instruction"] = instruction
            return None
        return render_directive(directive)

    def _deny_key(self, call: FrontierToolCall,
                    entry: dict | None = None) -> str:
        """Identity of a denied proposal: action + canonical args. A repeat
        with different args is a NEW proposal (full evaluation); only the
        exact repeat short-circuits."""
        action = (entry or {}).get("action")
        if action is None:
            gmeta = self._gateway_tools.get(call.name) or {}
            action = gmeta.get("action", call.name)
        try:
            args = json.dumps(call.arguments or {}, sort_keys=True,
                              default=str)
        except Exception:
            args = ""
        return f"{action}|{args}"

    def _ladder_slot(self, tool_name: str,
                     args: dict) -> tuple[str | None, str | None, str]:
        """Resolve (slot, error-prefix, exhaustion-instruction) for a tool.

        Legacy order tools keep byte-identical behavior (slot `order_id`,
        `order_not_found`, the pinned instruction). Domain tools carrying a
        ToolSpec.resource resolve (id_param, `{type}_not_found`, a generic
        instruction naming the resource). (None, None, "") when the call
        carries no id slot — the ladder stays inert."""
        if "order_id" in args:
            return ("order_id", "order_not_found",
                    "The order id could not be resolved after repeated "
                    "clarify attempts — propose escalate_to_human with a short "
                    "reason now; do not invent an order.")
        gateway = getattr(self.runner, "gateway", None)
        specs = getattr(gateway, "specs", None) or {}
        resource_spec = getattr(specs.get(tool_name), "resource", None)
        if resource_spec is not None and resource_spec[1] in args:
            rtype, id_param = resource_spec[0], resource_spec[1]
            label = id_param.replace("_", " ")
            return (id_param, f"{rtype}_not_found",
                    f"The {label} could not be resolved after repeated "
                    f"clarify attempts — propose escalate_to_human with a "
                    f"short reason now; do not invent {label}.")
        return None, None, ""

    def _dispatch_tool_call(self, call: FrontierToolCall, state: BlackboardState,
                            session_id: str,
                            frustration: Frustration) -> tuple[dict, dict | None, bool]:
        """Route one tool call. Returns (tool-result payload fed back to the
        brain, TurnResult.actions entry or None, escalated flag)."""
        gmeta = self._gateway_tools.get(call.name)
        if gmeta is not None:
            return self._run_governed(call, gmeta, state, session_id,
                                      frustration)
        # Read-only brain tool: explicit handler execution via the bridge —
        # POLICY-GATED (ADR-003): with a runner wired, the tool NAME is
        # evaluated as its own action and only ALLOW executes. Undeclared
        # names DENY by least privilege, so operators must declare
        # read-only tools in policies.yaml — a mis-registered mutating
        # handler can never execute ungoverned on the live path (the runner
        # is always wired there). Without a runner (unit tests, demos) the
        # legacy direct execution applies.
        if self.runner is not None:
            ctx = PolicyContext(
                authenticated=state.profile.authenticated,
                amount=None,
                signals={"risk_tier": state.profile.risk_tier,
                         "session_id": session_id,
                         "frustrated": frustration.frustrated,
                         "frustration_level": frustration.level},
            )
            decision = self.runner.policy.evaluate(call.name, ctx)
            if decision.verdict != "ALLOW":
                reasons = [f"bridge tool '{call.name}' not allowed: "
                           f"{decision.verdict}"] + list(decision.reasons)
                if self.decision_log is not None:
                    self.decision_log.record(DecisionEntry(
                        ts=now_ts(), conv_id=session_id, action=call.name,
                        verdict="DENY", reasons=list(reasons),
                        authenticated=state.profile.authenticated))
                entry = {"action": call.name, "tool": call.name,
                         "verdict": "DENY", "ok": False,
                         "error": reasons[0], "reasons": reasons}
                return ({"ok": False, "verdict": "DENY",
                         "reasons": reasons, "error": reasons[0]},
                        entry, False)
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
