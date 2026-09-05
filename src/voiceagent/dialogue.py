# src/voiceagent/dialogue.py — Task B: the clarify-and-dig not-found ladder.
"""Dialogue state for the governed turn loop: when a slot-bearing lookup
comes back not-found, the agent must CLARIFY AND DIG before giving up on the
first miss (the "order not found -> instant human" fix).

The DialogueTracker keeps a BOUNDED per-session, per-slot probe counter. Each
not-found probe advances one rung of the ladder (declared in the tenant's
policies.yaml top-level `not_found_ladder:` key, exposed through
PolicyEngine.not_found_ladder()):

    miss 1            -> ask_reconfirm    (re-state / re-confirm the id)
    miss 2..max       -> offer_alternates (declared alternate lookups)
    miss max+1        -> escalate         (the MANDATORY terminal)

Two reset semantics keep the counter bounded and honest:
- the slot is FILLED (a lookup for that slot succeeds) -> counter resets;
- the ladder exhausts -> the episode ends and the counter resets, so a later
  miss starts a fresh bounded ladder (never an unbounded loop).
Escalation itself stays a governed action: the tracker only reports the rung;
the orchestrator still routes any handoff through the PolicyEngine /
GovernedToolRunner / DecisionLog spine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Ladder rungs (directive kinds the orchestrator renders into the reply).
ASK_RECONFIRM = "ask_reconfirm"
OFFER_ALTERNATES = "offer_alternates"
ESCALATE = "escalate"

_DEFAULT_MAX_RETRIES = 2


@dataclass
class LadderDirective:
    """One rung of the not-found ladder, emitted by the tracker and rendered
    for the customer by the orchestrator (directive-style reply, same shape
    of determinism as the ask-slot flow)."""
    kind: str                       # ASK_RECONFIRM | OFFER_ALTERNATES | ESCALATE
    slot: str                       # e.g. "order_id"
    value: str = ""                 # the attempted (unresolved) value
    alternates: tuple[str, ...] = field(default_factory=tuple)


def _clamp_max_retries(max_retries: int) -> int:
    """Bounded by construction: a non-positive / non-int budget falls back to
    the platform default so the ladder always terminates."""
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) \
            or max_retries < 1:
        return _DEFAULT_MAX_RETRIES
    return max_retries


class DialogueTracker:
    """Per-session, per-slot not-found probe counters (bounded, resettable).

    State model mirrors the orchestrator's per-session dict style:
    {session_id: {slot: probe_count}} — in-memory, one entry per open
    clarify-and-dig episode, dropped as soon as the slot resolves or the
    ladder exhausts."""

    def __init__(self) -> None:
        self._probes: dict[str, dict[str, int]] = {}

    def not_found(self, session_id: str, slot: str, value: str = "", *,
                  max_retries: int = _DEFAULT_MAX_RETRIES,
                  alternates: "list[str] | tuple[str, ...]" = ()) \
            -> LadderDirective:
        """Record one not-found probe for (session, slot) and return the rung
        the turn must serve. Callers with offer_alternates disabled simply
        pass no alternates — the tracker then repeats the re-confirm ask
        until the (still bounded) retry budget is spent."""
        budget = _clamp_max_retries(max_retries)
        probes = self._probes.setdefault(session_id, {})
        n = probes.get(slot, 0) + 1
        if n > budget:
            # Mandatory terminal reached: end the episode (bounded), reset,
            # and hand the turn back for the governed escalation path.
            probes.pop(slot, None)
            if not probes:
                self._probes.pop(session_id, None)
            return LadderDirective(ESCALATE, slot, value)
        probes[slot] = n
        alts = tuple(str(a) for a in alternates if str(a).strip())
        if n == 1 or not alts:
            return LadderDirective(ASK_RECONFIRM, slot, value)
        return LadderDirective(OFFER_ALTERNATES, slot, value, alts)

    def found(self, session_id: str, slot: str) -> None:
        """The slot FILLED (a lookup succeeded): reset its probe counter so a
        future not-found for the same slot starts a fresh ladder."""
        probes = self._probes.get(session_id)
        if probes is None:
            return
        probes.pop(slot, None)
        if not probes:
            self._probes.pop(session_id, None)

    def probes(self, session_id: str, slot: str) -> int:
        """Current probe count for (session, slot) — inspection/testing seam."""
        return self._probes.get(session_id, {}).get(slot, 0)


def render_directive(d: LadderDirective) -> str:
    """Render a ladder directive as the customer-facing reply line. Short and
    voice-safe (spoken aloud); the unresolved reference is echoed so the
    customer knows exactly which id failed."""
    if d.kind == ASK_RECONFIRM:
        head = (f"I could not find an order with the reference {d.value}."
                if d.value else "I could not find that order.")
        return head + " Could you confirm the order ID once more?"
    if d.kind == OFFER_ALTERNATES:
        head = (f"I still could not find an order matching {d.value}."
                if d.value else "I still could not find that order.")
        alts = "; or ".join(d.alternates)
        return (head + " A few things we can try instead: " + alts
                + ". Or I can confirm the order ID with you once more.")
    return ""  # ESCALATE is not rendered by the tracker — the governed
    # escalate_to_human action owns that reply.
