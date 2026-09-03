# src/voiceagent/dialogue.py — Sprint A / WS3: task-oriented dialogue state.
"""Deterministic slot tracking for multi-turn task flows.

Small LLMs lose multi-turn state; a raw transcript prompt has no notion of
"the order id is collected, the date is not". This module owns the state:

    Slot: EMPTY -> FILLED -> CONFIRMED
    Workflow: named slots + an optional confirmation gate
    DialogueTracker.update(user_text, entities, auth_state)  # collect
    DialogueTracker.next_step() -> Directive                 # decide

Directives are DETERMINISTIC — ASK_SLOT / CONFIRM_ACTION / EXECUTE_READY /
ESCALATE_TO_HUMAN — and the agent's reply generator stays anchored to them
(phrase the ask, don't decide the task). Entity snapping (entities.
extract_order_id with the caller's candidate orders) repairs ASR garbles
before a slot is ever marked FILLED.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from voiceagent.entities import Entities, extract_order_id


@dataclass
class Slot:
    name: str
    value: object = None
    status: str = "EMPTY"  # EMPTY | FILLED | CONFIRMED


@dataclass
class Directive:
    kind: str  # ASK_SLOT | CONFIRM_ACTION | EXECUTE_READY | ESCALATE_TO_HUMAN
    slot: str | None = None
    hint: str | None = None
    action: str | None = None
    details: str | None = None
    payload: dict | None = None
    reason: str | None = None


WORKFLOWS: dict[str, dict] = {
    "order_status": {
        "slots": ["order_id"], "action": "order_status",
        "tool": "fetch_order_status", "confirm": False,
        "hints": {"order_id": "May I have your order ID?"},
    },
    "reschedule_delivery": {
        "slots": ["order_id", "new_date", "reconfirmed"], "action": "reschedule_delivery",
        "tool": "reschedule_delivery", "confirm": True,
        "hints": {"order_id": "May I have your order ID?",
                  "new_date": "Which day works for the redelivery?"},
    },
    "cancel_order": {
        "slots": ["order_id", "reason", "auth_verified"], "action": "cancel_order",
        "tool": "cancel_order", "confirm": True,
        "hints": {"order_id": "May I have your order ID?",
                  "reason": "May I ask why you want to cancel it?",
                  "auth_verified": "I need to verify your identity first — "
                                   "I've sent an OTP to your registered phone."},
    },
    "refund": {
        "slots": ["order_id", "amount", "reason", "auth_verified"],
        "action": "refund", "tool": "initiate_refund", "confirm": True,
        "hints": {"order_id": "May I have your order ID?",
                  "amount": "What refund amount are you expecting?",
                  "reason": "May I ask the reason for the refund?",
                  "auth_verified": "I need to verify your identity first — "
                                   "I've sent an OTP to your registered phone."},
    },
}

_REASON_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("damaged", "damaged"), ("defective", "damaged"), ("broken", "damaged"),
    ("late", "delayed"), ("delayed", "delayed"), ("delay", "delayed"),
    ("wrong item", "wrong_item"), ("wrong", "wrong_item"),
    ("changed my mind", "customer_cancelled"),
    ("not needed", "customer_cancelled"), ("no longer", "customer_cancelled"),
)
_YES = {"yes", "yeah", "yep", "sure", "ok", "okay", "haan", "haanji", "ji"}
_NO = {"no", "nope", "nahi", "nahin", "stop", "cancel that"}


def _parse_date_phrase(text: str) -> str | None:
    """Relative date phrases + ISO dates -> ISO date string. Support speech
    says 'tomorrow', not 2026-09-06."""
    import datetime
    low = text.lower()
    today = datetime.date.today()
    # Sprint A wiring: Hinglish/Hindi relative words. "kal" is
    # tomorrow-or-yesterday in Hindi; inside a reschedule workflow it is
    # always read as the future. Longest phrase first.
    if "day after tomorrow" in low or re.search(r"\bparso(n)?\b", low):
        return (today + datetime.timedelta(days=2)).isoformat()
    if "tomorrow" in low or re.search(r"\bkal\b", low):
        return (today + datetime.timedelta(days=1)).isoformat()
    if "today" in low or re.search(r"\baaj\b", low):
        return today.isoformat()
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return m.group(0) if m else None


def _parse_reason(text: str) -> str | None:
    low = text.lower()
    for kw, reason in _REASON_KEYWORDS:
        if kw in low:
            return reason
    return None


def _is_confirmation(text: str) -> bool | None:
    """True (yes) / False (no) / None (not a confirmation turn)."""
    low = text.lower().strip(" .!?")
    tokens = {tok.strip(".,!?;:\"'") for tok in low.split()}
    if tokens & _NO or low in _NO:
        return False
    if tokens & _YES or "confirm" in low:
        return True
    return None



@dataclass
class DialogueTracker:
    """Owns one task workflow's slot state for one conversation."""

    workflow: str
    candidate_orders: list[str] | None = None
    slots: dict[str, Slot] = field(default_factory=dict)
    _awaiting_confirmation: bool = False
    _confirmed: bool = False
    _declined: bool = False

    def __post_init__(self) -> None:
        if self.workflow not in WORKFLOWS:
            raise ValueError(f"unknown workflow: {self.workflow}")
        self.slots = {name: Slot(name)
                      for name in WORKFLOWS[self.workflow]["slots"]}

    # -- state collection ---------------------------------------------------

    def update(self, user_text: str, entities: Entities | None = None,
               auth_state: bool = False) -> None:
        """Fold one customer turn into the slot state. Fills slots from
        extracted/snapped entities and keyword parsing; handles the
        confirmation gate."""
        entities = entities or Entities()
        wf = WORKFLOWS[self.workflow]

        # Confirmation gate handling comes first: while awaiting
        # confirmation, a yes/no is the whole turn.
        if self._awaiting_confirmation:
            answer = _is_confirmation(user_text)
            if answer is True:
                self._confirmed = True
                if "reconfirmed" in self.slots:
                    self.slots["reconfirmed"].value = True
                    self.slots["reconfirmed"].status = "CONFIRMED"
            elif answer is False:
                self._declined = True
            # a non-committal turn while awaiting: stay in the gate
            return

        if "order_id" in self.slots and self.slots["order_id"].status == "EMPTY":
            oid = entities.order_id or extract_order_id(
                user_text, self.candidate_orders)
            if oid:
                self.slots["order_id"].value = oid
                self.slots["order_id"].status = "FILLED"

        if "amount" in self.slots and self.slots["amount"].status == "EMPTY":
            if entities.amount is not None:
                self.slots["amount"].value = entities.amount
                self.slots["amount"].status = "FILLED"

        if "reason" in self.slots and self.slots["reason"].status == "EMPTY":
            reason = _parse_reason(user_text)
            if reason:
                self.slots["reason"].value = reason
                self.slots["reason"].status = "FILLED"

        if "new_date" in self.slots and self.slots["new_date"].status == "EMPTY":
            date = _parse_date_phrase(user_text)
            if date:
                self.slots["new_date"].value = date
                self.slots["new_date"].status = "FILLED"

        if "auth_verified" in self.slots and \
                self.slots["auth_verified"].status == "EMPTY" and auth_state:
            self.slots["auth_verified"].value = True
            self.slots["auth_verified"].status = "FILLED"

    # -- next deterministic directive ----------------------------------------

    def next_step(self) -> Directive:
        wf = WORKFLOWS[self.workflow]
        if self._declined:
            return Directive(kind="ESCALATE_TO_HUMAN",
                             reason="customer declined the proposed action")
        # Ask for the first collectable slot ('reconfirmed' is the
        # confirmation gate, never asked for as data).
        askable = [s for s in self.slots.values()
                   if s.status == "EMPTY" and s.name != "reconfirmed"]
        if askable:
            slot = askable[0]
            return Directive(kind="ASK_SLOT", slot=slot.name,
                             hint=wf["hints"].get(slot.name))
        if wf["confirm"] and not self._confirmed:
            self._awaiting_confirmation = True
            return Directive(kind="CONFIRM_ACTION", action=wf["action"],
                             details=self._summary())
        payload = {name: s.value for name, s in self.slots.items()
                   if name != "reconfirmed"}
        return Directive(kind="EXECUTE_READY", action=wf["action"],
                         payload=payload)

    def _summary(self) -> str:
        parts = [f"{name}={slot.value}" for name, slot in self.slots.items()
                 if slot.value is not None and name != "reconfirmed"]
        return ", ".join(parts) if parts else "the proposed action"
