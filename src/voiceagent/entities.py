# src/voiceagent/entities.py
"""Deterministic entity extraction from customer text — the inputs the
policy engine needs (amount, order id) to make a real decision instead of
assuming "no amount, unauthenticated" for everything."""
from __future__ import annotations

import re
from dataclasses import dataclass

_AMOUNT_RE = re.compile(
    r"(?:₹|rs\.?|rupees?|रुपये?|रु\.?)?\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:₹|rs\.?|rupees?|रुपये?|रु\.?)?",
    re.IGNORECASE,
)
_ORDER_RE = re.compile(r"\b(?:ORD[-#]?\s*)(\d{4,10})\b", re.IGNORECASE)


@dataclass
class Entities:
    amount: float | None = None
    order_id: str | None = None


def extract_entities(text: str) -> Entities:
    """Extract a rupee amount and an order id (ORD-xxxxx) from customer text.
    Pure regex, no LLM — deterministic and cheap."""
    amount: float | None = None
    for m in _AMOUNT_RE.finditer(text):
        candidate = float(m.group(1).replace(",", ""))
        # Guard: a bare number like "4" in "plan 4" is not a refund amount.
        # Only accept amounts >= 100 (₹100 minimum meaningful transaction).
        if candidate >= 100:
            amount = candidate
            break

    order_id: str | None = None
    m = _ORDER_RE.search(text)
    if m:
        order_id = f"ORD-{m.group(1)}"

    return Entities(amount=amount, order_id=order_id)
