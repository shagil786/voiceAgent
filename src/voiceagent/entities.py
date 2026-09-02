# src/voiceagent/entities.py
"""Deterministic entity extraction from customer text — the inputs the
policy engine needs (amount, order id) to make a real decision instead of
assuming "no amount, unauthenticated" for everything.

M5b-3: ASR engines may speak numbers as WORDS (Qwen3-ASR writes
"ORD four thousand eight hundred twenty one"; IndicConformer writes Telugu
number words; whisper-hi may emit Devanagari digits ४८२१). Number-words are
normalized to digits before the regexes run — currently English scale form
("four thousand eight hundred twenty one"), digit-list form ("four eight
two one"), and Devanagari digits. Hindi/Telugu number WORDS need per-language
0-99 word tables (deferred — data task, not code).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Devanagari digits -> ASCII (whisper-hi / Qwen-hi sometimes emit ४८२१).
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lac": 100000,
           "million": 1000000, "crore": 10000000}
_NUM_TOKENS = set(_NUM_WORDS) | set(_SCALES) | {"and"}

_AMOUNT_RE = re.compile(
    r"(?:₹|rs\.?|rupees?|रुपये?|रु\.?)?\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:₹|rs\.?|rupees?|रुपये?|रु\.?)?",
    re.IGNORECASE,
)
_ORDER_RE = re.compile(r"\b(?:ORD[-#]?\s*)(\d{4,10})\b", re.IGNORECASE)
# \bORD\b: must not match the "ord" inside the word "order".
_ORDER_PREFIX_RE = re.compile(r"\bORD\b[-#\s:]*", re.IGNORECASE)
_PUNCT = ".,;:!?\"'()[]{}"


def _words_after(text: str, prefix_re: re.Pattern) -> list[str]:
    """Tokens following a prefix match, cut at the first non-number word."""
    m = prefix_re.search(text)
    if not m:
        return []
    out = []
    for tok in text[m.end():].split():
        w = tok.strip(_PUNCT).lower()
        if w in _NUM_TOKENS:
            out.append(w)
        else:
            break
    return out


@dataclass
class Entities:
    amount: float | None = None
    order_id: str | None = None


def words_to_number(tokens: list[str]) -> int | None:
    """English number-words -> int, scale form or digit-list form.

    Scale form:  ["four","thousand","eight","hundred","twenty","one"] -> 4821.
    Digit-list:  ["four","eight","two","one"] -> 4821 (IDs are spoken
    digit-by-digit). Returns None unless EVERY token is a number word —
    a partial match is not a number ("one agent" must not become 1).
    """
    clean = [t.strip(_PUNCT).lower() for t in tokens]
    scale_total, current, seen = 0, 0, False
    scale_seen = False
    unit_digits: list[int] = []  # consecutive unit words before any scale
    for w in clean:
        if w in _NUM_WORDS:
            v = _NUM_WORDS[w]
            current += v
            seen = True
            if not scale_seen and v < 10:
                unit_digits.append(v)
        elif w == "hundred":
            current = max(current, 1) * 100
            scale_seen = True
            unit_digits = []
        elif w in _SCALES:
            scale_total += max(current, 1) * _SCALES[w]
            current = 0
            scale_seen = True
            unit_digits = []
        elif w == "and":
            continue
        else:
            return None
    if not seen:
        return None
    # Digit-list reading: every token was a spoken digit ("four eight two one").
    if not scale_seen and len(unit_digits) == len(clean) and len(clean) >= 2:
        return int("".join(str(d) for d in unit_digits))
    return scale_total + current


def _order_id_span(text: str) -> tuple[str | None, tuple[int, int] | None]:
    """Order id + its char span, digit form ('ORD-4821') or number-words form
    ('ORD four thousand eight hundred twenty one')."""
    m = _ORDER_RE.search(text)
    if m:
        return f"ORD-{m.group(1)}", m.span()
    pm = _ORDER_PREFIX_RE.search(text)
    if not pm:
        return None, None
    end = pm.end()
    consumed: list[str] = []
    for tok in re.finditer(r"\S+", text[pm.end():]):
        w = tok.group(0).strip(_PUNCT).lower()
        if w in _NUM_TOKENS:
            consumed.append(w)
            end = pm.end() + tok.end()
        else:
            break
    n = words_to_number(consumed) if consumed else None
    if n is not None and 4 <= len(str(n)) <= 10:  # same shape as _ORDER_RE
        return f"ORD-{n}", (pm.start(), end)
    return None, None


def _amount_from_words(text: str) -> float | None:
    """'rupees five thousand and two hundred' -> 5000.00 (same >=₹100 guard
    as the digit path). Non-currency number words ("one agent") never
    qualify because the phrase must contain a scale word (hundred+)."""
    tokens = _words_after(text, re.compile(
        r"\b(?:₹|rs\.?|rupees?|रुपये?|रु\.?)\s*", re.IGNORECASE))
    if not any(t in _SCALES for t in tokens):
        return None
    n = words_to_number(tokens)
    return float(n) if n is not None and n >= 100 else None


def extract_entities(text: str) -> Entities:
    """Extract a rupee amount and an order id (ORD-xxxxx) from customer text.
    Pure regex + number-word normalization, no LLM — deterministic and cheap."""
    text = text.translate(_DEVANAGARI_DIGITS)

    order_id, order_span = _order_id_span(text)
    # Cut the order-id span so its number can't double as an amount
    # ("ORD-4821" must not read as ₹4821).
    order_text = text if order_span is None else \
        text[:order_span[0]] + " " + text[order_span[1]:]

    amount: float | None = None
    for m in _AMOUNT_RE.finditer(order_text):
        candidate = float(m.group(1).replace(",", ""))
        # Guard: a bare number like "4" in "plan 4" is not a refund amount.
        # Only accept amounts >= 100 (₹100 minimum meaningful transaction).
        if candidate >= 100:
            amount = candidate
            break
    if amount is None:
        amount = _amount_from_words(order_text)

    return Entities(amount=amount, order_id=order_id)
