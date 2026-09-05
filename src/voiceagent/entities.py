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

from voiceagent.tenant import DEFAULT_CURRENCY

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

# Hindi (Devanagari) number words 0-99 — irregular compounds, so this is a
# full table, not composition rules. Scales: सौ/हज़ार(हजार)/लाख/करोड़.
_HI_NUM_WORDS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "उन्नीस": 19,
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43,
    "चवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47,
    "अड़तालीस": 48, "उनचास": 49, "पचास": 50, "इक्यावन": 51, "बावन": 52,
    "तिरपन": 53, "चौवन": 54, "पचपन": 55, "छप्पन": 56, "सत्तावन": 57,
    "अट्ठावन": 58, "उनसठ": 59, "साठ": 60, "इकसठ": 61, "एकसठ": 61,
    "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66,
    "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70, "इकहत्तर": 71,
    "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85,
    "छियासी": 86, "सतासी": 87, "अठासी": 88, "नवासी": 89, "नब्बे": 90,
    "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98,
    "निन्यानवे": 99,
}
_HI_SCALES = {"सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100000,
              "करोड़": 10000000}

# Observed ASR garbles of Hindi number words (from real loopback voice
# transcripts, 2026-09-03): 'एकत्र' is how Qwen3-ASR heard 'इकहत्तर' in
# "ORD-55671" spoken as "पचपन हजार छह सौ इकहत्तर". Extend as new garbles
# are observed — each alias cites the transcript it came from.
_HI_GARBLES = {"एकत्र": "इकहत्तर", "एकतर": "इकहत्तर"}

_NUM_TOKENS = set(_NUM_WORDS) | set(_SCALES) | {"and"}

_ORDER_RE = re.compile(r"\b(?:ORD[-#]?\s*)(\d{4,10})\b", re.IGNORECASE)
# \bORD\b: must not match the "ord" inside the word "order".
_ORDER_PREFIX_RE = re.compile(r"\bORD\b[-#\s:]*", re.IGNORECASE)
_PUNCT = ".,;:!?\"'()[]{}"


def _canon_token(tok: str) -> str | None:
    """Canonical number token (en or hi) from a raw token, else None.
    Pure digits count too ('6 हजार' = 6000)."""
    w = tok.strip(_PUNCT).lower()
    if w in _HI_GARBLES:
        w = _HI_GARBLES[w]
    if (w in _NUM_WORDS or w in _SCALES or w in _HI_NUM_WORDS
            or w in _HI_SCALES or w == "and"):
        return w
    if w.isdigit():
        return w
    return None


def _token_value(w: str) -> int | None:
    if w in _NUM_WORDS:
        return _NUM_WORDS[w]
    if w in _HI_NUM_WORDS:
        return _HI_NUM_WORDS[w]
    if w.isdigit():
        return int(w)
    return None


def _words_after(text: str, prefix_re: re.Pattern) -> list[str]:
    """Canonical number tokens following a prefix match, cut at the first
    non-number word."""
    m = prefix_re.search(text)
    if not m:
        return []
    out = []
    for tok in text[m.end():].split():
        w = _canon_token(tok)
        if w is None:
            break
        out.append(w)
    return out


@dataclass
class Entities:
    amount: float | None = None
    order_id: str | None = None


def words_to_number(tokens: list[str]) -> int | None:
    """Bilingual (English + Hindi) number words -> int: scale form, digit-list
    form, and digit+scale combos ('6 हजार' = 6000).

    Scale form:  ["four","thousand","eight","hundred","twenty","one"] -> 4821.
    Digit-list:  ["four","eight","two","one"] -> 4821 (IDs spoken digit-wise).
    Hindi:       ["पचपन","हजार","छह","सौ","इकहत्तर"] -> 55671.
    Returns None unless EVERY token is a number word — a partial match is
    not a number ("one agent" must not become 1).
    """
    clean = []
    for t in tokens:
        w = _canon_token(t)
        if w is None:
            return None
        clean.append(w)
    scale_total, current, seen = 0, 0, False
    scale_seen = False
    unit_digits: list[int] = []  # consecutive unit words before any scale
    for w in clean:
        if w in _SCALES or w in _HI_SCALES:
            scale = _SCALES.get(w) or _HI_SCALES[w]
            scale_total += max(current, 1) * scale
            current = 0
            scale_seen = True
            unit_digits = []
        elif w == "hundred" or w == "सौ":
            current = max(current, 1) * 100
            scale_seen = True
            unit_digits = []
        elif w == "and":
            continue
        else:
            v = _token_value(w)
            if v is None:
                return None
            current += v
            seen = True
            if not scale_seen and 0 <= v < 10:
                unit_digits.append(v)
    if not seen:
        return None
    # Digit-list reading: every token was a spoken digit ("four eight two one").
    if (not scale_seen and len(unit_digits) == len(clean)
            and len(clean) >= 2):
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
        w = _canon_token(tok.group(0))
        if w is None:
            break
        consumed.append(w)
        end = pm.end() + tok.end()
    n = words_to_number(consumed) if consumed else None
    if n is not None and 4 <= len(str(n)) <= 10:  # same shape as _ORDER_RE
        return f"ORD-{n}", (pm.start(), end)
    return None, None


def _amount_from_bare_hi_phrase(order_text: str) -> float | None:
    """Hindi bare scale phrases ('6 हजार', 'पाँच हजार रुपये' without the
    currency word captured) — in support speech a 'X हजार' phrase is money.
    Requires हजार/लाख/करोड़ (सौ alone is too ambiguous without currency)."""
    runs: list[list[str]] = []
    run: list[str] = []
    for tok in order_text.split():
        w = _canon_token(tok)
        if w is None:
            if run:
                runs.append(run)
                run = []
        else:
            run.append(w)
    if run:
        runs.append(run)
    best: float | None = None
    for run in runs:
        if not any(w in _HI_SCALES and _HI_SCALES[w] >= 1000 for w in run):
            continue
        n = words_to_number(run)
        if n is None or n < 100:
            continue
        # Guard: long multi-part phrases without an explicit digit are
        # order-id-shaped ('पचपन हजार छह सौ इकहत्तर' = an ORD id, not
        # ₹55,671). Money shorthand is '6 हजार' (digit present) or a short
        # phrase ('पाँच हजार').
        if any(t.isdigit() for t in run) or len(run) <= 2:
            best = float(n)
    return best


# Currency WORD forms per currency symbol (regex fragments, `re.IGNORECASE`):
# the words that may introduce or follow a money amount for THAT currency.
# Scoped to the active currency only — "dollars" must not create amounts for
# a ₹ tenant. Keep the rupee alternation byte-identical (hi/Devanagari
# behaviour is pinned by tests).
_CURRENCY_WORDS: dict[str, tuple[str, ...]] = {
    "₹": (r"rs\.?", r"rupees?", r"रुपये?", r"रु\.?"),
    "$": (r"dollars?", r"usd?"),
    "€": (r"euros?", r"eur"),
}


def _currency_word_alts(currency: str) -> str:
    """Regex alternation of the currency's word forms (symbol included)."""
    return "|".join((re.escape(currency),)
                    + _CURRENCY_WORDS.get(currency, ()))


def _amount_from_words(text: str, currency: str = DEFAULT_CURRENCY) -> float | None:
    """Prefix form: 'rupees five thousand and two hundred' -> 5200.00. Same
    >=min guard as the digit path; non-currency number words ("one agent")
    never qualify because the phrase must contain a scale word (hundred+)."""
    tokens = _words_after(text, re.compile(
        r"\b(?:" + _currency_word_alts(currency) + r")\s*", re.IGNORECASE))
    if not any(t in _SCALES for t in tokens):
        return None
    n = words_to_number(tokens)
    return float(n) if n is not None and n >= 100 else None


def _amount_from_words_suffix(text: str, currency: str) -> float | None:
    """Suffix form: 'five thousand dollars' -> 5000.00 — the currency word
    FOLLOWS the phrase (English word order). Same scale-word + >=100 guard
    as the prefix form; words are scoped to the active currency."""
    words = _CURRENCY_WORDS.get(currency, ())
    if not words:
        return None
    cw = re.compile(r"^(?:" + "|".join(words) + r")$", re.IGNORECASE)
    tokens = text.split()
    best: float | None = None
    for i, tok in enumerate(tokens):
        if not cw.match(tok.strip(_PUNCT)):
            continue
        run: list[str] = []
        for t in reversed(tokens[:i]):
            w = _canon_token(t)
            if w is None:
                break
            run.append(w)
        run.reverse()
        if not run or not any(w in _SCALES for w in run):
            continue
        n = words_to_number(run)
        if n is not None and n >= 100:
            best = float(n)
            break
    return best


def extract_entities(text: str, currency: str = DEFAULT_CURRENCY,
                     min_amount: float = 100.0) -> Entities:
    """Extract an amount and an order id (ORD-xxxxx) from customer text.
    Pure regex + number-word normalization, no LLM — deterministic and cheap.

    currency/min_amount are tenant config (M6a): both default to the platform
    defaults (tenant.DEFAULT_CURRENCY, $ 100). The digit regex and the
    money-word patterns are scoped to the ACTIVE currency's word forms
    (dollars/USD for "$", rupees/रुपये for "₹", ...); a bare number >=
    min_amount still counts as an amount either way."""
    text = text.translate(_DEVANAGARI_DIGITS)
    sym = re.escape(currency)
    words = _currency_word_alts(currency)
    amount_re = re.compile(
        r"(?:" + sym + r"|" + words + r")?\s*"
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:" + sym + r"|" + words + r")?",
        re.IGNORECASE,
    )

    order_id, order_span = _order_id_span(text)
    # Cut the order-id span so its number can't double as an amount
    # ("ORD-4821" must not read as ₹4821).
    order_text = text if order_span is None else \
        text[:order_span[0]] + " " + text[order_span[1]:]

    amount: float | None = None
    for m in amount_re.finditer(order_text):
        candidate = float(m.group(1).replace(",", ""))
        # Guard: a bare number like "4" in "plan 4" is not a refund amount.
        # Only accept amounts >= min_amount (the smallest meaningful
        # transaction; ₹/$ 100 by default).
        if candidate >= min_amount:
            amount = candidate
            break
    if amount is None:
        amount = _amount_from_words(order_text, currency)
    if amount is None:
        amount = _amount_from_words_suffix(order_text, currency)
    if amount is None:
        amount = _amount_from_bare_hi_phrase(order_text)

    return Entities(amount=amount, order_id=order_id)


# ---------------------------------------------------------------------------
# Sprint A / WS1: phonetic & contextual entity snapping. ASR garbles order
# references ("or D7734", "ORD 7 7 3 4", "order वाली 4808") and pure regex
# misses them. When the customer's KNOWN candidate orders are available
# (from phone/account context — the telephony trunk knows who is calling),
# digit clusters in the text snap to the closest candidate above a
# confidence threshold; below it we return None rather than guess.
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_DIGIT_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u097F0-9]+")


def _digit_clusters(text: str) -> list[str]:
    """Digit material from the text as the caller spoke it: consecutive
    digit-bearing tokens merge into one cluster ('or D 7 7 3 4' -> '7734',
    'D7734' -> '7734'), so spaced/punctuated digits reconstruct cleanly.
    Letter-only and word tokens break the cluster (phone numbers stay
    separate from order references)."""
    clusters: list[str] = []
    cur = ""
    for tok in _DIGIT_TOKEN_RE.findall(text):
        digits = "".join(ch for ch in tok if ch.isdigit())
        if digits:
            cur += digits
        elif cur:
            clusters.append(cur)
            cur = ""
    if cur:
        clusters.append(cur)
    return clusters


SNAP_MIN_CONFIDENCE = 0.8


def _snap_order_id(text: str, candidate_orders: list[str],
                   min_confidence: float = SNAP_MIN_CONFIDENCE) -> str | None:
    """Snap digit clusters in the text to the closest candidate order
    (Levenshtein similarity on digit sequences, >= min_confidence). Returns
    the canonical candidate id, or None when nothing matches confidently —
    a wrong snap is worse than asking the customer again."""
    best, best_score = None, 0.0
    for cand in candidate_orders:
        cand_digits = "".join(ch for ch in str(cand) if ch.isdigit())
        if not cand_digits:
            continue
        for cluster in _digit_clusters(text):
            dist = _levenshtein(cluster, cand_digits)
            score = 1.0 - dist / max(len(cluster), len(cand_digits))
            if score > best_score:
                best, best_score = str(cand), score
    if best is not None and best_score >= min_confidence:
        return best
    return None


def extract_order_id(text: str,
                     candidate_orders: list[str] | None = None,
                     min_confidence: float = SNAP_MIN_CONFIDENCE) -> str | None:
    """Order-id extraction with contextual snapping.

    1. Exact paths first: clean 'ORD-XXXXX' digits, spaced digits after an
       ORD marker, and Hindi/English number-words (incl. Devanagari digits).
    2. If nothing exact and candidates are known, snap garbled digit
       clusters to the closest candidate above min_confidence.
    """
    order_id, _ = _order_id_span(text.translate(_DEVANAGARI_DIGITS))
    if order_id:
        return order_id
    if candidate_orders:
        return _snap_order_id(text.translate(_DEVANAGARI_DIGITS),
                              candidate_orders, min_confidence)
    return None
