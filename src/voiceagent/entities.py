# src/voiceagent/entities.py
"""Deterministic entity extraction from customer text — the inputs the
policy engine needs (amount, order id) to make a real decision instead of
assuming "no amount, unauthenticated" for everything.

M5b-3: ASR engines may speak numbers as WORDS (Qwen3-ASR writes
"ORD four thousand eight hundred twenty one"; IndicConformer writes Telugu
number words; whisper-hi may emit Devanagari digits ४८२१). Number-words are
normalized to digits before the regexes run: English scale form
("four thousand eight hundred twenty one"), digit-list form ("four eight
two one"), Devanagari digits, and — since 2026-09 — compositional
Telugu/Tamil/Bengali/Thai words (tens + units accumulate: ఇరవై ఒకటి ->
21) plus native-script digits for 9 scripts. Thai compounds are single
orthographic words, longest-match split first (see _space_thai_numbers).
New languages arrive as flat value tables + digit rows (data, never parser
branches); the scale engine reads the unified lookups only."""
from __future__ import annotations

import re
from dataclasses import dataclass

from voiceagent.tenant import DEFAULT_CURRENCY

# Native-script digits -> ASCII (whisper-hi / Qwen-hi sometimes emit ४८२१;
# Indic conformers emit Telugu/Tamil/Bengali/Thai digits the same way).
# One combined map applied everywhere Devanagari-only used to be.
_SCRIPT_DIGITS = {
    "०१२३४५६७८९": "0123456789",  # Devanagari
    "౦౧౨౩౪౫౬౭౮౯": "0123456789",  # Telugu
    "௦௧௨௩௪௫௬௭௮௯": "0123456789",  # Tamil
    "০১২৩৪৫৬৭৮৯": "0123456789",  # Bengali
    "๐๑๒๓๔๕๖๗๘๙": "0123456789",  # Thai
    "૦૧૨૩૪૫૬૭૮૯": "0123456789",  # Gujarati
    "೦೧೨೩೪೫೬೭೮೯": "0123456789",  # Kannada
    "൦൧൨൩൪൫൬൭൮൯": "0123456789",  # Malayalam
    "੦੧੨੩੪੫੬੭੮੯": "0123456789",  # Gurmukhi (Punjabi)
}
_NATIVE_DIGITS = str.maketrans("".join(_SCRIPT_DIGITS),
                               "".join(_SCRIPT_DIGITS.values()))
_DEVANAGARI_DIGITS = _NATIVE_DIGITS  # historical name (same map)

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

# Compositional number words (te/ta/bn/th): tens + units accumulate in the
# scale engine (ఇరవై ఒకటి -> 20+1), so these are FLAT value tables, not
# 0-99 enumerations. Initial sets (2026-09) — same standing as the
# synthetic lexicons elsewhere: expand from real transcripts as observed.
# Thai compounds are single orthographic words (ยี่สิบเอ็ด); see
# _space_thai_numbers, which longest-match splits them before this table.
_TE_WORDS = {
    "సున్నా": 0, "ఒకటి": 1, "ఒక": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4,
    "ఐదు": 5, "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9,
    "పది": 10, "పదకొండు": 11, "పన్నెండు": 12, "పదమూడు": 13,
    "పద్నాలుగు": 14, "పదిహేను": 15, "పదహారు": 16, "పదిహేడు": 17,
    "పద్దెనిమిది": 18, "పందొమ్మిది": 19, "ఇరవై": 20, "ముప్పై": 30,
    "నలభై": 40, "యాభై": 50, "అరవై": 60, "డెబ్బై": 70, "ఎనభై": 80,
    "తొంభై": 90,
}
_TE_SCALES = {"వంద": 100, "వెయ్యి": 1000, "వేలు": 1000, "లక్ష": 100000,
              "కోటి": 10000000}
_TA_WORDS = {
    "பூஜ்ஜியம்": 0, "சுழியம்": 0, "ஒன்று": 1, "ஒரு": 1, "இரண்டு": 2,
    "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5, "ஆறு": 6, "ஏழு": 7,
    "எட்டு": 8, "ஒன்பது": 9, "பத்து": 10, "பதினொன்று": 11,
    "பன்னிரண்டு": 12, "பதிமூன்று": 13, "பதினான்கு": 14, "பதினைந்து": 15,
    "பதினாறு": 16, "பதினேழு": 17, "பதினெட்டு": 18, "பத்தொன்பது": 19,
    "இருபது": 20, "முப்பது": 30, "நாற்பது": 40, "ஐம்பது": 50,
    "அறுபது": 60, "எழுபது": 70, "எண்பது": 80, "தொண்ணூறு": 90,
}
_TA_SCALES = {"நூறு": 100, "ஆயிரம்": 1000, "லட்சம்": 100000,
              "கோடி": 10000000}
_BN_WORDS = {
    "শূন্য": 0, "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5,
    "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9, "দশ": 10, "এগারো": 11,
    "বারো": 12, "তেরো": 13, "চৌদ্দ": 14, "পনেরো": 15, "ষোলো": 16,
    "সতেরো": 17, "আঠারো": 18, "উনিশ": 19, "বিশ": 20, "কুড়ি": 20,
    "তিরিশ": 30, "চল্লিশ": 40, "পঞ্চাশ": 50, "ষাট": 60, "সত্তর": 70,
    "আশি": 80, "নব্বই": 90,
}
_BN_SCALES = {"শত": 100, "হাজার": 1000, "লাখ": 100000, "লক্ষ": 100000,
              "কোটি": 10000000}
_TH_WORDS = {
    "ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "สาม": 3, "สี่": 4,
    "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9, "สิบ": 10,
    "ยี่สิบ": 20, "สามสิบ": 30, "สี่สิบ": 40, "ห้าสิบ": 50, "หกสิบ": 60,
    "เจ็ดสิบ": 70, "แปดสิบ": 80, "เก้าสิบ": 90,
}
_TH_SCALES = {"ร้อย": 100, "พัน": 1000, "หมื่น": 10000, "แสน": 100000,
              "ล้าน": 1000000}

# Unified lookups: every engine check below reads these, so a new language
# is a table above — never an `or w == ...` branch in the parser.
_WORD_VALUES: dict[str, int] = {**_NUM_WORDS, **_HI_NUM_WORDS, **_TE_WORDS,
                                **_TA_WORDS, **_BN_WORDS, **_TH_WORDS}
_SCALE_VALUES: dict[str, int] = {**_SCALES, **_HI_SCALES, **_TE_SCALES,
                                 **_TA_SCALES, **_BN_SCALES, **_TH_SCALES}
_HUNDRED_WORDS = frozenset({"hundred", "सौ", "వంద", "நூறு", "শত", "ร้อย"})

# Longest-match vocabulary for Thai runs (no whitespace in Thai script).
_THAI_VOCAB = tuple(sorted(set(_TH_WORDS) | set(_TH_SCALES), key=len,
                           reverse=True))


def _space_thai_numbers(text: str) -> str:
    """Insert spaces around Thai number words inside Thai-script runs.
    Thai compounds are single orthographic words (ยี่สิบเอ็ด = 21); greedy
    longest-match decomposes them into table entries the scale engine
    accumulates (20+1). Non-number Thai text passes through byte-identical.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if "\u0e00" <= ch <= "\u0e7f":
            matched = None
            for word in _THAI_VOCAB:
                if text.startswith(word, i):
                    matched = word
                    break
            if matched is not None:
                out.append(" " + matched + " ")
                i += len(matched)
                continue
            out.append(ch)
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)

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
    if w in _WORD_VALUES or w in _SCALE_VALUES or w == "and":
        return w
    if w.isdigit():
        return w
    return None


def _token_value(w: str) -> int | None:
    v = _WORD_VALUES.get(w)
    if v is not None:
        return v
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
        if w in _SCALE_VALUES:
            scale = _SCALE_VALUES[w]
            scale_total += max(current, 1) * scale
            current = 0
            scale_seen = True
            unit_digits = []
        elif w in _HUNDRED_WORDS:
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
    """Bare scale phrases ('6 हजार', 'ఐదు వేలు', 'ห้าพัน' without the
    currency word captured) — in support speech an 'X scale' phrase is
    money. Requires a NON-English scale >= 1000 (Hindi/Telugu/Tamil/
    Bengali/Thai ASR drops currency words; bare English scale words keep
    requiring explicit currency anchoring so amounts never leak across the
    currency isolation boundary). Hindi notes preserved: long multi-part
    phrases without an explicit digit are order-id-shaped ('पचपन हजार छह
    सौ इकहत्तर' = an ORD id, not ₹55,671). Money shorthand is '6 हजार'
    (digit present) or a short phrase ('पाँच हजार')."""
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
        # Bare-phrase convention is NON-English only: Hindi/Telugu/Tamil/
        # Bengali/Thai ASR often drops the currency word ("6 हजार"), while
        # bare English scale words ("five thousand dollars" on a ₹ tenant)
        # must keep requiring explicit currency anchoring — otherwise
        # amounts leak across the currency isolation boundary.
        if not any(w in _SCALE_VALUES and w not in _SCALES
                   and _SCALE_VALUES[w] >= 1000 for w in run):
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
# behaviour is pinned by tests). Later entries are additive extensions:
# Tamil/Telugu rupee forms (South-Indian tenants) and £/¥ word forms —
# each form mints amounts only for its own currency.
_CURRENCY_WORDS: dict[str, tuple[str, ...]] = {
    "₹": (r"rs\.?", r"rupees?", r"रुपये?", r"रु\.?",
          r"ரூபாய்", r"ரூ\.?", r"రూపాయలు", r"రూ\.?", r"টাকা"),
    "$": (r"dollars?", r"usd?"),
    "€": (r"euros?", r"eur"),
    "£": (r"pounds?", r"gbp?", r"sterling"),
    "¥": (r"yuan?", r"renminbi?", r"yen"),
    "฿": (r"บาท",),
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
    if not any(t in _SCALE_VALUES for t in tokens):
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
        if not run or not any(w in _SCALE_VALUES for w in run):
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
    text = _space_thai_numbers(text.translate(_NATIVE_DIGITS))
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


_DIGIT_TOKEN_RE = re.compile(
    r"[A-Za-z"
    r"\u0900-\u097F"  # Devanagari
    r"\u0C00-\u0C7F"  # Telugu
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0980-\u09FF"  # Bengali
    r"\u0E00-\u0E7F"  # Thai
    r"\u0A80-\u0AFF"  # Gujarati
    r"\u0C80-\u0CFF"  # Kannada
    r"\u0D00-\u0D7F"  # Malayalam
    r"\u0A00-\u0A7F"  # Gurmukhi
    r"0-9]+")


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
    order_id, _ = _order_id_span(
        _space_thai_numbers(text.translate(_NATIVE_DIGITS)))
    if order_id:
        return order_id
    if candidate_orders:
        return _snap_order_id(
            _space_thai_numbers(text.translate(_NATIVE_DIGITS)),
            candidate_orders, min_confidence)
    return None
