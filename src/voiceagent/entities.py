# src/voiceagent/entities.py
"""Deterministic entity extraction from customer text — the inputs the
policy engine needs (amount, record id) to make a real decision instead of
assuming "no amount, unauthenticated" for everything.

M5b-3: ASR engines may speak numbers as WORDS ("ORD four thousand eight
hundred twenty one") or native-script digits. Number-words normalize to
digits before the regexes run: scale form, digit-list form, and
compositional tens+units forms all accumulate in one scale engine over
unified lookups. Every language table (words, scales, digits, currency
forms) loads from data/lang/*.yaml; record-ID shapes load from the
tenant bundle's entities.yaml. New languages/industries arrive as data
files — never parser branches, never literals here. Thai compounds are
single orthographic words, longest-match split first (vocabulary for the
splitter also comes from the loaded tables)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from voiceagent.tenant import DEFAULT_CURRENCY

# Language tables (number words, scales, digits, garbles) live in
# data/lang/*.yaml and load here — add-a-language = add-a-file. The names
# below are the engine's unified lookups over whatever the files declare;
# no per-language branch exists anywhere in this parser.
from voiceagent.langdata import (  # noqa: E402  (data loader is stdlib+yaml)
    digit_map,
    garble_map,
    number_lookups,
    tables as _lang_tables,
    tokenizer_ranges,
)

_WORD_VALUES, _SCALE_VALUES, _HUNDRED_WORDS = number_lookups()
_GARBLES = garble_map()  # ASR-mishearing -> canonical word, all languages
# English scales, for the bare-phrase currency-isolation rule below
# ("five thousand" bare must not mint ₹ — only non-English scales do).
_EN_SCALES = frozenset(
    (_lang_tables().get("en", {}).get("scales") or {}))
_NATIVE_DIGITS = str.maketrans(
    "".join(sorted(digit_map())), "".join(digit_map()[c]
                                          for c in sorted(digit_map())))
_DEVANAGARI_DIGITS = _NATIVE_DIGITS  # historical name (same map)

# Longest-match vocabulary for Thai runs (no whitespace in Thai script):
# every loaded word+scale written in the Thai block, longest first.
_TH_WORDS_TH = frozenset(
    w for w in list(_WORD_VALUES) + list(_SCALE_VALUES)
    if w and "\u0e00" <= w[0] <= "\u0e7f")
_THAI_VOCAB = tuple(
    sorted(_TH_WORDS_TH, key=len, reverse=True))


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

_NUM_TOKENS = set(_WORD_VALUES) | set(_SCALE_VALUES) | {"and"}

# Record-ID shapes: tenant data (Tenant.record_id_shapes over the bundle's
# entities.yaml). No ID literal lives in this module — the default bundle's
# ORD declaration is the no-shapes fallback, loaded once (repo-root
# anchored, like langdata). A tenant that declares nothing gets the default
# bundle's shapes: the DEFAULT DEPLOYMENT is ecommerce, the platform is not.
_DEFAULT_ID_SHAPES: list[dict] | None = None


def _default_id_shapes() -> list[dict]:
    global _DEFAULT_ID_SHAPES
    if _DEFAULT_ID_SHAPES is None:
        from voiceagent.tenant import Tenant
        from pathlib import Path as _Path
        root = (_Path(__file__).resolve().parents[2]
                / "data" / "tenants" / "default")
        _DEFAULT_ID_SHAPES = Tenant.load(root).record_id_shapes() or []
    return _DEFAULT_ID_SHAPES


def _resolve_shapes(id_shapes: list[dict] | None) -> list[dict]:
    return _default_id_shapes() if id_shapes is None else id_shapes
_PUNCT = ".,;:!?\"'()[]{}"


def _canon_token(tok: str) -> str | None:
    """Canonical number token (any loaded language) from a raw token, else
    None. Pure digits count too ('6 हजार' = 6000)."""
    w = tok.strip(_PUNCT).lower()
    if w in _GARBLES:
        w = _GARBLES[w]
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


def _record_id_span(text: str, shapes: list[dict]
                    ) -> tuple[str | None, tuple[int, int] | None]:
    """First record id + its char span across the tenant's declared shapes:
    digit form ('ORD-4821' / 'APT-1042') or number-words form after a
    declared prefix ('ORD four thousand eight hundred twenty one'). The
    output prefix is the shape's normalized code — the platform never
    invents or assumes one."""
    for shape in shapes:
        code = shape["code"]
        lo, hi = shape["min_digits"], shape["max_digits"]
        m = re.compile(shape["digit_pattern"],
                       re.IGNORECASE).search(text)
        if m:
            return f"{code}-{m.group(1)}", m.span()
        prefix = shape.get("prefix_pattern")
        if not prefix:
            continue
        pm = re.compile(prefix, re.IGNORECASE).search(text)
        if not pm:
            continue
        end = pm.end()
        consumed: list[str] = []
        for tok in re.finditer(r"\S+", text[pm.end():]):
            w = _canon_token(tok.group(0))
            if w is None:
                break
            consumed.append(w)
            end = pm.end() + tok.end()
        n = words_to_number(consumed) if consumed else None
        if n is not None and lo <= len(str(n)) <= hi:
            return f"{code}-{n}", (pm.start(), end)
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
        if not any(w in _SCALE_VALUES and w not in _EN_SCALES
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
# Currency word forms per symbol — loaded from data/lang/*.yaml
# (`currency_words:`): each language declares its own money words, and each
# form mints amounts only for its own currency (tenant isolation).
from voiceagent.langdata import currency_words as _load_currency

_CURRENCY_WORDS: dict[str, tuple[str, ...]] = {
    sym: tuple(forms) for sym, forms in _load_currency().items()}


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
                     min_amount: float = 100.0,
                     id_shapes: list[dict] | None = None) -> Entities:
    """Extract an amount and a record id from customer text. Pure regex +
    number-word normalization, no LLM — deterministic and cheap.

    currency/min_amount are tenant config (M6a). id_shapes are the tenant's
    declared ID shapes (Tenant.record_id_shapes); None resolves to the
    DEFAULT BUNDLE's declaration — the platform itself knows no ID shape,
    so a non-default industry passes its own shapes and ORD never appears.
    The digit regex and the money-word patterns are scoped to the ACTIVE
    currency's word forms; a bare number >= min_amount still counts either
    way."""
    text = _space_thai_numbers(text.translate(_NATIVE_DIGITS))
    sym = re.escape(currency)
    words = _currency_word_alts(currency)
    amount_re = re.compile(
        r"(?:" + sym + r"|" + words + r")?\s*"
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:" + sym + r"|" + words + r")?",
        re.IGNORECASE,
    )

    order_id, order_span = _record_id_span(text, _resolve_shapes(id_shapes))
    # Cut the record-id span so its number can't double as an amount
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


def _token_class() -> str:
    parts = ["A-Za-z"]
    for lo, hi in tokenizer_ranges():
        parts.append("\\u%04x-\\u%04x" % (lo, hi))
    return "".join(parts) + "0-9"


_DIGIT_TOKEN_RE = re.compile("[" + _token_class() + "]+")


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
                     min_confidence: float = SNAP_MIN_CONFIDENCE,
                     id_shapes: list[dict] | None = None) -> str | None:
    """Record-id extraction with contextual snapping. (The name is legacy —
    the id CODE comes from the tenant's declared shapes, so this returns
    APT-1042 on the clinic bundle.)

    1. Exact paths first: clean prefixed digits, spaced digits after a
       declared prefix marker, and number-words in any loaded language
       (incl. native-script digits).
    2. If nothing exact and candidates are known, snap garbled digit
       clusters to the closest candidate above min_confidence.
    """
    order_id, _ = _record_id_span(
        _space_thai_numbers(text.translate(_NATIVE_DIGITS)),
        _resolve_shapes(id_shapes))
    if order_id:
        return order_id
    if candidate_orders:
        return _snap_order_id(
            _space_thai_numbers(text.translate(_NATIVE_DIGITS)),
            candidate_orders, min_confidence)
    return None
