# src/voiceagent/langid.py
"""M5a: stdlib-only language identification for text turns.

Native Unicode-script detection runs first — the script with the most
characters wins (majority rule, not single-character), and ANY meaningful
native-script presence beats Latin so code-switched text like
"मेरा recharge क्यों fail हुआ?" is treated as native (hi) and gets the
reply-language directive. Pure Latin text is "hinglish" when it carries >= 2
DISTINCT tokens from a small Hinglish lexicon, else "en".

Global target set (es/fr/de/pt): Latin-script languages are detected from
GLOBAL_LEXICONS with the same >=2-distinct-token contract, checked after
native scripts and before hinglish. Without them the reply-language
guardrail failed every es/fr/de/pt turn (langid could only answer
en/hinglish/hi/te), and the canned-reply fallback served those customers
Hindi text. Selection takes the lexicon with the MOST distinct hits: es and
pt share core support vocabulary (pedido, reembolso, para), so a fixed
check order would misroute one of the two on its own shared words.

Note: languages sharing one script block resolve via their lang file's
`detect_as` field (explicit data); callers that know better may pass
language="mr" explicitly (the agent directive accepts any native-script
code).
"""
from __future__ import annotations

import re

# Languages the reply-language directive fires for (non-Latin scripts),
# script ranges, and Latin detection lexicons: ALL loaded from data/
# (data/lang/*.yaml claims + data/scripts.yaml ranges). Add-a-language =
# add-a-file; this module holds detection LOGIC only (majority rule,
# most-hits-wins, distinct-token thresholds), zero language data.
from voiceagent.langdata import (detect_lexicons as _load_lexicons,
                                 native_script_codes as _load_native,
                                 script_claims as _load_claims)

NATIVE_SCRIPT_LANGS = _load_native()

# (detected_code, lo, hi): every claimed script resolves through its lang
# file's `detect_as` (shared blocks name their detected code explicitly —
# no per-language branch here; see data/lang/*.yaml).
_SCRIPT_RANGES = tuple(_load_claims())

# Latin-script lexicons grouped by detection stage. The global stage runs
# before hinglish (a turn carrying 2+ global hits is that language even if
# it also carries hinglish tokens); within global, most-hits-wins with
# sorted-code tie order (es precedes pt on their shared support words).
_LEXICONS = _load_lexicons()
GLOBAL_LEXICONS = {
    code: toks for code, (stage, toks) in sorted(_LEXICONS.items())
    if stage == "global"}
HINGLISH_LEXICON = frozenset().union(*[
    toks for code, (stage, toks) in _LEXICONS.items()
    if stage == "hinglish"] or [frozenset()])

# Native-script characters required before the native script beats Latin.
_MAJORITY_MIN = 2

# Lexicon design notes (the lists themselves live in data/lang/*.yaml
# `detect_tokens`): exact-token match, >= 2 DISTINCT hits — function words
# carry the signal, domain nouns disambiguate support turns. Accented AND
# unaccented forms are listed (ASR strips diacritics). "die" stays OUT of
# the German list (common English verb — der/das/nicht carry de alone).

# Distinct lexicon tokens required for a Latin-script global language.
_LEXICON_MIN = 2

# Latin letters including common Western diacritics: accented forms
# ("dónde", "não", "Rückerstattung") must tokenize as single words.
_TOKEN_RE = re.compile(r"[a-z\u00e0-\u024f]+")


def detect_language(text: str) -> str:
    """Return one of: en, hinglish, es, fr, de, pt, hi, ta, te, bn, mr, gu,
    kn, ml, pa, th."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if counts:
        best = max(counts, key=lambda k: counts[k])
        if counts[best] >= _MAJORITY_MIN:
            return best
    tokens = set(_TOKEN_RE.findall(text.lower()))
    best_lang, best_hits = None, _LEXICON_MIN - 1
    for lang, lexicon in GLOBAL_LEXICONS.items():
        hits = len(tokens & lexicon)
        if hits > best_hits:
            best_lang, best_hits = lang, hits
    if best_lang is not None:
        return best_lang
    if len(tokens & HINGLISH_LEXICON) >= 2:
        return "hinglish"
    return "en"
