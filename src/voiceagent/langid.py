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

Note: Marathi shares the Devanagari block with Hindi, so Devanagari-script
text is reported as "hi"; callers that know better may pass language="mr"
explicitly (the agent directive accepts any native-script code).
"""
from __future__ import annotations

import re

# Languages the reply-language directive fires for (non-Latin scripts).
NATIVE_SCRIPT_LANGS = frozenset(
    {"hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"})

# Unicode block per language. Devanagari maps to hi (Marathi is script-
# identical); every other block is unambiguous.
_SCRIPT_RANGES = (
    ("hi", 0x0900, 0x097F),  # Devanagari
    ("bn", 0x0980, 0x09FF),  # Bengali
    ("pa", 0x0A00, 0x0A7F),  # Gurmukhi
    ("gu", 0x0A80, 0x0AFF),  # Gujarati
    ("ta", 0x0B80, 0x0BFF),  # Tamil
    ("te", 0x0C00, 0x0C7F),  # Telugu
    ("kn", 0x0C80, 0x0CFF),  # Kannada
    ("ml", 0x0D00, 0x0D7F),  # Malayalam
)

# Native-script characters required before the native script beats Latin.
_MAJORITY_MIN = 2

# Small Latin-script Hinglish lexicon. Hits are DISTINCT tokens, so common
# English words on the list ("do") cannot flip detection on their own.
HINGLISH_LEXICON = frozenset({
    "kya", "hai", "nahi", "nhi", "kab", "kahan", "kaha", "kyu", "kyun",
    "mera", "meri", "mere", "karo", "kro", "kar", "krna", "karna", "karke",
    "karu", "do", "chahiye", "aayega", "aayegi", "milenga", "milega",
    "batao", "bhai", "lag", "gaya", "gayi", "hua", "hui", "ho", "raha",
    "rahi", "kitna", "kitni", "abhi", "wapas", "paisa", "paise",
})

# Latin-script lexicons for the global target set (LLM-authored SYNTHETIC
# token lists — real-traffic validation pending). Same contract as
# HINGLISH_LEXICON: exact-token match, >= 2 DISTINCT hits. Function words
# carry the signal; domain nouns (pedido, commande, Bestellung) make
# support-desk turns unambiguous. Accented AND unaccented forms are listed
# because ASR output often strips diacritics ("dónde" -> "donde"). Tokens
# shared with English ("los", "la") are single-edged: the 2-distinct-hits
# rule keeps "Los Angeles order status" English. "die" is deliberately
# ABSENT from the German lexicon: it is a common English verb, and an
# English sentence carrying it alongside one more German word ("the die is
# cast, das is fine") would flip to de and serve German text to an English
# caller — der/das/nicht carry the German signal without that collision.
GLOBAL_LEXICONS = {
    "es": frozenset({
        "que", "para", "pero", "gracias", "quiero", "mi", "mis", "pedido",
        "factura", "reembolso", "dónde", "donde", "cuándo", "cuando",
    }),
    "fr": frozenset({
        "commande", "remboursement", "livraison", "merci", "mais", "vous",
        "mon", "ma", "pas", "le", "la", "les", "je", "veux", "voudrais",
    }),
    "de": frozenset({
        "der", "das", "nicht", "und", "bitte", "danke", "mein",
        "meine", "bestellung", "lieferung", "rückerstattung",
        "rueckerstattung", "ruckerstattung", "ich", "eine", "einen",
        "mochte", "möchte", "wann", "kommt", "für", "fuer",
    }),
    "pt": frozenset({
        "obrigado", "obrigada", "não", "nao", "meu", "minha", "você",
        "voce", "quando", "onde", "mas", "entrega", "fatura", "pedido",
        "reembolso", "quero",
    }),
}

# Distinct lexicon tokens required for a Latin-script global language.
_LEXICON_MIN = 2

# Latin letters including common Western diacritics: accented forms
# ("dónde", "não", "Rückerstattung") must tokenize as single words.
_TOKEN_RE = re.compile(r"[a-z\u00e0-\u024f]+")


def detect_language(text: str) -> str:
    """Return one of: en, hinglish, es, fr, de, pt, hi, ta, te, bn, mr, gu,
    kn, ml, pa."""
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
