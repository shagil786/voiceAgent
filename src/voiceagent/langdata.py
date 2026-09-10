# src/voiceagent/langdata.py — platform language data loader.
"""One loader for every per-language table: number words/scales, script
digits, sentiment phrases, ASR garbles. Source of truth is data/lang/
(one file per language code); the parsers (entities, sentiment) read the
loaded tables as opaque values — adding a language is adding a file, never
a code branch.

Missing/unreadable files contribute nothing (fail-open): the engine still
parses digits + whatever languages did load. Anchored to the repo root
(src/voiceagent/ -> parents[2]), the same convention as runtime.py —
resolution never depends on the process cwd.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
LANG_DIR = _REPO_ROOT / "data" / "lang"


def load_lang_tables(directory: str | Path = LANG_DIR) -> dict[str, dict]:
    """code -> {words, scales, hundred, digits, sentiment, garbles}.
    Every section optional (a digits-only file is valid); malformed files
    are skipped, never fatal."""
    import yaml
    out: dict[str, dict] = {}
    d = Path(directory)
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.yaml")):
        try:
            raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        code = str(raw.get("code") or f.stem)
        numbers = raw.get("numbers") or {}
        if not isinstance(numbers, dict):
            numbers = {}
        words = numbers.get("words") or {}
        scales = numbers.get("scales") or {}
        hundred = numbers.get("hundred") or []
        sentiment = raw.get("sentiment") or []
        garbles = raw.get("garbles") or {}
        digits = raw.get("digits") or ""
        companions = raw.get("companions") or []
        scripts = raw.get("scripts") or []
        detect_as = raw.get("detect_as") or code
        detect_tokens = raw.get("detect_tokens") or []
        detect_stage = raw.get("detect_stage") or ""
        currency_words = raw.get("currency_words") or {}
        out[code] = {
            "words": {str(k): int(v) for k, v in words.items()
                      if isinstance(v, int)},
            "scales": {str(k): int(v) for k, v in scales.items()
                       if isinstance(v, int)},
            "hundred": [str(h) for h in hundred if h],
            "digits": str(digits),
            "sentiment": [str(s) for s in sentiment if s],
            "garbles": {str(k): str(v) for k, v in garbles.items()},
            "companions": [str(c) for c in companions if c],
            "scripts": [str(s) for s in scripts if s],
            "detect_as": str(detect_as),
            "detect_tokens": [str(t) for t in detect_tokens if t],
            "detect_stage": str(detect_stage),
            "currency_words": {
                str(sym): [str(f) for f in (forms or []) if f]
                for sym, forms in currency_words.items()
                if isinstance(currency_words, dict)},
        }
    return out


_TABLES: dict[str, dict] | None = None


def tables() -> dict[str, dict]:
    """Process-wide loaded tables (loaded once, like intent exemplars)."""
    global _TABLES
    if _TABLES is None:
        _TABLES = load_lang_tables()
    return _TABLES


def number_lookups() -> tuple[dict[str, int], dict[str, int], frozenset]:
    """(word->value, scale->value, hundred-words) merged across languages."""
    words: dict[str, int] = {}
    scales: dict[str, int] = {}
    hundred: set[str] = set()
    for entry in tables().values():
        words.update(entry["words"])
        scales.update(entry["scales"])
        hundred.update(entry["hundred"])
    return words, scales, frozenset(hundred)


def digit_map() -> dict[str, str]:
    """Native-script digit char -> ASCII, merged across language files."""
    out: dict[str, str] = {}
    for entry in tables().values():
        row = entry["digits"]
        if len(row) == 10:
            out.update(zip(row, "0123456789"))
    return out


def sentiment_lexicon() -> dict[str, tuple[str, ...]]:
    """code -> phrase tuple (only codes that declare sentiment)."""
    return {code: tuple(entry["sentiment"]) for code, entry in tables().items()
            if entry["sentiment"]}


def companions_for(code: str) -> tuple[str, ...]:
    """Sibling-script codes scanned alongside `code` (code-switching)."""
    entry = tables().get(code) or {}
    return tuple(entry.get("companions") or ())


def currency_words() -> dict[str, list[str]]:
    """Currency symbol -> regex word-forms, merged across lang files (each
    language declares its own money words; a form mints amounts only for
    its own currency). Order-preserving dedupe."""
    out: dict[str, list[str]] = {}
    for entry in tables().values():
        for sym, forms in (entry.get("currency_words") or {}).items():
            bucket = out.setdefault(str(sym), [])
            for f in forms:
                if f not in bucket:
                    bucket.append(str(f))
    return out


def script_table(directory: str | Path | None = None) -> dict[str, list]:
    """Script name -> [(lo, hi)] codepoint ranges from data/scripts.yaml
    (unicode mechanical facts, not language knowledge)."""
    import yaml
    d = (Path(directory) if directory else
         Path(__file__).resolve().parents[2] / "data" / "scripts.yaml")
    try:
        raw = yaml.safe_load(d.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    scripts = raw.get("scripts") or {}
    out: dict[str, list] = {}
    for name, ranges in scripts.items():
        pairs = []
        for r in ranges or []:
            try:
                lo, hi = int(r[0]), int(r[1])
            except Exception:
                continue
            pairs.append((lo, hi))
        if pairs:
            out[str(name)] = pairs
    return out


def script_claims() -> list[tuple[str, int, int]]:
    """(detected_code, lo, hi) for langid + tokenizer: every script claimed
    by a lang file resolves through its `detect_as` (Marathi claims
    Devanagari but detects as hi — explicit data, no code branch). Scripts
    no file claims (Arabic) contribute ranges to the tokenizer but no
    detection mapping."""
    st = script_table()
    claims: list[tuple[str, int, int]] = []
    for code in sorted(tables()):
        entry = tables()[code]
        for script in entry.get("scripts") or []:
            for lo, hi in st.get(script, []):
                claims.append((entry.get("detect_as") or code, lo, hi))
    return claims


def native_script_codes() -> frozenset:
    """Codes declaring a native script (the reply-language directive set)."""
    return frozenset(code for code, e in tables().items() if e.get("scripts"))


def tokenizer_ranges() -> list[tuple[int, int]]:
    """Every script range in data/scripts.yaml — claimed or not — for the
    entity tokenizer's character class (tokenization is mechanical; a
    script needs no language claim to tokenize). Sorted for stable regex."""
    seen: set[tuple[int, int]] = set()
    for ranges in script_table().values():
        seen.update(ranges)
    return sorted(seen)


def detect_lexicons() -> dict[str, tuple[str, frozenset]]:
    """Latin-script detection lexicons: code -> (stage, tokens).
    Stages ('global', 'hinglish') order the checks; within a stage the
    lexicon with the MOST distinct hits wins (ties keep sorted-code order —
    es precedes pt, preserving the documented shared-vocabulary behavior).
    No language is named in the consumer."""
    return {code: (e["detect_stage"], frozenset(e["detect_tokens"]))
            for code, e in tables().items()
            if e.get("detect_tokens") and e.get("detect_stage")}


def garble_map() -> dict[str, str]:
    """ASR-mishearing -> canonical number word, merged across languages."""
    out: dict[str, str] = {}
    for entry in tables().values():
        out.update(entry["garbles"])
    return out
