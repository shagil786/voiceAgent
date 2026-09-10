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


def garble_map() -> dict[str, str]:
    """ASR-mishearing -> canonical number word, merged across languages."""
    out: dict[str, str] = {}
    for entry in tables().values():
        out.update(entry["garbles"])
    return out
