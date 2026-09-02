# src/voiceagent/sentiment.py
"""Deterministic frustration detection — the 'Sentiment Agent' without an
LLM pass: zero added latency, fully auditable, testable.

Why lexicons, not a model: every millisecond of the turn budget belongs to
ASR and reply generation, and the outcome must be explainable in the decision
log ("escalated: frustrated=true, hits=[ridiculous, worst]"). Lexicon matches
plus typographic intensity (SHOUTING CAPS, '!!!') cover the support-desk
signal well enough to route on; whether frustration DOES route to a human is
a policy decision (escalate_when in policies.yaml), not a code decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Frustration phrases per language family. 'hinglish' text is Romanized
# Hindi, 'hi' is Devanagari; European languages cover the US/EU market.
# English is always scanned too — support frustration code-switches.
LEXICON: dict[str, tuple[str, ...]] = {
    "en": (
        "angry", "furious", "ridiculous", "useless", "worst", "terrible",
        "horrible", "unacceptable", "pathetic", "garbage", "awful",
        "sick of", "fed up", "nonsense", "scam", "cheated", "cheating",
        "incompetent", "last warning", "speak to a manager", "talk to a manager",
        "escalate", "demand", "never again", "cancel everything",
    ),
    "hinglish": (
        "bakwas", "ghatiya", "bekaar", "bekara", "faltu", "pareshan",
        "pareshaan", "tang aa", "jhooth", "thag", "gussa", "naraz",
        "bar bar", "baar baar", "kitna din", "kitne din", "nahi ho raha",
        "last time bol",
    ),
    "hi": (
        "गुस्सा", "नाराज़", "नाराज", "परेशान", "बेकार", "घटिया", "बकवास",
        "झूठ", "ठग", "बार बार", "बार-बार", "फालतू", "कितने दिन", "कितना दिन",
        "अपमान",
    ),
    "es": (
        "enojado", "furioso", "inaceptable", "ridículo", "ridiculo",
        "estafa", "estafado", "hartado", "harto", "pésimo", "pesimo",
        "inútil", "inutil", "basta",
    ),
    "fr": (
        "en colère", "colere", "furieux", "inacceptable", "ridicule",
        "arnaque", "j'en ai marre", "marre", "nul",
    ),
    "de": (
        "wütend", "wutend", "furchtbar", "unakzeptabel", "lächerlich",
        "lacherlich", "betrogen", "es reicht",
    ),
}

_INTENSITY_BANGS = re.compile(r"[!]{2,}|[?]{2,}")
_SHOUTING = re.compile(r"\b[A-Z]{4,}\b")


@dataclass
class Frustration:
    level: str  # "none" | "mild" | "high"
    hits: list[str] = field(default_factory=list)
    intensity: list[str] = field(default_factory=list)

    @property
    def frustrated(self) -> bool:
        return self.level != "none"


def _phrases_for(language: str | None) -> list[str]:
    if not language:
        # No language known: scan every lexicon (conservative — support
        # frustration is too important to miss over a detection miss).
        seen: set[str] = set()
        merged: list[str] = []
        for phrases in LEXICON.values():
            for p in phrases:
                if p not in seen:
                    seen.add(p)
                    merged.append(p)
        return merged
    phrases = list(LEXICON["en"])
    if language != "en":
        phrases += LEXICON.get(language, ())
        # hinglish and hi speakers code-switch into each other's script.
        if language == "hinglish":
            phrases += LEXICON["hi"]
        elif language == "hi":
            phrases += LEXICON["hinglish"]
    return phrases


def detect_frustration(text: str, language: str | None = None,
                       extra_phrases: list[str] | None = None) -> Frustration:
    """Detect frustration in customer text. Deterministic: distinct lexicon
    hits + typographic intensity -> none/mild/high. Unknown languages get
    the English lexicon + intensity only. extra_phrases = learned lexicon
    entries (SentimentStore) — the detector knows more over time."""
    hits: list[str] = []
    low = text.lower()
    phrases = set(_phrases_for(language))
    if extra_phrases:
        phrases |= {p.strip().lower() for p in extra_phrases if p.strip()}
    for phrase in phrases:
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"\b"
        if re.search(pattern, low) or (phrase in low and not phrase.isalpha()):
            hits.append(phrase)
    hits.sort()

    intensity: list[str] = []
    if _INTENSITY_BANGS.search(text):
        intensity.append("repeated !/?")
    shouts = [w for w in _SHOUTING.findall(text) if w.isalpha()]
    if shouts:
        intensity.append("shouting caps")
    intensity.sort()

    h, i = len(hits), len(intensity)
    if h >= 2 or (h >= 1 and i >= 1) or i >= 2:
        level = "high"
    elif h == 1 or i == 1:
        level = "mild"
    else:
        level = "none"
    return Frustration(level=level, hits=hits, intensity=intensity)


# ---------------------------------------------------------------------------
# M6b: the lexicon LEARNS. Novel frustration expressions — intensity signals
# (SHOUTING CAPS, '!!!') with no known phrase match — are captured as
# candidates in SQLite; reviewed candidates are promoted into the live
# lexicon, so the detector knows more words next week than today. Promotion
# is human-approved (auditable), auto-capture is deterministic.
# ---------------------------------------------------------------------------

class SentimentStore:
    """SQLite-backed learnable lexicon. Same runtime-data pattern as
    SQLiteMemory: lives in gitignored data/out/, safe to recreate."""

    def __init__(self, path: str = "data/out/sentiment.db"):
        import sqlite3
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sentiment_phrases ("
            "phrase TEXT NOT NULL, lang TEXT NOT NULL, source TEXT NOT NULL,"
            "added_at TEXT NOT NULL, PRIMARY KEY (phrase, lang))")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sentiment_candidates ("
            "candidate TEXT NOT NULL, lang TEXT NOT NULL, source TEXT NOT NULL,"
            "count INTEGER NOT NULL DEFAULT 1,"
            "first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,"
            "PRIMARY KEY (candidate, lang))")
        self._conn.commit()

    def learned_phrases(self, language: str | None) -> list[str]:
        """Live learned phrases for a language plus language-agnostic ones."""
        cur = self._conn.execute(
            "SELECT phrase FROM sentiment_phrases WHERE lang IN (?, '_any')",
            (language or "_none",))
        return [r[0] for r in cur.fetchall()]

    def add_phrase(self, phrase: str, language: str | None,
                   source: str = "manual") -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO sentiment_phrases VALUES (?, ?, ?, ?)",
            (phrase.strip().lower(), language or "_any", source, now))
        self._conn.commit()

    def capture_candidates(self, candidates: list[str], language: str | None,
                           source: str = "intensity") -> int:
        """Record novel frustration expressions for review. Returns the
        number of NEW candidates (repeats only bump count/last_seen)."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        lang = language or "_any"
        new = 0
        for c in candidates:
            c = c.strip().lower()
            if len(c) < 4:
                continue
            cur = self._conn.execute(
                "SELECT 1 FROM sentiment_candidates WHERE candidate=? AND lang=?",
                (c, lang))
            if cur.fetchone() is None:
                new += 1
            self._conn.execute(
                "INSERT INTO sentiment_candidates VALUES (?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(candidate, lang) DO UPDATE SET count = count + 1, "
                "last_seen = excluded.last_seen",
                (c, lang, source, now, now))
        self._conn.commit()
        return new

    def candidates(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT candidate, lang, count, source, first_seen, last_seen "
            "FROM sentiment_candidates ORDER BY count DESC")
        return [dict(zip(("candidate", "lang", "count", "source",
                          "first_seen", "last_seen"), r))
                for r in cur.fetchall()]

    def promote(self, candidate: str, language: str | None) -> bool:
        """Review outcome: promote a candidate into the live lexicon."""
        cand = candidate.strip().lower()
        cur = self._conn.execute(
            "SELECT 1 FROM sentiment_candidates WHERE candidate=? AND lang=?",
            (cand, language or "_any"))
        if cur.fetchone() is None:
            return False
        self.add_phrase(cand, language, source="promoted")
        self._conn.execute(
            "DELETE FROM sentiment_candidates WHERE candidate=? AND lang=?",
            (cand, language or "_any"))
        self._conn.commit()
        return True


def candidate_phrases_from(text: str) -> list[str]:
    """Novel-expression candidates from an unmatched frustrated turn: the
    SHOUTING CAPS tokens (the customer's own words for the feeling)."""
    return [w.lower() for w in _SHOUTING.findall(text) if w.isalpha()]
