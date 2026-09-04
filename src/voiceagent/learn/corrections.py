"""Deterministic correction classifier (no LLM): lexicon leads + type
precedence, mirroring sentiment.py. Owner/customer scoping is structural:
customer text can only ever be a candidate, never global."""
from __future__ import annotations

from dataclasses import dataclass

_LEADS = ("no,", "no ", "don't", "do not", "never", "wrong", "incorrect",
          "actually", "i told you", "you said", "stop saying", "not ", "but ")
_POLICY = ("promise", "never ", "always", "escalat", "refund", "approv",
           "discount", "price")
_EXEMPLAR = ("say ", "respond with", "like this", "for example")
_TONE = ("tone", "rude", "polite", "shorter", "short ", "hindi", "english",
         "hinglish", "language")

@dataclass
class Correction:
    is_correction: bool
    patch_type: str  # tone | fact | policy | exemplar | none
    quote: str
    scope: str  # global | candidate | none

def classify_correction(user_text: str, last_agent_text: str = "",
                        is_owner: bool = False) -> Correction:
    low = user_text.lower()
    quote = user_text.strip()[:280]
    # Ruling R2: correction requires an explicit lead word. Type keywords
    # classify only — they never trigger on their own (precision over
    # recall: e.g. "what's the price?" must never become a correction).
    if not any(l in low for l in _LEADS):
        return Correction(False, "none", quote, "none")
    if any(k in low for k in _POLICY):
        ptype = "policy"
    elif any(k in low for k in _EXEMPLAR):
        ptype = "exemplar"
    elif any(k in low for k in _TONE):
        ptype = "tone"
    else:
        ptype = "fact"
    scope = "global" if is_owner else "candidate"
    return Correction(True, ptype, quote, scope)
