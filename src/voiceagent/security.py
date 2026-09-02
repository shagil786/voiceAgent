# src/voiceagent/security.py
"""Prompt-injection defense for the support agent (M6b).

Structural defense already in place: the action decision is DETERMINISTIC
(embedding classifier + policy engine) — an injected instruction cannot
execute anything, because the LLM's ACTION lines are ignored when a
classifier is present. What injection CAN still do is steer the natural-
language reply (tone, fake promises, leaked prompt text). This module:

1. DETECTS classic injection patterns (instruction override, identity/
   roleplay override, prompt extraction, fake chat-template markers,
   jailbreak memes) — deterministically, zero LLM cost.
2. SANITIZES the text that reaches the LLM prompt: fake template markers
   and fake system tags are stripped so a customer cannot forge meta-turns.
3. SIGNALS the policy engine (injection_suspected) — whether an injection
   attempt escalates to a human is tenant policy (escalate_when), and every
   flag lands in the decision log for the audit trail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (pattern, name) — matched case-insensitively against the customer text.
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|old)\s+"
     r"(instructions|rules|prompts|messages)", "instruction_override"),
    (r"disregard\s+(all\s+|the\s+)?(previous|prior|above|your)?\s*"
     r"(instructions|rules)", "instruction_override"),
    (r"forget\s+(everything|all|your)\s+(you|instructions|rules)", "instruction_override"),
    (r"you'?re now|you are now", "identity_override"),
    (r"pretend\s+(you'?re|you are|to be)", "roleplay_override"),
    (r"act\s+as\s+(a|an|the|if)", "roleplay_override"),
    (r"reveal\s+(your|the)\s+(prompt|instructions|system)", "prompt_extraction"),
    (r"(your|the)\s+(system\s+prompt|initial\s+instructions)", "prompt_extraction"),
    (r"repeat\s+(everything|the\s+text|your\s+instructions)\s+"
     r"(above|before|from)", "prompt_extraction"),
    (r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|eot\|>", "fake_template_marker"),
    (r"^\s*system\s*:", "fake_system_tag"),
    (r"^\s*assistant\s*:", "fake_system_tag"),
    (r"developer\s+mode|\bDAN\s+mode|jailbreak", "jailbreak"),
    (r"new\s+rules?\s*:", "instruction_override"),
)

_FAKE_MARKERS = re.compile(
    r"<\|[^|]*\|>|^\s*(system|assistant|developer)\s*:\s*",
    re.MULTILINE | re.IGNORECASE)


@dataclass
class InjectionSignal:
    detected: bool
    patterns: list[str] = field(default_factory=list)


def detect_injection(text: str) -> InjectionSignal:
    """Deterministic injection scan of customer text. Benign support text
    ("how do I cancel", "my refund was ignored") must never match — the
    patterns anchor on instruction-directed meta-language, not topic words."""
    patterns = []
    for pattern, name in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            patterns.append(name)
    return InjectionSignal(detected=bool(patterns), patterns=sorted(set(patterns)))


def sanitize_for_prompt(text: str) -> str:
    """Strip forged meta-turns from customer text before it enters the LLM
    prompt: fake chat-template markers and fake system/assistant line tags.
    Customer CONTENT (order ids, amounts, questions) is preserved."""
    cleaned = _FAKE_MARKERS.sub(" ", text)
    return re.sub(r"  +", " ", cleaned).strip()
