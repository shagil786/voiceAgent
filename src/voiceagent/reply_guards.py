# src/voiceagent/reply_guards.py — the deterministic reply-guard pipeline.
"""Task D1 (architecture debt): every deterministic post-generation guard the
Agent applies to an LLM reply, moved VERBATIM out of voiceagent.agent's
catch-all — no behavior change. The pipeline, in the order Agent.handle runs
it:

    ACTION-scaffolding scrub (strip_action_lines / extract_action)
    -> echo guardrail (extract_required_references + _patch_reply; the fact
       groups resolve through echo_spec_registry from tool-contract data)
    -> reply-language guardrail (_acceptable_reply_langs) with the ONE
       governed frontier re-render (repair_reply) as the guide-not-replace
       path, and the deterministic canned fallback (_canned_reply)
    -> empty-reply safety net + the M6a empathy check (_already_apologetic)

The canned/empathy/noted TEXT TABLES are demo tenant data and live in
voiceagent.demo_data; this module is the guard LOGIC that consumes them.
agent.py re-exports the public names, so `from voiceagent.agent import X`
keeps working unchanged."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from voiceagent.demo_data import REPLY_TEMPLATES

if TYPE_CHECKING:  # Turn is duck-typed at runtime (no import cycle)
    from voiceagent.memory import Turn

ACTION_RE = re.compile(r"ACTION:\s*([a-z_]+)", re.IGNORECASE)
# Order IDs / reference numbers the customer may state (Latin or Devanagari).
ORDER_ID_RE = re.compile(
    r"\b(?:ORD[-#]?\s*)?(\d{4,10})\b", re.IGNORECASE
)

# Intent keywords that must appear in the reply when the customer states them
# are TOOL-CONTRACT data now (Sprint A3): ToolSpec.facts on the deployment's
# declared specs (DEFAULT_TOOL_SPECS + tenant tools.yaml overrides), with the
# demo tenant contracts (voiceagent.demo_data.DEMO_TENANT_CONTRACT_SPECS) as
# the no-bundle fallback.


def echo_spec_registry(specs: "dict | None", demo: bool = True) -> dict:
    """ONE shared resolution of the echo guardrail's fact groups, used by
    Agent.__init__ and extract_required_references (single source, no
    copy-paste). `specs` is the wired tool surface (gateway.specs) or None;
    `demo` is True whenever no real tenant bundle is declared — the demo
    tenant contracts are then merged in (over the code defaults, under the
    wired specs) so the historical keyword guarantees are enforced on every
    demo wiring. A declared tenant bundle suppresses the demo contracts
    entirely: only the bundle's own specs apply."""
    from voiceagent.demo_data import DEMO_TENANT_CONTRACT_SPECS
    from voiceagent.tools import DEFAULT_TOOL_SPECS
    registry: dict = {}
    if demo:
        registry.update(DEFAULT_TOOL_SPECS)
        registry.update(DEMO_TENANT_CONTRACT_SPECS)
    if specs:
        registry.update(specs)
    return registry


def extract_required_references(user_text: str,
                                specs: "dict | None" = None,
                                demo: bool = True) -> list[str]:
    """References the reply must contain: the customer's stated order id(s)
    and any declared tool-contract fact the customer stated. Shared with
    chat.py (turn records) and the echo guardrail. The scan is
    FIRST-MATCH-PER-SPEC (one fact per spec, then move on) — exactly the
    historical KEYWORD_FACTS group semantics: e.g. "my recharge failed" pins
    only 'fail' (recharge's first fact), never both 'fail' and 'recharge'.
    specs=None resolves the demo registry (the historical default for callers
    with no tool surface)."""
    registry = echo_spec_registry(specs, demo)
    refs: list[str] = []
    for m in ORDER_ID_RE.finditer(user_text):
        refs.append(m.group(0))
    lower = user_text.lower()
    for spec in registry.values():
        for f in spec.facts:
            if f in lower:
                if f not in refs:  # cross-spec duplicates stay single
                    refs.append(f)
                break  # first match per spec — historical group semantics
    return refs


def extract_action(text: str) -> str | None:
    m = ACTION_RE.search(text)
    return m.group(1).lower() if m else None


def find_order_id(text: str) -> str | None:
    """First order-id match in text ('ORD-1234' or bare digits), else None.
    The single entry point to ORDER_ID_RE outside this module."""
    m = ORDER_ID_RE.search(text)
    return m.group(0) if m else None


def find_recent_order_id(history: list["Turn"]) -> str | None:
    """Most recent order id in a conversation (scan newest -> oldest)."""
    for t in reversed(history):
        oid = find_order_id(t.text)
        if oid:
            return oid
    return None


# ---------------------------------------------------------------------------
# M5b-4: reply-language guardrail. The prompt directive ("Reply in the
# customer's language") is unreliable on 0.5B models — the live fresh-caller
# voice check showed hi/te customers receiving English replies, and the
# empty-reply safety net was English-only. After generation, verify the
# reply's language; on mismatch substitute a deterministic canned reply in
# the customer's language (per-intent where available), then re-apply the
# echo guardrail so the customer's reference still appears. en turns are
# never touched — the text-path benchmark stays byte-identical.
# ---------------------------------------------------------------------------

def _acceptable_reply_langs(language: str | None) -> frozenset | None:
    """Reply languages a turn may legitimately come back in; None disables
    the guardrail (every en/None turn). hinglish accepts Roman hinglish or
    Devanagari Hindi (a Hindi speaker reads both natively); native languages
    are strict."""
    if not language or language == "en":
        return None
    if language == "hinglish":
        return frozenset({"hinglish", "hi"})
    return frozenset({language})


def _ref_for_template(refs: list[str]) -> str:
    """The customer's order-id-shaped reference (keywords are not refs)."""
    for r in refs:
        if r.upper().startswith("ORD") or r.isdigit():
            return r
    return ""

def _canned_reply(action: str | None, language: str, refs: list[str]) -> str:
    # A language outside the template tables gets neutral English — the old
    # "else 'hi'" fallback served Hindi text to es/fr/de/pt customers.
    lang_key = language if language in _SERVED_REPLY_LANGS else "en"
    table = REPLY_TEMPLATES.get(action or "") or REPLY_TEMPLATES["default"]
    tpl = table.get(lang_key) or REPLY_TEMPLATES["default"][lang_key]
    return tpl.format(ref=_ref_for_template(refs))

# The es/fr/de/pt templates are LLM-authored SYNTHETIC phrasings (es: LatAm-
# neutral; fr: EU vous-form; de: formal Sie; pt: Brazilian-neutral você) —
# real-traffic validation pending, same as the intent exemplars.

_SERVED_REPLY_LANGS = frozenset(REPLY_TEMPLATES["default"])


def _patch_reply(reply: str, required: list[str]) -> str:
    """Deterministic guardrail: prepend a confirmation sentence that echoes
    any customer reference the LLM failed to include. Returns reply unchanged
    if nothing is missing."""
    missing = [r for r in required if r.lower() not in reply.lower()]
    if not missing:
        return reply
    head = reply.split("\n\n", 1)[0]
    confirm = (
        f"I understand — this is regarding {', '.join(missing)}. "
    )
    if reply.strip().startswith(("ACTION:", "response", " thinking")):
        return confirm.strip() + "\n\n" + reply.strip()
    return confirm + reply


def strip_action_lines(text: str) -> str:
    """Remove the LLM's ACTION scaffolding lines from a customer-visible
    reply (and collapse the blank-line runs they leave behind). The action
    decision comes from the deterministic classifier (or fallback
    extract_action), so the ACTION line itself must never reach the customer.
    Call only after the action has been captured and the echo guardrail has
    run."""
    kept = [ln for ln in text.split("\n") if not ACTION_RE.search(ln)]
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


_APOLOGY_MARKERS = ("sorry", "apolog", "khed", "kshama", "माफ", "खेद",
                    "క్షమించ", "lamento", "disculp", "désolé", "desole",
                    "leid", "entschuldig")


def _already_apologetic(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _APOLOGY_MARKERS)


# ---------------------------------------------------------------------------
# Task B: the ONE governed frontier re-render of a guardrail-violating reply.
# Module-level form of the former Agent._repair_reply (Task D1 move): the LLM
# handle, the compiled system prompt and the chat-template flag are passed in
# by the caller instead of living on Agent. `language` stays in the signature
# for the historical call shape.
# ---------------------------------------------------------------------------

def repair_reply(llm, system_prompt: str, use_template: bool,
                 violating_reply: str, user_text: str,
                 language: str, allowed_langs: frozenset,
                 required_refs: list[str]) -> str:
    """Task B: ONE governed re-render of a guardrail-violating frontier
    reply. The repair prompt carries the ORIGINAL frontier reply, the
    ORIGINAL user turn, the allowed language(s) and the required
    references (missing ones called out for verbatim inclusion); persona
    never_say / may_promise constraints travel through the compiled
    system prompt, same as the main turn. Sync, one extra frontier round
    max — the caller re-checks the guards and falls back to the canned
    path when the re-render still violates; exceptions are the caller's
    fail-open concern and never reach the customer."""
    missing = [r for r in required_refs
               if r.lower() not in violating_reply.lower()]
    lines = [
        "Your previous reply violated this conversation's reply "
        "constraints. Rewrite it now.",
        f"Previous reply that violated the constraints: {violating_reply}",
        f"The customer said: {user_text}",
        "- Write the reply ONLY in language code(s): "
        + ", ".join(sorted(allowed_langs)) + ".",
    ]
    if required_refs:
        lines.append("- Keep these customer references verbatim: "
                     + ", ".join(required_refs) + ".")
    if missing:
        lines.append("- These required references were MISSING and must "
                     "appear verbatim: " + ", ".join(missing) + ".")
    lines.append("- Respect every persona constraint in your "
                 "instructions: never say or promise anything not "
                 "permitted there.")
    lines.append("Reply with the rewritten reply text only.")
    instruction = "\n".join(lines)
    if use_template:
        prompt = llm.chat_template(system_prompt, "", instruction)
    else:
        prompt = (f"{system_prompt}\n\nContext:\n\n"
                  f"Customer: {instruction}\nAssistant:")
    stop = getattr(llm, "stop_tokens", None)
    text = llm.generate(prompt, max_tokens=300, stop=stop)
    post = getattr(llm, "postprocess", None)
    clean = post(text) if callable(post) else text
    # The re-rendered reply is judged as a customer-visible reply: ACTION
    # scaffolding is scrubbed before the guards re-check it.
    return strip_action_lines(clean)
