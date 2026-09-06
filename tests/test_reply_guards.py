# tests/test_reply_guards.py — Task D1: agent.py's catch-all split.
"""The deterministic reply-guard pipeline now lives in voiceagent.reply_guards
and the demo reply templates (Acme/e-commerce text) live in
voiceagent.demo_data. agent.py re-exports the public names, so
`from voiceagent.agent import X` keeps working — these tests pin BOTH the new
module surface and the backward-compat re-exports, plus the core-cleanliness
rule: core agent.py ships no demo template text."""
from __future__ import annotations

import inspect

from voiceagent import agent, demo_data, reply_guards
from voiceagent.demo_data import REPLY_TEMPLATES


# --- the new module surface ---------------------------------------------------

def test_canned_reply_lives_in_reply_guards():
    out = reply_guards._canned_reply("order_status", "en", ["ORD-77812"])
    assert out == ("Your order ORD-77812 has been checked. The latest "
                   "status will arrive in your app and by SMS shortly.")
    # uncovered language falls back to English, not Hindi
    out2 = reply_guards._canned_reply("order_status", "ta", ["ORD-77812"])
    assert out2 == REPLY_TEMPLATES["order_status"]["en"].format(ref="ORD-77812")


def test_patch_reply_lives_in_reply_guards():
    out = reply_guards._patch_reply("Sure, checking now.", ["ORD-77812"])
    assert out.startswith("I understand — this is regarding ORD-77812.")


def test_language_guard_lives_in_reply_guards():
    assert reply_guards._acceptable_reply_langs("en") is None
    assert reply_guards._acceptable_reply_langs("hinglish") == \
        frozenset({"hinglish", "hi"})
    assert reply_guards._acceptable_reply_langs("te") == frozenset({"te"})


def test_echo_guard_lives_in_reply_guards():
    refs = reply_guards.extract_required_references(
        "my order ORD-77812 recharge failed")
    assert "ORD-77812" in refs
    assert "fail" in refs and "recharge" not in refs  # first-match-per-spec


def test_action_scrub_and_extraction_live_in_reply_guards():
    assert reply_guards.extract_action("foo\nACTION: refund\nbar") == "refund"
    assert reply_guards.extract_action("no action") is None
    assert reply_guards.strip_action_lines(
        "Answer one.\n\nACTION: refund\n\nAnswer two.") == \
        "Answer one.\n\nAnswer two."


def test_order_id_helpers_live_in_reply_guards():
    assert reply_guards.find_order_id("order ORD-77812 please") == "ORD-77812"
    assert reply_guards.find_recent_order_id([]) is None


# --- demo templates moved to demo_data ----------------------------------------

def test_demo_templates_live_in_demo_data():
    assert "order_status" in demo_data.REPLY_TEMPLATES
    assert demo_data.NOTED_REPLIES["en"] == "Your request has been noted."
    assert demo_data.EMPATHY_PREFIXES["en"].startswith("I'm really sorry")
    assert demo_data.DEMO_TENANT_ACTIONS  # the pre-existing demo vocabulary
    assert demo_data.DEMO_TENANT_CONTRACT_SPECS


# --- public-API compatibility: agent.py re-exports -----------------------------

def test_agent_reexports_guard_and_template_names():
    # `from voiceagent.agent import X` keeps working — and re-exports are the
    # SAME objects, not copies.
    from voiceagent.agent import (  # noqa: F401
        EMPATHY_PREFIXES, NOTED_REPLIES, REPLY_TEMPLATES, _acceptable_reply_langs,
        _canned_reply, _patch_reply,
    )
    assert agent.REPLY_TEMPLATES is demo_data.REPLY_TEMPLATES
    assert agent.NOTED_REPLIES is demo_data.NOTED_REPLIES
    assert agent.EMPATHY_PREFIXES is demo_data.EMPATHY_PREFIXES
    assert agent._canned_reply is reply_guards._canned_reply
    assert agent._patch_reply is reply_guards._patch_reply
    assert agent._acceptable_reply_langs is reply_guards._acceptable_reply_langs
    assert agent.extract_required_references is \
        reply_guards.extract_required_references
    assert agent.extract_action is reply_guards.extract_action
    assert agent.strip_action_lines is reply_guards.strip_action_lines
    assert agent.find_order_id is reply_guards.find_order_id


# --- core cleanliness: agent.py ships no demo text ------------------------------

def test_agent_module_contains_no_demo_template_text():
    src = inspect.getsource(agent)
    # a Devanagari demo template line must not live in core agent.py
    assert "आपके ऑर्डर" not in src
    # the demo empathy/noted English strings must not be defined in core
    assert "I'm really sorry about the trouble" not in src
    assert "Your request has been noted." not in src
    # demo reply tables are consumed via the demo_data import
    assert demo_data.__name__ in src
