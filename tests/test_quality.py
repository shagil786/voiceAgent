"""Tests for voiceagent.quality — the judge that grades conversations.

The honesty contract under test: every score carries its source, an
unparseable or unavailable judge degrades to the deterministic heuristic,
and the heuristic never presents itself as a model opinion.
"""
import pytest

from voiceagent.quality import (
    _parse_scores,
    _spine_text,
    build_judge_from_env,
    heuristic_score,
    judge_conversation,
)


def _row(action="order_lookup", verdict="ALLOW", reasons=("phone match",)):
    return {"action": action, "verdict": verdict, "reasons": list(reasons)}


# ── spine rendering ──────────────────────────────────────────────────

def test_spine_text_lists_each_decision():
    text = _spine_text([_row(), _row("cancel_order", "DENY", ("shipped",))])
    assert "- order_lookup -> ALLOW (phone match)" in text
    assert "- cancel_order -> DENY (shipped)" in text


def test_spine_text_tolerates_missing_reasons():
    assert "- order_lookup -> ALLOW ()" in _spine_text([{"action": "order_lookup", "verdict": "ALLOW"}])


# ── heuristic ────────────────────────────────────────────────────────

def test_heuristic_empty_spine_scores_nothing():
    out = heuristic_score([])
    assert out["source"] == "heuristic"
    assert out["overall"] is None
    assert out["reasoning"] == "no decisions recorded"


def test_heuristic_clean_spine_scores_high():
    out = heuristic_score([_row(), _row("fetch_status")])
    assert out["source"] == "heuristic"
    assert out["tool_choice"] == 8
    assert out["verdict_quality"] == 7
    assert out["overall"] == pytest.approx((8 + 7 + 6) / 3, abs=0.05)


def test_heuristic_missing_reasons_drags_both_axes():
    rows = [{"action": "x", "verdict": "ALLOW", "reasons": []},
            {"action": "y", "verdict": "ALLOW", "reasons": []}]
    out = heuristic_score(rows)
    assert out["tool_choice"] == 5
    assert out["verdict_quality"] == 4


def test_heuristic_counts_escalations_and_blocks_in_reasoning():
    rows = [_row(), _row("fraud_check", "ESCALATE", ("risk",)), _row("refund", "DENY", ("shipped",))]
    out = heuristic_score(rows)
    assert "1 escalation(s), 1 blocked" in out["reasoning"]


def test_heuristic_all_escalations_is_not_balanced():
    out = heuristic_score([_row("a", "ESCALATE", ("r",)), _row("b", "ESCALATE", ("r",))])
    assert out["escalation_judgment"] == 6


# ── strict-JSON parsing ──────────────────────────────────────────────

_GOOD = '{"tool_choice": 8, "verdict_quality": 7, "escalation_judgment": 9, "overall": 9, "reasoning": "clean run"}'


def test_parse_scores_accepts_strict_json():
    out = _parse_scores(_GOOD)
    assert out == {"tool_choice": 8.0, "verdict_quality": 7.0,
                   "escalation_judgment": 9.0, "overall": 9.0,
                   "reasoning": "clean run"}


def test_parse_scores_extracts_json_from_prose_wrapping():
    out = _parse_scores('Sure, here you go:\n' + _GOOD + '\nHope that helps.')
    assert out is not None and out["overall"] == 9.0


def test_parse_scores_rejects_missing_keys():
    assert _parse_scores('{"tool_choice": 8, "overall": 8}') is None


def test_parse_scores_rejects_non_numeric():
    assert _parse_scores('{"tool_choice": "great", "verdict_quality": 7, "escalation_judgment": 7, "overall": 7}') is None


def test_parse_scores_rejects_garbage():
    assert _parse_scores("no json here at all") is None
    assert _parse_scores("") is None


def test_parse_scores_clamps_to_range():
    out = _parse_scores('{"tool_choice": 99, "verdict_quality": -3, "escalation_judgment": 7, "overall": 7}')
    assert out is not None
    assert out["tool_choice"] == 10.0
    assert out["verdict_quality"] == 0.0


def test_parse_scores_truncates_long_reasoning():
    out = _parse_scores('{"tool_choice": 8, "verdict_quality": 7, "escalation_judgment": 7, "overall": 7, "reasoning": "%s"}' % ("x" * 500))
    assert out is not None
    assert len(out["reasoning"]) == 300


# ── judge entry point ────────────────────────────────────────────────

class _StubLLM:
    def __init__(self, text=None, exc=None):
        self.text = text
        self.exc = exc
        self.prompts = []

    def chat_template(self, system, role, prompt):
        self.prompts.append((system, role, prompt))

    def generate(self, prompt, max_tokens=220):
        if self.exc is not None:
            raise self.exc
        return self.text


def test_no_llm_means_heuristic():
    out = judge_conversation([_row()], None)
    assert out["source"] == "heuristic"
    assert out["overall"] is not None


def test_llm_scores_carry_llm_judge_source():
    llm = _StubLLM(_GOOD)
    out = judge_conversation([_row()], llm)
    assert out["source"] == "llm-judge"
    assert out["overall"] == pytest.approx((8 + 7 + 9) / 3, abs=0.05)
    assert out["reasoning"] == "clean run"
    assert llm.prompts and "order_lookup" in llm.prompts[0][2]


def test_llm_overall_is_recomputed_not_trusted():
    doctored = _GOOD.replace('"overall": 9', '"overall": 1')
    out = judge_conversation([_row()], _StubLLM(doctored))
    assert out["overall"] == pytest.approx((8 + 7 + 9) / 3, abs=0.05)


def test_unparseable_judge_output_falls_back_and_says_so():
    out = judge_conversation([_row()], _StubLLM("I can't score this, sorry"))
    assert out["source"] == "heuristic"
    assert out["reasoning"].startswith("judge output unparseable")


def test_judge_exception_falls_back_silently():
    out = judge_conversation([_row()], _StubLLM(exc=RuntimeError("down")))
    assert out["source"] == "heuristic"


# ── env wiring ───────────────────────────────────────────────────────

def test_no_frontier_url_means_no_judge():
    assert build_judge_from_env({}) is None
    assert build_judge_from_env({"VOICEAGENT_FRONTIER_URL": "  "}) is None
