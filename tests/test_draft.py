"""Brain-assisted onboarding drafts (deploy/draft.py): the frontier READS the
business content and proposes surfaces; gap questions go back to the owner.
Every test here runs WITHOUT a model (stub LLM or none) — the brain path
must degrade to the deterministic compiler silently."""
import json

CHUNKS = [{"source": "owner_paste",
           "text": "Sunrise Dental Clinic offers root canals, cleanings and "
                   "braces, open 9am-6pm weekdays. Patients ask about prices, "
                   "book visits by phone, and cancel with booking reference "
                   "APT-1042. We guarantee painless treatment."}]


class StubLLM:
    def __init__(self, payload):
        self._payload = payload
    def generate(self, prompt, max_tokens=2000):
        assert "Sunrise" in prompt  # content actually reaches the brain
        return json.dumps(self._payload)


DRAFT = {
    "tools": [{
        "tool_name": "fetch_appointment_status",
        "description": "Look up a patient's appointment",
        "params": ["appointment_id"], "action": "fetch_appointment_status",
        "operation": "__fetch__", "operation_params": {},
        "resource_type": "appointment", "id_param": "appointment_id",
        "preconditions": [], "facts": ["appointment"],
        "side_effects": False, "risk_class": "read"}],
    "intents": {"fetch_appointment_status":
                ["where is my appointment", "appointment kab hai"]},
    "entities": {"record_ids": [{
        "code": "APT", "digit_pattern": "\\b(?:APT[-#]?\\s*)(\\d{4,10})\\b",
        "prefix_pattern": "\\bAPT\\b[-#\\s:]*", "bare_digits": True,
        "min_digits": 4, "max_digits": 10}]},
    "policies": {"fetch_appointment_status": {"require_approval": False}},
    "evals": [{"user": "what are your hours"}],
    "questions": [],
}


def test_brain_draft_produces_validated_surfaces():
    from voiceagent.deploy.draft import draft_bundle_surfaces
    out = draft_bundle_surfaces(CHUNKS, {"offering": "dental clinic"},
                                llm=StubLLM(DRAFT))
    assert out["drafted"] is True
    assert out["tools"][0]["name"] == "fetch_appointment_status"
    assert out["tools"][0]["state"] == "PROPOSED"  # never approved here
    assert out["intents"]["fetch_appointment_status"][0] == \
        "where is my appointment"
    assert out["entities"]["record_ids"][0]["code"] == "APT"
    # Risky promise in content -> confirm question, even with a full draft.
    ids = [q["id"] for q in out["questions"]]
    assert "never_promise" in ids
    assert all({"id", "prompt", "why", "kind", "answer_key"} <= set(q)
               for q in out["questions"])


def test_invalid_drafts_become_questions_not_silent_garbage():
    from voiceagent.deploy.draft import draft_bundle_surfaces
    bad = dict(DRAFT)
    bad["tools"] = [{"tool_name": "x", "operation": "nope"}]
    bad["entities"] = {"record_ids": [{
        "code": "ZZ", "digit_pattern": "(unclosed",
        "min_digits": 4, "max_digits": 10}]}
    out = draft_bundle_surfaces(CHUNKS, {}, llm=StubLLM(bad))
    assert out["tools"] == []
    assert out["entities"] is None
    ids = [q["id"] for q in out["questions"]]
    assert "tool_x" in ids and "id_example" in ids
    # And the deterministic gaps still fire on an empty interview.
    for need in ("offering", "top_asks", "handoff_triggers", "greeting",
                 "languages", "erp_url"):
        assert need in ids


def test_no_brain_falls_back_to_template_plus_questions():
    from voiceagent.deploy.draft import draft_bundle_surfaces, preview_bundle
    out = draft_bundle_surfaces(CHUNKS, {"offering": "dental"}, llm=None)
    # No VOICEAGENT_LLM_* in this env -> deterministic path.
    assert out["drafted"] is False
    assert len(out["questions"]) >= 4
    prev = preview_bundle("preview", CHUNKS, {"offering": "dental"},
                          llm=None)
    assert prev["tools"]  # v1 template tools survive
    assert prev["questions"] and prev["drafted"] is False
    assert "template preview" in prev["note"]


def test_garbage_brain_output_is_fail_open():
    from voiceagent.deploy.draft import draft_bundle_surfaces

    class Babble:
        def generate(self, prompt, max_tokens=2000):
            return "Sure! Here is my analysis: <not json at all>"
    out = draft_bundle_surfaces(CHUNKS, {}, llm=Babble())
    assert out["drafted"] is False
    assert any(q["id"] == "offering" for q in out["questions"])


def test_preview_keys_cover_dashboard_contract():
    from voiceagent.deploy.draft import preview_bundle
    out = preview_bundle("preview", CHUNKS, {}, llm=StubLLM(DRAFT))
    for key in ("deploy_id", "spec", "knowledge", "tools", "policies",
                "evals", "intents", "entities", "questions", "drafted",
                "note"):
        assert key in out, f"preview missing {key}"
