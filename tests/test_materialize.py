"""Last-mile onboarding (deploy/materialize.py): approved draft surfaces
become a tenant bundle that the REAL validator passes. Nothing is written
unless verification is clean."""
import json

SURFACES = {
    "tools": [{
        "name": "fetch_visit_status", "description": "Look up a visit",
        "params": ["visit_id"], "action": "fetch_visit_status",
        "operation": "__fetch__", "operation_params": {},
        "resource_type": "visit", "id_param": "visit_id",
        "filter_param": None, "filter_key": None, "preconditions": [],
        "facts": ["visit"], "side_effects": False, "risk_class": "read"}],
    "intents": {"fetch_visit_status":
                ["where is my visit", "visit kab hai", "status batao"]},
    "entities": {"record_ids": [{
        "code": "VST", "digit_pattern": "\\b(?:VST[-#]?\\s*)(\\d{4,10})\\b",
        "prefix_pattern": "\\bVST\\b[-#\\s:]*", "bare_digits": True,
        "min_digits": 4, "max_digits": 10}]},
    "policies": {"fetch_visit_status": {"require_approval": False}},
}
INTERVIEW = {"offering": "Sunrise Vet Clinic", "languages": ["en", "hi"],
             "currency": "₹", "greeting": "Hello, Sunrise Vet here!",
             "handoff_triggers": ["angry caller"],
             "never_promise": ["same-day slots"]}
CHUNKS = [{"source": "owner_paste",
           "text": "Sunrise Vet Clinic treats dogs and cats, open 9-6. "
                   "Ask about prices, book visits, cancel with VST-1042."}]


def test_materialize_writes_verified_bundle(tmp_path):
    from voiceagent.deploy.materialize import materialize_tenant_bundle
    from voiceagent.tenant import Tenant
    out = materialize_tenant_bundle(SURFACES, INTERVIEW, CHUNKS, "vet",
                                    tmp_path / "tenant")
    assert out["ok"] and out["errors"] == [], out["errors"]
    assert out["tenant"] == "vet"
    root = tmp_path / "tenant"
    assert (root / "tenant.json").exists()
    assert (root / "intents" / "fetch_visit_status.yaml").exists()
    assert (root / "entities.yaml").exists()
    assert (root / "proposals.yaml").exists()
    assert (root / "policies.yaml").exists()
    assert (root / "knowledge" / "00.md").exists()
    # The written bundle loads through the REAL seams.
    t = Tenant.load(root)
    shapes = t.record_id_shapes() or []
    assert shapes and shapes[0]["code"] == "VST"
    assert "fetch_visit_status" in (t.action_vocabulary() or [])
    cfg = json.loads((root / "tenant.json").read_text())
    assert cfg["persona"]["languages"] == ["en", "hi"]
    assert cfg["currency"] == "₹"
    assert cfg["persona"]["greeting"] == "Hello, Sunrise Vet here!"


def test_materialize_writes_nothing_on_invalid(tmp_path):
    from voiceagent.deploy.materialize import materialize_tenant_bundle
    bad = dict(SURFACES)
    bad["entities"] = {"record_ids": [{
        "code": "ZZ", "digit_pattern": "(unclosed",
        "min_digits": 4, "max_digits": 10}]}
    out = materialize_tenant_bundle(bad, INTERVIEW, CHUNKS, "vet",
                                    tmp_path / "tenant")
    assert out["ok"] is False and out["errors"]
    assert not (tmp_path / "tenant").exists()


def test_materialize_refuses_to_overwrite(tmp_path):
    from voiceagent.deploy.materialize import materialize_tenant_bundle
    (tmp_path / "tenant").mkdir()
    out = materialize_tenant_bundle(SURFACES, INTERVIEW, CHUNKS, "vet",
                                    tmp_path / "tenant")
    assert out["ok"] is False and "overwrite" in out["errors"][0]


def test_empty_surfaces_still_verify(tmp_path):
    # Answers-only mode (no tools/ids): knowledge + guardrails only.
    from voiceagent.deploy.materialize import materialize_tenant_bundle
    out = materialize_tenant_bundle(
        {"tools": [], "intents": {}, "entities": None, "policies": {}},
        {"offering": "Info Desk"}, CHUNKS, "info", tmp_path / "t")
    assert out["ok"], out["errors"]


def test_v1_tool_shapes_translate_and_valves_skip(tmp_path):
    # v1 compiler tools carry parameters.properties (not params); platform
    # valves never become tenant proposals; the rest verifies clean.
    from voiceagent.deploy.materialize import materialize_tenant_bundle
    surfaces = {"tools": [
        {"name": "escalate_to_human", "description": "handoff",
         "parameters": {"type": "object", "properties": {}},
         "policy_action": "escalate_to_human"},
        {"name": "booking", "description": "Handle: booking",
         "parameters": {"type": "object",
                        "properties": {"query": {"type": "string"}}},
         "policy_action": "booking"}],
        "intents": {}, "entities": None, "policies": {}}
    out = materialize_tenant_bundle(surfaces, {"offering": "vet"},
                                    CHUNKS, "vet2", tmp_path / "t")
    assert out["ok"], out["errors"]
    text = (tmp_path / "t" / "proposals.yaml").read_text()
    assert "escalate_to_human" not in text  # platform-owned surface
    assert "booking" in text and "query" in text  # params translated
    assert out["skipped"] == []


def test_template_preview_materializes_end_to_end(tmp_path):
    # The no-brain path: v1 tools (parameters.properties) flow through
    # preview_bundle into a verifying tenant — the exact shape that failed
    # live review (params lost between compiler and materializer).
    from voiceagent.deploy.draft import preview_bundle
    from voiceagent.deploy.materialize import materialize_tenant_bundle
    chunks = [{"source": "owner_paste",
               "text": "Sunrise Vet treats dogs. Cancel with VST-1042."}]
    prev = preview_bundle("e2e", chunks, {"offering": "vet"}, llm=None)
    assert prev["drafted"] is False
    out = materialize_tenant_bundle(
        {"tools": prev["tools"], "intents": prev["intents"],
         "entities": prev["entities"], "policies": prev["policies"]},
        {"offering": "vet"}, chunks, "vet3", tmp_path / "t")
    assert out["ok"], out["errors"]
    assert out["skipped"] == []
