from voiceagent.deploy.compiler import compile_bundle
from voiceagent.deploy.ingest import ingest_owner_paste

INTERVIEW = {
    "offering": "Sharma Realty sells 2/3BHK flats in Whitefield",
    "top_asks": ["2BHK price?", "site visit slot?", "loan help?",
                 "floor plans?", "possession date?"],
    "never_promise": ["never promise loan approval", "never quote final price"],
    "handoff_triggers": ["customer asks legal", "budget above 2cr"],
}


def test_compiler_emits_gated_bundle_with_10_evals():
    chunks = [ingest_owner_paste("2BHK from 85L. Site visits 10am-6pm.")]
    b = compile_bundle("sharma-realty", chunks, INTERVIEW)
    assert b.schema_version == 1 and b.deploy_id == "sharma-realty"
    assert b.tools and all(t.state == "PROPOSED" for t in b.tools)
    assert len(b.evals) == 10
    assert "answer" in b.spec.get("patterns", [])
    for name in ("escalate_to_human",):
        assert b.policies[name] == {"allow": True}


def test_external_tools_default_require_approval():
    chunks = [ingest_owner_paste("We book site visits.")]
    b = compile_bundle("x", chunks, INTERVIEW)
    # escalate_to_human is always-allowed by design; external tools default require_approval
    for t in b.tools[1:]:
        action = t.policy_action or t.name
        assert b.policies.get(action, {}).get("require_approval") is True
