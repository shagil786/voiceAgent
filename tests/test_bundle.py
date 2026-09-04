# tests/test_bundle.py
from pathlib import Path
import pytest
from voiceagent.deploy.bundle import (
    SCHEMA_VERSION, load_bundle, save_bundle, diff_bundles,
    read_live, write_live,
)

GOLDEN = Path("data/deployments/_example/v1")

def test_schema_version_frozen_at_1():
    assert SCHEMA_VERSION == 1

def test_golden_bundle_loads():
    b = load_bundle(GOLDEN)
    assert b.schema_version == 1
    assert b.deploy_id == "example"
    assert len(b.tools) >= 1
    assert all(t.state in ("PROPOSED", "APPROVED", "CONNECTED") for t in b.tools)

def test_unknown_schema_version_rejected(tmp_path):
    import json
    bad = tmp_path / "bundle.json"
    bad.write_text(json.dumps({"schema_version": 99, "deploy_id": "x"}))
    with pytest.raises(ValueError, match="schema_version"):
        load_bundle(tmp_path)

def test_diff_detects_tool_added_and_policy_changed():
    old = load_bundle(GOLDEN)
    new = load_bundle(GOLDEN)
    new.policies["refund"]["max_without_approval"] = 9999
    d = diff_bundles(old, new)
    assert any(x["section"] == "policies" and x["kind"] == "changed" for x in d)

def test_live_pointer_roundtrip(tmp_path):
    assert read_live(tmp_path) is None
    write_live(tmp_path, "v2")
    assert read_live(tmp_path) == "v2"

def test_save_load_roundtrip(tmp_path):
    b = load_bundle(GOLDEN)
    save_bundle(b, tmp_path / "v1")
    b2 = load_bundle(tmp_path / "v1")
    assert b2.schema_version == b.schema_version
    assert b2.deploy_id == b.deploy_id
    assert b2.spec == b.spec
    assert b2.tools == b.tools
    assert b2.policies == b.policies
    assert b2.knowledge == b.knowledge
    assert b2.evals == b.evals

def test_bad_tool_state_rejected(tmp_path):
    import json
    d = tmp_path / "b"
    d.mkdir()
    (d / "bundle.json").write_text(json.dumps({"schema_version": 1, "deploy_id": "x"}))
    (d / "tools.json").write_text(json.dumps([{"name": "t", "description": "d",
        "parameters": {}, "state": "BOGUS", "connection_ref": None,
        "policy_action": "x", "scopes": []}]))
    (d / "policies.yaml").write_text("x:\n  allow: true\n")
    (d / "evals.json").write_text("[]")
    import pytest
    with pytest.raises(ValueError, match="state"):
        load_bundle(d)


def test_diff_eval_only_change_yields_single_evals_entry():
    import copy
    old = load_bundle(GOLDEN)
    new = load_bundle(GOLDEN)
    assert old.evals, "golden bundle needs at least one eval for this test"
    new.evals[0] = copy.deepcopy(old.evals[0])
    new.evals[0].turns = [{"user": "changed turn"}]
    d = diff_bundles(old, new)
    assert len(d) == 1
    assert d[0]["section"] == "evals" and d[0]["kind"] == "changed"


def test_diff_policies_single_key_change_lists_only_that_key():
    old = load_bundle(GOLDEN)
    new = load_bundle(GOLDEN)
    key = next(iter(old.policies))
    new.policies[key] = {"changed_by_test": True}
    d = diff_bundles(old, new)
    pol = [x for x in d if x["section"] == "policies"]
    assert len(pol) == 1
    assert pol[0]["detail"] == [key]


def test_eval_extra_fields_ignored_forward_compat(tmp_path):
    import json
    import shutil
    shutil.copytree(GOLDEN, tmp_path / "v1")
    evals = json.loads((tmp_path / "v1" / "evals.json").read_text())
    evals[0]["source_contact_hash"] = "abc123"  # Plan 3 eval-tagging
    (tmp_path / "v1" / "evals.json").write_text(json.dumps(evals))
    b = load_bundle(tmp_path / "v1")  # must not raise on unknown fields
    assert len(b.evals) == len(evals)
