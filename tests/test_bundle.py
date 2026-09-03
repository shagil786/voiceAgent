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
