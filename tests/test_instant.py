# tests/test_instant.py
import pytest
from voiceagent.deploy.bundle import load_bundle
from voiceagent.learn.instant import (
    apply_owner_correction, instant_correct, next_version)
from voiceagent.learn.corrections import classify_correction

GOLDEN = "data/deployments/_example/v1"

def test_fact_patch_appends_knowledge_with_source():
    b = load_bundle(GOLDEN)
    c = classify_correction("No, the fee is 499", is_owner=True)
    new, log = apply_owner_correction(b, c)
    assert any(ch["source"].startswith("owner_correction:") for ch in new.knowledge)
    assert len(b.knowledge) + 1 == len(new.knowledge)  # copy-on-write
    assert log["patch_type"] == "fact"

def test_policy_patch_flags_dsl_review_on_amounts():
    b = load_bundle(GOLDEN)
    new, log = apply_owner_correction(
        b, classify_correction("Never promise refunds above 5000", is_owner=True))
    assert "Never promise refunds above 5000" in new.spec.get("never_promise", [])
    assert log["needs_dsl_review"] is True

def test_customer_scope_rejected_and_versions_increment(tmp_path):
    b = load_bundle(GOLDEN)
    cust = classify_correction("No, mine is 3BHK", is_owner=False)
    with pytest.raises(ValueError, match="owner scope"):
        apply_owner_correction(b, cust)
    assert next_version(tmp_path) == "v1"
    (tmp_path / "v3").mkdir()
    assert next_version(tmp_path) == "v4"

def test_instant_correct_goes_live_and_fast(tmp_path):
    import shutil, time
    from voiceagent.deploy.bundle import load_bundle, save_bundle
    shutil.copytree(GOLDEN, tmp_path / "v1")
    b = load_bundle(tmp_path / "v1")
    # golden ships 2 evals incl. an action-assert that fail-closes without a
    # wired runner; go_live needs ≥10 all-pass — so stage 10 contains-only
    # evals in-fixture (R3). This exercises the real live path (patch →
    # checks → pointer flip); golden fail-closed semantics stay covered in
    # selfcheck tests.
    from voiceagent.deploy.bundle import EvalCheck
    b.evals = [EvalCheck(name=f"live-{i:02d}", turns=[{"user": "Hello"}],
                         assert_={"contains": "Hello"}) for i in range(10)]
    save_bundle(b, tmp_path / "v1")
    t0 = time.monotonic()
    out = instant_correct(str(tmp_path), "No, visits run 10am to 6pm")
    dt = time.monotonic() - t0
    assert out["live"] is True and out["passed"] is True
    assert dt < 60  # §7.1-4 headroom: stub tier runs in ms; 60s guards CI flakes
    from voiceagent.deploy.bundle import read_live
    assert read_live(str(tmp_path)) == out["version"]
