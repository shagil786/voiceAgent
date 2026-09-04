# tests/test_batch_cli.py
import sys
sys.path.insert(0, "scripts")
from batch_learn import approve, mine
from voiceagent.learn.outcomes import JsonlOutcomes, OutcomeLabel
from voiceagent.learn.profiles import Profile, SQLiteProfiles

def test_mine_approve_end_to_end(tmp_path):
    import shutil
    from voiceagent.deploy.bundle import load_bundle
    from voiceagent.learn.batch import EvalCheck
    shutil.copytree("data/deployments/_example/v1", tmp_path / "v1")
    b = load_bundle(tmp_path / "v1")
    b.evals = [EvalCheck(name=f"e{i:02d}", turns=[{"user": "Hi"}],
                         assert_={"contains": "Hi"}) for i in range(10)]
    from voiceagent.deploy.bundle import save_bundle
    save_bundle(b, tmp_path / "v1")
    db = SQLiteProfiles(str(tmp_path / "p.db"))
    from voiceagent.learn.batch import hash_contact
    for i in range(3):
        key = f"+910000000{i}"
        db.put(Profile(key=key, alias="", prefs=[], corrections=[], open_items=[],
                       pending_global=[{"quote": "No, fee is 499", "patch_type": "fact",
                                        "session_id": "s", "ts": "t"}],
                       consent={}, updated_at="2026-09-04T00:00:00"))
    keys = [f"+910000000{i}" for i in range(3)]
    path = mine(str(tmp_path), "nope.jsonl", str(tmp_path / "p.db"), keys=keys)
    import json
    props = json.loads(__import__("pathlib").Path(path).read_text())
    assert len(props) == 1 and props[0]["kind"] == "knowledge_gap"
    out = approve(str(tmp_path), ids=[props[0]["id"]])
    assert out["live"] is True and out["applied"] == [props[0]["id"]]
