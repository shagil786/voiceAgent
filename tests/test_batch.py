from voiceagent.learn.batch import hash_contact, mine_proposals, normalize_quote

def _cand(quote, ptype="fact", h="h1", ts="2026-09-01T00:00:00"):
    return {"quote": quote, "patch_type": ptype, "session_id": "s",
            "ts": ts, "contact_hash": h}

def test_hash_and_normalize():
    assert len(hash_contact("+911")) == 64 and hash_contact("+911") != hash_contact("+912")
    assert normalize_quote("  NO,  the Fee is 499!! ") == "no the fee is 499"

def test_anonymity_gate_needs_3x3():
    same = [_cand("No, fee is 499", h="h1"), _cand("No! fee is 499?", h="h1"),
            _cand("no fee is 499.", h="h1")]
    assert mine_proposals(same, []) == []  # 1 hash → nothing
    trio = [_cand("No, fee is 499", h="h1"), _cand("NO fee is 499!", h="h2"),
            _cand("no, fee is 499.", h="h3")]
    props = mine_proposals(trio, [])
    assert len(props) == 1 and props[0]["kind"] == "knowledge_gap"
    assert props[0]["evidence"]["distinct_hashes"] == 3
    assert props[0]["id"] == "knowledge_gap-000"

def test_full_distinct_count_with_capped_display():
    cands = [_cand("No, fee is 499", h=f"h{i:02d}") for i in range(30)]
    props = mine_proposals(cands, [])
    assert len(props) == 1
    ev = props[0]["evidence"]
    assert ev["distinct_hashes"] == 30
    assert len(ev["hashes"]) == 25
    assert len(ev["all_hashes"]) == 30
    assert ev["hashes"] == sorted(ev["all_hashes"])[:25]

def test_mixed_hashless_group_yields_nothing():
    cands = [_cand("No, fee is 499", h="h1"), _cand("No! fee is 499?", h="h2")]
    for i in range(5):
        c = _cand("no fee is 499.", h="hx")
        del c["contact_hash"]
        cands.append(c)
    assert mine_proposals(cands, []) == []  # 2 hashed + 5 hashless → nothing

def test_policy_majority_and_cap():
    cands = [_cand(f"No, never promise X{i%2}", ptype="policy", h=f"h{i}") for i in range(6)]
    props = mine_proposals(cands, [])
    assert props and all(p["kind"] == "threshold" for p in props)
    assert all(p["patch"]["needs_dsl_review"] is True for p in props)

def test_apply_and_purge_roundtrip():
    from voiceagent.deploy.bundle import load_bundle
    from voiceagent.learn.batch import apply_approved, purge_contact
    b = load_bundle("data/deployments/_example/v1")
    approvals = [
        {"id": "exemplar-000", "kind": "exemplar", "title": "t", "detail": "d",
         "evidence": {"count": 3, "distinct_hashes": 3, "hashes": ["abc"],
                      "sample_quotes": []},
         "patch": {"user": "2BHK price?", "assert_contains": "2BHK price?"},
         "status": "approved"},
        {"id": "wording-000", "kind": "wording", "title": "t", "detail": "d",
         "evidence": {"count": 1, "distinct_hashes": 1, "hashes": [],
                      "sample_quotes": []},
         "patch": {"tone_notes_add": ""}, "status": "approved"},
    ]
    new, log = apply_approved(b, approvals)
    assert [a for a in log["applied"]] == ["exemplar-000"]
    assert log["skipped"][0]["id"] == "wording-000"
    assert new.spec["eval_sources"]["batch-exemplar-000"] == "abc"
    assert len(b.evals) == len(load_bundle("data/deployments/_example/v1").evals)
    pruned, n = purge_contact(new, "abc")
    assert n == 1 and not any("batch-exemplar-000" in e.name for e in pruned.evals)
    same, n0 = purge_contact(new, "nope")
    assert n0 == 0 and same is new

def test_flood_single_hash_yields_nothing_and_no_raw_keys():
    import json
    from voiceagent.learn.batch import mine_proposals
    cands = [{"quote": f"No, item {i} broke", "patch_type": "fact",
              "session_id": "s", "ts": "t", "contact_hash": "solo"} for i in range(100)]
    assert mine_proposals(cands, []) == []
    from voiceagent.learn.batch import hash_contact
    trio = [{"quote": "No, fee is 499", "patch_type": "fact", "session_id": "s",
             "ts": "t", "contact_hash": h} for h in ("a", "b", "c")]
    props = mine_proposals(trio, [])
    blob = json.dumps(props)
    assert "+911" not in blob and len(props) == 1

def test_proposal_cap_and_deterministic_order():
    from voiceagent.learn.batch import mine_proposals
    cands = []
    for g in range(60):
        for h in ("h1", "h2", "h3"):
            cands.append({"quote": f"No, thing{g} failed", "patch_type": "fact",
                          "session_id": "s", "ts": "t", "contact_hash": f"{h}-{g}"})
    props = mine_proposals(cands, [])
    assert len(props) == 50
    assert [p["id"] for p in props] == sorted(p["id"] for p in props)
