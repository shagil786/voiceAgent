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

def test_policy_majority_and_cap():
    cands = [_cand(f"No, never promise X{i%2}", ptype="policy", h=f"h{i}") for i in range(6)]
    props = mine_proposals(cands, [])
    assert props and all(p["kind"] == "threshold" for p in props)
    assert all(p["patch"]["needs_dsl_review"] is True for p in props)
