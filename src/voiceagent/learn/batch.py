"""Batch-learn miner: cluster candidates into anonymized proposals (no LLM)."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone

STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "have",
    "from", "they", "them", "then", "your", "about", "into",
})

_PTYPE_ORDER = ("fact", "policy", "tone", "exemplar")
_PTYPE_RANK = {t: i for i, t in enumerate(_PTYPE_ORDER)}
_KIND_BY_PTYPE = {
    "fact": "knowledge_gap",
    "policy": "threshold",
    "tone": "wording",
    "exemplar": "exemplar",
}


def hash_contact(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def normalize_quote(text: str) -> str:
    out = re.sub(r"[^a-z0-9 ]", "", text.lower())
    out = re.sub(r"\s+", " ", out).strip()
    return out[:200]


def _longest_quote(members: list[dict]) -> str:
    return max((str(m.get("quote", "")) for m in members), key=len, default="")


# Downstream contract (Task 3 / spec §4.5a purge): the applier joins the FULL
# hash list into spec.eval_sources so purge_contact attributes every
# contributing contact. `hashes` is display-capped at 25; `all_hashes` is the
# uncapped sorted full list the applier must join. `distinct_hashes` is always
# the FULL distinct count, never the truncated-list length.
def _evidence(members: list[dict], all_hashes: list[str]) -> dict:
    return {
        "count": len(members),
        "distinct_hashes": len(all_hashes),
        "hashes": all_hashes[:25],
        "all_hashes": list(all_hashes),
        "sample_quotes": [str(m.get("quote", ""))[:120] for m in members[:3]],
    }


def _majority_ptype(members: list[dict]) -> str:
    counts = Counter(m.get("patch_type", "fact") for m in members)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], _PTYPE_RANK.get(kv[0], 99)))
    winner = ranked[0][0] if ranked else "fact"
    return winner if winner in _KIND_BY_PTYPE else "fact"


def _patch_for(kind: str, longest: str) -> dict:
    if kind == "knowledge_gap":
        today = datetime.now(timezone.utc).date().isoformat()
        return {"text": longest, "source": f"batch:{today}"}
    if kind == "threshold":
        return {"never_promise_add": longest, "needs_dsl_review": True}
    if kind == "wording":
        return {"tone_notes_add": longest}
    return {"user": longest, "assert_contains": longest[:12]}


def _keyword_proposal(outcomes: list) -> dict | None:
    neg = [o for o in (outcomes or [])
           if getattr(o, "label", "") in ("thumbs_down", "escalated")
           and getattr(o, "note", "")]
    if len(neg) < 3:
        return None
    word_sessions: dict[str, set[str]] = {}
    for o in neg:
        words = {w for w in re.findall(r"[a-z0-9]+", o.note.lower())
                 if len(w) >= 4 and w not in STOPWORDS}
        for w in words:
            word_sessions.setdefault(w, set()).add(o.session_id)
    qualified = {w: s for w, s in word_sessions.items() if len(s) >= 3}
    if not qualified:
        return None
    word = sorted(qualified, key=lambda w: (-len(qualified[w]), w))[0]
    matching = [o for o in neg
                if word in re.findall(r"[a-z0-9]+", o.note.lower())]
    all_hashes = sorted({o.contact_hash for o in matching if o.contact_hash})
    return {
        "kind": "wording",
        "title": f"Repeated {word} complaints",
        "detail": f"{len(matching)} negative outcomes mention '{word}'",
        "evidence": {
            "count": len(matching),
            "distinct_hashes": len(all_hashes),
            "hashes": all_hashes[:25],
            "all_hashes": list(all_hashes),
            "sample_quotes": [o.note[:120] for o in matching[:3]],
        },
        "patch": {"tone_notes_add": ""},
        "status": "proposed",
    }


def mine_proposals(candidates: list[dict], outcomes: list,
                   bundle=None) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for c in candidates or []:
        groups.setdefault(normalize_quote(str(c.get("quote", ""))), []).append(c)

    hashed_by_template = {
        t: [m for m in members if m.get("contact_hash")]
        for t, members in groups.items()
    }
    ordered = sorted(groups.items(),
                     key=lambda kv: (-len(hashed_by_template[kv[0]]), kv[0]))
    proposals: list[dict] = []
    for template, members in ordered:
        hashed = hashed_by_template[template]
        distinct = sorted({m["contact_hash"] for m in hashed})
        if len(hashed) < 3 or len(distinct) < 3:
            continue
        ptype = _majority_ptype(hashed)
        kind = _KIND_BY_PTYPE[ptype]
        longest = _longest_quote(hashed)
        proposals.append({
            "kind": kind,
            "title": longest[:60],
            "detail": (f"{len(hashed)} reports from "
                       f"{len(set(m['contact_hash'] for m in hashed))} "
                       f"contacts; majority {ptype}"),
            "evidence": _evidence(hashed, distinct),
            "patch": _patch_for(kind, longest),
            "status": "proposed",
        })

    keyword = _keyword_proposal(outcomes)
    if keyword:
        proposals = proposals[:49]
        proposals.append(keyword)
    proposals = proposals[:50]

    proposals.sort(key=lambda p: (p["kind"], p["title"]))
    counters: dict[str, int] = {}
    for p in proposals:
        n = counters.get(p["kind"], 0)
        p["id"] = f"{p['kind']}-{n:03d}"
        counters[p["kind"]] = n + 1
    return proposals
