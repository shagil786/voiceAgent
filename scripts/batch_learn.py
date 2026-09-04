"""Nightly batch-learn CLI: mine proposals → owner approve → versioned bundle.

On-demand job (v1; scheduling is ops, not code). Candidate enumeration uses
an explicit `--keys` contact list: the profile-store protocol has no key
listing by design, so the owner passes the contact list (or recent-call
export) to mine. Real mining reads the SQLite profiles DB (`--profiles-db`);
without it the store is an (empty-by-construction) InMemoryProfiles.

Customer text NEVER writes the global bundle directly; only owner-approved
proposals become versions. Contact hashes are SHA-256 hex; raw keys never
enter proposals, evals, or logs.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.deploy.bundle import load_bundle, read_live, save_bundle
from voiceagent.deploy.selfcheck import go_live, run_self_checks
from voiceagent.learn.batch import (
    apply_approved,
    hash_contact,
    mine_proposals,
    purge_contact,
    read_proposals,
    write_proposals,
)
from voiceagent.learn.instant import next_version
from voiceagent.learn.outcomes import InMemoryOutcomes, JsonlOutcomes
from voiceagent.learn.profiles import InMemoryProfiles, SQLiteProfiles


def _load_outcomes(outcomes_path):
    """Outcomes store; a missing/unreadable file means empty, never a crash."""
    try:
        return JsonlOutcomes(outcomes_path) if outcomes_path else InMemoryOutcomes()
    except Exception:
        return InMemoryOutcomes()


def _load_profiles(profiles_db):
    """SQLite store when a DB path is given, else empty in-memory store."""
    if not profiles_db:
        return InMemoryProfiles()
    try:
        return SQLiteProfiles(profiles_db)
    except Exception:
        return InMemoryProfiles()


def _base_version(deploy_dir: Path) -> str:
    """Live pointer when set, else the highest `v*` dir (instant pattern)."""
    live = read_live(deploy_dir)
    if live is not None:
        return live
    nums = [int(p.name[1:]) for p in deploy_dir.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()]
    if not nums:
        raise FileNotFoundError(f"no bundle version in {deploy_dir}")
    return f"v{max(nums)}"


def _latest_proposals_file(deploy_dir: Path) -> Path:
    """Lexicographically max file in `<deploy>/proposals/`."""
    pdir = deploy_dir / "proposals"
    files = sorted(p for p in pdir.iterdir() if p.is_file()) if pdir.exists() else []
    if not files:
        raise FileNotFoundError(f"no proposals in {pdir}")
    return files[-1]


def mine(deploy_dir, outcomes_path, profiles_db, keys=None) -> str:
    """Mine candidates into `<deploy>/proposals/<YYYY-MM-DD>.json`; return path.

    Tolerant: missing outcomes file / profiles DB yield empty inputs, never
    a crash (proposals then come only from whatever inputs exist).
    """
    deploy = Path(deploy_dir)
    outcomes = _load_outcomes(outcomes_path)
    try:
        labels = outcomes.query()
    except Exception:
        labels = []
    store = _load_profiles(profiles_db)
    candidates: list[dict] = []
    for key in keys or []:
        try:
            profile = store.get(key)
        except Exception:
            continue
        if profile is None:
            continue
        contact_hash = hash_contact(key)
        for entry in profile.pending_global or []:
            candidates.append({
                "quote": entry.get("quote", ""),
                "patch_type": entry.get("patch_type", "fact"),
                "session_id": entry.get("session_id", ""),
                "ts": entry.get("ts", ""),
                "contact_hash": contact_hash,
            })
    proposals = mine_proposals(candidates, labels)
    today = datetime.now(timezone.utc).date().isoformat()
    path = deploy / "proposals" / f"{today}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_proposals(str(path), proposals)
    return str(path)


def approve(deploy_dir, ids=None, approve_all=False, make_brain=None) -> dict:
    """Approve proposals → new bundle version → self-checks → go-live.

    Fail-closed: the new version is always saved (auditable); the live
    pointer flips only when `go_live` approves (≥10 all-pass).
    """
    deploy = Path(deploy_dir)
    proposals_path = _latest_proposals_file(deploy)
    proposals = read_proposals(str(proposals_path))
    if ids == "all":
        approve_all = True
    wanted = None if approve_all else set(ids or [])
    approvals: list[dict] = []
    for prop in proposals:
        if approve_all:
            if prop.get("status") == "proposed":
                prop["status"] = "approved"
                approvals.append(prop)
        elif wanted is not None and prop.get("id") in wanted:
            prop["status"] = "approved"
            approvals.append(prop)
    base = _base_version(deploy)
    new, changelog = apply_approved(load_bundle(deploy / base), approvals)
    version = next_version(deploy)
    save_bundle(new, deploy / version)
    results = run_self_checks(new, make_brain)
    live = go_live(str(deploy), version, results)
    write_proposals(str(proposals_path), proposals)
    return {"version": version, "applied": changelog.get("applied", []),
            "skipped": changelog.get("skipped", []), "live": live}


def purge(deploy_dir, contact_hash, make_brain=None) -> dict:
    """Remove evals derived from `contact_hash` → new version → checks → live.

    Checks still run (cheap stub tier); if evals drop below 10, `go_live`
    fails closed correctly and the pointer stays untouched.
    """
    deploy = Path(deploy_dir)
    base = _base_version(deploy)
    pruned, count = purge_contact(load_bundle(deploy / base), contact_hash)
    version = next_version(deploy)
    save_bundle(pruned, deploy / version)
    results = run_self_checks(pruned, make_brain)
    live = go_live(str(deploy), version, results)
    return {"version": version, "removed": count, "live": live}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch-learn: mine/approve/purge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mine = sub.add_parser("mine", help="mine proposals from outcomes + profiles")
    p_mine.add_argument("--deploy", required=True)
    p_mine.add_argument("--outcomes", required=True)
    p_mine.add_argument("--profiles-db", default=None)
    p_mine.add_argument("--keys", default="",
                        help="comma-separated contact keys to enumerate")

    p_approve = sub.add_parser("approve", help="approve proposals into a version")
    p_approve.add_argument("--deploy", required=True)
    group = p_approve.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", help="comma-separated proposal ids")
    group.add_argument("--all", action="store_true",
                       help="approve all proposed items")

    p_purge = sub.add_parser("purge", help="purge a contact's evals by hash")
    p_purge.add_argument("--deploy", required=True)
    p_purge.add_argument("--hash", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "mine":
        keys = [k for k in args.keys.split(",") if k] if args.keys else []
        print(mine(args.deploy, args.outcomes, args.profiles_db, keys=keys))
    elif args.cmd == "approve":
        import json
        out = approve(args.deploy,
                      ids=args.ids.split(",") if args.ids else None,
                      approve_all=args.all)
        print(json.dumps(out, indent=2))
    elif args.cmd == "purge":
        import json
        print(json.dumps(purge(args.deploy, args.hash), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
