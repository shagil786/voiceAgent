"""Instant owner-correction patches (§4.4): owner quote → versioned bundle + go-live.

Owner corrections must not contain customer PII — bundle versions are permanent and outside TTL/delete scope.

Self-checks run the stub tier by default (`make_brain=None` → selfcheck
default brain); any live-brain variance is Plan 4 scope.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

from voiceagent.deploy.bundle import Bundle, load_bundle, read_live, save_bundle
from voiceagent.deploy.selfcheck import go_live, run_self_checks
from voiceagent.learn.corrections import Correction, classify_correction
from voiceagent.memory import now_ts

# Safety invariant (spec §4.4 "touching CONNECTED scopes stays proposed"
# needs no extra code): instant patches only ever edit `spec`/`knowledge` —
# they never modify `tools.json` or policy DSL entries, so CONNECTED tool
# scopes are unreachable by construction.
_DSL_HINTS = ("threshold", "amount", "₹", "rs.", "percent", "%",
              "above", "under", "over ")


def apply_owner_correction(bundle: Bundle, correction: Correction,
                           context: str = "") -> tuple[Bundle, dict]:
    """Copy-on-write patch of one owner correction; returns (new, changelog)."""
    if correction.scope != "global":
        raise ValueError("instant patch requires owner scope")
    ts = now_ts()
    new = copy.deepcopy(bundle)
    quote = correction.quote
    needs_dsl_review = False
    if correction.patch_type == "tone":
        new.spec.setdefault("tone_notes", []).append(quote)
    elif correction.patch_type == "fact":
        new.knowledge.append({"text": quote, "source": f"owner_correction:{ts}",
                              "crawled_at": ts})
    elif correction.patch_type == "policy":
        new.spec.setdefault("never_promise", []).append(quote)
        needs_dsl_review = any(h in quote.lower() for h in _DSL_HINTS)
    elif correction.patch_type == "exemplar":
        new.knowledge.append({"text": "Exemplar guidance: " + quote,
                              "source": f"owner_exemplar:{ts}",
                              "crawled_at": ts})
    log = {"ts": ts, "actor": "owner", "quote": quote,
           "patch_type": correction.patch_type, "context": context,
           "needs_dsl_review": needs_dsl_review}
    return new, log


def _version_numbers(deploy_dir: Path) -> list[int]:
    if not deploy_dir.exists():
        return []
    out = []
    for p in deploy_dir.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
            out.append(int(p.name[1:]))
    return out


def next_version(deploy_dir: str | Path) -> str:
    """Next `vN` name: `v{max+1}` over `v<int>` dirs, `v1` when none."""
    nums = _version_numbers(Path(deploy_dir))
    return f"v{max(nums) + 1}" if nums else "v1"


def instant_correct(deploy_dir: str | Path, quote: str, context: str = "",
                    make_brain: Callable[[], Any] | None = None,
                    actor: str = "owner") -> dict:
    """Owner quote → classify → patch → stub-tier checks → save → go_live.

    Base version is the live pointer when set, else the highest `v*` dir.
    The new version is always saved (proposed, auditable); the live pointer
    flips only when `go_live` approves (≥10 all-pass). Returns `{version,
    passed, checks, changelog, live, reason}` (`reason` is None on success).
    """
    if actor != "owner":
        raise ValueError("instant_correct is owner-only")
    correction = classify_correction(quote, "", is_owner=True)
    if not correction.is_correction:
        changelog = {"actor": actor, "quote": correction.quote,
                     "patch_type": "none", "context": context,
                     "version": None, "passed": False, "live": False}
        return {"version": None, "passed": False, "checks": [],
                "changelog": changelog, "live": False,
                "reason": "not a correction"}
    d = Path(deploy_dir)
    live = read_live(d)
    if live is not None:
        base = live
    else:
        nums = _version_numbers(d)
        base = f"v{max(nums)}" if nums else "v1"
    new, log = apply_owner_correction(load_bundle(d / base), correction,
                                      context)
    log["actor"] = actor
    results = run_self_checks(new, make_brain)
    passed = bool(results) and all(r["passed"] for r in results)
    version = next_version(d)
    save_bundle(new, d / version)
    went_live = go_live(str(d), version, results)
    reason = None if went_live else "self-checks did not approve go-live"
    log.update({"version": version, "passed": passed, "live": went_live,
                "needs_dsl_review": log["needs_dsl_review"]})
    return {"version": version, "passed": passed, "checks": results,
            "changelog": log, "live": went_live, "reason": reason}
