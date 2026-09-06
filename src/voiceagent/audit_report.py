"""Shadow policy report: mine persisted call outcomes and PROPOSE policy
adjustments — as a markdown report for human review. NOTHING here mutates
agent config (ADR-001: learning proposes, humans dispose).

Sources: the persistent DecisionLog (VOICEAGENT_AUDIT_DB) and the intent
memory store (VOICEAGENT_MEMORY_DB). Data threshold: below MIN_CALLS the
report says so — patterns from a handful of calls are noise, and a noisy
suggestion is worse than none.

CLI:  .venv/bin/python -m voiceagent.audit_report \
        --audit-db data/out/audit.sqlite --memory-db data/out/memory.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field

MIN_CALLS = 25          # below this: honest "insufficient data" report
LOW_RATING = 6.0        # avg rating at/below this flags satisfaction
HIGH_ESCALATION_RATE = 0.5   # deny-then-escalate ratio that triggers a review


@dataclass
class Report:
    markdown: str
    calls: int = 0
    sufficient: bool = False
    suggestions: list[str] = field(default_factory=list)


def _rows(db_path: str, sql: str, args: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def _decision_rows(audit_db: str, days: int) -> list[dict]:
    rows = _rows(
        audit_db,
        "SELECT ts, conv_id, action, verdict, reasons, amount FROM decision_log"
        " WHERE ts >= datetime('now', ?) ORDER BY id",
        (f"-{days} days",))
    out = []
    for ts, conv_id, action, verdict, reasons, amount in rows:
        try:
            reason_list = json.loads(reasons or "[]")
        except (ValueError, TypeError):
            reason_list = []
        out.append({"ts": ts, "conv_id": conv_id, "action": action,
                    "verdict": verdict, "reasons": reason_list,
                    "amount": amount})
    return out


def build_report(audit_db: str | None, memory_db: str | None,
                 days: int = 30) -> Report:
    """Deterministic shadow report over persisted outcomes. Both DBs are
    optional; missing sources render as empty sections, never errors."""
    lines: list[str] = ["# Shadow Policy Report", ""]
    suggestions: list[str] = []

    decisions = _decision_rows(audit_db, days) if audit_db else []
    calls = len({d["conv_id"] for d in decisions if d["conv_id"]})

    # memory side
    episodes: list[tuple] = []
    ratings: list[tuple] = []
    if memory_db:
        try:
            episodes = _rows(memory_db,
                             "SELECT label, confidence, outcome FROM episodes")
            ratings = _rows(memory_db,
                            "SELECT rating, comment FROM ratings")
        except sqlite3.Error:
            pass

    sufficient = calls >= MIN_CALLS
    lines.append(f"- governed calls analyzed: **{calls}**")
    lines.append(f"- governed actions: **{len(decisions)}**")
    lines.append(f"- captured learning episodes: **{len(episodes)}**")
    lines.append(f"- caller ratings: **{len(ratings)}**")
    lines.append("")
    if not sufficient:
        lines.append(f"> **Insufficient data**: {MIN_CALLS}+ governed calls are"
                     " needed before patterns are signal instead of noise."
                     " Keep calling; rerun after.")
        return Report(markdown="\n".join(lines), calls=calls,
                      sufficient=False, suggestions=[])

    # -- action verdict breakdown -------------------------------------------
    lines.append("## Action outcomes")
    lines.append("")
    lines.append("| action | ALLOW | DENY | ESCALATE | errors |")
    lines.append("|---|---|---|---|---|")
    by_action: dict[str, Counter] = defaultdict(Counter)
    for d in decisions:
        by_action[d["action"]][d["verdict"]] += 1
        if not d["verdict"] == "ALLOW" or "error" in str(d["reasons"]).lower():
            pass
    errored = Counter()
    for d in decisions:
        if any("error" in str(r).lower() or "timeout" in str(r).lower()
               for r in d["reasons"]):
            errored[d["action"]] += 1
    for action in sorted(by_action):
        c = by_action[action]
        lines.append(f"| {action} | {c['ALLOW']} | {c['DENY']} | "
                     f"{c['ESCALATE']} | {errored[action]} |")
    lines.append("")

    # -- deny -> escalation pattern (the core suggestion heuristic) ----------
    lines.append("## Deny-then-escalate patterns")
    lines.append("")
    deny_convs: dict[str, set] = defaultdict(set)
    for d in decisions:
        if d["verdict"] == "DENY":
            deny_convs[d["action"]].add(d["conv_id"])
    escalations = {d["conv_id"] for d in decisions
                   if d["action"] == "escalate_to_human"
                   and d["verdict"] == "ALLOW"}
    for action in sorted(deny_convs):
        convs = deny_convs[action]
        escalated = convs & escalations
        rate = len(escalated) / len(convs) if convs else 0.0
        lines.append(f"- `{action}`: {len(convs)} denied calls, "
                     f"{len(escalated)} escalated afterwards ({rate:.0%})")
        if len(convs) >= 5 and rate >= HIGH_ESCALATION_RATE:
            suggestions.append(
                f"`{action}`: {rate:.0%} of denied calls escalated to a human "
                f"afterwards. Review the policy rule for `{action}` — the "
                f"declared restriction may be tighter than the operation "
                f"actually requires. Applies as an explicit policies.yaml "
                f"diff after your approval.")
    lines.append("")

    # -- feedback -------------------------------------------------------------
    lines.append("## Caller feedback")
    lines.append("")
    if ratings:
        vals = [float(r[0]) for r in ratings]
        avg = sum(vals) / len(vals)
        lines.append(f"- {len(vals)} ratings, average **{avg:.1f}/10**, "
                     f"min {min(vals):.0f}, max {max(vals):.0f}")
        comments = [str(r[1]) for r in ratings if r[1]]
        for c in comments[:5]:
            lines.append(f"- comment: {c!r}")
        if avg <= LOW_RATING:
            suggestions.append(
                f"Average satisfaction is {avg:.1f}/10 (threshold "
                f"{LOW_RATING}). Read the low-rating call transcripts in the "
                f"audit DB before changing any policy — find what the low "
                f"ratings share.")
    else:
        lines.append("- no ratings captured yet")
    lines.append("")

    # -- understanding quality -------------------------------------------------
    if episodes:
        confs = [float(e[1]) for e in episodes]
        unknown = Counter(e[0] or "(unknown)" for e in episodes
                          if float(e[1]) < 0.35 or not e[0])
        lines.append("## Understanding quality (learning candidates)")
        lines.append("")
        lines.append(f"- {len(episodes)} captured episodes, "
                     f"avg confidence {sum(confs) / len(confs):.2f}")
        top_unknown = unknown.most_common(5)
        if top_unknown:
            lines.append("- most-captured unclear labels: "
                         + ", ".join(f"{lbl} ({n})" for lbl, n in top_unknown))
        lines.append("")

    # -- suggestions ------------------------------------------------------------
    lines.append("## Suggestions (PROPOSALS — nothing applies without your "
                 "approval)")
    lines.append("")
    if suggestions:
        for s in suggestions:
            lines.append(f"- {s}")
    else:
        lines.append("- no policy-affecting patterns crossed a threshold this "
                     "period")
    lines.append("")

    return Report(markdown="\n".join(lines), calls=calls, sufficient=True,
                  suggestions=suggestions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-db", default=None)
    ap.add_argument("--memory-db", default=None)
    ap.add_argument("--out", default=None, help="write markdown here "
                    "(default: stdout)")
    args = ap.parse_args()
    report = build_report(args.audit_db, args.memory_db)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report.markdown)
        print(f"report written: {args.out}")
    else:
        print(report.markdown)


if __name__ == "__main__":
    main()
