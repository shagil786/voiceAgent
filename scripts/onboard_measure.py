#!/usr/bin/env python3
"""Onboarding drill: measurement only (no go_live, pointer untouched).

Drill: ingest (cap) -> compile -> save tmp v1 -> run_self_checks (stub tier).
Timings per phase via time.monotonic. `go_live` is imported for reuse but
deliberately never called here; the live pointer is left untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from voiceagent.deploy.bundle import save_bundle  # noqa: E402
from voiceagent.deploy.compiler import compile_bundle  # noqa: E402
from voiceagent.deploy.ingest import (  # noqa: E402
    fetch_site,
    ingest_owner_paste,
    rank_chunks,
)
from voiceagent.deploy.selfcheck import go_live, run_self_checks  # noqa: E402,F401
from voiceagent.learn.instant import next_version  # noqa: E402


def measure_onboard(seed_url_or_paste: str, interview: dict, fetcher=None) -> dict:
    """Run the onboarding drill and return phase metrics.

    Returns {pages, chunks, compile_ms, checks_ms, total_ms, evals, tools}.
    Uses a stub-tier `run_self_checks` default; never calls `go_live`.
    """
    t0 = time.monotonic()
    if seed_url_or_paste.startswith("http"):
        crawled = fetch_site(seed_url_or_paste, fetcher=fetcher)
        pasted: list[dict] = []
    else:
        crawled = []
        pasted = [ingest_owner_paste(seed_url_or_paste)]
    ranked = rank_chunks(pasted, crawled)
    pages = len([c for c in crawled if not c["source"].startswith("gap:")])
    if seed_url_or_paste.startswith("http"):
        chunks_n = len(ranked)
    else:
        pages = len(ranked)
        chunks_n = len(ranked)

    t1 = time.monotonic()
    bundle = compile_bundle("onboard-drill", ranked, interview)
    t2 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="onboard-drill-") as tmp:
        save_bundle(bundle, Path(tmp) / next_version(tmp))
        t3 = time.monotonic()
        run_self_checks(bundle)
        t4 = time.monotonic()

    compile_ms = (t2 - t1) * 1000
    checks_ms = (t4 - t3) * 1000
    total_ms = (t4 - t0) * 1000
    return {
        "pages": pages,
        "chunks": chunks_n,
        "compile_ms": compile_ms,
        "checks_ms": checks_ms,
        "total_ms": total_ms,
        "evals": len(bundle.evals),
        "tools": len(bundle.tools),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure onboarding drill (no go-live).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Seed URL to crawl (capped).")
    src.add_argument("--paste-file", help="File with owner-pasted text.")
    ap.add_argument("interview_json", help="JSON file with interview dict.")
    args = ap.parse_args(argv)

    interview = json.loads(Path(args.interview_json).read_text(encoding="utf-8"))
    if args.url:
        seed = args.url
    else:
        seed = Path(args.paste_file).read_text(encoding="utf-8")
    out = measure_onboard(seed, interview)
    print(f"{'phase':<10}{'value':>12}")
    for k in ("pages", "chunks", "compile_ms", "checks_ms", "total_ms", "evals", "tools"):
        v = out[k]
        print(f"{k:<10}{v:>12.1f}" if isinstance(v, float) else f"{k:<10}{v:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
