#!/usr/bin/env python3
"""scripts/forget_caller.py — retention + right-to-be-forgotten CLI.

Usage:
    .venv/bin/python scripts/forget_caller.py --erase-session CALL-ROOM-1
    .venv/bin/python scripts/forget_caller.py --purge [--days 90]

--erase-session deletes one conversation's audit entries, memory episodes
and ratings (per-store counts printed). --purge deletes rows older than
VOICEAGENT_DATA_RETENTION_DAYS (or --days). DB paths come from .env
(VOICEAGENT_AUDIT_DB / VOICEAGENT_MEMORY_DB); nothing deletes by default.
The org's ERP/CRM is the system of record — erasure there is the org's own
procedure (see docs/DATA_FLOWS.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--erase-session", metavar="CONV_ID",
                       help="erase one conversation everywhere")
    group.add_argument("--purge", action="store_true",
                       help="delete rows older than the retention window")
    parser.add_argument("--days", type=int, default=None,
                        help="retention window override for --purge")
    parser.add_argument("--tenant", default=None,
                        help="tenant scope for --erase-session memory rows")
    parser.add_argument("--audit-db", default=None,
                        help="audit DB override (default: VOICEAGENT_AUDIT_DB)")
    parser.add_argument("--memory-db", default=None,
                        help="memory DB override (default: VOICEAGENT_MEMORY_DB)")
    parser.add_argument("--chat-db", default=None,
                        help="chat transcript DB override "
                             "(default: VOICEAGENT_CHAT_MEMORY_DB)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from voiceagent import retention as ret

    if args.erase_session:
        out = ret.erase_session(args.erase_session, tenant=args.tenant,
                                audit_db=args.audit_db,
                                memory_db=args.memory_db,
                                chat_db=args.chat_db)
        print(json.dumps({"erased": args.erase_session, "removed": out},
                         indent=2))
        return 0
    days = args.days
    if days is None:
        days = ret.retention_days()
    if days is None:
        print("ERROR: no retention window — set VOICEAGENT_DATA_RETENTION_DAYS"
              " in .env or pass --days N (nothing deletes by default).",
              file=sys.stderr)
        return 2
    out = ret.purge_expired(days=days, audit_db=args.audit_db,
                            memory_db=args.memory_db,
                            chat_db=args.chat_db)
    print(json.dumps({"retention_days": days, "removed": out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
