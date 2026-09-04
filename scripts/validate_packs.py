import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.packs import (
    CANONICAL_PATTERNS,
    detect_patterns,
    load_pack,
    load_vertical,
)

VERTICALS = ["luxury_automotive", "b2b_saas", "insurance", "real_estate", "cards"]


def _pack_checks():
    checks = []
    for name in CANONICAL_PATTERNS:
        checks.append(
            (f"pack {name} loads with required keys",
             lambda n=name: set(load_pack(n)) >= {"pattern", "tools", "policies", "disclosures", "probes"}),
        )
        checks.append(
            (f"pack {name} has 2+ probes",
             lambda n=name: len(load_pack(n).get("probes") or []) >= 2),
        )
        checks.append(
            (f"pack {name} has 1+ tools",
             lambda n=name: len(load_pack(n).get("tools") or []) >= 1),
        )
    return checks


def _vertical_checks():
    checks = []
    for domain_id in VERTICALS:
        checks.append(
            (f"vertical {domain_id} has catalog + disclosures",
             lambda d=domain_id: bool(load_vertical(d).catalog)
             and bool(load_vertical(d).statutory_disclosures)),
        )
    return checks


CHECKS = [
    *_pack_checks(),
    *_vertical_checks(),
    ("detect defaults to answer",
     lambda: detect_patterns({"offering": "", "top_asks": []}) == ["answer"]),
    ("detect finds qualify",
     lambda: "qualify" in detect_patterns({"offering": "sell flats", "top_asks": ["site visit slot?"]})),
]

if __name__ == "__main__":
    failed = 0
    for name, fn in CHECKS:
        try:
            ok = bool(fn())
        except Exception as e:  # noqa: BLE001 - validator reports, never crashes
            ok = False
            name = f"{name} ({e})"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)
