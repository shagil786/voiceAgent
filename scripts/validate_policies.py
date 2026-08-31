import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.policy import PolicyEngine, PolicyContext, load_policies

CHECKS = [
    ("fraud escalates", lambda e: e.evaluate("fraud").verdict == "ESCALATE"),
    ("high_value_refund escalates when authed",
     lambda e: e.evaluate("high_value_refund", PolicyContext(authenticated=True)).verdict == "ESCALATE"),
    ("refund under limit + authed allows",
     lambda e: e.evaluate("refund", PolicyContext(amount=1000, authenticated=True)).verdict == "ALLOW"),
    ("refund over limit requires human",
     lambda e: e.evaluate("refund", PolicyContext(amount=20000, authenticated=True)).verdict == "REQUIRE_HUMAN_APPROVAL"),
    ("refund without auth requires auth",
     lambda e: e.evaluate("refund", PolicyContext(amount=1000, authenticated=False)).verdict == "REQUIRE_AUTH"),
    ("order_status allows",
     lambda e: e.evaluate("order_status").verdict == "ALLOW"),
    ("unknown action denied (least privilege)",
     lambda e: e.evaluate("not_a_real_action").verdict == "DENY"),
]

if __name__ == "__main__":
    policies = load_policies("data/policies/policies.yaml")
    eng = PolicyEngine(policies)
    failed = 0
    for name, fn in CHECKS:
        ok = fn(eng)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)
