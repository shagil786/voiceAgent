# scripts/validate_tenant.py — CI gate for a tenant bundle.
"""Validates a tenant bundle (data/tenants/<name>/) before deploy:

- tenant.json parses; persona fields have the right types
- intents/*.yaml files are lists of non-empty strings; filenames are valid
  intent names (lowercase snake_case)
- policies.yaml (if present) passes the same structural checks as the
  platform policies (scripts/validate_policies.py logic)

The tenant bundle is the Control Plane's per-customer artifact: it is
validated in CI, versioned in git, and deployed atomically. A bundle that
fails validation must never reach an agent.

Usage: .venv/bin/python scripts/validate_tenant.py [data/tenants/example-acme]
"""
import re
import sys
from pathlib import Path

INTENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def validate(root: Path) -> list[str]:
    errors: list[str] = []

    cfg = root / "tenant.json"
    if not cfg.exists():
        return [f"missing {cfg}"]
    import json
    try:
        data = json.loads(cfg.read_text())
    except json.JSONDecodeError as e:
        return [f"{cfg}: invalid JSON: {e}"]
    persona = data.get("persona", {})
    if isinstance(persona, dict):
        for key in ("may_promise", "never_say", "languages"):
            val = persona.get(key)
            if val is not None and not (isinstance(val, list)
                                        and all(isinstance(x, str) for x in val)):
                errors.append(f"{cfg}: persona.{key} must be a list of strings")
        if isinstance(persona.get("languages"), list) and not persona["languages"]:
            errors.append(f"{cfg}: persona.languages is empty")

    intents_dir = root / "intents"
    if intents_dir.is_dir():
        import yaml
        for f in sorted(intents_dir.glob("*.yaml")):
            if not INTENT_NAME_RE.match(f.stem):
                errors.append(f"{f}: filename must be a lowercase intent name")
                continue
            try:
                items = yaml.safe_load(f.read_text())
            except Exception as e:  # noqa: BLE001 — report, don't crash CI
                errors.append(f"{f}: invalid YAML: {e}")
                continue
            if not isinstance(items, list) or not items:
                errors.append(f"{f}: must be a non-empty YAML list")
                continue
            bad = [x for x in items if not isinstance(x, str) or not x.strip()]
            if bad:
                errors.append(f"{f}: {len(bad)} non-string/empty exemplars")

    pol = root / "policies.yaml"
    if pol.exists():
        import yaml
        try:
            pdata = yaml.safe_load(pol.read_text())
        except Exception as e:  # noqa: BLE001
            errors.append(f"{pol}: invalid YAML: {e}")
        else:
            if not isinstance(pdata, dict):
                errors.append(f"{pol}: must be a YAML mapping")
            else:
                for action, rule in pdata.items():
                    if not INTENT_NAME_RE.match(str(action)):
                        errors.append(f"{pol}: invalid action key '{action}'")
                    if isinstance(rule, dict):
                        when = rule.get("escalate_when")
                        if when is not None and not isinstance(when, dict):
                            errors.append(
                                f"{pol}: escalate_when for '{action}' must be "
                                f"a mapping of signal -> value")
    return errors


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [
        p for p in Path("data/tenants").glob("*") if p.is_dir()]
    if not roots:
        print("no tenant bundles found — nothing to validate")
        return 0
    failed = False
    for root in roots:
        errors = validate(root)
        status = "PASS" if not errors else "FAIL"
        print(f"[{status}] {root}")
        for e in errors:
            print(f"    - {e}")
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
