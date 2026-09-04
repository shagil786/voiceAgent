"""Secrets scan: fail on any hardcoded provider key in src/ + scripts/.

Escape hatch: append `# noqa: secrets` to a line to exclude it from the
scan (split on the marker before matching). Unused in v1 — allowlist none.
"""
def test_no_hardcoded_secrets():
    import re
    from pathlib import Path
    pat = re.compile(r"sk-[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16}|xox[bpas]-")
    hits = [f"{p}:{i}" for p in (list(Path('src').rglob('*.py')) + list(Path('scripts').rglob('*.py')))
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if pat.search(line.split("# noqa: secrets")[0])]
    assert hits == []
