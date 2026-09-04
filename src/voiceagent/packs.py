"""Operator packs as data: strict YAML loader for patterns + verticals.

Packs are data only — adding a vertical or pattern never edits Python.
Unknown fields are rejected (strict load); unknown pattern names fall back
to ``answer`` via :func:`detect_patterns`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from voiceagent.swarm.specialist import SpecialistSpec, SpecialistTool


PACK_DIR = Path(__file__).resolve().parents[2] / "data" / "packs"
VERTICAL_DIR = PACK_DIR / "verticals"

PACK_FIELDS = frozenset({"pattern", "tools", "policies", "disclosures", "probes"})
VERTICAL_FIELDS = frozenset({
    "domain_id", "name", "role_description", "system_prompt",
    "catalog", "tools", "statutory_disclosures",
})

CANONICAL_PATTERNS = ["answer", "resolve", "qualify", "follow_up", "draft_action"]

_PATTERN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "qualify": ("visit", "book", "demo", "trial", "slot"),
    "follow_up": ("remind", "nudge", "callback", "pending", "abandoned"),
    "draft_action": ("offer", "deal", "order", "refund", "cancel", "return"),
    "resolve": ("support", "help", "issue", "broken", "complaint", "status"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file (lazy import mirrors ``policy.load_policies``)."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"pack file {path} must contain a YAML mapping")
    return loaded


def _validate(d: dict[str, Any], allowed: frozenset[str] | set[str], what: str) -> None:
    """Strict load: reject any field outside the allowed set."""
    unknown = set(d) - set(allowed)
    if unknown:
        raise ValueError(f"unknown field {sorted(unknown)} in {what}")


def load_pack(name: str) -> dict[str, Any]:
    """Load an operator pattern pack by name.

    Raises:
        FileNotFoundError: on unknown pack names.
        ValueError: on unknown fields (strict load).
    """
    path = PACK_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown pack: {name}")
    data = _read_yaml(path)
    _validate(data, PACK_FIELDS, f"pack {name}")
    return data


def _find_vertical_file(domain_id: str) -> Path:
    """Resolve a domain_id to its YAML file: direct ``<id>.yaml`` first,
    then scan all vertical files for a matching ``domain_id`` key."""
    direct = VERTICAL_DIR / f"{domain_id}.yaml"
    if direct.is_file():
        return direct
    for path in sorted(VERTICAL_DIR.glob("*.yaml")):
        try:
            if _read_yaml(path).get("domain_id") == domain_id:
                return path
        except (ValueError, OSError):
            continue
    raise FileNotFoundError(f"unknown vertical: {domain_id}")


def load_vertical(domain_id: str) -> SpecialistSpec:
    """Load a vertical pack and build its :class:`SpecialistSpec`.

    Raises:
        FileNotFoundError: on unknown domain ids.
        ValueError: on unknown fields (strict load).
    """
    data = _read_yaml(_find_vertical_file(domain_id))
    _validate(data, VERTICAL_FIELDS, f"vertical {domain_id}")
    tools = [
        SpecialistTool(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("parameters", {}),
        )
        for t in (data.get("tools") or [])
    ]
    return SpecialistSpec(
        domain_id=data["domain_id"],
        name=data["name"],
        role_description=data.get("role_description", ""),
        system_prompt=data.get("system_prompt", ""),
        catalog=data.get("catalog", []) or [],
        tools=tools,
        statutory_disclosures=data.get("statutory_disclosures", []) or [],
    )


def detect_patterns(interview: dict[str, Any]) -> list[str]:
    """Keyword scan over offering + top_asks → subset of the 5 pattern names.

    ``answer`` is always included first; empty interviews return ``["answer"]``.
    """
    offering = str(interview.get("offering") or "")
    top_asks = interview.get("top_asks") or []
    text = " ".join([offering] + [str(a) for a in top_asks]).lower()
    found = ["answer"]
    for pattern in CANONICAL_PATTERNS[1:]:
        if any(kw in text for kw in _PATTERN_KEYWORDS[pattern]):
            found.append(pattern)
    return found
