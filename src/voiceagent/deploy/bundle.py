"""Frozen bundle schema v1. Loader rejects unknown versions, never guesses."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA_VERSION = 1
TOOL_STATES = ("PROPOSED", "APPROVED", "CONNECTED")
LIVE_POINTER = "live"

@dataclass
class ToolEntry:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    state: str = "PROPOSED"
    connection_ref: str | None = None
    policy_action: str = ""
    scopes: list[str] = field(default_factory=list)

@dataclass
class EvalCheck:
    name: str
    turns: list[dict] = field(default_factory=list)  # [{user: str}]
    assert_: dict = field(default_factory=dict)      # {contains?, action?, verdict?}

@dataclass
class Bundle:
    schema_version: int
    deploy_id: str
    spec: dict = field(default_factory=dict)
    tools: list[ToolEntry] = field(default_factory=list)
    policies: dict = field(default_factory=dict)
    knowledge: list[dict] = field(default_factory=list)
    evals: list[EvalCheck] = field(default_factory=list)

def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def _load_policies_yaml(p: Path) -> dict:
    """Mirror src/voiceagent/policy.py's YAML loading pattern (yaml.safe_load,
    no new dependency), but strict: a missing file raises FileNotFoundError
    and a non-mapping result raises ValueError — never synthesize defaults."""
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if isinstance(loaded, dict) and loaded:
        return loaded
    raise ValueError(f"policy file {p} did not load as a non-empty mapping")

def _dump_policies_yaml(policies: dict, p: Path) -> None:
    """Write policies using the same YAML approach as _load_policies_yaml
    (yaml.safe_dump, no new dependency)."""
    import yaml
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(policies, f, default_flow_style=False, sort_keys=False)

def load_bundle(path: str | Path) -> Bundle:
    d = Path(path)
    meta = _read_json(d / "bundle.json")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema_version {meta.get('schema_version')!r}; "
            f"this code reads {SCHEMA_VERSION}")
    tools = [ToolEntry(**t) for t in _read_json(d / "tools.json")]
    for t in tools:
        if t.state not in TOOL_STATES:
            raise ValueError(f"tool {t.name!r} has bad state {t.state!r}")
    evals = [EvalCheck(name=e["name"], turns=e.get("turns", []),
                       assert_=e.get("assert", {}))
             for e in _read_json(d / "evals.json")]
    pol_path = d / "policies.yaml"
    policies = _load_policies_yaml(pol_path)
    kdir = d / "knowledge"
    knowledge = [_read_json(p) for p in sorted(kdir.glob("*.json"))] if kdir.exists() else []
    return Bundle(schema_version=1, deploy_id=meta.get("deploy_id", d.parent.name),
                  spec=meta.get("spec", {}), tools=tools,
                  policies=policies, knowledge=knowledge, evals=evals)

def save_bundle(bundle: Bundle, path: str | Path) -> None:
    d = Path(path); d.mkdir(parents=True, exist_ok=True)
    (d / "bundle.json").write_text(json.dumps(
        {"schema_version": SCHEMA_VERSION, "deploy_id": bundle.deploy_id,
         "spec": bundle.spec}, indent=2, sort_keys=True), encoding="utf-8")
    (d / "tools.json").write_text(json.dumps(
        [asdict(t) for t in bundle.tools], indent=2, sort_keys=True), encoding="utf-8")
    (d / "evals.json").write_text(json.dumps(
        [{"name": e.name, "turns": e.turns, "assert": e.assert_}
         for e in bundle.evals], indent=2, sort_keys=True), encoding="utf-8")
    _dump_policies_yaml(bundle.policies, d / "policies.yaml")
    kdir = d / "knowledge"; kdir.mkdir(exist_ok=True)
    for i, ch in enumerate(bundle.knowledge):
        (kdir / f"{i:03d}.json").write_text(
            json.dumps(ch, indent=2, sort_keys=True), encoding="utf-8")

def diff_bundles(old: Bundle, new: Bundle) -> list[dict]:
    out: list[dict] = []
    if old.spec != new.spec:
        out.append({"section": "spec", "kind": "changed",
                    "detail": sorted(set(new.spec) | set(old.spec))})
    old_t, new_t = {t.name: t for t in old.tools}, {t.name: t for t in new.tools}
    for n in new_t.keys() - old_t.keys():
        out.append({"section": "tools", "kind": "added", "detail": n})
    for n in old_t.keys() - new_t.keys():
        out.append({"section": "tools", "kind": "removed", "detail": n})
    for n in old_t.keys() & new_t.keys():
        if asdict(old_t[n]) != asdict(new_t[n]):
            out.append({"section": "tools", "kind": "changed", "detail": n})
    if old.policies != new.policies:
        out.append({"section": "policies", "kind": "changed",
                    "detail": sorted(set(new.policies) | set(old.policies))})
    if old.knowledge != new.knowledge:
        out.append({"section": "knowledge", "kind": "changed",
                    "detail": f"{len(old.knowledge)}->{len(new.knowledge)} chunks"})
    return out

def read_live(deploy_dir: str | Path) -> str | None:
    p = Path(deploy_dir) / LIVE_POINTER
    return p.read_text(encoding="utf-8").strip() if p.exists() else None

def write_live(deploy_dir: str | Path, version: str) -> None:
    Path(deploy_dir, LIVE_POINTER).write_text(version + "\n", encoding="utf-8")
