"""Outcome labels: owner/reported verdicts per session (batch-learn input)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

LABELS = ("resolved", "escalated", "thumbs_up", "thumbs_down")

@dataclass
class OutcomeLabel:
    session_id: str
    label: str  # one of LABELS
    ts: str
    note: str = ""
    contact_hash: str | None = None

class InMemoryOutcomes:
    def __init__(self):
        self._rows: list[OutcomeLabel] = []
    def record(self, label: OutcomeLabel) -> None:
        if label.label not in LABELS:
            raise ValueError(f"bad label {label.label!r}")
        self._rows.append(label)
    def query(self, label: str | None = None,
              session_id: str | None = None) -> list[OutcomeLabel]:
        out = self._rows
        if label is not None:
            out = [o for o in out if o.label == label]
        if session_id is not None:
            out = [o for o in out if o.session_id == session_id]
        return list(out)

class JsonlOutcomes(InMemoryOutcomes):
    def __init__(self, path: str):
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._rows.append(OutcomeLabel(**d))
    def record(self, label: OutcomeLabel) -> None:
        super().record(label)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(label)) + "\n")
