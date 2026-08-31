# src/voiceagent/decisionlog.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DecisionEntry:
    ts: str
    conv_id: str
    action: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    amount: float | None = None
    authenticated: bool = False


class DecisionLog:
    """Append-only audit trail of every policy decision the agent made."""

    def __init__(self) -> None:
        self._entries: list[DecisionEntry] = []

    def record(self, entry: DecisionEntry) -> None:
        self._entries.append(entry)

    def entries(self) -> list[DecisionEntry]:
        return list(self._entries)

    def query(self, action: str | None = None, verdict: str | None = None) -> list[DecisionEntry]:
        out = self._entries
        if action is not None:
            out = [e for e in out if e.action == action]
        if verdict is not None:
            out = [e for e in out if e.verdict == verdict]
        return list(out)

    def to_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps([asdict(e) for e in self._entries], indent=2),
            encoding="utf-8",
        )

    def to_csv(self, path: str) -> None:
        if not self._entries:
            Path(path).write_text("", encoding="utf-8")
            return
        fields = ["ts", "conv_id", "action", "verdict", "amount", "authenticated", "reasons"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in self._entries:
                row = asdict(e)
                row["reasons"] = "|".join(e.reasons)
                writer.writerow(row)