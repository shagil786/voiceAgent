# src/voiceagent/metrics.py — dependency-free runtime counters.
"""In-process per-turn metrics: latency + policy-verdict counts.

Design: always-on counters, no I/O, ~0 overhead when unread. The
orchestrator owns one `Metrics` and records a single sample per
`handle_turn`; anything that needs persistence scrapes `snapshot()`.
Stdlib only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Ring cap: the latency list never grows past this (oldest dropped).
MAX_SAMPLES = 10_000


@dataclass
class Metrics:
    """Turn counters: count, latency samples (ms), verdict histogram."""
    turns: int = 0
    lat_ms: list[int] = field(default_factory=list)
    verdicts: Counter = field(default_factory=Counter)

    def record(self, latency_s: float, verdict: str) -> None:
        """Record one turn: latency in seconds + primary policy verdict."""
        self.turns += 1
        self.lat_ms.append(int(round(latency_s * 1000)))
        del self.lat_ms[:-MAX_SAMPLES]  # ring-drop oldest past the cap
        self.verdicts[verdict] += 1

    def snapshot(self) -> dict:
        """Plain-data shape: {turns, avg_latency_ms, verdicts: {v: n}}."""
        avg = round(sum(self.lat_ms) / len(self.lat_ms)) if self.lat_ms else 0
        return {"turns": self.turns, "avg_latency_ms": avg,
                "verdicts": dict(self.verdicts)}
