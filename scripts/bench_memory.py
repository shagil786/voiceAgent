"""Micro-benchmark (M4a): working-memory overhead vs a ~440ms/turn LLM call.
Times n appends and n history(last_n=4) reads per store and prints ops/sec.
Usage: python scripts/bench_memory.py [n]   (default n=1000)"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.memory import InMemoryMemory, SQLiteMemory, Turn

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
TURN = Turn(ts="2026-09-02T00:00:00", role="user",
            text="where is my order ORD-1234? it has not arrived yet")


def bench(store) -> tuple[float, float]:
    """Returns (appends/s, history reads/s) for n operations each."""
    t0 = time.perf_counter()
    for _ in range(N):
        store.append("bench", TURN)
    append_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(N):
        store.history("bench", last_n=4)
    read_s = time.perf_counter() - t0
    return N / append_s, N / read_s


def main():
    with tempfile.TemporaryDirectory() as d:
        results = [
            ("InMemoryMemory", bench(InMemoryMemory())),
            ("SQLiteMemory", bench(SQLiteMemory(str(Path(d) / "bench.db")))),
        ]
    print(f"n={N} appends + n={N} history(last_n=4) reads per store")
    print(f"{'store':<16}{'appends/s':>12}{'reads/s':>12}")
    for name, (appends_s, reads_s) in results:
        print(f"{name:<16}{appends_s:>12.0f}{reads_s:>12.0f}")


if __name__ == "__main__":
    main()
