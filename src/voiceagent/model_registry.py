"""Capacity-capped, idle-unloading registry of loaded models — the lazy
home for capability services (spec 2026-09-14-service-split-design.md).

Eviction drops the handle reference; weights free via GC and re-load hits
the local HF cache (no re-download). Thread-safe: a short lock guards the
map, per-key locks serialize concurrent first-loads so a cold key never
loads twice and unrelated keys never block each other.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable


class LoadedModelLRU:
    def __init__(self, capacity: int, idle_unload_s: float,
                 clock: Callable[[], float] = time.monotonic):
        self._capacity = max(1, int(capacity))
        self._idle_s = float(idle_unload_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._loading: dict[str, threading.Lock] = {}
        self._entries: OrderedDict[str, tuple[object, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, loader: Callable[[], object]) -> object:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                obj, _ = entry
                self._entries.move_to_end(key)
                self._entries[key] = (obj, self._clock())
                self._hits += 1
                return obj
            self._misses += 1
            load_lock = self._loading.setdefault(key, threading.Lock())
        with load_lock:
            with self._lock:  # re-check: another thread may have loaded
                entry = self._entries.get(key)
                if entry is not None:
                    obj, _ = entry
                    self._entries.move_to_end(key)
                    self._entries[key] = (obj, self._clock())
                    return obj
                obj = loader()
                self._evict_for_capacity(protected=key)
                self._entries[key] = (obj, self._clock())
                return obj

    def _evict_for_capacity(self, protected: str) -> None:
        while len(self._entries) >= self._capacity:
            oldest = next(iter(self._entries))
            if oldest == protected and len(self._entries) == 1:
                break
            del self._entries[oldest]
            self._evictions += 1

    def evict_idle(self) -> int:
        with self._lock:
            now = self._clock()
            stale = [k for k, (_, last) in self._entries.items()
                     if now - last > self._idle_s]
            for k in stale:
                del self._entries[k]
            return len(stale)

    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses,
                    "evictions": self._evictions,
                    "capacity": self._capacity,
                    "loaded_keys": list(self._entries)}
