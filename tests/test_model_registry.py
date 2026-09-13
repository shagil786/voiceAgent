# tests/test_model_registry.py — capacity + idle eviction + per-key load
# serialization, all with fake models and a fake clock (no real weights).
import threading
import time

import pytest

from voiceagent.model_registry import LoadedModelLRU


class FakeModel:
    loads = 0

    def __init__(self, key):
        self.key = key
        FakeModel.loads += 1


@pytest.fixture(autouse=True)
def _fresh_load_counter():
    # FakeModel.loads is a class-level counter; without a per-test reset the
    # absolute assertions in test_first_get/test_remove only hold in
    # isolation. (Deviation from the brief: minimal fix for shared state.)
    FakeModel.loads = 0
    yield
    FakeModel.loads = 0


def test_first_get_loads_second_get_hits():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    a = reg.get("qwen", lambda: FakeModel("qwen"))
    b = reg.get("qwen", lambda: FakeModel("qwen"))
    assert a is b and FakeModel.loads == 1
    assert reg.stats()["hits"] == 1 and reg.stats()["misses"] == 1


def test_capacity_evicts_oldest():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    reg.get("a", lambda: FakeModel("a"))
    reg.get("b", lambda: FakeModel("b"))
    reg.get("a", lambda: FakeModel("a"))   # touch a -> b is now oldest
    reg.get("c", lambda: FakeModel("c"))   # evicts b
    assert reg.stats()["evictions"] == 1
    assert reg.stats()["loaded_keys"] == ["a", "c"]


def test_evict_idle_drops_only_stale_entries():
    now = [1000.0]
    reg = LoadedModelLRU(capacity=2, idle_unload_s=10, clock=lambda: now[0])
    reg.get("a", lambda: FakeModel("a"))
    now[0] = 1005.0
    reg.get("b", lambda: FakeModel("b"))
    now[0] = 1012.0                        # a idle 12s > 10, b idle 7s
    assert reg.evict_idle() == 1
    assert reg.stats()["loaded_keys"] == ["b"]


def test_per_key_load_is_serialized_under_concurrency():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    gate = threading.Event()
    loads = []

    def slow_loader():
        loads.append(threading.current_thread().name)
        gate.wait(timeout=5)
        return FakeModel("qwen")

    results = []

    def worker():
        results.append(reg.get("qwen", slow_loader))

    t1 = threading.Thread(target=worker)
    t1.start()
    time.sleep(0.05)                       # let t1 enter the loader
    t2 = threading.Thread(target=worker)
    t2.start()
    time.sleep(0.05)
    gate.set()
    t1.join(); t2.join()
    assert results[0] is results[1]
    assert len(loads) == 1                 # second caller reused the load


def test_remove():
    reg = LoadedModelLRU(capacity=2, idle_unload_s=600)
    reg.get("a", lambda: FakeModel("a"))
    assert reg.remove("a") is True
    assert reg.remove("a") is False
    assert reg.get("a", lambda: FakeModel("a")) is not None
    assert FakeModel.loads == 2
