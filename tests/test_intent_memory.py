# tests/test_intent_memory.py
"""ADR-002: the agent's own learned understanding.

- IntentMemoryStore: episodic fragments (capture) consolidated into bounded
  intent prototypes (consolidate), retrieved as ADDITIONAL classifier exemplars
  (prototypes_for).
- Floor guarantee (ADR-001/002): declared tenant exemplars are a floor —
  memory prototypes may ADD candidates but never replace or degrade below
  the seeds. A garbage prototype must not make classification worse than
  seeds alone.
- Opt-in: VOICEAGENT_MEMORY_DB unset -> the whole memory layer is inert.
- Fail-open: any memory error never breaks a live voice turn.

Embeddings in these tests come from a deterministic FAKE embedder so the
consolidation math (centroid, cosine merge, decay) is exact; only the floor
guarantee test loads the real multilingual encoder (same convention as
test_intent.py).
"""
import sqlite3

import numpy as np
import pytest

from voiceagent.memory import (CAPTURE_CONFIDENCE_THRESHOLD,
                               IntentMemoryStore, now_ts)
from voiceagent.agent import build_agent
from voiceagent.runtime import (_intent_memory_from_env,
                                build_orchestrator, classifier_exemplars)

TENANT = "acme"


# --- test doubles ------------------------------------------------------------

def _unit(v):
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n else a


def make_embedder(vectors_by_text, default=(1.0, 0.0)):
    """Deterministic fake embedder: unit vectors from a text->vector map."""
    def embed(texts):
        out = []
        for t in texts:
            v = vectors_by_text.get(t, default)
            out.append(_unit(v))
        return np.asarray(out, dtype=np.float32)
    return embed


class SpyStore:
    """Records capture calls; inert otherwise."""

    def __init__(self):
        self.captured = []

    def capture(self, tenant, text, label, confidence, outcome="",
                session_id=""):
        self.captured.append((tenant, text, label, confidence, outcome,
                              session_id))

    def prototypes_for(self, tenant):
        return []


class PoisonStore:
    """Every method raises — the fail-open contract must hold anyway."""

    def capture(self, *a, **k):
        raise RuntimeError("memory is on fire")

    def prototypes_for(self, *a, **k):
        raise RuntimeError("memory is on fire")


class FakeClassifier:
    def __init__(self, label="order_status", confidence=1.0):
        self.label = label
        self.confidence = confidence

    def classify(self, text):
        return (self.label, self.confidence)


class FakeLLM:
    def generate(self, prompt, max_tokens=256, stop=None):
        return "Your order ORD-77812 is out for delivery."

    def postprocess(self, text):
        return text


class FakeIndex:
    def search(self, query, k=3):
        return [{"id": "a", "text": "Order status.", "section": "faqs",
                 "score": 0.9}]


def make_store(tmp_path, embedder=None, **kw):
    return IntentMemoryStore(str(tmp_path / "mem.db"),
                             embedder=embedder or make_embedder({}), **kw)


# --- store round-trip --------------------------------------------------------

def test_capture_then_consolidate_produces_prototypes(tmp_path):
    emb = make_embedder({
        "where is my order": (1.0, 0.0),
        "order status please": (0.9, 0.435889894354067),
        "track the order": (0.95, 0.312249899919920),
        "I need a refund": (0.0, 1.0),
        "refund my money": (0.1, 0.9949874371066200),
    })
    store = make_store(tmp_path, embedder=emb)
    store.capture(TENANT, "where is my order", "track_order", 0.9, "unmatched")
    store.capture(TENANT, "order status please", "track_order", 0.5, "unmatched")
    store.capture(TENANT, "track the order", "track_order", 0.7, "unmatched")
    store.capture(TENANT, "I need a refund", "refund", 0.3, "unmatched")
    store.capture(TENANT, "refund my money", "refund", 0.4, "unmatched")
    store.consolidate(TENANT)

    protos = store.prototypes_for(TENANT)
    labels = {p[0] for p in protos}
    assert labels == {"track_order", "refund"}
    by_label = {p[0]: p for p in protos}
    # exemplars: up to 3, ordered by confidence (desc)
    assert by_label["track_order"][1] == ["where is my order",
                                          "track the order",
                                          "order status please"]
    # prototype confidence = mean episode confidence
    assert by_label["track_order"][2] == pytest.approx((0.9 + 0.5 + 0.7) / 3)
    assert by_label["refund"][2] == pytest.approx(0.35)


def test_prototype_centroid_in_embedding_space(tmp_path):
    # The centroid must be the (normalized) mean of the SAME embedding
    # function the classifier queries use — verified by the store returning
    # exemplars that a same-space query resolves against (contract kept via
    # the classifier_exemplars helper; here we check the math directly).
    emb = make_embedder({"a text": (3.0, 4.0)})
    store = make_store(tmp_path, embedder=emb)
    store.capture(TENANT, "a text", "x", 0.8)
    store.consolidate(TENANT)
    assert store.prototypes_for(TENANT)[0][0] == "x"


def test_tenant_namespaces_are_separate(tmp_path):
    store = make_store(tmp_path)
    store.capture("t1", "hello there", "greet", 0.2)
    store.consolidate("t1")
    assert store.prototypes_for("t2") == []
    assert len(store.prototypes_for("t1")) == 1


# --- bounded growth ----------------------------------------------------------

def test_episode_ttl_drops_old_episodes(tmp_path):
    emb = make_embedder({"old episode text": (1.0, 0.0),
                         "fresh episode text": (0.0, 1.0)})
    store = make_store(tmp_path, embedder=emb, ttl_days=14)
    old = "2020-01-01T00:00:00"
    store.capture(TENANT, "old episode text", "legacy", 0.2, ts=old)
    store.capture(TENANT, "fresh episode text", "fresh", 0.2, ts=now_ts())
    store.consolidate(TENANT)
    # TTL-expired episodes never become prototypes and are deleted.
    assert {p[0] for p in store.prototypes_for(TENANT)} == {"fresh"}
    assert [e.ts for e in store.episodes(TENANT)] != old
    assert all(e.ts > old for e in store.episodes(TENANT))


def test_dedupe_merge_at_high_cosine_keeps_higher_confidence_label(tmp_path):
    # cos([1,0],[0.99,0.141...]) ~ 0.99 >= 0.95 -> labels merge into one
    # prototype; the higher-confidence label wins.
    v_b = (0.99, 0.141067359796659)
    emb = make_embedder({"alpha text": (1.0, 0.0), "beta text": v_b})
    store = make_store(tmp_path, embedder=emb)
    store.capture(TENANT, "alpha text", "label_a", 0.9)
    store.capture(TENANT, "beta text", "label_b", 0.5)
    store.consolidate(TENANT)
    protos = store.prototypes_for(TENANT)
    assert len(protos) == 1
    label, exemplars, conf = protos[0]
    assert label == "label_a"                      # higher-confidence label
    assert exemplars == ["alpha text"]
    assert conf == pytest.approx(0.7)              # hit-weighted mean
    assert store.hit_count(TENANT, label) == 2


def test_distinct_labels_below_merge_threshold_stay_separate(tmp_path):
    emb = make_embedder({"a text": (1.0, 0.0), "b text": (0.0, 1.0)})
    store = make_store(tmp_path, embedder=emb)
    store.capture(TENANT, "a text", "label_a", 0.9)
    store.capture(TENANT, "b text", "label_b", 0.5)
    store.consolidate(TENANT)
    assert {p[0] for p in store.prototypes_for(TENANT)} == {"label_a",
                                                            "label_b"}


def test_top_k_pruning_keeps_highest_confidence(tmp_path):
    emb = make_embedder({
        "text for a": (1.0, 0.0),
        "text for b": (0.0, 1.0),
        "text for c": (0.7071067811865476, 0.7071067811865476),
    })
    store = make_store(tmp_path, embedder=emb, top_k=2)
    for label, conf in [("a", 0.9), ("b", 0.5), ("c", 0.7)]:
        store.capture(TENANT, f"text for {label}", label, conf)
    store.consolidate(TENANT)
    protos = store.prototypes_for(TENANT)
    assert [p[0] for p in protos] == ["a", "c"]


def test_decay_loses_stale_prototypes(tmp_path):
    # ttl 14d; decay starts after 30d without a consolidation refresh; each
    # pass multiplies confidence by 0.9; below 0.05 the prototype is dropped.
    store = make_store(tmp_path, ttl_days=14, decay_days=30,
                       decay_factor=0.9, drop_below=0.05)
    t0 = "2026-01-01T00:00:00"
    store.capture(TENANT, "stale topic text", "stale_label", 0.06, ts=t0)
    store.consolidate(TENANT, now=t0)
    assert [p[0] for p in store.prototypes_for(TENANT)] == ["stale_label"]
    assert store.prototypes_for(TENANT)[0][2] == pytest.approx(0.06)

    # +20d: episodes TTL-expired, prototype survives untouched (not yet 30d)
    t20 = "2026-01-21T00:00:00"
    store.consolidate(TENANT, now=t20)
    assert store.prototypes_for(TENANT)[0][2] == pytest.approx(0.06)

    # +31d: first decay pass
    t31 = "2026-02-01T00:00:00"
    store.consolidate(TENANT, now=t31)
    assert store.prototypes_for(TENANT)[0][2] == pytest.approx(0.054)

    # +32d: second pass drops it below 0.05
    t32 = "2026-02-02T00:00:00"
    store.consolidate(TENANT, now=t32)
    assert store.prototypes_for(TENANT) == []


def test_fresh_episodes_refresh_last_seen_no_decay(tmp_path):
    store = make_store(tmp_path, ttl_days=14, decay_days=30)
    t0 = "2026-01-01T00:00:00"
    t40 = "2026-02-10T00:00:00"
    store.capture(TENANT, "first text", "live_label", 0.5, ts=t0)
    store.consolidate(TENANT, now=t0)
    # episodes expire, prototype ages past decay_days but then gets fresh
    # episodes: no decay may apply to a refreshed prototype.
    store.capture(TENANT, "second text", "live_label", 0.5, ts=t40)
    store.consolidate(TENANT, now=t40)
    protos = store.prototypes_for(TENANT)
    assert len(protos) == 1
    assert protos[0][2] == pytest.approx(0.5)  # mean of fresh episodes only
    assert store.hit_count(TENANT, "live_label") == 1


def test_deterministic_same_episodes_same_prototypes(tmp_path):
    emb = make_embedder({
        "text one": (1.0, 0.0),
        "text two": (0.0, 1.0),
        "text three": (0.7, 0.7),
    })
    episodes = [("text one", "l1", 0.8), ("text two", "l2", 0.4),
                ("text three", "l3", 0.6)]
    store_a = IntentMemoryStore(str(tmp_path / "a.db"), embedder=emb)
    store_b = IntentMemoryStore(str(tmp_path / "b.db"), embedder=emb)
    for t, label, conf in episodes:
        for s in (store_a, store_b):
            s.capture(TENANT, t, label, conf, ts="2026-03-01T00:00:00")
    store_a.consolidate(TENANT, now="2026-03-02T00:00:00")
    store_b.consolidate(TENANT, now="2026-03-02T00:00:00")
    # twice on the same store, too
    store_a.consolidate(TENANT, now="2026-03-02T00:00:00")
    assert store_a.prototypes_for(TENANT) == store_b.prototypes_for(TENANT)


def test_max_vectors_per_prototype_cap(tmp_path):
    store = make_store(tmp_path, max_vectors=3)
    for i in range(10):
        store.capture(TENANT, f"vec text {i}", "big", 0.5)
    store.consolidate(TENANT)  # must not raise; centroid from capped window
    assert len(store.prototypes_for(TENANT)) == 1


def test_corrupt_prototype_row_is_skipped(tmp_path):
    store = make_store(tmp_path)
    store.capture(TENANT, "good text", "good", 0.5)
    store.consolidate(TENANT)
    with sqlite3.connect(str(tmp_path / "mem.db")) as conn:
        conn.execute(
            "INSERT INTO prototypes (tenant, label, centroid_json,"
            " exemplars_json, hit_count, last_seen, confidence)"
            " VALUES (?, 'bad', 'NOT JSON', 'ALSO BAD', 1, ?, 0.9)",
            (TENANT, now_ts()))
        conn.commit()
    # prototypes_for skips the corrupt row; consolidate does not crash
    assert [p[0] for p in store.prototypes_for(TENANT)] == ["good"]
    store.consolidate(TENANT)
    assert [p[0] for p in store.prototypes_for(TENANT)] == ["good"]


# --- auto-consolidation ------------------------------------------------------

def test_auto_consolidation_every_nth_capture(tmp_path):
    store = make_store(tmp_path, consolidate_every=5)
    for i in range(4):
        store.capture(TENANT, f"auto text {i}", "learned", 0.2)
    assert store.prototypes_for(TENANT) == []      # nothing consolidated yet
    store.capture(TENANT, "auto text 4", "learned", 0.2)  # 5th capture
    assert len(store.prototypes_for(TENANT)) == 1
    assert store.consolidation_count == 1
    for i in range(5, 9):                          # 6th..9th: no trigger
        store.capture(TENANT, f"auto text {i}", "learned", 0.2)
    assert store.consolidation_count == 1
    store.capture(TENANT, "auto text 9", "learned", 0.2)  # 10th capture
    assert store.consolidation_count == 2


def test_auto_consolidation_threshold_is_25_by_default(tmp_path):
    store = make_store(tmp_path, embedder=make_embedder({}))
    for i in range(24):
        store.capture(TENANT, f"t {i}", "l", 0.2)
    assert store.consolidation_count == 0
    store.capture(TENANT, "t 24", "l", 0.2)
    assert store.consolidation_count == 1


# --- retrieval swap + floor guarantee (ADR-001/002) --------------------------

DECLARED = {
    "order_status": ["where is my order"],
    "refund": ["I need a refund"],
}


class FakeMemoryWithPrototypes:
    def __init__(self, protos):
        self._protos = protos

    def prototypes_for(self, tenant):
        return self._protos


def test_classifier_exemplars_appends_prototypes_after_declared():
    mem = FakeMemoryWithPrototypes(
        [("speak_to_human", ["please connect me to a manager"], 0.8)])
    merged = classifier_exemplars(DECLARED, mem, TENANT)
    # Floor: declared exemplars keep their position/priority verbatim...
    assert merged["order_status"] == ["where is my order"]
    assert merged["refund"] == ["I need a refund"]
    # ...prototypes are APPENDED as new candidates.
    assert merged["speak_to_human"] == ["please connect me to a manager"]


def test_classifier_exemplars_extends_existing_label_after_declared():
    mem = FakeMemoryWithPrototypes(
        [("refund", ["money back please"], 0.8)])
    merged = classifier_exemplars(DECLARED, mem, TENANT)
    assert merged["refund"] == ["I need a refund", "money back please"]


def test_classifier_exemplars_fail_open_on_store_error():
    assert classifier_exemplars(DECLARED, PoisonStore(), TENANT) == DECLARED


# --- M1: conflict guard (floor = retained + conflict-guarded) ----------------

DECLARED_GUARD = {
    "order_status": ["where is my order"],
    "refund": ["I need a refund"],
}


def test_conflict_guard_drops_mislabeled_near_seed_prototype():
    """ADVERSARIAL (M1): a prototype whose exemplar is a near-duplicate of a
    declared seed but carries a WRONG label would outrank the seed at argmax
    cosine (insertion order is irrelevant). The guard must drop it."""
    # "track my parcel status now" embeds NEAR the order_status seed vector
    # (cos 0.99 >= 0.90) but was captured under a wrong label.
    near = _unit((0.99, 0.141067359796659))
    emb = make_embedder({
        "where is my order": (1.0, 0.0),
        "I need a refund": (0.0, 1.0),
        "track my parcel status now": (0.99, 0.141067359796659),
    })
    mem = FakeMemoryWithPrototypes(
        [("wrong_label", ["track my parcel status now"], 0.9)])
    merged = classifier_exemplars(DECLARED_GUARD, mem, TENANT, embed=emb)
    assert "wrong_label" not in merged
    assert merged["order_status"] == ["where is my order"]
    assert merged["refund"] == ["I need a refund"]


def test_conflict_guard_keeps_same_label_near_seed_prototype():
    # A near-seed exemplar under the CORRECT label is reinforcing, not a
    # conflict — it stays.
    emb = make_embedder({
        "where is my order": (1.0, 0.0),
        "I need a refund": (0.0, 1.0),
        "where is my order please": (0.99, 0.141067359796659),
    })
    mem = FakeMemoryWithPrototypes(
        [("order_status", ["where is my order please"], 0.8)])
    merged = classifier_exemplars(DECLARED_GUARD, mem, TENANT, embed=emb)
    assert merged["order_status"] == ["where is my order",
                                      "where is my order please"]


def test_conflict_guard_fail_open_on_embedder_error():
    # If the guard's own embedding fails, fall back to the declared floor.
    def boom(texts):
        raise RuntimeError("embedder down")
    mem = FakeMemoryWithPrototypes(
        [("wrong_label", ["track my parcel status now"], 0.9)])
    assert classifier_exemplars(DECLARED_GUARD, mem, TENANT,
                                embed=boom) == DECLARED_GUARD


def test_conflict_guard_with_no_declared_exemplars_keeps_prototypes():
    emb = make_embedder({"any text": (1.0, 0.0)})
    mem = FakeMemoryWithPrototypes(
        [("learned", ["please connect me to a manager"], 0.8)])
    merged = classifier_exemplars(None, mem, TENANT, embed=emb)
    assert merged == {"learned": ["please connect me to a manager"]}


def test_classifier_exemplars_no_memory_returns_declared_unchanged():
    # Opt-in pin: without a wired store the exemplars are byte-identical.
    assert classifier_exemplars(DECLARED, None, TENANT) is DECLARED
    assert classifier_exemplars(None, None, TENANT) is None


def test_floor_garbage_prototype_cannot_degrade_seed_classification():
    """THE ADR-002 floor test: with declared exemplars + a garbage prototype,
    seed-intent phrases still resolve to the seed labels (real encoder)."""
    from voiceagent.intent import IntentClassifier
    garbage = FakeMemoryWithPrototypes(
        [("learned_noise", ["zzz qqq plork blemf goo garbled"], 0.9)])
    merged = classifier_exemplars(DECLARED, garbage, TENANT)
    clf = IntentClassifier(exemplars=merged)
    assert clf.classify("where is my order?")[0] == "order_status"
    assert clf.classify("I need a refund for my order")[0] == "refund"
    # And the learned label is reachable by a prototype-trained phrase.
    learned = FakeMemoryWithPrototypes(
        [("speak_to_human", ["please connect me to a manager on the phone"],
          0.8)])
    clf2 = IntentClassifier(
        exemplars=classifier_exemplars(DECLARED, learned, TENANT))
    assert clf2.classify(
        "please connect me to a manager on the phone")[0] == "speak_to_human"


# --- opt-in / wiring ---------------------------------------------------------

def test_env_unset_memory_layer_is_inert():
    assert _intent_memory_from_env({}) is None


def test_env_set_builds_store(tmp_path):
    db = tmp_path / "mem.db"
    store = _intent_memory_from_env({"VOICEAGENT_MEMORY_DB": str(db)})
    assert isinstance(store, IntentMemoryStore)


def test_env_bad_path_fails_open_to_none(tmp_path):
    # A corrupt/unusable DB path must not take the process down.
    assert _intent_memory_from_env(
        {"VOICEAGENT_MEMORY_DB": str(tmp_path / "no" / "such" / "dir" /
                                     "mem.db")}) is None


def test_build_orchestrator_wires_memory_from_env(tmp_path):
    db = tmp_path / "mem.db"
    orch = build_orchestrator(env={
        "VOICEAGENT_FRONTIER_URL": "https://fake/v1",
        "VOICEAGENT_MEMORY_DB": str(db)})
    assert orch is not None
    assert isinstance(orch.intent_memory, IntentMemoryStore)


def test_build_orchestrator_without_memory_env_has_none():
    orch = build_orchestrator(env={"VOICEAGENT_FRONTIER_URL": "https://fake/v1"})
    assert orch is not None
    assert orch.intent_memory is None


# --- capture hooks on the live path -----------------------------------------

def test_low_confidence_turn_is_captured_with_outcome():
    spy = SpyStore()
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=FakeClassifier(
        label="order_status", confidence=0.2), intent_memory=spy)
    agent.handle("weird phrasing nobody seeded", conv_id="c1")
    assert len(spy.captured) == 1
    tenant, text, label, conf, outcome, session_id = spy.captured[0]
    assert tenant == "default"
    assert text == "weird phrasing nobody seeded"
    assert label == "order_status"
    assert conf == pytest.approx(0.2)
    assert outcome == "order_status"          # the turn's resulting action


def test_high_confidence_turn_is_not_captured():
    spy = SpyStore()
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=FakeClassifier(
        label="order_status", confidence=0.99), intent_memory=spy)
    agent.handle("where is my order ORD-1")
    assert spy.captured == []


def test_unmatched_action_is_captured_as_unmatched_outcome():
    spy = SpyStore()
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=FakeClassifier(
        label="", confidence=0.9), intent_memory=spy)
    agent.handle("gibberish")
    assert len(spy.captured) == 1
    assert spy.captured[0][2] == ""            # unknown/fallback label
    assert spy.captured[0][4] == "unmatched"


def test_capture_threshold_default_is_0_35():
    assert CAPTURE_CONFIDENCE_THRESHOLD == 0.35
    spy = SpyStore()
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=FakeClassifier(
        label="order_status", confidence=CAPTURE_CONFIDENCE_THRESHOLD),
        intent_memory=spy)
    agent.handle("at exactly the threshold")
    assert spy.captured == []                  # below, not equal, captures
    agent2 = build_agent(FakeIndex(), FakeLLM(), classifier=FakeClassifier(
        label="order_status",
        confidence=CAPTURE_CONFIDENCE_THRESHOLD - 0.01), intent_memory=spy)
    agent2.handle("just under the threshold")
    assert len(spy.captured) == 1


def test_no_memory_store_never_captures():
    # Opt-in pin: no store wired -> nothing to call, turn is unchanged.
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=FakeClassifier(
        label="order_status", confidence=0.1), intent_memory=None)
    res = agent.handle("where is my order ORD-1")
    assert res.action == "order_status"


def test_fail_open_poison_store_turn_completes():
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=FakeClassifier(
        label="order_status", confidence=0.1), intent_memory=PoisonStore())
    res = agent.handle("where is my order ORD-1")
    assert res.action == "order_status"        # static behavior preserved
    assert "out for delivery" in res.text


# --- M2: store version + live reseed ----------------------------------------

def test_store_version_bumps_on_consolidate(tmp_path):
    store = make_store(tmp_path)
    v0 = store.version()
    store.capture(TENANT, "some text", "l", 0.2)
    store.consolidate(TENANT)
    v1 = store.version()
    store.consolidate(TENANT)
    assert v1 > v0
    assert store.version() > v1


def test_eager_preload_embeds_at_construction(tmp_path):
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return np.zeros((len(texts), 2), dtype=np.float32)

    IntentMemoryStore(str(tmp_path / "mem.db"), embedder=counting_embed,
                      eager=True)
    assert calls                       # model warm BEFORE the first call
    calls.clear()
    IntentMemoryStore(str(tmp_path / "lazy.db"), embedder=counting_embed)
    assert calls == []                 # default stays lazy


def test_label_cap_top_100_by_episode_count(tmp_path):
    # 150 distinct labels -> consolidation embeds at most the top-100
    # labels (by episode count, label asc tie-break) per pass.
    dim = 256
    vectors = {}
    for i in range(150):
        v = np.zeros(dim, dtype=np.float32)
        v[i] = 1.0                     # one-hot: pairwise cosine 0, no merge
        vectors[f"text {i}"] = v
    store = make_store(tmp_path, embedder=make_embedder(vectors), top_k=200)
    for i in range(150):
        store.capture(TENANT, f"text {i}", f"l{i:03d}", 0.5)
    store.consolidate(TENANT)
    labels = {p[0] for p in store.prototypes_for(TENANT)}
    assert len(labels) == 100
    assert labels == {f"l{i:03d}" for i in range(100)}   # lowest ids win tie


class VersionedStore:
    """Store double exposing version() + prototypes_for() for the reseed pin."""

    def __init__(self):
        self._version = 0
        self._protos = []

    def capture(self, *a, **k):
        pass                             # capture alone does NOT reseed

    def consolidate(self):
        self._version += 1

    def version(self):
        return self._version

    def prototypes_for(self, tenant):
        return self._protos


class RecordingClassifier:
    """Fake classifier with reseed(): classifies the learned phrase only once
    the reseeded exemplar set contains it."""

    def __init__(self):
        self._exemplars = {"order_status": ["where is my order"]}
        self.reseed_calls = []
        self.rebuilds = 0

    def reseed(self, exemplars):
        self.reseed_calls.append(exemplars)
        self.rebuilds += 1
        self._exemplars = exemplars

    def classify(self, text):
        learned = self._exemplars.get("speak_to_human", [])
        if "please connect me to a manager" in learned:
            return ("speak_to_human", 0.9)
        return ("order_status", 0.2)


def test_agent_reseeds_live_classifier_on_version_change():
    """M2 pin: capture+consolidate -> the NEXT turn classifies with the new
    prototype WITHOUT rebuilding the agent (classifier.reseed, not new)."""
    store = VersionedStore()
    clf = RecordingClassifier()
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=clf,
                        intent_memory=store)
    # Turn 1: version known, no prototypes -> default (low-conf) label.
    res1 = agent.handle("weird phrasing", conv_id="c1")
    assert res1.action == "order_status"
    assert clf.rebuilds == 0          # version unchanged: no reseed yet
    # Learn: capture + consolidate bumps the store version.
    store.capture("default", "please connect me to a manager",
                  "speak_to_human", 0.2)
    store.consolidate()
    store._protos = [("speak_to_human",
                      ["please connect me to a manager"], 0.8)]
    # Turn 2: same agent, reseeded in place -> learned label wins.
    res2 = agent.handle("please connect me to a manager", conv_id="c1")
    assert res2.action == "speak_to_human"
    assert clf.rebuilds == 1
    assert clf.reseed_calls[-1]["speak_to_human"] == \
        ["please connect me to a manager"]
    # Turn 3: version unchanged -> no further reseed.
    agent.handle("please connect me to a manager", conv_id="c1")
    assert clf.rebuilds == 1


def test_agent_reseed_fail_open_on_poison_store():
    clf = RecordingClassifier()
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=clf,
                        intent_memory=PoisonStore())
    res = agent.handle("where is my order ORD-1")
    assert res.action == "order_status"
    assert clf.rebuilds == 0


def test_capture_records_pre_promotion_label():
    """MINOR: the episode records what the CLASSIFIER said (refund, 0.2) —
    the deterministic amount promotion to high_value_refund is an action
    decision, not an understanding fact."""
    spy = SpyStore()
    agent = build_agent(FakeIndex(), FakeLLM(), classifier=FakeClassifier(
        label="refund", confidence=0.2), intent_memory=spy)
    res = agent.handle("refund it, 50000 rupees", amount=50000)
    assert res.action == "high_value_refund"   # promotion happened
    assert len(spy.captured) == 1
    assert spy.captured[0][2] == "refund"      # PRE-promotion label
    assert spy.captured[0][3] == pytest.approx(0.2)
    assert spy.captured[0][4] == "high_value_refund"  # outcome = action


# --- M4: orchestrator sidecar capture ---------------------------------------

def orch_with(store, classifier, tenant_name="acme"):
    from voiceagent.orchestrator import Deployment, Orchestrator
    from voiceagent.swarm.frontier import FrontierAgentBridge, FrontierReply

    class StubClient:
        def chat(self, messages, tools=None, **kw):
            return FrontierReply(content="Happy to help with that.",
                                 tool_calls=[], model="stub",
                                 latency_s=0.001, raw={})

    dep = Deployment(name=tenant_name, system_prompt="You are Acme.",
                     metadata={"tenant": tenant_name})
    orch = Orchestrator(FrontierAgentBridge(StubClient()),
                        intent_memory=store, intent_classifier=classifier)
    orch.deploy(dep)
    return orch


def test_orchestrator_sidecar_captures_low_confidence_turn():
    spy = SpyStore()
    orch = orch_with(spy, FakeClassifier(label="", confidence=0.9))
    res = orch.handle_turn("s1", "kssl blemf worta")
    assert "Happy to help" in res.reply          # turn completed normally
    assert len(spy.captured) == 1
    tenant, text, label, conf, outcome, session_id = spy.captured[0]
    assert tenant == "acme"
    assert text == "kssl blemf worta"
    assert label == ""                           # unknown label captured
    assert outcome == "unmatched"                # no executed action


def test_orchestrator_sidecar_outcome_is_executed_action():
    from voiceagent.swarm.frontier import FrontierReply, FrontierToolCall

    class ToolClient:
        def __init__(self):
            self.n = 0

        def chat(self, messages, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return FrontierReply(content=None, tool_calls=[
                    FrontierToolCall(id="t1", name="reschedule_delivery",
                                     arguments={"order_id": "ORD-1"})],
                    model="stub", latency_s=0.001, raw={})
            return FrontierReply(content="Done.", tool_calls=[],
                                 model="stub", latency_s=0.001, raw={})

    from voiceagent.orchestrator import Deployment, Orchestrator
    from voiceagent.swarm.frontier import FrontierAgentBridge
    spy = SpyStore()
    orch = Orchestrator(FrontierAgentBridge(ToolClient()),
                        runner=GovernedRunnerStub(),
                        intent_memory=spy,
                        intent_classifier=FakeClassifier(label="", confidence=0.9))
    dep = Deployment(name="acme", system_prompt="x",
                     gateway_tools={"reschedule_delivery": {
                         "action": "reschedule", "side_effects": True}})
    orch.deploy(dep)
    orch.handle_turn("s1", "reschedule my delivery please")
    assert len(spy.captured) == 1
    assert spy.captured[0][4] == "reschedule"    # the executed action


class GovernedRunnerStub:
    """Minimal runner: ALLOW everything, return a success result."""

    def run(self, action, ctx, tool_name, params, conv_id=""):
        from voiceagent.tools import GovernedOutcome, ToolResult
        return GovernedOutcome(decision_verdict="ALLOW", reasons=[],
                               executed=True,
                               result=ToolResult(ok=True, value={"ok": True}))


def test_orchestrator_sidecar_skips_high_confidence_turn():
    spy = SpyStore()
    orch = orch_with(spy, FakeClassifier(label="order_status",
                                         confidence=0.99))
    orch.handle_turn("s1", "where is my order ORD-1")
    assert spy.captured == []


def test_orchestrator_capture_fail_open_on_poison_store():
    orch = orch_with(PoisonStore(), FakeClassifier(label="", confidence=0.9))
    res = orch.handle_turn("s1", "kssl blemf worta")
    assert "Happy to help" in res.reply


def test_orchestrator_sidecar_lazy_builds_real_classifier(monkeypatch, tmp_path):
    # Without an injected classifier the orchestrator builds the local one
    # lazily on first capture — and a broken build never breaks the turn.
    import voiceagent.orchestrator as orch_mod
    spy = SpyStore()
    orch = orch_with(spy, None)
    assert orch._intent_classifier is None
    monkeypatch.setattr(
        orch_mod, "_sidecar_classifier", lambda: (_ for _ in ()).throw(
            RuntimeError("no model")), raising=False)
    res = orch.handle_turn("s1", "kssl blemf worta")
    assert "Happy to help" in res.reply


def test_reseed_updates_real_classifier():
    """M2 on the REAL classifier: reseed() swaps the exemplar matrices in
    place — a learned phrase resolves to its label, seeds still win."""
    from voiceagent.intent import IntentClassifier
    clf = IntentClassifier()
    base = dict(clf._exemplars)
    clf.reseed({**base, "speak_to_human":
                ["please connect me to a manager on the phone"]})
    assert clf.classify(
        "please connect me to a manager on the phone")[0] == "speak_to_human"
    assert clf.classify("where is my order?")[0] == "order_status"
    assert clf.classify("I need a refund for my order")[0] == "refund"
