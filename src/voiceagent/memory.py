# src/voiceagent/memory.py
"""Short-term working memory (M4a): per-conversation turn history.

Two stdlib-only backends behind one protocol:
- InMemoryMemory: dict of deques — tests, CLI, ephemeral demos.
- SQLiteMemory: one file in WAL mode — survives restarts. Concurrency model:
  a single shared connection (check_same_thread=False) guarded by one lock.
  The stdlib server serializes requests anyway, and every method holds the
  lock for one short statement, so worker threads never interleave reads
  and writes.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


def now_ts() -> str:
    """Timestamp string for Turn.ts (same format as the decision log)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Turn:
    ts: str
    role: str  # "user" | "agent"
    text: str
    action: str | None = None
    verdict: str | None = None
    refs: list[str] = field(default_factory=list)


@runtime_checkable
class ConversationMemory(Protocol):
    def append(self, conv_id: str, turn: Turn) -> None: ...
    def history(self, conv_id: str, last_n: int | None = None) -> list[Turn]: ...
    def clear(self, conv_id: str) -> None: ...


class InMemoryMemory:
    """dict[conv_id -> deque[Turn]], oldest evicted at maxlen_per_conv."""

    def __init__(self, maxlen_per_conv: int = 100):
        self._maxlen = maxlen_per_conv
        self._convs: dict[str, deque[Turn]] = {}

    def append(self, conv_id: str, turn: Turn) -> None:
        self._convs.setdefault(conv_id, deque(maxlen=self._maxlen)).append(turn)

    def history(self, conv_id: str, last_n: int | None = None) -> list[Turn]:
        turns = list(self._convs.get(conv_id, ()))
        return turns[-last_n:] if last_n is not None else turns

    def clear(self, conv_id: str) -> None:
        self._convs.pop(conv_id, None)


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS turns ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " conv_id TEXT NOT NULL,"
    " ts TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL,"
    " action TEXT, verdict TEXT, refs_json TEXT NOT NULL DEFAULT '[]')",
    "CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conv_id)",
)


class SQLiteMemory:
    """SQLite-backed ConversationMemory. refs are stored as a JSON list in
    refs_json; history() reads oldest-first (last_n takes the newest rows)."""

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # WAL's recommended pairing: commits no longer fsync per write.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        self._conn.commit()

    def append(self, conv_id: str, turn: Turn) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (conv_id, ts, role, text, action, verdict,"
                " refs_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conv_id, turn.ts, turn.role, turn.text, turn.action,
                 turn.verdict, json.dumps(turn.refs)))
            self._conn.commit()

    def history(self, conv_id: str, last_n: int | None = None) -> list[Turn]:
        sql = ("SELECT ts, role, text, action, verdict, refs_json FROM turns"
               " WHERE conv_id = ? ORDER BY id")
        params: list = [conv_id]
        if last_n is not None:
            sql += " DESC LIMIT ?"
            params.append(last_n)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        if last_n is not None:
            rows.reverse()  # DESC LIMIT gives the newest n; restore order
        return [Turn(ts=r[0], role=r[1], text=r[2], action=r[3], verdict=r[4],
                     refs=json.loads(r[5])) for r in rows]

    def clear(self, conv_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM turns WHERE conv_id = ?", (conv_id,))
            self._conn.commit()


def public_dict(turn: Turn) -> dict:
    """API view of a turn (the /api/history shape; internal refs omitted)."""
    return {"ts": turn.ts, "role": turn.role, "text": turn.text,
            "action": turn.action, "verdict": turn.verdict}


# --- ADR-002: learned intent memory (episodic fragments + prototypes) --------

# Capture policy: live turns whose classifier confidence falls below this
# threshold (or produce no/unknown label) are captured as episodic fragments.
CAPTURE_CONFIDENCE_THRESHOLD = 0.35

# Consolidation policy (all overridable per store instance; these are the
# defaults the env-wired store uses).
EPISODE_TTL_DAYS = 14        # episodic fragments are ephemeral (ADR-002)
CONSOLIDATE_EVERY = 25       # every Nth capture triggers a consolidation pass
PROTOTYPE_TOP_K = 50         # bounded growth: max prototypes per tenant
MAX_LABELS_PER_PASS = 25     # consolidation embeds at most this many labels
MAX_VECTORS_PER_PROTOTYPE = 30   # centroid window cap (most recent episodes)
# Pass embedding budget: labels processed until this many episode vectors
# have been embedded (25 labels x 30 vectors worst case ~= seconds on CPU,
# never the 10k-vector mega-pass possible with uncapped windows).
MAX_EMBEDS_PER_PASS = 300
EXEMPLARS_PER_PROTOTYPE = 3  # representative texts kept per prototype
# NOTE on the merge threshold: 0.95 sits above LaBSE's typical same-intent
# similarity, BUT LaBSE is known to over-score Romanized code-mixed Hindi
# (hinglish) phrases — two DIFFERENT hinglish intents can clear 0.95 and
# merge (the M5b sweep measured exactly this collapse in the latin space).
# If hinglish prototypes start merging wrongly, route the merge cosine per
# script (route_space) instead of one global threshold.
MERGE_COSINE = 0.95          # prototypes with centroid cosine >= this merge
DECAY_AFTER_DAYS = 30        # prototypes not refreshed within this decay
DECAY_FACTOR = 0.9           # per-pass confidence multiplier when stale
DROP_BELOW = 0.05            # decayed prototypes below this are dropped


@dataclass
class Episode:
    """One episodic fragment: a candidate turn captured during a live call.
    Ephemeral by design — TTL-bounded, never a permanent transcript."""
    id: int
    tenant: str
    ts: str
    text: str
    label: str
    confidence: float
    outcome: str


def _ts_shift(ts: str, days: int) -> str:
    """ts shifted by `days` (negative = past), same ISO format. Unparseable
    input returns ts unchanged (fail-open: no artificial TTL culling)."""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return ts
    return (dt + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


# Shared per-space encoders (M5b dual-space): one lazy SentenceTransformer
# per knowledge embedding space, process-wide — an extension of the old
# single-model _SHARED_EMBEDDER singleton, not a second cache.
_SHARED_EMBEDDERS: dict[str, "SentenceTransformer"] = {}


def space_embed(texts: list[str], space: str) -> "np.ndarray":
    """THE shared per-space embedding function (M5b dual-space): knowledge's
    SPACE_EMBEDDERS[space] via SentenceTransformer.encode(...,
    normalize_embeddings=True), exactly like IntentClassifier._build's
    per-space matrices. 'native' (LaBSE) is the historical default_embed
    below; 'latin' (MiniLM) serves en/hinglish queries. One lazy model per
    space, shared process-wide; the heavy import stays lazy so this module
    stays importable with zero heavy deps."""
    model = _SHARED_EMBEDDERS.get(space)
    if model is None:
        from sentence_transformers import SentenceTransformer
        from voiceagent.knowledge import SPACE_EMBEDDERS
        model = SentenceTransformer(SPACE_EMBEDDERS[space])
        _SHARED_EMBEDDERS[space] = model
    return np.asarray(
        model.encode(list(texts), normalize_embeddings=True),
        dtype=np.float32)


def default_embed(texts: list[str]) -> "np.ndarray":
    """THE shared embedding function for learned memory (ADR-002): the
    native (LaBSE) space — the SAME model and the SAME normalized-encode
    call the intent classifier uses for its primary (native-script) matrix
    (knowledge.DEFAULT_EMBEDDER, kept under the historical name so every
    existing entry point keeps meaning "LaBSE"; see space_embed for the
    latin counterpart). Prototype centroid vectors therefore live in the
    same space as classifier queries; no new embedding model is introduced."""
    from voiceagent.knowledge import NATIVE_SPACE
    return space_embed(texts, NATIVE_SPACE)

# M1 conflict guard: a prototype exemplar this close to a DECLARED exemplar
# of a DIFFERENT label would outrank the seed at argmax cosine (insertion
# order is irrelevant to the classifier) — such candidates are dropped.
CONFLICT_COSINE = 0.90


def classifier_exemplars(declared: dict[str, list[str]] | None,
                         intent_memory: Any | None, tenant: str,
                         embed: Callable[[list[str]], "np.ndarray"] | None = None,
                         ) -> dict[str, list[str]] | None:
    """Retrieval swap (ADR-001/002): merge the memory store's learned
    prototypes into the DECLARED tenant exemplars that seed the classifier.

    The floor is RETAINED + CONFLICT-GUARDED, not dominant: the classifier is
    argmax cosine over ALL exemplars, so a mislabeled prototype that is a
    near-duplicate of a declared seed could outrank it. The guard therefore
    drops any prototype candidate whose exemplar has cosine >= CONFLICT_COSINE
    to a declared exemplar of a DIFFERENT label (declared exemplars are few;
    they are embedded once per call, in the same shared space as the
    centroids). Same-label near-seed prototypes are reinforcing and stay.

    intent_memory=None returns the declared dict UNCHANGED (same object —
    the opt-in pin). Store or embedder errors fail open to the declared
    exemplars alone (logged, never raised)."""
    if intent_memory is None:
        return declared
    if declared is None:
        merged: dict[str, list[str]] = {}
        declared_texts: list[str] = []
        declared_labels: list[str] = []
    else:
        merged = {label: list(exs) for label, exs in declared.items()}
        declared_texts = [t for exs in declared.values() for t in exs]
        declared_labels = [label for label, exs in declared.items()
                           for _ in exs]
    try:
        prototypes = intent_memory.prototypes_for(tenant)
        if declared_texts and prototypes:
            emb = embed or default_embed
            dvecs = np.asarray(emb(declared_texts), dtype=np.float32)
        for label, exemplars, _confidence in prototypes:
            drop = False
            if declared_texts:
                pvecs = np.asarray(emb(list(exemplars)), dtype=np.float32)
                sims = pvecs @ dvecs.T                    # (p, d)
                for i, text in enumerate(exemplars):
                    for j, dlabel in enumerate(declared_labels):
                        if (dlabel != label
                                and float(sims[i, j]) >= CONFLICT_COSINE):
                            logger.warning(
                                "intent memory: dropped prototype '%s' — "
                                "exemplar %r conflicts with declared label "
                                "'%s' (cosine >= %.2f)",
                                label, text, dlabel, CONFLICT_COSINE)
                            drop = True
                            break
                    if drop:
                        break
            if drop:
                continue
            existing = merged.setdefault(label, [])
            for text in exemplars:
                if text not in existing:
                    existing.append(text)
    except Exception:
        logger.warning("intent memory: retrieval swap failed — serving "
                       "declared exemplars only", exc_info=True)
        return declared
    return merged


def _sidecar_classifier():
    """Factory for the Orchestrator's memory-only classifier (ADR-002 M4):
    the frontier-brain path produces no (label, confidence) pair, so a LOCAL
    classifier runs purely to feed episodic capture — never for decisions.
    Module-level so tests can patch it."""
    from voiceagent.intent import IntentClassifier
    return IntentClassifier()


class IntentMemoryStore:
    """SQLite-backed learned intent memory (ADR-002): episodic fragments
    captured automatically during live turns, consolidated into bounded
    intent prototypes, retrieved by the classifier as ADDITIONAL exemplars.

    Schema: episodes(id, tenant, session_id, ts, text, label, confidence, outcome) and
    prototypes(tenant, label, centroid_json, exemplars_json, hit_count,
    last_seen, confidence) — vectors are JSON float lists produced by
    default_embed (the classifier's own embedding function).

    One connection guarded by a lock (same concurrency model as
    SqliteDecisionLog): writers (voice turn thread, consolidation, tests)
    serialize safely and every write commits.

    Opt-in: built by runtime._intent_memory_from_env when VOICEAGENT_MEMORY_DB
    is set (wired into the Agent path for capture + retrieval and into the
    Orchestrator for sidecar capture); unset keeps the entire memory layer
    inert (zero behavior change). The store itself is HONEST (methods may
    raise); fail-open wrapping lives at the call sites — a memory error must
    never break a voice turn.

    Floor guarantee (ADR-001/002): this store only ever ADDS candidates —
    prototypes_for() output is appended AFTER the tenant's declared exemplars
    by classifier_exemplars (runtime re-export) and conflict-guarded there
    (near-seed prototypes with a conflicting label are dropped). The floor is
    RETAINED + GUARDED, not dominant: argmax cosine can still rank a learned
    prototype above a declared seed for a genuinely new-looking query —
    that is the point of learning."""

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS episodes ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " tenant TEXT NOT NULL,"
        " ts TEXT NOT NULL,"
        " text TEXT NOT NULL,"
        " label TEXT NOT NULL,"
        " confidence REAL NOT NULL DEFAULT 0.0,"
        " outcome TEXT NOT NULL DEFAULT '',"
        " session_id TEXT NOT NULL DEFAULT '')",
        "CREATE TABLE IF NOT EXISTS ratings ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " tenant TEXT NOT NULL,"
        " session_id TEXT NOT NULL,"
        " ts TEXT NOT NULL,"
        " rating REAL NOT NULL,"
        " comment TEXT NOT NULL DEFAULT '')",
        "CREATE INDEX IF NOT EXISTS idx_episodes_tenant"
        " ON episodes(tenant, label)",
        "CREATE TABLE IF NOT EXISTS prototypes ("
        " tenant TEXT NOT NULL,"
        " label TEXT NOT NULL,"
        " centroid_json TEXT NOT NULL,"
        " exemplars_json TEXT NOT NULL,"
        " hit_count INTEGER NOT NULL DEFAULT 0,"
        " last_seen TEXT NOT NULL,"
        " confidence REAL NOT NULL DEFAULT 0.0,"
        " PRIMARY KEY (tenant, label))",
    )

    def __init__(self, path: str,
                 *, embedder: Callable[[list[str]], "np.ndarray"] | None = None,
                 eager: bool = False,
                 ttl_days: int = EPISODE_TTL_DAYS,
                 consolidate_every: int = CONSOLIDATE_EVERY,
                 top_k: int = PROTOTYPE_TOP_K,
                 max_vectors: int = MAX_VECTORS_PER_PROTOTYPE,
                 exemplar_count: int = EXEMPLARS_PER_PROTOTYPE,
                 merge_cosine: float = MERGE_COSINE,
                 decay_days: int = DECAY_AFTER_DAYS,
                 decay_factor: float = DECAY_FACTOR,
                 drop_below: float = DROP_BELOW) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            for stmt in self._SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()
        self._embed = embedder or default_embed
        self._ttl_days = ttl_days
        self._consolidate_every = consolidate_every
        self._top_k = top_k
        self._max_vectors = max_vectors
        self._exemplar_count = exemplar_count
        self._merge_cosine = merge_cosine
        self._decay_days = decay_days
        self._decay_factor = decay_factor
        self._drop_below = drop_below
        self._captures: dict[str, int] = {}
        self._version = 0              # bumped on every consolidation pass
        self.consolidation_count = 0   # testability: how many passes ran
        if eager:
            # M3: when wired from the env path, load the shared embedder NOW
            # (construction happens at process start) — the first capture must
            # never load a SentenceTransformer mid-call.
            self._embed(["memory warmup"])

    # -- capture --------------------------------------------------------------

    def capture(self, tenant: str, text: str, label: str, confidence: float,
                outcome: str = "", ts: str | None = None,
                session_id: str = "") -> None:
        """Record one episodic fragment. Duplicates are acceptable (the
        consolidation averages them out). Every consolidate_every-th capture
        for this tenant triggers a consolidation pass inline — SQLite-local
        and fast at this scale (no background thread to reason about)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes (tenant, ts, text, label, confidence,"
                " outcome, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tenant, ts or now_ts(), text, label, float(confidence),
                 outcome, session_id))
            self._conn.commit()
            count = self._captures.get(tenant, 0) + 1
            self._captures[tenant] = count
        if self._consolidate_every > 0 and count % self._consolidate_every == 0:
            self.consolidate(tenant)

    def record_rating(self, tenant: str, session_id: str, rating: float,
                      comment: str = "") -> None:
        """Caller satisfaction for one call: a ratings row + the rating is
        folded into every episode that session captured (outcome annotation),
        so feedback literally re-weights what the agent learns from."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO ratings (tenant, session_id, ts, rating, comment)"
                " VALUES (?, ?, ?, ?, ?)",
                (tenant, session_id, now_ts(), float(rating), comment))
            self._conn.execute(
                "UPDATE episodes SET outcome = outcome || ? "
                "WHERE tenant = ? AND session_id = ?",
                (f" | feedback:{rating:g}", tenant, session_id))
            self._conn.commit()

    def erase_session(self, session_id: str,
                      tenant: str | None = None) -> dict[str, int]:
        """Right-to-be-forgotten: delete one session's episodes + ratings
        (optionally scoped to a tenant). Returns per-table row counts."""
        out: dict[str, int] = {}
        with self._lock:
            for table in ("episodes", "ratings"):
                if tenant is None:
                    cur = self._conn.execute(
                        f"DELETE FROM {table} WHERE session_id = ?",
                        (session_id,))
                else:
                    cur = self._conn.execute(
                        f"DELETE FROM {table} WHERE session_id = ?"
                        " AND tenant = ?", (session_id, tenant))
                out[table] = cur.rowcount
            self._conn.commit()
        return out

    def prune_episodes_before(self, cutoff_iso: str) -> int:
        """Retention: delete episodes older than an ISO timestamp, any
        tenant. Returns the row count removed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM episodes WHERE ts < ?", (cutoff_iso,))
            self._conn.commit()
            return cur.rowcount

    # -- consolidation --------------------------------------------------------

    def consolidate(self, tenant: str, now: str | None = None) -> None:
        """Deterministic consolidation pass for one tenant: TTL-cull episodes,
        group by label, build prototypes (centroid = normalized mean of the
        same-space episode vectors, capped at max_vectors most recent;
        exemplars = up to 3 highest-confidence / most recent texts; confidence
        = mean), merge near-duplicate prototypes across labels at
        merge_cosine (keeping the higher-confidence label), decay prototypes
        not refreshed within decay_days (dropped below drop_below), and prune
        to the top-K by (confidence, last_seen). Same episode set -> same
        prototype set."""
        try:
            self._consolidate(tenant, now or now_ts())
        except Exception:
            with self._lock:      # never leave a half-written prototype set
                self._conn.rollback()
            raise
        self.consolidation_count += 1
        self._version += 1        # M2: live reseed watches this counter

    def version(self) -> int:
        """Monotonic counter bumped on every successful consolidation pass —
        the live Agent's cheap change signal for in-place classifier reseed
        (checked at most once per turn, fail-open)."""
        return self._version

    def _consolidate(self, tenant: str, now: str) -> None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, text, label, confidence FROM episodes"
                " WHERE tenant = ? ORDER BY id", (tenant,)).fetchall()

        # 1. TTL cull: episodic fragments are ephemeral (ADR-002).
        cutoff = _ts_shift(now, days=-self._ttl_days)
        stale_ids = [r[0] for r in rows if r[1] < cutoff]
        if stale_ids:
            with self._lock:
                self._conn.executemany(
                    "DELETE FROM episodes WHERE id = ?",
                    [(i,) for i in stale_ids])
                self._conn.commit()
        rows = [r for r in rows if r[1] >= cutoff]

        by_label: dict[str, list[tuple]] = {}
        # row shape: (id, ts, text, label, confidence) -> r[0..4]
        for r in rows:
            by_label.setdefault(r[3], []).append(r)

        # 1b. M3 bound: at most MAX_LABELS_PER_PASS distinct labels get
        #     embedded per pass (top by episode count, label asc tie-break).
        #     Labels beyond the cap are NOT rebuilt this pass — their existing
        #     prototypes fall through to the survivor/decay path below.
        if len(by_label) > MAX_LABELS_PER_PASS:
            keep = sorted(by_label,
                          key=lambda lbl: (-len(by_label[lbl]), lbl)
                          )[:MAX_LABELS_PER_PASS]
            by_label = {lbl: by_label[lbl] for lbl in keep}

        # 2. Build one prototype per label (sorted labels -> deterministic).
        built: list[dict] = []
        embed_budget = MAX_EMBEDS_PER_PASS
        for label in sorted(by_label):
            eps = by_label[label]
            window = eps[-self._max_vectors:]     # most recent win the cap
            # Per-pass embed budget: a label that does not fit is left to the
            # survivor/decay path and rebuilt next pass — this pass can never
            # embed more than MAX_EMBEDS_PER_PASS texts, so an inline
            # consolidation on a turn thread stays sub-second-ish.
            if len(window) > embed_budget:
                window = window[-embed_budget:]
            embed_budget -= len(window)
            vectors = np.asarray(
                self._embed([r[2] for r in window]), dtype=np.float32)
            centroid = vectors.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm:
                centroid = centroid / norm
            exemplars: list[str] = []
            seen: set[str] = set()
            for r in sorted(eps, key=lambda e: (-e[4], -e[0])):
                if r[2] in seen:
                    continue
                seen.add(r[2])
                exemplars.append(r[2])
                if len(exemplars) >= self._exemplar_count:
                    break
            built.append({
                "label": label,
                "centroid": centroid,
                "exemplars": exemplars,
                "hit_count": len(eps),
                "confidence": sum(e[4] for e in eps) / len(eps),
                "last_seen": max(e[1] for e in eps),
            })

        # 3. Merge near-duplicate prototypes across labels (cosine >=
        #    merge_cosine): the higher-confidence label survives.
        merged: list[dict] = []
        for p in built:
            target = None
            for m in merged:
                if float(np.dot(p["centroid"], m["centroid"])) \
                        >= self._merge_cosine:
                    target = m
                    break
            if target is None:
                merged.append(p)
                continue
            total = target["hit_count"] + p["hit_count"]
            winner = p if p["confidence"] > target["confidence"] else target
            target.update({
                "label": winner["label"],
                "centroid": winner["centroid"],
                "exemplars": winner["exemplars"],
                "hit_count": total,
                "confidence": (target["confidence"] * target["hit_count"]
                               + p["confidence"] * p["hit_count"]) / total,
                "last_seen": max(target["last_seen"], p["last_seen"]),
            })

        # 4. Carry over prototypes whose label had no episodes this pass,
        #    applying decay to stale ones (not refreshed within decay_days).
        with self._lock:
            old_rows = self._conn.execute(
                "SELECT label, centroid_json, exemplars_json, hit_count,"
                " last_seen, confidence FROM prototypes WHERE tenant = ?",
                (tenant,)).fetchall()
        decay_cutoff = _ts_shift(now, days=-self._decay_days)
        survivors: list[dict] = []
        for label, centroid_json, exemplars_json, hit_count, last_seen, \
                confidence in old_rows:
            if label in by_label:
                continue                # rebuilt fresh from episodes above
            try:
                centroid = [float(x) for x in json.loads(centroid_json)]
                exemplars = [str(x) for x in json.loads(exemplars_json)]
                conf = float(confidence)
            except (ValueError, TypeError):
                continue                # corrupt row: dropped, never surfaced
            if not exemplars:
                continue
            if last_seen < decay_cutoff:
                conf *= self._decay_factor   # decays every pass it stays stale
            if conf < self._drop_below:
                continue
            survivors.append({"label": label, "centroid": centroid,
                              "exemplars": exemplars,
                              "hit_count": int(hit_count),
                              "last_seen": last_seen, "confidence": conf})

        # 5. Bounded growth: top-K by (confidence desc, last_seen desc,
        #    label asc) — chained stable sorts for a deterministic order.
        protos = merged + survivors
        protos.sort(key=lambda p: p["label"])
        protos.sort(key=lambda p: p["last_seen"], reverse=True)
        protos.sort(key=lambda p: p["confidence"], reverse=True)
        protos = protos[:self._top_k]

        with self._lock:
            self._conn.execute("DELETE FROM prototypes WHERE tenant = ?",
                               (tenant,))
            for p in protos:
                self._conn.execute(
                    "INSERT OR REPLACE INTO prototypes (tenant, label,"
                    " centroid_json, exemplars_json, hit_count, last_seen,"
                    " confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (tenant, p["label"],
                     json.dumps([float(x) for x in p["centroid"]]),
                     json.dumps(p["exemplars"]), p["hit_count"],
                     p["last_seen"], float(p["confidence"])))
            self._conn.commit()

    # -- retrieval ------------------------------------------------------------

    def prototypes_for(self, tenant: str) -> list[tuple[str, list[str], float]]:
        """(label, exemplar_texts, confidence) for classifier seeding —
        appended AFTER the tenant's declared exemplars by
        runtime.classifier_exemplars (the declared set is the floor,
        ADR-001/002). Corrupt rows are skipped, never surfaced."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, exemplars_json, confidence FROM prototypes"
                " WHERE tenant = ? ORDER BY confidence DESC, label ASC",
                (tenant,)).fetchall()
        out: list[tuple[str, list[str], float]] = []
        for label, raw, conf in rows:
            try:
                exemplars = [str(x) for x in json.loads(raw)]
            except (ValueError, TypeError):
                continue
            if exemplars:
                out.append((label, exemplars, float(conf)))
        return out

    # -- introspection (tests / ops) ------------------------------------------

    def episodes(self, tenant: str) -> list[Episode]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, tenant, ts, text, label, confidence, outcome"
                " FROM episodes WHERE tenant = ? ORDER BY id",
                (tenant,)).fetchall()
        return [Episode(id=r[0], tenant=r[1], ts=r[2], text=r[3], label=r[4],
                        confidence=r[5], outcome=r[6]) for r in rows]

    def hit_count(self, tenant: str, label: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT hit_count FROM prototypes WHERE tenant = ?"
                " AND label = ?", (tenant, label)).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
