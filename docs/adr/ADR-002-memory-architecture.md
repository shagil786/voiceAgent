# ADR-002: Memory architecture (episodic fragments + consolidated prototypes)

## Context
Understanding must be learned by the running agent from real conversations
(ADR-001), with bounded growth.

## Decision
- **Episodic fragments** (ephemeral): one record per candidate turn — embedded
  text snippet, detected intent, confidence, outcome, timestamp. TTL-bounded;
  raw transcripts are never permanent.
- **Consolidation pass** (background): merges fragments into intent
  prototypes (label + centroid vector + K-best exemplar snippets + hit count +
  last-seen + confidence). Near-duplicates merge; unused prototypes decay.
  Growth bounded by top-K prototypes per tenant.
- **Retrieval**: the intent classifier queries living prototypes instead of a
  static exemplar array. New prototypes serve calls immediately.
- **Capture**: low-confidence / unknown-intent turns are captured
  automatically during live calls.

## Consequences
+ Bounded growth; self-improving understanding; no deploy to learn.
− Consolidation logic must be tested against drift (a bad merge must never
  make the classifier worse than the seed exemplars — seed exemplars are
  retained as a floor).
− **Floor semantics, honestly stated**: the classifier is argmax cosine over
  ALL exemplars, so exemplar insertion order gives the declared set
  RETENTION, not priority. The floor is "retained + conflict-guarded":
  retrieval drops any prototype whose exemplar is a near-duplicate
  (cosine >= 0.90, same shared encoder) of a declared exemplar under a
  DIFFERENT label. A learned prototype CAN still outrank a declared seed
  for a genuinely new-looking query — that is intended learning, guarded
  against mislabeled near-seed noise only.

## Current deviations (as implemented)
- **Frontier-path capture**: the Orchestrator's frontier brain produces no
  (label, confidence) pair, so a LOCAL sidecar classifier runs purely for
  episodic capture (never for decisions); retrieval on the frontier path is
  pending. The Agent path both captures (its own classifier) and retrieves.
- **Retrieval snapshot**: classifier exemplars were merged once at build
  time; the Agent path now reseeds its live classifier in place when the
  memory store's prototype version changes (checked at most once per turn,
  fail-open). Other retrieval sites (e.g. the voice demo) still snapshot at
  construction.
- **Capture scope**: only low-confidence (< 0.35) or unknown-label turns are
  captured, on both paths; episodes are TTL-bounded at 14 days so raw
  transcripts stay ephemeral.
