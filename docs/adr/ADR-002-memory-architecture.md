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
