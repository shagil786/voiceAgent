"""Test-suite-wide configuration.

Historical note (2026-09-14 service split): this module used to import
torch at session start because macOS segfaults when faiss loads before
torch's OpenMP runtime. The guard now lives at the ONLY faiss import site
(voiceagent/knowledge.py, sentence-transformers before `import faiss`),
and the default test tier must not load ML libraries at all — see
tests/test_import_contract.py.
"""

# macOS OpenMP order guard (restored 2026-09-14, full-suite finding): torch
# 2.14.0 and faiss-cpu 1.9.0 each bundle libomp.dylib; when faiss initializes
# first, the first parallel faiss search aborts (OMP Error #15 -> Fatal
# Python error; reproduced minimal). The stable order is the repo's standing
# rule — import sentence_transformers (hence torch) BEFORE any faiss import,
# same as knowledge._ml_imports. Bare `import torch` is NOT equivalent
# (verified: torch->faiss->search aborts; ST->faiss->search is stable and
# byte-identical). Needed at SESSION START, not only inside _ml_imports:
# test_index_cache imports faiss at module level, and torch-typed stub tests
# (test_asr_routing) load torch at runtime. Library import only — no model
# weights — so the fast tier's no-weights contract holds; the
# import-contract probes run in fresh subprocesses and are unaffected.
import sentence_transformers  # noqa: E402  (must precede any faiss import)
