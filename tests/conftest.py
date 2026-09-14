"""Test-suite-wide configuration.

Historical note (2026-09-14 service split): this module used to import
torch at session start because macOS segfaults when faiss loads before
torch's OpenMP runtime. The guard now lives at the ONLY faiss import site
(voiceagent/knowledge.py, sentence-transformers before `import faiss`),
and the default test tier must not load ML libraries at all — see
tests/test_import_contract.py.
"""
