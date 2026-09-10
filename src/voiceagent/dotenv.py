# src/voiceagent/dotenv.py — the ONE .env loader every script uses.
"""Stdlib KEY=value loader (repo pattern): '#' comments, optional quotes,
never overrides variables the operator already exported. VOICEAGENT_DOTENV_PATH
overrides the file location — tests point it at a missing path so subprocess
checks stay hermetic; an explicitly missing override is honored (no
fallback), an unset/empty one uses the caller's default path.

Import side-effect rule (suite-wide lesson): call this ONLY inside main() /
entry points. A module-level load_dotenv() call leaks .env into any process
that imports the module — it once rerouted every TTS voice in the test
suite via VOICEAGENT_TTS_VOICES.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path) -> None:
    override = os.environ.get("VOICEAGENT_DOTENV_PATH", "").strip()
    if override:
        path = Path(override)
    else:
        path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
