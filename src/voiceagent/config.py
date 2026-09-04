"""Central runtime config — the hardcoded-paths/secrets seam.

Every tunable the operators need (model cache dir, candidate models, TTS
voices, embedding space, frontier endpoint, HF token) resolves in ONE place
with ONE precedence rule: env > tenant JSON > code defaults. Call sites read
a RuntimeConfig instead of os.environ / literals scattered across modules.

Design notes:
- stdlib only (os, dataclasses, json, pathlib); no new dependencies.
- `env` is an injectable mapping so tests never touch os.environ.
- Defaults are verbatim copies of today's hardcoded behavior (NOT guesses):
    - candidate model stems: the "name" fields of llm.CANDIDATE_MODELS.
    - voices: a copy of tts.VOICE_REGISTRY (piper voice names).
  They are COPIED rather than imported so this module never pulls in heavy
  third-party deps (llm.py imports llama_cpp at module top).
- Tenant voices are opportunistic: tenant.py ignores unknown keys, so when
  `tenant` is a TenantConfig the voices live only in the tenant.json file.
  load_config therefore also accepts a Mapping (parsed tenant.json dict) or
  a path to tenant.json and reads its "voices" key. Tenant overrides voices
  only when non-empty; env wins per-key over the merged dict.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# Verbatim copy of the "name" fields of llm.CANDIDATE_MODELS (src/voiceagent/llm.py).
DEFAULT_CANDIDATE_MODELS = [
    "qwen3-0.6b-q4",
    "qwen2.5-0.5b-q4",
    "qwen2.5-1.5b-q4",
    "qwen2.5-0.5b-hinglish-q4",
]

# Verbatim copy of tts.VOICE_REGISTRY (src/voiceagent/tts.py): text language
# -> piper voice name. ("hinglish" is NOT a registry key — TTSHandle routes
# it to the "hi" voice via HINGLISH_VOICE_LANG.)
DEFAULT_VOICES = {
    "en": "en_US-lessac-medium",
    "hi": "hi_IN-pratham-medium",
    "te": "te_IN-maya-medium",
}

DEFAULT_MODELS_DIR = "data/models"
DEFAULT_EMBEDDING_SPACE = "latin"  # mirrors knowledge.LATIN_SPACE


@dataclass
class RuntimeConfig:
    """Resolved runtime configuration (env > tenant JSON > code defaults)."""
    models_dir: str = DEFAULT_MODELS_DIR
    candidate_models: list[str] = field(
        default_factory=lambda: list(DEFAULT_CANDIDATE_MODELS))
    voices: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_VOICES))
    embedding_space: str = DEFAULT_EMBEDDING_SPACE
    frontier_url: str | None = None
    frontier_model: str | None = None
    frontier_key: str | None = field(default=None, repr=False)
    hf_token: str | None = field(default=None, repr=False)


def _parse_list(raw: str | None) -> list[str] | None:
    """Comma-separated env value -> stripped non-empty items (None if unset)."""
    if raw is None:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items


def _parse_voices(raw: str | None) -> dict[str, str] | None:
    """Comma `lang:path` pairs -> dict, splitting on the FIRST ':' only.

    (ONNX paths never contain ':', so first-':' split is safe; documented.)
    None when unset; empty dict when set-but-empty (treated as no-override).
    """
    if raw is None:
        return None
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        lang, _, path = pair.partition(":")
        lang, path = lang.strip(), path.strip()
        if lang and path:
            out[lang] = path
    return out


def _tenant_voices(tenant) -> dict[str, str] | None:
    """Opportunistic tenant-level voices (tenant.py ignores unknown keys, so
    a TenantConfig object never carries them — read the JSON instead).

    Accepts a TenantConfig (getattr fallback), a Mapping (parsed tenant.json),
    or a path to tenant.json. Returns None when no non-empty voices found.
    """
    if tenant is None:
        return None
    if isinstance(tenant, (str, Path)):
        p = Path(tenant)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        voices = data.get("voices") if isinstance(data, dict) else None
    elif isinstance(tenant, Mapping):
        voices = tenant.get("voices")
    else:  # TenantConfig (or similar): opportunistic attribute read only.
        voices = getattr(tenant, "voices", None)
    if isinstance(voices, dict) and voices:
        return {str(k): str(v) for k, v in voices.items() if v}
    return None


def load_config(env: Mapping[str, str] | None = None,
                tenant=None) -> RuntimeConfig:
    """Resolve a RuntimeConfig: env vars override tenant JSON voices, which
    override code defaults. `env=None` reads os.environ.

    Tenant-voices contract: a `TenantConfig` object cannot carry voices
    (tenant.py ignores unknown keys) — tenant voices arrive via a Mapping
    (parsed tenant.json dict) or a path-string tenant pointing at
    tenant.json; a TenantConfig object contributes no voices."""
    e = os.environ if env is None else env

    models_dir = e.get("VOICEAGENT_MODELS_DIR") or DEFAULT_MODELS_DIR

    cand = _parse_list(e.get("VOICEAGENT_CANDIDATE_MODELS"))
    candidate_models = cand if cand else list(DEFAULT_CANDIDATE_MODELS)

    voices: dict[str, str] = dict(DEFAULT_VOICES)
    t_voices = _tenant_voices(tenant)
    if t_voices:
        voices.update(t_voices)
    e_voices = _parse_voices(e.get("VOICEAGENT_VOICES"))
    if e_voices:
        voices.update(e_voices)

    embedding_space = e.get("VOICEAGENT_EMBEDDING_SPACE") or DEFAULT_EMBEDDING_SPACE

    frontier_url = e.get("VOICEAGENT_FRONTIER_URL") or None
    frontier_model = e.get("VOICEAGENT_FRONTIER_MODEL") or None
    frontier_key = e.get("VOICEAGENT_FRONTIER_KEY") or None
    hf_token = e.get("VOICEAGENT_HF_TOKEN") or None

    return RuntimeConfig(
        models_dir=models_dir,
        candidate_models=candidate_models,
        voices=voices,
        embedding_space=embedding_space,
        frontier_url=frontier_url,
        frontier_model=frontier_model,
        frontier_key=frontier_key,
        hf_token=hf_token,
    )
