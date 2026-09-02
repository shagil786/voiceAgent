# src/voiceagent/tenant.py
"""Per-deployment configuration — the de-hardcoding seam (M6a).

A tenant is a business running the agent: its persona, currency, and (over
time) intent exemplars, knowledge base, voices, and language set are DATA
for that tenant, never code. Onboarding a customer = loading their config,
not forking the codebase. Defaults reproduce the historical built-in
behavior exactly, so the text benchmark stays byte-identical when no
tenant file exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PERSONA = ("customer support assistant for an Indian ecommerce "
                   "company")
DEFAULT_CURRENCY = "₹"
TENANT_CONFIG_PATH = "data/tenants/default.json"


@dataclass
class TenantConfig:
    name: str = "default"
    persona: str = DEFAULT_PERSONA
    currency: str = DEFAULT_CURRENCY

    @classmethod
    def load(cls, path: str | Path = TENANT_CONFIG_PATH) -> "TenantConfig":
        """Load tenant config from JSON; missing file -> historical
        defaults. Unknown keys are ignored (forward compatibility)."""
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(
            name=data.get("name", "default"),
            persona=data.get("persona", DEFAULT_PERSONA),
            currency=data.get("currency", DEFAULT_CURRENCY),
        )
