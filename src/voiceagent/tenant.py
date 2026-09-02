# src/voiceagent/tenant.py
"""Per-deployment configuration — the de-hardcoding seam (M6a/M6b).

A tenant is a business running the agent: identity, persona, and (over time)
intent exemplars, knowledge base, voices, and language set are DATA for that
tenant, never code. Onboarding a customer = loading their config, not
forking the codebase.

Design notes:
- The persona is STRUCTURED (role, tone, promise permissions, forbidden
  claims, languages) and COMPILED into the system prompt. A free-text
  persona string would be un-auditable and un-testable; structured fields
  let CI assert things like "the agent never promises guaranteed refunds".
- The file is the source-of-truth format (git-ops-able, offline-friendly —
  right for self-hosted deployments); a dashboard/API becomes the editing
  surface later, with this file as import/export.
- Static config (this file) = what the operator declares. Learned state
  (e.g. the sentiment lexicon DB) = what the system has learned. The two
  are deliberately different stores.
- Defaults reproduce the historical built-in behavior exactly, so the text
  benchmark stays byte-identical when no tenant file exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PERSONA_ROLE = ("customer support assistant for an Indian ecommerce "
                        "company")
DEFAULT_CURRENCY = "₹"
TENANT_CONFIG_PATH = "data/tenants/default/tenant.json"


@dataclass
class Persona:
    """Structured agent identity — compiled into the system prompt and
    reviewable/assertable by compliance tooling."""
    role: str = DEFAULT_PERSONA_ROLE
    tone: str | None = None
    # Explicit promise permissions: the agent may not promise anything that
    # is not on this list (enforced by prompt compilation + eval assertions).
    may_promise: list[str] = field(default_factory=list)
    # Compliance-critical forbidden claims.
    never_say: list[str] = field(default_factory=list)
    # Languages this deployment serves (None = platform default set).
    languages: list[str] | None = None


@dataclass
class TenantConfig:
    name: str = "default"
    persona: Persona = field(default_factory=Persona)
    currency: str = DEFAULT_CURRENCY

    @classmethod
    def load(cls, path: str | Path = TENANT_CONFIG_PATH) -> "TenantConfig":
        """Load tenant config from JSON; missing file -> historical
        defaults. A legacy flat-string `persona` key is accepted and wrapped
        as Persona(role=...). Unknown keys are ignored."""
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        raw_persona = data.get("persona", {})
        if isinstance(raw_persona, str):  # legacy flat format
            persona = Persona(role=raw_persona)
        else:
            persona = Persona(
                role=raw_persona.get("role", DEFAULT_PERSONA_ROLE),
                tone=raw_persona.get("tone"),
                may_promise=list(raw_persona.get("may_promise", [])),
                never_say=list(raw_persona.get("never_say", [])),
                languages=raw_persona.get("languages"),
            )
        return cls(
            name=data.get("name", "default"),
            persona=persona,
            currency=data.get("currency", DEFAULT_CURRENCY),
        )


class Tenant:
    """A tenant BUNDLE: one namespace per customer, with typed config
    surfaces, each in its natural format:

        <root>/
          tenant.json           # identity: structured persona + locale
          intents/<intent>.yaml # per-intent exemplars (human-reviewed data)
          policies.yaml         # action verdicts, thresholds, escalate_when
          knowledge/            # KB source documents
          (learned state is NOT here — it lives in DBs, per store)

    Every surface is OPTIONAL: a missing surface means "fall back to the
    built-in default", so the default tenant reproduces historical behavior
    byte-identically and a new tenant overrides only what it declares. This
    bundle — validated (scripts/validate_tenant.py), versioned in git,
    deployed atomically — IS the Control Plane's per-customer artifact.
    """

    def __init__(self, root: str | Path = "data/tenants/default"):
        self.root = Path(root)
        self.config = TenantConfig.load(self.root / "tenant.json")
        self.name = self.config.name if self.root.exists() else self.config.name

    @classmethod
    def load(cls, root: str | Path = "data/tenants/default") -> "Tenant":
        """Load a tenant bundle; a missing root is the default tenant with
        every surface falling back to platform built-ins."""
        return cls(root)

    @property
    def exists(self) -> bool:
        return self.root.exists()

    def intent_exemplars(self) -> dict[str, list[str]] | None:
        """Exemplars from intents/<intent>.yaml (a YAML list per file,
        filename = intent name), merged. None when the bundle declares
        none — callers fall back to the built-in INTENT_EXEMPLARS."""
        d = self.root / "intents"
        if not d.is_dir():
            return None
        merged: dict[str, list[str]] = {}
        for f in sorted(d.glob("*.yaml")):
            import yaml
            data = yaml.safe_load(f.read_text())
            if not isinstance(data, list):
                raise ValueError(f"{f}: expected a YAML list of exemplars")
            merged[f.stem] = [str(x) for x in data]
        return merged or None

    def policy_file(self) -> str | None:
        p = self.root / "policies.yaml"
        return str(p) if p.exists() else None

    def knowledge_dir(self) -> str | None:
        d = self.root / "knowledge"
        return str(d) if d.is_dir() else None

    def language_set(self) -> list[str] | None:
        return self.config.persona.languages
