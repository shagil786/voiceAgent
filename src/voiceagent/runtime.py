# src/voiceagent/runtime.py — the one place the governed Orchestrator is built.
"""The governed agent runtime: a single assembly seam for the Orchestrator so
every entry point (LiveKit worker, REPL, tests) drops in the SAME brain.

Contract (non-negotiable, mirrors the limb plan): LiveKit / HTTP are transport;
`Orchestrator.handle_turn` is the ONLY brain. This module wires the frontier
brain + governed tool runner + policy engine + a Deployment's tool surface into
one `Orchestrator`, so no entry point can accidentally serve the legacy Agent
or an ungoverned swarm path.

Importable with zero heavy deps (no livekit, no llama.cpp, no network): the
frontier client is only constructed when `config_from_env()` resolves a URL, and
model loads happen lazily inside the integration adapters, not here.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

from voiceagent.decisionlog import DecisionLog
from voiceagent.memory import InMemoryMemory
from voiceagent.orchestrator import Deployment, Orchestrator
from voiceagent.policy import PolicyEngine, load_policies
from voiceagent.swarm.frontier import (
    FrontierAgentBridge,
    FrontierClient,
    config_from_env,
)
from voiceagent.tools import (
    DEFAULT_TOOL_SPECS,
    GovernedToolRunner,
    MockERP,
    ToolGateway,
    specs_with_yaml_facts,
)
from voiceagent.tenant import DEFAULT_CURRENCY, Tenant, compile_persona_block

# Default policy file + deployment name. A real deployment overrides the
# Deployment (system prompt, gateway tools, knowledge) per business; the policy
# file lives in git as the company's support/compliance artifact.
DEFAULT_POLICY_PATH = "data/policies/policies.yaml"

# The platform's BUILT-IN demo tenant: a COMMITTED bundle under
# data/tenants/default/ (unlike customer bundles such as pizzapal, which are
# gitignored — this one IS the platform's shipped demo tenant). Task E moved
# the demo identity/knowledge/ERP fixture out of voiceagent.demo_data into
# this bundle, so the no-tenant path loads real tenant DATA through the same
# Tenant machinery as any named customer instead of importing demo content.
# Anchored to the repo root (src/voiceagent/ -> parents[2]) so resolution does
# not depend on the process cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TENANT_BUNDLE = _REPO_ROOT / "data" / "tenants" / "default"

# The default (demo) DEPLOYMENT name — domain data, a literal now that the
# bundle itself is named "default" (the Deployment name is the historical
# "acme_support", byte-identical by test pin).
DEFAULT_DEPLOYMENT_NAME = "acme_support"

# Platform-level GOVERNANCE for the frontier system prompt — domain-agnostic
# by design: no tool names, no channel names, no business vocabulary. The
# spoken-aloud brevity rule, the propose-vs-policy contract, authenticate-
# context-from-session, never-invent facts / URLs / tracking links / reference
# numbers, promise-only-what-your-tools-do, and the escalation guidance
# (escalate_to_human is the platform-owned safety valve, always proposeable —
# ADR-003). This is platform code-level policy — tenant bundles supply
# identity/persona AROUND it, never instead of it. The ACTION EXAMPLES are
# deliberately NOT here: they are derived at deployment-compile time from each
# deployment's actual composed surface (platform_prompt below), so a
# non-ecommerce tenant never inherits another business's vocabulary.
PLATFORM_GOVERNANCE = (
    "Be concise and warm — your replies are spoken aloud. You may propose "
    "governed actions from your tool surface — the policy layer decides; if "
    "a verdict blocks you, explain it plainly to the customer. Authenticate "
    "context comes from the session; never invent facts — fetch or verify "
    "them with your tools. Never invent URLs, tracking links, or reference "
    "numbers: if the customer asks for a tracking link, offer to send it "
    "through an available channel instead of reading one out. Only promise "
    "actions that exist in your tool surface — never say you are doing "
    "something you have no tool for. If the customer is upset or asks for a "
    "human agent, propose escalate_to_human with a short reason.")

# The action-examples sentence is CAPPED: it is illustrative guidance, not the
# proposal surface (that is computed from the tool specs — ADR-003), and an
# unbounded list would bloat every turn of a wide-surface deployment.
_MAX_ACTION_EXAMPLES = 8


def _action_examples_sentence(gateway_surface: Iterable[str]) -> str | None:
    """Sorted, capped tool names from a deployment's COMPOSED gateway surface.
    Sorted for stable prompts and CI diffs; capped at _MAX_ACTION_EXAMPLES
    (the "include" wording keeps the truncation honest). Returns None when the
    surface yields nothing proposeable — the caller emits governance-only and
    never invents example actions."""
    names = sorted(set(gateway_surface))[:_MAX_ACTION_EXAMPLES]
    if not names:
        return None
    return ("Governed actions you may propose include: "
            + ", ".join(names) + ".")


def platform_prompt(gateway_surface: Iterable[str] = ()) -> str:
    """The platform block of a frontier system prompt: PLATFORM_GOVERNANCE
    plus the action-examples sentence DERIVED from the deployment's composed
    gateway surface (a gateway-tools mapping passes as its keys, so the exact
    dict wired into the Deployment can be handed in unchanged). Callers must
    pass the same surface the Deployment wires as gateway_tools — the prompt
    may illustrate the surface, never out-promise it. Empty surface ->
    governance-only."""
    examples = _action_examples_sentence(gateway_surface)
    if examples is None:
        return PLATFORM_GOVERNANCE
    return PLATFORM_GOVERNANCE + " " + examples


# The built-in tool surface: DERIVED from DEFAULT_TOOL_SPECS so a new tool
# binding is automatically proposeable by the brain — the proposal surface can
# never drift from the execution bindings again (order_lookup was invisible
# for a day because this used to be a hand-maintained dict). Crafted
# descriptions below override the generic wording where the action needs one.

def _auto_gateway_tools() -> dict[str, dict]:
    """Pure derivation: every DEFAULT_TOOL_SPECS binding becomes a brain-
    proposeable tool. ALL metadata (action, side_effects, description,
    parameters) comes from the ToolSpec declared next to the binding — zero
    hand-maintained entries here, so adding a tool anywhere never requires
    touching runtime.py. Tenant tools.yaml may still override descriptions
    (data beats defaults)."""
    from voiceagent.tools import DEFAULT_TOOL_SPECS
    out: dict[str, dict] = {}
    for name, spec in DEFAULT_TOOL_SPECS.items():
        meta: dict = {
            "action": spec.action or name,
            "side_effects": spec.side_effects,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in spec.params},
                "required": list(spec.params),
            },
        }
        if spec.description:
            meta["description"] = spec.description
        out[name] = meta
    return out


BUILTIN_GATEWAY_TOOLS: dict[str, dict] = _auto_gateway_tools()


def _load_default_bundle_knowledge() -> dict[str, str]:
    """The default bundle's KB documents (knowledge/*.md), loaded straight
    from the committed bundle — the historical DEMO_BUILTIN_KNOWLEDGE, now
    tenant data. Empty when the bundle is unavailable (installed-package
    edge): the fallback is a matter of correctness only for the repo layout
    this module ships in."""
    d = DEFAULT_TENANT_BUNDLE / "knowledge"
    if not d.is_dir():
        return {}
    return _cap_knowledge({f.stem: f.read_text(encoding="utf-8")
                           for f in sorted(d.glob("*.md"))})


# Total injected knowledge is capped because deploy() joins it into the
# system prompt — an unbounded KB would bloat every single turn.
# (RAG phase 1: the SAME number is the chunked-retrieval switch threshold —
# see _knowledge_for; the constant lives in knowledge_rag so both seams
# share one value.)
from voiceagent.knowledge_rag import (  # noqa: E402  (light: numpy only)
    DEFAULT_CHUNKS_CACHE_PATH,
    KNOWLEDGE_BUDGET_CHARS,
    build_chunked_index,
    cap_knowledge,
)

MAX_KNOWLEDGE_CHARS = KNOWLEDGE_BUDGET_CHARS


def gateway_tools_from_yaml(path: str | Path) -> dict[str, dict]:
    """Load a tenant bundle's tools.yaml into a Deployment.gateway_tools
    surface — the DEPLOYMENT surface (what the brain may propose), NOT the
    ToolGateway.from_yaml params/preconditions shape (how tools execute).
    Fail fast: tool names are code bindings (DEFAULT_TOOL_SPECS) so an
    unknown declaration can never silently no-op, and escalate_to_human must
    stay proposeable in any declared surface — the safety valve."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or not isinstance(raw.get("tools", {}),
                                                   dict):
        raise ValueError(
            "tools.yaml: expected a top-level 'tools' mapping of "
            "tool name -> spec")
    tools: dict = raw.get("tools") or {}
    unknown = sorted(set(tools) - set(DEFAULT_TOOL_SPECS))
    if unknown:
        raise ValueError(
            f"tools.yaml: unknown tool(s) {', '.join(unknown)} — bindings "
            "are code (DEFAULT_TOOL_SPECS), declarations are data")
    if "escalate_to_human" not in tools:
        raise ValueError(
            "tools.yaml: 'escalate_to_human' must be declared — the brain "
            "must always be able to propose a human handoff")
    surface: dict[str, dict] = {}
    for name, meta in tools.items():
        if not isinstance(meta, dict) or not meta.get("action"):
            raise ValueError(f"tools.yaml: '{name}' must declare an 'action'")
        # Sprint A3: optional per-tool reply contract (`facts:`) — validated
        # here AND in ToolGateway (voiceagent.tools.parse_facts) so the CI
        # gate and both loaders share one type contract.
        if "facts" in meta:
            from voiceagent.tools import parse_facts
            meta["facts"] = list(parse_facts(meta["facts"],
                                             f"tools.yaml '{name}'"))
        # Task E: optional param-type / numeric-bound constraints — validated
        # here (CI gate) AND in ToolGateway.from_yaml / specs_with_yaml_facts
        # (enforcement), same one-contract pattern as `facts`.
        if "param_types" in meta:
            from voiceagent.tools import parse_param_types
            parse_param_types(meta["param_types"], f"tools.yaml '{name}'")
        if "param_bounds" in meta:
            from voiceagent.tools import parse_param_bounds
            parse_param_bounds(meta["param_bounds"], f"tools.yaml '{name}'")
        surface[name] = dict(meta)
    return surface


def _cap_knowledge(knowledge: dict[str, str]) -> dict[str, str]:
    """Sorted-id prefix of the knowledge that fits under MAX_KNOWLEDGE_CHARS.
    Whole-file granularity: a file that does not fit is dropped together with
    everything after it — a truncated FAQ could assert the opposite of the
    text it cut off. (Delegates to knowledge_rag.cap_knowledge, which also
    serves as the chunked-retrieval fail-open fallback.)"""
    return cap_knowledge(knowledge, MAX_KNOWLEDGE_CHARS)


def _knowledge_for(
    raw: dict[str, str],
    *,
    embedder: Any | None = None,
    cache_path: str | Path | None = None,
) -> tuple[dict[str, str], Any | None]:
    """THE RETRIEVAL SWITCH (RAG phase 1). Returns (knowledge, chunked):

    - Total KB <= MAX_KNOWLEDGE_CHARS -> (whole-file capped dict, None):
      EXACTLY today's behavior — byte-identical prompt, knowledge_ids as
      pinned.
    - Total KB over the budget -> ({}, ChunkedKnowledge): the deployment
      stores CHUNKS (with source ids) + normalized embeddings (cached on
      disk with a bumped cache version) and the orchestrator retrieves
      top-K per turn instead of injecting the whole KB.
    - Any chunk-index build failure (e.g. the embedder is unavailable)
      fails OPEN to the historical whole-file cap — a broken embedder must
      never take a deployment down."""
    if sum(len(text) for text in raw.values()) <= MAX_KNOWLEDGE_CHARS:
        return _cap_knowledge(raw), None
    try:
        return {}, build_chunked_index(
            raw, embedder=embedder,
            cache_path=(DEFAULT_CHUNKS_CACHE_PATH if cache_path is None
                        else cache_path))
    except Exception:
        logging.getLogger(__name__).warning(
            "chunked knowledge build failed; falling back to the "
            "whole-file cap (fail-open)", exc_info=True)
        return _cap_knowledge(raw), None


BUILTIN_KNOWLEDGE: dict[str, str] = _load_default_bundle_knowledge()


def _bundle_gateway_tools(tenant: Tenant) -> dict[str, dict]:
    p = tenant.root / "tools.yaml"
    if p.exists():
        return gateway_tools_from_yaml(p)
    return dict(BUILTIN_GATEWAY_TOOLS)


def _bundle_knowledge(
    tenant: Tenant,
    *,
    embedder: Any | None = None,
    cache_path: str | Path | None = None,
) -> tuple[dict[str, str], Any | None]:
    """The tenant's knowledge through the retrieval switch (_knowledge_for):
    whole-file capped dict for a KB under the budget, chunked index for a KB
    over it. Returns (knowledge, chunked_knowledge)."""
    d = tenant.knowledge_dir()
    if d is None:
        return _knowledge_for(dict(BUILTIN_KNOWLEDGE), embedder=embedder,
                              cache_path=cache_path)
    files = {f.stem: f.read_text(encoding="utf-8")
             for f in sorted(Path(d).glob("*.md"))}
    if not files:
        return _knowledge_for(dict(BUILTIN_KNOWLEDGE), embedder=embedder,
                              cache_path=cache_path)
    return _knowledge_for(files, embedder=embedder, cache_path=cache_path)


def _resolve_tenant(tenant: str | None,
                    env: dict[str, str] | None) -> Tenant | None:
    """Explicit `tenant` arg wins, then VOICEAGENT_TENANT (same env
    precedence as config_from_env: the passed env dict replaces os.environ).
    A value containing '/' is a bundle path; otherwise it is a bundle NAME
    under data/tenants/. No value -> None (built-in deployment)."""
    e = os.environ if env is None else env
    value = tenant or e.get("VOICEAGENT_TENANT") or None
    if not value:
        return None
    root = Path(value) if "/" in value else Path("data/tenants") / value
    bundle = Tenant.load(root)
    # Tenant.load falls back to platform defaults for a MISSING root, which
    # here would silently serve default-identity + platform-wide policy — a
    # typo'd VOICEAGENT_TENANT must never widen policy, so fail fast instead.
    if not bundle.exists:
        raise ValueError(
            f"tenant bundle not found: {root} — VOICEAGENT_TENANT must name "
            "a bundle under data/tenants/ or a bundle path")
    return bundle


def make_deployment(
    tenant: "Tenant | None" = None,
    policy_path: str = DEFAULT_POLICY_PATH,
    *,
    knowledge_embedder: Any | None = None,
    knowledge_cache_path: str | Path | None = None,
) -> Deployment:
    """Build the governed Deployment: prompt + gateway tool surface + inline
    knowledge. With a tenant bundle, identity/persona, tool surface,
    knowledge and metadata all come from data/tenants/<name>/ — onboarding a
    customer is data, not code.

    RAG phase 1: knowledge goes through the retrieval switch (_knowledge_for).
    A KB over MAX_KNOWLEDGE_CHARS produces Deployment.chunked_knowledge (a
    chunk index with cached embeddings) and an EMPTY knowledge dict — the
    orchestrator retrieves top-K chunks per turn. `knowledge_embedder`/
    `knowledge_cache_path` inject the encoder/cache for tests; None defers
    to the shared lazy default embedder and the platform cache path.

    tenant=None loads the COMMITTED default bundle (data/tenants/default/)
    through the SAME Tenant.load machinery as named tenants — the platform's
    built-in demo tenant is a bundle, not a Python import (Task E). The
    composed no-tenant Deployment keeps its historical byte-identical shape
    (name `acme_support`, identity-first prompt, built-in tool surface, no
    metadata/actions): the bundle supplies the bytes, the composition is the
    pinned platform default. (policy_path is accepted for API symmetry; the
    policy engine is wired in build_orchestrator.)"""
    if tenant is None:
        tenant = Tenant.load(DEFAULT_TENANT_BUNDLE)
        knowledge, chunked = _bundle_knowledge(
            tenant, embedder=knowledge_embedder,
            cache_path=knowledge_cache_path)
        return Deployment(
            name=DEFAULT_DEPLOYMENT_NAME,
            # The identity sentence compiles from the bundle's declared
            # persona (flat-string form -> "You are <role>.") — the same
            # compiler named tenants go through. Action examples derive from
            # BUILTIN_GATEWAY_TOOLS — the exact surface wired as gateway_tools
            # below (the composed-surface source of truth for the builtin
            # deployment), so the demo prompt lists what the demo brain may
            # actually propose.
            system_prompt=compile_persona_block(tenant.config.persona)
                          + " " + platform_prompt(BUILTIN_GATEWAY_TOOLS),
            gateway_tools=dict(BUILTIN_GATEWAY_TOOLS),
            knowledge=knowledge,
            chunked_knowledge=chunked,
        )
    knowledge, chunked = _bundle_knowledge(
        tenant, embedder=knowledge_embedder,
        cache_path=knowledge_cache_path)
    gateway_tools = _bundle_gateway_tools(tenant)
    return Deployment(
        name=tenant.config.name,
        # Action examples derive from the COMPOSED GATEWAY SURFACE — this
        # exact dict is wired as gateway_tools below. NOT
        # Tenant.action_vocabulary(): ADR-003 makes the tool-spec-derived
        # surface the brain's proposal surface, while the vocabulary is the
        # wider declared taxonomy (intents/ + info-only extras) — advertising
        # it would name actions no tool can execute, out-promising the
        # surface the governance text itself forbids.
        system_prompt=platform_prompt(gateway_tools) + "\n\n"
                      + compile_persona_block(tenant.config.persona),
        gateway_tools=gateway_tools,
        knowledge=knowledge,
        chunked_knowledge=chunked,
        # Declared greeting: instant pickup line (tenant data); '' keeps the
        # governed greeting-turn path.
        greeting=str(getattr(tenant.config, "greeting", "") or ""),
        # Sprint A1: the bundle DECLARES its action vocabulary (intents/ +
        # tools.yaml + optional tenant.json extras); None when it declares
        # nothing. No business list ships in core.
        actions=tenant.action_vocabulary(),
        metadata={"languages": tenant.language_set(),
                  "tenant": tenant.config.name},
    )


def _audit_log_from_env(env: dict[str, str] | None):
    """Task D3: the audit trail is PERSISTENT when VOICEAGENT_AUDIT_DB names
    a SQLite path (wired into BOTH the runner and the orchestrator audit
    seams); absent config keeps the in-memory DecisionLog — zero config
    change for existing deployments. Reads the passed env dict (falling back
    to os.environ) with the same precedence as the frontier config."""
    e = os.environ if env is None else env
    audit_db = e.get("VOICEAGENT_AUDIT_DB")
    if audit_db:
        from voiceagent.decisionlog import SqliteDecisionLog
        return SqliteDecisionLog(audit_db)
    return DecisionLog()


def _intent_memory_from_env(env: dict[str, str] | None):
    """ADR-002: the learned intent memory is OPT-IN — VOICEAGENT_MEMORY_DB
    naming a SQLite path builds an IntentMemoryStore (episodic fragments +
    consolidated prototypes, wired into the live turn path like the audit
    log); unset config returns None and the whole memory layer is inert
    (zero behavior change for existing deployments). A unusable DB path
    fails OPEN to None: a bad path must never take the process down. The
    store is built EAGER-embedding (M3): the SentenceTransformer loads at
    process start, never mid-call."""
    e = os.environ if env is None else env
    memory_db = e.get("VOICEAGENT_MEMORY_DB")
    if not memory_db:
        return None
    from voiceagent.memory import IntentMemoryStore
    try:
        return IntentMemoryStore(memory_db, eager=True)
    except Exception:
        return None


# Retrieval swap (ADR-001/002) lives in voiceagent.memory next to the shared
# embedder it conflict-guards with; re-exported here so every wiring site
# (and older imports) keeps one name: runtime.classifier_exemplars.
from voiceagent.memory import classifier_exemplars  # noqa: E402


def build_orchestrator(
    env: dict[str, str] | None = None,
    policy_path: str = DEFAULT_POLICY_PATH,
    *,
    tenant: str | None = None,
    erp: Any | None = None,
    memory: Any | None = None,
    decision_log: Any | None = None,
    deployment: Deployment | None = None,
    max_tool_rounds: int = 3,
    intent_memory: Any | None = None,
) -> Orchestrator | None:
    """Assemble the governed Orchestrator. Returns None when no frontier brain
    is configured (VOICEAGENT_FRONTIER_URL unset) so callers fail FAST with an
    explicit message instead of crashing later on `orchestrator.handle_turn`.

    `env` overrides os.environ for config resolution (lets tests pin a stub
    frontier). `tenant` selects a tenant bundle by NAME under data/tenants/ or
    by bundle PATH; when it resolves (explicit arg, else VOICEAGENT_TENANT),
    the bundle's policies.yaml and Deployment drive the brain. `erp`/`memory`/
    `decision_log` are injectable so a real backend or a test double can be
    substituted without touching the wiring. `intent_memory` (ADR-002) is the
    learned intent memory; None defers to VOICEAGENT_MEMORY_DB (opt-in —
    unset keeps the memory layer fully inert).
    """
    cfg = config_from_env(env)
    if cfg is None:
        return None

    bundle = _resolve_tenant(tenant, env)
    log = decision_log or _audit_log_from_env(env)
    intent_memory = intent_memory or _intent_memory_from_env(env)
    # The bundle's policy file IS the least-privilege artifact: undeclared
    # actions get a DENY fed back to the brain. Only a bundle that declares no
    # policy file falls back to the platform policy_path.
    policy_src = (bundle.policy_file() or policy_path) if bundle else policy_path
    # Currency is tenant data: the bundle declares it, the platform default
    # covers a no-tenant deployment. Feeds the policy reason strings.
    currency = bundle.config.currency if bundle else DEFAULT_CURRENCY
    policy = PolicyEngine(load_policies(policy_src), currency=currency)
    # Sprint A3: a bundle may declare per-tool reply contracts (`facts:` in
    # tools.yaml) — merge them into the gateway specs so the EXECUTED tool's
    # spec carries the deployment's contract data. No tools.yaml -> the code
    # defaults (DEFAULT_TOOL_SPECS) apply unchanged.
    specs = None
    if bundle is not None and (bundle.root / "tools.yaml").exists():
        specs = specs_with_yaml_facts(bundle.root / "tools.yaml")
    runner = GovernedToolRunner(
        ToolGateway(erp=erp or MockERP(), specs=specs), policy,
        decision_log=log)
    brain = FrontierAgentBridge(FrontierClient(cfg))
    dep = deployment or make_deployment(tenant=bundle,
                                        policy_path=policy_path)
    orch = Orchestrator(
        brain, runner=runner, memory=memory or InMemoryMemory(),
        decision_log=log, max_tool_rounds=max_tool_rounds,
        actions=dep.actions,  # Sprint A1: resolved vocabulary into the brain
        intent_memory=intent_memory)  # ADR-002: learned intent memory
    orch.deploy(dep)
    return orch
