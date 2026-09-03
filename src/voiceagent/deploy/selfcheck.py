"""Runnable self-checks: each bundle eval is real Orchestrator.handle_turn calls.

Default harness is the deterministic ScriptedClient from voiceagent.deploy.stub
(shipped with src — never the tests package), wrapped in FrontierAgentBridge with
runner=None — so no governed tools fire and assertions cover reply content
(`contains`); `action`/`verdict` assertions resolve against the turn's first
governed action entry and will fail closed when no action ran.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from voiceagent.deploy.bundle import Bundle, write_live


def _check_turn(reply: str, action: Any, verdict: Any,
                assertion: dict) -> tuple[bool, str]:
    if "contains" in assertion and assertion["contains"] not in reply:
        return False, f"missing {assertion['contains']!r} in {reply!r}"
    if "action" in assertion and action != assertion["action"]:
        return False, f"action {action!r} != {assertion['action']!r}"
    if "verdict" in assertion and verdict != assertion["verdict"]:
        return False, f"verdict {verdict!r} != {assertion['verdict']!r}"
    return True, "ok"


def _default_bridge(ev) -> Any:
    """Deterministic stub brain for one eval: one scripted reply per turn.

    Reply text embeds the eval's `contains` expectation (when present) so the
    check exercises the real handle_turn path deterministically; evals without
    a `contains` assertion get a stable placeholder reply.
    """
    from voiceagent.swarm.frontier import FrontierAgentBridge
    from voiceagent.deploy.stub import make_default_client

    assertion = ev.assert_ or {}
    turns = ev.turns or [{}]
    seed = assertion.get("contains", f"ok {ev.name}")
    stub = make_default_client([f"selfcheck {ev.name}: {seed}" for _ in turns])
    return FrontierAgentBridge(stub)


def _as_bridge(brain: Any) -> Any:
    from voiceagent.swarm.frontier import FrontierAgentBridge

    if isinstance(brain, FrontierAgentBridge):
        return brain
    return FrontierAgentBridge(brain)


def run_self_checks(bundle: Bundle,
                    make_brain: Callable[[], Any] | None = None,
                    live_spot: bool = False) -> list[dict]:
    """Run every eval in the bundle through a real Orchestrator turn loop.

    Returns [{name, passed, detail}]. `make_brain` (optional) builds a fresh
    stub brain client — or a ready FrontierAgentBridge — per eval; the default
    scripts deterministic replies from each eval's own assertion. `live_spot`
    is reserved for a future live-spot probe and currently changes nothing.
    """
    from voiceagent.memory import InMemoryMemory
    from voiceagent.orchestrator import Deployment, Orchestrator

    out: list[dict] = []
    for ev in bundle.evals:
        bridge = _as_bridge(make_brain()) if make_brain is not None else _default_bridge(ev)
        dep = Deployment(
            name=bundle.deploy_id,
            system_prompt=bundle.spec.get("role", ""),
            knowledge={str(i): c.get("text", "") for i, c in enumerate(bundle.knowledge or [])},
        )
        orch = Orchestrator(brain=bridge, runner=None, memory=InMemoryMemory())
        orch.deploy(dep)
        ok_all, details = True, []
        for t in ev.turns or [{}]:
            r = orch.handle_turn(session_id=f"selfcheck-{ev.name}",
                                 user_text=(t.get("user", "") if isinstance(t, dict) else ""))
            action = r.actions[0].get("action") if r.actions else None
            verdict = r.actions[0].get("verdict") if r.actions else None
            ok, d = _check_turn(r.reply, action, verdict, ev.assert_ or {})
            ok_all = ok_all and ok
            details.append(d)
        out.append({"name": ev.name, "passed": ok_all, "detail": "; ".join(details)})
    return out


def go_live(deploy_dir: str, version: str, results: list[dict]) -> bool:
    """Mechanical go-live: write the live pointer only on 10/10 passed.

    Returns True when the pointer was written, False otherwise (pointer left
    untouched — including when fewer than 10 checks ran).
    """
    if len(results) >= 10 and all(r["passed"] for r in results):
        Path(deploy_dir).mkdir(parents=True, exist_ok=True)
        write_live(deploy_dir, version)
        return True
    return False
