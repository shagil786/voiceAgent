"""Approval gate: PROPOSED -> APPROVED -> CONNECTED lifecycle + dry-run.

Pure functions over Bundle (copy-on-write via copy.deepcopy — callers never
see mutation). Only CONNECTED tools are callable; scope widening resets a
CONNECTED tool to APPROVED and clears its dry-run. Stored dry-run probes are
owner-visible with secrets redacted.
"""
from __future__ import annotations

import copy
import re

from voiceagent.deploy.bundle import Bundle, ToolEntry

_REDACT_KEYS = re.compile(r"key|token|secret|password|auth", re.IGNORECASE)


def redact(obj):
    """Recursively replace STRING values whose key matches key|token|secret|
    password|auth (case-insensitive) with "[REDACTED]". Non-string values
    (flags, numbers, nested dicts/lists) pass through or recurse, so record
    fields like auth_ok: True are never corrupted."""
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if isinstance(v, str) and _REDACT_KEYS.search(str(k))
                else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def _find_tool(bundle: Bundle, name: str) -> ToolEntry:
    for t in bundle.tools:
        if t.name == name:
            return t
    raise ValueError(f"unknown tool {name!r}")


def approve_knowledge(bundle: Bundle) -> Bundle:
    out = copy.deepcopy(bundle)
    out.spec["knowledge_approved"] = True
    return out


def approve_tool(bundle: Bundle, name: str) -> Bundle:
    out = copy.deepcopy(bundle)
    _find_tool(out, name).state = "APPROVED"
    return out


def tool_state(bundle: Bundle, name: str) -> str:
    return _find_tool(bundle, name).state


def get_dry_run(bundle: Bundle, name: str) -> dict | None:
    return copy.deepcopy(_find_tool(bundle, name).dry_run)


def record_dry_run(bundle: Bundle, name: str, probe: dict,
                   confirmed_by: str) -> Bundle:
    if _find_tool(bundle, name).state != "APPROVED":
        raise ValueError(
            f"tool {name!r} must be APPROVED before dry-run; "
            "call approve_tool first")
    if not confirmed_by or not confirmed_by.strip():
        raise ValueError("dry-run requires a non-empty confirmed_by (owner)")
    if not isinstance(probe, dict) or probe.get("auth_ok") is not True:
        raise ValueError("dry-run probe requires auth_ok is True")
    benign = probe.get("benign_call")
    if (not isinstance(benign, dict) or "request" not in benign
            or "response" not in benign):
        raise ValueError("dry-run probe requires benign_call {request, response}")
    out = copy.deepcopy(bundle)
    entry = _find_tool(out, name)
    entry.dry_run = redact(copy.deepcopy(probe))
    entry.state = "CONNECTED"
    return out


def widen_scope(bundle: Bundle, name: str, scopes: list[str]) -> Bundle:
    out = copy.deepcopy(bundle)
    entry = _find_tool(out, name)
    entry.scopes = list(scopes)
    if entry.state == "CONNECTED":
        entry.state = "APPROVED"
        entry.dry_run = None
    return out
