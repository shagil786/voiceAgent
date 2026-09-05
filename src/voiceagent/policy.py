# src/voiceagent/policy.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The reason strings reach the brain and the customer, so amounts are shown
# in the deployment's currency — tenant data, not a hardcoded symbol. The
# platform default follows tenant.DEFAULT_CURRENCY (single source).
from voiceagent.tenant import DEFAULT_CURRENCY


DEFAULT_POLICIES = {
    "escalate": ["fraud", "legal", "chargeback", "high_value_refund"],
    "refund": {"require_auth": True, "max_without_approval": 5000},
    "high_value_refund": {"require_auth": True, "escalate": True},
    "order_status": {"allow": True},
    "refund_info": {"allow": True},
    "delivery_eta": {"allow": True},
    "order_cancellation": {"require_auth": True, "allowed_until": "shipped"},
    "account_changes": {"require_auth": True, "require_otp": True},
    "billing": {"allow": True},
    "recharge": {"allow": True},
    "payment_declined": {"allow": True},
    "otp": {"require_auth": True},
}

# Platform-default high-value-refund threshold, used ONLY when no policy file
# declares the top-level `high_value_refund_threshold` key. A tenant declares
# its own value in policies.yaml (validated by scripts/validate_tenant.py);
# business thresholds are data, never inline literals in agent code.
DEFAULT_HIGH_VALUE_REFUND_THRESHOLD = 5000


def load_policies(path: str) -> dict:
    """Load a YAML policy file. Falls back to DEFAULT_POLICIES on error so
    a missing/broken file never crashes the agent (the audit log records it)."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict) and loaded:
            return loaded
    except FileNotFoundError:
        pass
    return dict(DEFAULT_POLICIES)


@dataclass
class PolicyContext:
    amount: float | None = None
    authenticated: bool = False
    otp_verified: bool = False
    # M6a: open-ended context signals (frustrated, frustration_level,
    # customer_tier, repeat_calls, ...) set by deterministic detectors or the
    # channel. Policies reference them via `escalate_when:` — conditions are
    # DATA (YAML), so sentiment/state-based routing needs no code change.
    signals: dict = field(default_factory=dict)


@dataclass
class Decision:
    verdict: str  # ALLOW | DENY | REQUIRE_AUTH | REQUIRE_HUMAN_APPROVAL | ESCALATE
    reasons: list[str] = field(default_factory=list)


class PolicyEngine:
    def __init__(self, policies: dict | None = None,
                 currency: str = DEFAULT_CURRENCY):
        self.policies = policies or dict(DEFAULT_POLICIES)
        self.currency = currency

    def known_actions(self) -> list[str]:
        """Action vocabulary the policy explicitly declares, via an optional
        top-level `actions:` list in the policy file. Rule keys are NOT the
        vocabulary: many supported actions have no rule (least-privilege
        DENY) and rule names can differ from action names (order_cancellation
        vs cancel_order), so an empty result means "not declared" and callers
        keep their own default list."""
        acts = self.policies.get("actions")
        if not isinstance(acts, list):
            return []
        return [a for a in acts if isinstance(a, str)]

    def high_value_refund_threshold(self) -> float:
        """The amount at/above which a refund IS a high_value_refund —
        declared per tenant as the top-level `high_value_refund_threshold`
        key in policies.yaml (business thresholds are policy data, evaluated
        through this engine so the tenant's currency is wired). A malformed
        or undeclared value falls back to the platform default, never
        crashes the turn."""
        v = self.policies.get("high_value_refund_threshold")
        if (isinstance(v, (int, float)) and not isinstance(v, bool)
                and v > 0):
            return v
        return DEFAULT_HIGH_VALUE_REFUND_THRESHOLD

    def not_found_ladder(self) -> dict | None:
        """The clarify-and-dig ladder for not-found slot lookups (Task B),
        declared per tenant as the top-level `not_found_ladder:` key in
        policies.yaml: {max_retries: int >= 1, offer_alternates: bool,
        alternates: [str, ...]} (e.g. "check the recent orders placed on this
        phone number"). Absent or malformed -> None: the pre-ladder behavior
        (the raw not-found result fed back to the brain on the first miss) is
        the default, so existing deployments keep their semantics."""
        v = self.policies.get("not_found_ladder")
        if not isinstance(v, dict) or not v:
            return None
        max_retries = v.get("max_retries", 2)
        if (not isinstance(max_retries, int)
                or isinstance(max_retries, bool) or max_retries < 1):
            return None
        alternates = v.get("alternates")
        return {
            "max_retries": max_retries,
            "offer_alternates": bool(v.get("offer_alternates", True)),
            "alternates": ([str(a) for a in alternates]
                           if isinstance(alternates, list) else []),
        }

    def evaluate(self, action: str, ctx: PolicyContext | None = None) -> Decision:
        ctx = ctx or PolicyContext()
        escalate = set(self.policies.get("escalate", []))
        if action in escalate:
            return Decision("ESCALATE", [f"action '{action}' requires human escalation"])

        policy = self.policies.get(action)
        if policy is None:
            return Decision("DENY", [f"no policy defined for action '{action}' (least privilege)"])
        if not isinstance(policy, dict):
            return Decision("ALLOW", [f"policy for '{action}' is a bare allow"])

        # M6a: data-driven conditional escalation (e.g. escalate_when:
        # {frustrated: true}) — a frustrated customer goes to a human before
        # being asked for OTP or amounts by a bot. All listed signals must
        # match the turn's context.
        escalate_when = policy.get("escalate_when")
        if (isinstance(escalate_when, dict) and escalate_when
                and all(ctx.signals.get(k) == v
                        for k, v in escalate_when.items())):
            return Decision(
                "ESCALATE",
                [f"action '{action}' escalated by condition {escalate_when}"],
            )

        if policy.get("require_auth") and not ctx.authenticated:
            return Decision("REQUIRE_AUTH", [f"action '{action}' requires customer authentication"])
        if policy.get("require_otp") and not ctx.otp_verified:
            return Decision("REQUIRE_AUTH", [f"action '{action}' requires OTP verification"])

        max_amount = policy.get("max_without_approval")
        if max_amount is not None and ctx.amount is not None and ctx.amount > max_amount:
            c = self.currency
            return Decision(
                "REQUIRE_HUMAN_APPROVAL",
                [f"amount {c}{ctx.amount:,.0f} exceeds {c}{max_amount:,.0f} without approval"],
            )

        if policy.get("escalate"):
            return Decision("ESCALATE", [f"action '{action}' configured to escalate"])
        if policy.get("allow", False):
            return Decision("ALLOW", [f"action '{action}' allowed by policy"])
        return Decision("ALLOW", [f"action '{action}' allowed"])
