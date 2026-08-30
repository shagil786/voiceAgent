from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field

DATASET_SCHEMA = [
    "id", "language", "intent", "user_text", "expected_action",
    "key_facts", "escalate",
]

INTENTS = [
    "order_status", "refund", "cancel_order", "address_change",
    "payment_declined", "recharge", "billing", "return", "replacement",
    "otp", "fraud", "account_closure", "delivery_delay", "product_info",
    "invoice", "plan_change", "roaming", "network_issue", "complaint",
    "high_value_refund",
]

# (language, intent, user_text, expected_action, key_facts, escalate)
_SEED_TEMPLATES = [
    ("en", "order_status", "Where is my order #ORD-77812?",
     "order_status", ["ORD-77812"], False),
    ("en", "refund", "I need a refund for order #ORD-22109.",
     "refund", ["ORD-22109"], False),
    ("en", "payment_declined", "Why was my payment declined?",
     "payment_declined", ["declined"], False),
    ("hinglish", "order_status", "Bhai mera order abhi tak nahi aaya, order id ORD-55671 hai.",
     "order_status", ["ORD-55671"], False),
    ("hinglish", "refund", "Actually can you refund my order, order #ORD-99032?",
     "refund", ["ORD-99032"], False),
    ("hi", "recharge", "मेरा recharge क्यों fail हुआ?",
     "recharge", ["fail"], False),
    ("hi", "billing", "मुझे अपना bill समझ नहीं आया।",
     "billing", ["bill"], False),
    ("en", "high_value_refund", "I want a refund of ₹25,000 for order #ORD-11223.",
     "high_value_refund", ["ORD-11223"], True),
    ("en", "fraud", "Someone used my account. Block it now.",
     "fraud", ["block"], True),
    ("hinglish", "otp", "OTP nahi aaya mere phone pe, resend karo.",
     "otp", ["otp"], False),
]

def _mutate(text: str, rng: random.Random) -> str:
    """Return the seed text as-is; templates already cover variation.
    Kept as a hook for later augmentation without changing the schema."""
    return text

def generate_eval_set(out_path: str, n: int = 1000, seed: int = 42) -> int:
    rng = random.Random(seed)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_SCHEMA)
        for i in range(n):
            lang, intent, text, action, facts, escalate = rng.choice(_SEED_TEMPLATES)
            order_id = f"ORD-{rng.randint(10000, 99999)}"
            text = text.replace("ORD-77812", order_id)
            facts = [order_id if f == "ORD-77812" else f for f in facts]
            writer.writerow([
                f"conv-{i:04d}", lang, intent, _mutate(text, rng),
                action, "|".join(facts), escalate,
            ])
    return n

def load_conversations(path: str) -> list["Conversation"]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(Conversation(
                id=row["id"], language=row["language"], intent=row["intent"],
                user_text=row["user_text"], expected_action=row["expected_action"],
                key_facts=[k for k in row["key_facts"].split("|") if k],
                escalate=row["escalate"].lower() == "true",
            ))
    return out

@dataclass
class Conversation:
    id: str
    language: str
    intent: str
    user_text: str
    expected_action: str
    key_facts: list[str] = field(default_factory=list)
    escalate: bool = False