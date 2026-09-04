# src/voiceagent/finetune_data.py
from __future__ import annotations

import csv
import json

SYSTEM = (
    # Neutral persona, matching the runtime default in agent.py. NOTE: the
    # existing Kaggle fine-tuned model was trained on the OLD persona text
    # ("...for an Indian ecommerce company"); future fine-tunes use this.
    "You are a customer support assistant. "
    "Answer directly and concisely. Answer ONLY from the provided context. "
    "Always address the customer's specific reference (order id) in your reply. "
    "If the request requires an action, end your reply with a line: "
    "ACTION: <action_name>."
)


def synthesize_reply(key_facts: list[str], expected_action: str) -> str:
    ref = next((f for f in key_facts if f.startswith("ORD-")), None)
    if ref:
        body = f"Your request regarding {ref} is being handled."
    else:
        body = f"Your request regarding {expected_action} is being handled."
    return f"{body}\nACTION: {expected_action}"


def prepare_finetune_data(csv_path: str, out_jsonl: str) -> int:
    n = 0
    with open(csv_path, newline="", encoding="utf-8") as f, \
         open(out_jsonl, "w", encoding="utf-8") as out:
        for row in csv.DictReader(f):
            facts = [k for k in row.get("key_facts", "").split("|") if k]
            action = row.get("expected_action", "order_status")
            assistant = synthesize_reply(facts, action)
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row.get("user_text", "")},
                {"role": "assistant", "content": assistant},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n += 1
    return n
