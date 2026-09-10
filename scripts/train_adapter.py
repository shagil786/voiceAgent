#!/usr/bin/env python3
"""Train a per-tenant LoRA adapter for the reply-format layer.

WHY (review decision): full fine-tuning would bake one tenant's data into
shared weights and destroy the generic platform. A LoRA adapter per tenant
(base model frozen) teaches exactly what the guardrails keep compensating
for — ACTION-line format discipline and reply-language following — while
multi-tenancy stays intact: swap adapters per deployment, base untouched.

Data: JSONL rows from voiceagent.finetune_data.prepare_finetune_data
({"messages": [{role, content}]}). The adapter learns the tenant's reply
shape, never its facts (facts stay in knowledge/, retrieved at runtime).

Usage:
    .venv/bin/python scripts/train_adapter.py \
        --data data/finetune/acme.jsonl --model Qwen/Qwen2.5-0.5B-Instruct \
        --out data/adapters/acme --tenant acme --steps 200
    # smoke (no downloads, CPU seconds): covered by tests/test_train_adapter.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_RANK = 8
DEFAULT_ALPHA = 16
DEFAULT_LR = 2e-4
DEFAULT_TARGETS = ("q_proj", "v_proj")


def load_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no training rows in {path}")
    return rows


def render_messages(messages: list[dict], tokenizer) -> str:
    """Chat template when the tokenizer has one, else a plain transcript."""
    try:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        pass
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}"
                     for m in messages)


def encode_rows(rows: list[dict], tokenizer, max_len: int = 512) -> list[dict]:
    enc: list[dict] = []
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    for r in rows:
        text = render_messages(r.get("messages", []), tokenizer)
        ids = tokenizer.encode(text, add_special_tokens=True)[:max_len]
        mask = [1] * len(ids)
        pad_n = max_len - len(ids)
        enc.append({"input_ids": ids + [pad] * pad_n,
                    "attention_mask": mask + [0] * pad_n,
                    "labels": ids + [-100] * pad_n})
    return enc


def train(data: str | Path, model_id: str, out_dir: str | Path, *,
          tenant: str = "default", rank: int = DEFAULT_RANK,
          alpha: int = DEFAULT_ALPHA, lr: float = DEFAULT_LR,
          max_steps: int = 200, target_modules=None,
          model=None, tokenizer=None) -> dict:
    """Full loop, injectable model/tokenizer for hermetic tests. Returns
    run stats + writes the adapter (safetensors + card) to out_dir."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    rows = load_rows(data)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id)
    target_modules = list(target_modules or DEFAULT_TARGETS)
    cfg = LoraConfig(r=rank, lora_alpha=alpha,
                     target_modules=target_modules,
                     lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    dataset = encode_rows(rows, tokenizer)
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=lr,
        max_steps=max_steps, logging_steps=10, save_steps=max_steps,
        save_total_limit=1, report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    data_hash = hashlib.sha256(
        Path(data).read_bytes()).hexdigest()[:16]
    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    card = {"tenant": tenant, "base_model": model_id,
            "rank": rank, "alpha": alpha, "lr": lr,
            "max_steps": max_steps, "target_modules": target_modules,
            "rows": len(rows), "data_sha256": data_hash,
            "trainable_params": n_trainable}
    (out / "adapter_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="finetune JSONL rows")
    ap.add_argument("--model", required=True, help="base causal-LM id")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--alpha", type=int, default=DEFAULT_ALPHA)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    args = ap.parse_args()
    card = train(args.data, args.model, args.out, tenant=args.tenant,
                 rank=args.rank, alpha=args.alpha, lr=args.lr,
                 max_steps=args.steps, target_modules=args.targets)
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
