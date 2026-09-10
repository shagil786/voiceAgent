"""Adapter training loop (scripts/train_adapter.py): hermetic smoke — a tiny
random GPT-2 + stub tokenizer proves data prep, LoRA wrapping, a training
step, save/load, and the adapter card without downloads or GPUs."""
import json

import torch


class StubTok:
    """Whitespace tokenizer: encode/decode + pad/eos ids. No downloads."""
    def __init__(self):
        self._v = {"<pad>": 0, "<eos>": 1}
    @property
    def pad_token_id(self):
        return 0
    @property
    def eos_token_id(self):
        return 1
    def encode(self, text, add_special_tokens=True):
        ids = []
        for w in text.split():
            if w not in self._v:
                self._v[w] = len(self._v)
            ids.append(self._v[w])
        return ids + ([1] if add_special_tokens else [])
    def __len__(self):
        return max(128, len(self._v))


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=128)
    return GPT2LMHeadModel(cfg)


def _rows(tmp_path):
    p = tmp_path / "rows.jsonl"
    rows = [
        {"messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Where is my order ORD-1?"},
            {"role": "assistant",
             "content": "Your order ORD-1 ships today.\nACTION: order_status"}]},
        {"messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Refund ORD-2 please."},
            {"role": "assistant",
             "content": "Refund for ORD-2 noted.\nACTION: initiate_refund"}]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                 encoding="utf-8")
    return p


def test_train_loop_end_to_end(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from train_adapter import encode_rows, load_rows, train
    data = _rows(tmp_path)
    assert len(load_rows(data)) == 2
    tok = StubTok()
    enc = encode_rows(load_rows(data), tok, max_len=32)
    assert len(enc[0]["input_ids"]) == 32
    assert enc[0]["labels"][0] != -100  # real tokens, padded tail masked
    card = train(data, "tiny-random-gpt2", tmp_path / "adapter",
                 tenant="smoke", rank=4, max_steps=2,
                 target_modules=["c_attn"],
                 model=_tiny_model(), tokenizer=tok)
    assert card["tenant"] == "smoke" and card["rows"] == 2
    assert card["trainable_params"] > 0
    assert (tmp_path / "adapter" / "adapter_card.json").exists()
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    # The saved adapter reloads onto the base and runs.
    from peft import PeftModel
    base = _tiny_model()
    m = PeftModel.from_pretrained(base, str(tmp_path / "adapter"))
    m.eval()
    with torch.no_grad():
        out = m(torch.tensor([[1, 2, 3]]))
    assert out.logits.shape[-1] == 128


def test_empty_data_refuses(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from train_adapter import load_rows
    import pytest
    p = tmp_path / "empty.jsonl"
    p.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no training rows"):
        load_rows(p)
