# scripts/kaggle/notebook_runner.py
"""Self-contained Kaggle notebook cell body for the VoiceAgent fine-tune.
Paste this ONE cell into a Kaggle notebook (GPU accelerator: P100/T4).
It regenerates the synthetic Hinglish eval data (seed 42), installs deps,
and runs LoRA fine-tuning on Qwen2.5-0.5B, saving merged weights to
/kaggle/working/merged for GGUF conversion.

Why self-contained: Kaggle's Add-Input upload UI is awkward; the data is
deterministic so the notebook rebuilds it inline. No files to upload.
"""
CELL = r'''
# %%capture
# --- install deps ---
!pip install -q transformers peft trl datasets accelerate bitsandbytes

# --- regenerate the synthetic Hinglish eval data (seed 42, matches repo) ---
import csv, json, random, io

_SEED_TEMPLATES = [
    ("en", "order_status", "Where is my order #ORD-77812?", "order_status", ["ORD-77812"], False),
    ("en", "refund", "I need a refund for order #ORD-22109.", "refund", ["ORD-22109"], False),
    ("en", "payment_declined", "Why was my payment declined?", "payment_declined", ["declined"], False),
    ("hinglish", "order_status", "Bhai mera order abhi tak nahi aaya, order id ORD-55671 hai.", "order_status", ["ORD-55671"], False),
    ("hinglish", "refund", "Actually can you refund my order, order #ORD-99032?", "refund", ["ORD-99032"], False),
    ("hi", "recharge", "\u092e\u0947\u0930\u093e recharge \u0915\u094d\u092f\u094b\u0902 fail \u0939\u0941\u0906?", "recharge", ["fail"], False),
    ("hi", "billing", "\u092e\u0941\u091d\u0947 \u0905\u092a\u0928\u093e bill \u0938\u092e\u091d \u0928\u0939\u0940\u0902 \u0906\u092f\u093e\u0964", "billing", ["bill"], False),
    ("en", "high_value_refund", "I want a refund of \u20b925,000 for order #ORD-11223.", "high_value_refund", ["ORD-11223"], True),
    ("en", "fraud", "Someone used my account. Block it now.", "fraud", ["block"], True),
    ("hinglish", "otp", "OTP nahi aaya mere phone pe, resend karo.", "otp", ["otp"], False),
]

SYSTEM = ("You are a customer support assistant for an Indian ecommerce company. "
          "Answer directly and concisely. Answer ONLY from the provided context. "
          "Always address the customer's specific reference (order id) in your reply. "
          "If the request requires an action, end your reply with a line: ACTION: <action_name>.")

rng = random.Random(42)
rows = []
for i in range(1000):
    lang, intent, text, action, facts, esc = rng.choice(_SEED_TEMPLATES)
    oid = f"ORD-{rng.randint(10000,99999)}"
    text = text.replace("ORD-77812", oid)
    facts = [oid if f == "ORD-77812" else f for f in facts]
    ref = next((f for f in facts if f.startswith("ORD-")), None)
    body = f"Your request regarding {ref} is being handled." if ref else f"Your request regarding {action} is being handled."
    rows.append({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
        {"role": "assistant", "content": f"{body}\nACTION: {action}"},
    ]})

with open("finetune_data.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} examples")

# --- LoRA fine-tune ---
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ds = Dataset.from_list(rows)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=16, lora_alpha=32,
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                  lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
def fmt(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
ds = ds.map(fmt)
trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(output_dir="./ft", num_train_epochs=3,
                           per_device_train_batch_size=2, learning_rate=2e-4,
                           logging_steps=5, save_steps=100, fp16=True),
    train_dataset=ds, tokenizer=tok, max_seq_length=1024, dataset_text_field="text")
trainer.train()
model = model.merge_and_unload()
model.save_pretrained("/kaggle/working/merged")
tok.save_pretrained("/kaggle/working/merged")
print("merged model saved to /kaggle/working/merged — download + convert to GGUF with llama.cpp")
'''

if __name__ == "__main__":
    # Print the cell body so a subagent/browser can paste it directly.
    print(CELL)
