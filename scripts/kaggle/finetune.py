"""LoRA fine-tune for VoiceAgent (run on Kaggle GPU, NOT locally).

Targets: Qwen2.5-0.5B-Instruct or Qwen3-0.6B. Trains the model to reply
grounded + emit the ACTION line. Output: a merged HF weights dir you convert
to GGUF with llama.cpp.

Example (Kaggle notebook / terminal with GPU):
    pip install -q transformers peft trl datasets accelerate bitsandbytes
    python finetune.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --data finetune_data.jsonl --out merged
"""
import argparse
import json

from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="finetune_data.jsonl")
    ap.add_argument("--out", default="merged")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    ds = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype="float16")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj",
                      "v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    def fmt(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(fmt)

    trainer = SFTTrainer(
        model=model, args=TrainingArguments(
            output_dir="./ft", num_train_epochs=args.epochs,
            per_device_train_batch_size=2, learning_rate=args.lr,
            logging_steps=5, save_steps=100, fp16=True),
        train_dataset=ds, tokenizer=tokenizer, max_seq_length=1024,
        dataset_text_field="text")
    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"merged model saved to {args.out} — convert to GGUF with llama.cpp")


if __name__ == "__main__":
    main()
