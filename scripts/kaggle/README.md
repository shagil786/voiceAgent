# Kaggle fine-tune pipeline

The 0.5B model already passes the gate (91% resolution). This pipeline makes
it *better* at Hinglish + grounded replies by fine-tuning on our eval data on
a free Kaggle GPU.

## 1. Prepare the data (run locally)
    source .venv/bin/activate
    python -c "import sys; sys.path.insert(0,'src'); from voiceagent.finetune_data import prepare_finetune_data; print(prepare_finetune_data('data/eval/conversations.csv','scripts/kaggle/finetune_data.jsonl'))"

## 2. Train on Kaggle (GPU)
Upload `finetune_data.jsonl` and `scripts/kaggle/finetune.py` to a Kaggle
notebook with GPU P100/T4 accelerator (free), then run:
    pip install -q transformers peft trl datasets accelerate bitsandbytes
    python finetune.py --model Qwen/Qwen2.5-0.5B-Instruct --data finetune_data.jsonl --out merged
Training ~500-1000 samples on 0.5B takes a few minutes on a T4.

## 3. Convert to GGUF (llama.cpp)
    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp && pip install -r requirements.txt
    python convert_hf_to_gguf.py ../scripts/kaggle/merged -o qwen2.5-0.5b-hinglish-q4_k_m.gguf --outtype q4_k_m
Place the .gguf in data/models/, add it to CANDIDATE_MODELS in
src/voiceagent/llm.py, and re-run the benchmark to compare.

## 4. Re-benchmark
    python scripts/run_benchmark.py 200
Target: Hinglish resolution above the current ~91% baseline with no latency
regression (0.42s).
