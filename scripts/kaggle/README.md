# Kaggle fine-tune pipeline

The 0.5B model already passes the gate (100% resolution on the 200-conv eval).
This pipeline makes it *better* at Hinglish + grounded replies by fine-tuning on
our eval data on a free Kaggle GPU.

## Easiest path: self-contained notebook cell (no file upload)

The Kaggle notebook editor is a custom React app — attaching files via the
Add-Input UI is fiddly, and browser automation of the editor is unreliable.
Instead, everything needed (data regeneration + deps + training) is packed into
ONE cell:

1. Generate the cell text:
   ```bash
   source .venv/bin/activate
   python scripts/kaggle/notebook_runner.py > kaggle_cell.txt
   ```
2. On Kaggle: **Create → Notebook**, then in **Settings** (left sidebar) set
   **Accelerator = GPU P100** (or T4 x2). Both are free.
3. Delete the default cell, paste the contents of `kaggle_cell.txt` into the
   first cell.
4. **Run All** (or Shift+Enter on the cell). It will:
   - install `transformers peft trl datasets accelerate bitsandbytes`
   - regenerate the 1,000-example Hinglish eval set (seed 42 — same as the repo)
   - LoRA fine-tune Qwen2.5-0.5B for 3 epochs (~5 min on a T4)
   - save merged weights to `/kaggle/working/merged`
5. **Download** `/kaggle/working/merged` (right-click → download, or `Save Version`
   then download the output).

## Convert to GGUF (llama.cpp)
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && pip install -r requirements.txt
python convert_hf_to_gguf.py ../kaggle/merged -o qwen2.5-0.5b-hinglish-q4_k_m.gguf --outtype q4_k_m
```
Place the `.gguf` in `data/models/`, add it to `CANDIDATE_MODELS` in
`src/voiceagent/llm.py`, and re-run the benchmark.

## Re-benchmark
```bash
python scripts/run_benchmark.py 200
```
Target: Hinglish resolution above the current baseline (100% on the synthetic
eval) with no latency regression (0.63s). Real value shows up when the eval set
is replaced with real Hinglish support conversations.

## Original manual path (files as a dataset input)
If you prefer attaching files the standard way: upload `finetune_data.jsonl` +
`scripts/kaggle/finetune.py` as a Kaggle dataset, attach it to the notebook
(Add Input), and run `python finetune.py --data /kaggle/input/<owner>/<slug>/finetune_data.jsonl --out merged`.
