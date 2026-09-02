# scripts/measure_qwen_asr.py — M5b-2 ASR bake-off: Qwen3-ASR-0.6B.
"""Runs the M5b-1 loopback set (en/hi/te x 5 + the hinglish demo query,
ground truth known — we synthesized them) through Qwen/Qwen3-ASR-0.6B-hf
(Apache-2.0, 52 languages, LLM-decoder ASR from the Qwen3-Omni family) and
reports corpus WER + median latency per language, with the detected language
per sample. Compare with data/out/asr-measurement.json (whisper small) and
data/out/asr-measurement-indic.json (IndicConformer).

Qwen3-ASR needs transformers>=5.13 (the project venv pins 4.46), so this
runs in its OWN venv — zero risk to the project environment:

  python3 -m venv /tmp/venv-asr-bakeoff
  /tmp/venv-asr-bakeoff/bin/pip install -q "transformers>=5.13.0" torch librosa
  /tmp/venv-asr-bakeoff/bin/python scripts/measure_qwen_asr.py   # from repo root

No language hint is forced: the model does its own LID (the "parsed" decode
returns it) — mirrors the blind production voice path.
Writes data/out/asr-measurement-qwen.json.
"""
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO / "data/out/tts-m5b-samples"
M5B1_JSON = REPO / "data/out/asr-measurement.json"
OUT_JSON = REPO / "data/out/asr-measurement-qwen.json"
MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"


def normalize(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def wer_edits(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int]:
    prev = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words, 1):
        cur = [i] + [0] * len(hyp_words)
        for j, hw in enumerate(hyp_words, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(ref_words)


def _build_items() -> list[dict]:
    src = json.loads(M5B1_JSON.read_text())
    items = []
    for lang in ("en", "hi", "te"):
        for e in src["audio"][lang]:
            items.append({"lang": lang, "i": e["i"],
                          "wav": str(SAMPLES_DIR / f"{lang}_{e['i']}.wav"),
                          "ref": e["text"]})
    hing = REPO / "data/out/voice-m5b2/query-hinglish.wav"
    if hing.exists():
        items.append({"lang": "hinglish", "i": 0, "wav": str(hing),
                      "ref": "Bhai mera order abhi tak nahi aaya, order id ORD-55671 hai"})
    return items


def main() -> int:
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    print(f"loading {MODEL_ID} (float32, CPU) ...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID,
                                                     torch_dtype=torch.float32)
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    def transcribe(path: str) -> tuple[str, str | None]:
        inputs = processor.apply_transcription_request(audio=path)
        inputs = inputs.to(model.device, model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        try:
            parsed = processor.decode(gen[0], return_format="parsed")
            return parsed.get("transcription", "").strip(), parsed.get("language")
        except Exception:
            return processor.decode(gen[0], skip_special_tokens=True).strip(), None

    items = _build_items()
    result: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "caveat": ("LOOPBACK: piper TTS audio (clean, no mic/channel noise) — "
                   "WER is optimistic; compare engines relative to each other. "
                   "M5b-1 whisper-small reference: en 0.038, hi 0.423, te 1.067."),
        "wer_metric": "corpus WER per lang: sum(edits)/sum(ref words)",
        "latency_metric": "median wall time of the single WER pass per utterance",
        "samples": [],
        "paths": {},
    }

    langs = sorted({it["lang"] for it in items})
    edits = {l: 0 for l in langs}
    ref_words = {l: 0 for l in langs}
    per_ms = {l: [] for l in langs}

    for it in items:
        t0 = time.perf_counter()
        hyp, detected = transcribe(it["wav"])
        ms = (time.perf_counter() - t0) * 1000
        e, n = wer_edits(normalize(it["ref"]).split(), normalize(hyp).split())
        edits[it["lang"]] += e
        ref_words[it["lang"]] += n
        per_ms[it["lang"]].append(ms)
        result["samples"].append({**it, "hyp": hyp, "detected": detected,
                                  "wer": round(e / n, 4) if n else None,
                                  "latency_s": round(ms / 1000, 3)})
        print(f"  qwen {it['lang']}#{it['i']}: {ms/1000:.2f}s det={detected} "
              f"wer={result['samples'][-1]['wer']} hyp={hyp!r}")

    result["paths"]["qwen3-asr-0.6b"] = {
        "wer": {l: round(edits[l] / ref_words[l], 4) for l in langs if ref_words[l]},
        "median_latency_s": {l: round(statistics.median(per_ms[l]) / 1000, 3)
                             for l in langs if per_ms[l]},
    }

    q = result["paths"]["qwen3-asr-0.6b"]
    rows = ["| engine | en WER | hi WER | te WER | hinglish WER |",
            "|---|---|---|---|---|",
            "| whisper small (M5b-1) | 0.038 | 0.423 | 1.067 | — |",
            f"| Qwen3-ASR-0.6B | {q['wer'].get('en', float('nan')):.3f} | "
            f"{q['wer'].get('hi', float('nan')):.3f} | "
            f"{q['wer'].get('te', float('nan')):.3f} | "
            f"{q['wer'].get('hinglish', float('nan')):.3f} |"]
    print("\n" + "\n".join(rows))
    print("\nmedian latency (s): " + json.dumps(q["median_latency_s"]))
    print(result["caveat"])

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
