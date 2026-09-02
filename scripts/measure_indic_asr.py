# scripts/measure_indic_asr.py — M5b-2: IndicConformer loopback measurement.
"""Measures the M5b-1 loopback set (data/out/tts-m5b-samples) through the
KNOWN-language routed path — get_asr_for_language('te') -> IndicConformer —
and reports whether IndicConformer actually fixes the Telugu failure that
motivated it (whisper te WER >= 1.0 at every size, M5b-1).

The blind voice-loop path (language unknown before ASR) deliberately does
NOT auto-reroute: whisper's detection cannot separate Hinglish from native
Indic audio (hinglish demo query detected as te @0.74 / pa @0.74; a REAL te
sample detected at only 0.69 — see voiceagent.asr). te/ta are served when
the deployment knows its language (telephony trunk config); en/hi reference
numbers come from data/out/asr-measurement.json (whisper small).

CAVEAT (inherited from M5b-1): loopback = piper TTS audio, clean, no mic or
channel noise — WER is optimistic; the table compares paths relative to
each other. Tamil has no piper voice (M5b-1), so no ta loopback audio.

IndicConformer downloads ~2.4 GB from HF on first use AND IS GATED: accept
the license at https://huggingface.co/ai4bharat/indic-conformer-600m-
multilingual, then authenticate (huggingface-cli login or HF_TOKEN).
Run from the repo root. Writes data/out/asr-measurement-indic.json.
"""
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_asr import normalize, wer_edits  # noqa: E402

OUT_JSON = Path("data/out/asr-measurement-indic.json")
M5B1_JSON = Path("data/out/asr-measurement.json")
SAMPLES_DIR = Path("data/out/tts-m5b-samples")


def _load_refs_and_files() -> list[dict]:
    """(lang, i, ref, wav) for the te loopback samples, refs from M5b-1 json."""
    src = json.loads(M5B1_JSON.read_text())
    items = []
    for e in src["audio"]["te"]:
        wav = SAMPLES_DIR / f"te_{e['i']}.wav"
        assert wav.exists(), f"missing loopback sample {wav}"
        items.append({"lang": "te", "i": e["i"], "wav": str(wav),
                      "ref": e["text"]})
    return items


def main() -> int:
    from voiceagent.asr import get_asr_for_language

    items = _load_refs_and_files()
    result: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "caveat": ("LOOPBACK: piper TTS audio (clean, no mic/channel noise) — "
                   "WER is optimistic; compare paths relative to each other. "
                   "M5b-1 whisper-small reference: en 0.038, hi 0.423, te 1.067."),
        "wer_metric": "corpus WER: sum(edits)/sum(ref words)",
        "latency_metric": "median wall time of the single WER pass per utterance",
        "paths": {},
    }

    print("indic-direct: get_asr_for_language('te') -> IndicConformer "
          "(~2.4 GB gated download on first use)")
    handle = get_asr_for_language("te")
    samples = []
    edits, ref_words, per_ms = 0, 0, []
    for it in items:
        t0 = time.perf_counter()
        hyp = handle.transcribe(it["wav"], language="te")
        ms = (time.perf_counter() - t0) * 1000
        e, n = wer_edits(normalize(it["ref"]).split(), normalize(hyp).split())
        edits += e
        ref_words += n
        per_ms.append(ms)
        samples.append({**it, "hyp": hyp,
                        "wer": round(e / n, 4) if n else None,
                        "latency_s": round(ms / 1000, 3)})
        print(f"  indic te#{it['i']}: {ms/1000:.2f}s "
              f"wer={samples[-1]['wer']} hyp={hyp!r}")
    result["paths"]["indic-direct"] = {
        "wer": {"te": round(edits / ref_words, 4)},
        "median_latency_s": round(statistics.median(per_ms) / 1000, 3),
        "samples": samples,
    }

    m5b1 = json.loads(M5B1_JSON.read_text())["models"]["small"]["wer"]
    ind = result["paths"]["indic-direct"]
    rows = [
        "| path | te WER | median latency (s) |",
        "|---|---|---|",
        f"| whisper small (M5b-1, broken) | {m5b1['te']:.3f} | 27.90 |",
        f"| IndicConformer known-lang (M5b-2) | {ind['wer']['te']:.3f} | "
        f"{ind['median_latency_s']:.2f} |",
    ]
    print("\n" + "\n".join(rows))
    print(result["caveat"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
