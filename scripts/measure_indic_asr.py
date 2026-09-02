# scripts/measure_indic_asr.py — M5b-2: routed-ASR loopback measurement.
"""Re-measures the M5b-1 loopback set (data/out/tts-m5b-samples) through the
M5b-2 production auto path and reports whether IndicConformer actually fixes
the Telugu failure that motivated it (whisper te WER >= 1.0 at every size).

Paths measured:
- auto — transcribe_wav_auto(wav): the blind production voice-loop entry.
  One whisper-small pass auto-detects; detected te skips whisper's decode and
  re-transcribes with IndicConformer. en/hi decode on whisper small.
- indic-direct — get_asr_for_language("te").transcribe(wav, language="te"):
  the known-language Indic path (telephony-style), engine-only baseline.

Compare against data/out/asr-measurement.json (whisper tiny/base/small).

CAVEAT (inherited from M5b-1): loopback = piper TTS audio, clean, no mic or
channel noise — WER is optimistic; the table compares paths relative to each
other. Tamil has no piper voice (M5b-1), so no ta loopback audio exists.

IndicConformer downloads ~2.4 GB from HF on first use and runs on CPU.
Run from the repo root. Writes data/out/asr-measurement-indic.json.
"""
import json
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_asr import LANGS, SAMPLES_DIR, normalize, wer_edits  # noqa: E402

OUT_JSON = Path("data/out/asr-measurement-indic.json")
SAMPLES_REPS = 0  # extra timed reps per utterance; the WER pass is timed too


def main() -> int:
    from voiceagent.asr import get_asr_for_language, transcribe_wav_auto

    items = []
    for lang in LANGS:
        for wav in sorted(SAMPLES_DIR.glob(f"{lang}_*.wav")):
            with wave.open(str(wav), "rb") as w:
                dur = w.getnframes() / float(w.getframerate())
            i = int(wav.stem.split("_")[1])
            items.append({"lang": lang, "i": i, "wav": str(wav),
                          "ref": SENTENCE_OF[(lang, i)], "dur_s": round(dur, 3)})
    assert len(items) == 15, f"expected 15 loopback samples, found {len(items)}"

    result: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "caveat": ("LOOPBACK: piper TTS audio (clean, no mic/channel noise) — "
                   "WER is optimistic; compare paths relative to each other. "
                   "M5b-1 whisper-small reference: en 0.038, hi 0.423, te 1.067."),
        "wer_metric": "corpus WER per (path, lang): sum(edits)/sum(ref words)",
        "latency_metric": "median wall time of the single WER pass per utterance",
        "paths": {},
    }

    # --- auto path (production): en/hi/te --------------------------------
    print("auto path: whisper small auto-detect + IndicConformer reroute")
    samples = []
    edits = {l: 0 for l in LANGS}
    ref_words = {l: 0 for l in LANGS}
    per_lang_ms = {l: [] for l in LANGS}
    for it in items:
        t0 = time.perf_counter()
        hyp = transcribe_wav_auto(it["wav"])
        ms = (time.perf_counter() - t0) * 1000
        e, n = wer_edits(normalize(it["ref"]).split(), normalize(hyp).split())
        edits[it["lang"]] += e
        ref_words[it["lang"]] += n
        per_lang_ms[it["lang"]].append(ms)
        samples.append({**it, "hyp": hyp,
                        "wer": round(e / n, 4) if n else None,
                        "latency_s": round(ms / 1000, 3)})
        print(f"  auto {it['lang']}#{it['i']}: {ms/1000:.2f}s "
              f"wer={samples[-1]['wer']} hyp={hyp!r}")
    result["paths"]["auto"] = {
        "wer": {l: round(edits[l] / ref_words[l], 4) for l in LANGS},
        "median_latency_s": {l: round(statistics.median(per_lang_ms[l]) / 1000, 3)
                             for l in LANGS},
        "samples": samples,
    }

    # --- indic-direct (te only; en/hi never touch the conformer) ---------
    print("indic-direct: get_asr_for_language('te') -> IndicConformer")
    handle = get_asr_for_language("te")
    samples = []
    edits, ref_words, per_ms = 0, 0, []
    for it in items:
        if it["lang"] != "te":
            continue
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
        "median_latency_s": {"te": round(statistics.median(per_ms) / 1000, 3)},
        "samples": samples,
    }

    # markdown table ------------------------------------------------------
    auto = result["paths"]["auto"]
    ind = result["paths"]["indic-direct"]
    rows = [
        "| path | en WER | hi WER | te WER | median latency (s) |",
        "|---|---|---|---|---|",
        f"| auto (whisper small + reroute) | {auto['wer']['en']:.3f} | "
        f"{auto['wer']['hi']:.3f} | {auto['wer']['te']:.3f} | — |",
        f"| auto te-only latency | — | — | — | {auto['median_latency_s']['te']:.2f} |",
        f"| indic-direct (te, known lang) | — | — | {ind['wer']['te']:.3f} | "
        f"{ind['median_latency_s']['te']:.2f} |",
    ]
    print("\n" + "\n".join(rows))
    print("\nen/hi median via auto: "
          + json.dumps({l: auto["median_latency_s"][l] for l in ("en", "hi")}))
    print(result["caveat"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_JSON}")
    return 0


def _load_refs() -> dict:
    """Reference texts from the M5b-1 measurement json (same samples)."""
    src = json.loads(Path("data/out/asr-measurement.json").read_text())
    refs = {}
    for lang, entries in src["audio"].items():
        for e in entries:
            refs[(lang, e["i"])] = e["text"]
    return refs


SENTENCE_OF = _load_refs()

if __name__ == "__main__":
    raise SystemExit(main())
