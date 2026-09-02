# scripts/measure_asr.py — M5b-1: loopback ASR measurement.
"""Loopback ASR measurement to pick a model size for the voice path.

Builds a small loopback set — 5 fixed support utterances each for en, hi, te
(some carrying an order id) — synthesizes them with the multilingual
TTSHandle (so ground-truth text is known), transcribes with faster-whisper
tiny/base/small, and reports WER per (model, language) plus transcription
latency (median of 3) per model.

CAVEAT: loopback = piper TTS output (clean 22.05 kHz audio, no mic, no room
noise). WER here is optimistic; the table is for comparing models relative
to each other, not absolute production accuracy.

Models download from HF on first use (small is ~480 MB). Run from the repo
root. Writes data/out/asr-measurement.json and prints a compact table.
"""
import json
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.tts import TTSHandle  # noqa: E402

OUT_JSON = Path("data/out/asr-measurement.json")
SAMPLES_DIR = Path("data/out/tts-m5b-samples")
MODELS = ("tiny", "base", "small")
LANGS = ("en", "hi", "te")
LATENCY_REPS = 3  # timed transcriptions per sentence, median taken

SENTENCES = {
    "en": [
        "Hello, I want to check the status of my order ORD-4821.",
        "My payment failed but the money was deducted from my account.",
        "When will my refund be credited back to me?",
        "I need to change the delivery address for my order ORD-7734.",
        "Please connect me to a human agent right away.",
    ],
    "hi": [
        "नमस्ते, मुझे अपने ऑर्डर ORD-4821 का स्टेटस चेक करना है।",
        "मेरा पेमेंट फेल हो गया लेकिन पैसे मेरे खाते से कट गए हैं।",
        "मेरा रिफंड कब मेरे खाते में आएगा?",
        "मुझे ऑर्डर ORD-7734 के लिए डिलीवरी का पता बदलना है।",
        "कृपया मुझे तुरंत किसी वास्तविक एजेंट से जोड़ें।",
    ],
    "te": [
        "నమస్కారం, నా ఆర్డర్ ORD-4821 స్టేటస్ నాకు తెలియాలి.",
        "నా పేమెంట్ ఫెయిల్ అయింది కానీ డబ్బు నా ఖాతా నుండి కట్ అయింది.",
        "నా రీఫండ్ ఎప్పుడు నా ఖాతాలో వస్తుంది?",
        "నా ఆర్డర్ ORD-7734 కోసం డెలివరీ చిరునామా మార్చాలి.",
        "దయచేసి నన్ను వెంటనే ఒక నిజమైన ఏజెంట్‌కి కనెక్ట్ చేయండి.",
    ],
}


def normalize(text: str) -> str:
    """Lowercase, keep alphanumerics only (so 'ORD-4821' == 'ord 4821' and
    script punctuation drops out), collapse whitespace."""
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def wer_edits(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int]:
    """(edit_distance, ref_word_count) via word-level Levenshtein (stdlib)."""
    prev = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words, 1):
        cur = [i] + [0] * len(hyp_words)
        for j, hw in enumerate(hyp_words, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(ref_words)


def transcribe(model, wav_path: str) -> tuple[str, str]:
    """Run faster-whisper, return (text, detected_language). No language hint:
    mirrors the production voice path, which must detect language from audio."""
    segments, info = model.transcribe(wav_path)
    return " ".join(s.text for s in segments).strip(), info.language


def synthesize_samples() -> dict[str, list[dict]]:
    """Synthesize the loopback set once; record wav bytes + duration per file."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    handle = TTSHandle()
    audio = {}
    for lang, texts in SENTENCES.items():
        entries = []
        for i, text in enumerate(texts):
            wav = SAMPLES_DIR / f"{lang}_{i}.wav"
            handle.speak(text, language=lang, out_path=str(wav))
            with wave.open(str(wav), "rb") as w:
                dur = w.getnframes() / float(w.getframerate())
            entries.append({"i": i, "wav": str(wav), "text": text,
                            "bytes": wav.stat().st_size, "dur_s": round(dur, 3)})
        audio[lang] = entries
    return audio


def main() -> int:
    print("synthesizing loopback set (en/hi/te x 5) ...")
    audio = synthesize_samples()

    # flatten the set, keeping (lang, i, ref, wav_path)
    items = [(lang, e["i"], e["text"], e["wav"])
             for lang in LANGS for e in audio[lang]]

    result: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "caveat": ("LOOPBACK: piper TTS audio (clean, no mic/channel noise) — "
                   "WER is optimistic; compare models relative to each other."),
        "wer_metric": "corpus WER per (model, lang): sum(edits)/sum(ref words)",
        "latency_metric": f"median of {LATENCY_REPS} timed transcriptions",
        "audio": audio,
        "models": {},
    }

    for model_name in MODELS:
        print(f"loading faster-whisper '{model_name}' ...")
        from faster_whisper import WhisperModel  # import per model: big first load
        model = WhisperModel(model_name, device="cpu", compute_type="int8")

        edits: dict[str, int] = {l: 0 for l in LANGS}
        ref_words: dict[str, int] = {l: 0 for l in LANGS}
        per_sentence_ms: dict[tuple[str, int], list[float]] = {}
        samples = []

        for lang, i, ref, wav in items:
            hyp, det = transcribe(model, wav)  # WER pass (also warms caches)
            e, n = wer_edits(normalize(ref).split(), normalize(hyp).split())
            edits[lang] += e
            ref_words[lang] += n
            samples.append({"lang": lang, "i": i, "ref": ref, "hyp": hyp,
                            "detected_lang": det,
                            "wer": round(e / n, 4) if n else None})
            # timed passes (same utterance repeats fine — deterministic audio)
            times = []
            for _ in range(LATENCY_REPS):
                t0 = time.perf_counter()
                transcribe(model, wav)
                times.append((time.perf_counter() - t0) * 1000)
            per_sentence_ms[(lang, i)] = times

        wer = {l: round(edits[l] / ref_words[l], 4) for l in LANGS}
        lang_med_ms = {l: statistics.median(
            ms for (lang, _), lst in per_sentence_ms.items() if lang == l
            for ms in lst) for l in LANGS}
        all_ms = [ms for lst in per_sentence_ms.values() for ms in lst]
        result["models"][model_name] = {
            "wer": wer,
            "median_transcription_s": {l: round(ms / 1000, 3)
                                       for l, ms in lang_med_ms.items()},
            "median_transcription_s_all": round(statistics.median(all_ms) / 1000, 3),
            "samples": samples,
        }
        del model  # free VRAM/CPU threads before loading the next size

    # markdown table -----------------------------------------------------
    hdr = "| model | en WER | hi WER | te WER | median latency (s) |"
    sep = "|---|---|---|---|---|"
    rows = [hdr, sep]
    for m in MODELS:
        r = result["models"][m]
        rows.append(f"| {m} | {r['wer']['en']:.3f} | {r['wer']['hi']:.3f} | "
                    f"{r['wer']['te']:.3f} | {r['median_transcription_s_all']:.2f} |")
    print("\n".join(rows))
    print("\n" + result["caveat"])
    print("per-language median latency (s): "
          + json.dumps({m: result["models"][m]["median_transcription_s"]
                        for m in MODELS}))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
