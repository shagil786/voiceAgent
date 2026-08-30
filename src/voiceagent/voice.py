# src/voiceagent/voice.py
from __future__ import annotations

import time
from pathlib import Path

_tts = None
_asr = None

# piper-tts 1.7.0's PiperVoice.load takes a model .onnx path, not a voice
# name. We download a small English voice once and cache it under data/models/.
PIPER_VOICE_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                   "en/en_US/lessac/medium/en_US-lessac-medium.onnx")
PIPER_CONFIG_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json")


def _ensure_piper_model(model_dir: str = "data/models") -> Path:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    onnx = Path(model_dir) / "en_US-lessac-medium.onnx"
    cfg = Path(model_dir) / "en_US-lessac-medium.onnx.json"
    if not onnx.exists():
        import urllib.request
        print(f"downloading piper voice {PIPER_VOICE_URL} ...")
        urllib.request.urlretrieve(PIPER_VOICE_URL, onnx)
    if not cfg.exists():
        import urllib.request
        print(f"downloading piper config {PIPER_CONFIG_URL} ...")
        urllib.request.urlretrieve(PIPER_CONFIG_URL, cfg)
    return onnx


def _get_tts():
    global _tts
    if _tts is None:
        from piper import PiperVoice  # piper-tts
        onnx = _ensure_piper_model()
        _tts = PiperVoice.load(onnx)
    return _tts


def tts_latency(text: str = "Namaste, aapka order kal deliver ho jayega.") -> float:
    t0 = time.time()
    try:
        tts = _get_tts()
        # piper 1.7.0 requires a real wave.Wave_write object (None crashes).
        # Write to a throwaway file and delete it so we only measure synthesis.
        import tempfile
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name
        with wave.open(tmp_name, "wb") as wf:
            tts.synthesize_wav(text, wf)
        Path(tmp_name).unlink(missing_ok=True)
    except Exception as e:  # pragma: no cover - piper may be unavailable
        print(f"[voice] tts unavailable: {e}")
        return 0.0
    return time.time() - t0


def asr_latency(audio_path: str | None = None) -> tuple[float, str]:
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        _asr = WhisperModel("tiny", device="cpu", compute_type="int8")
    t0 = time.time()
    if audio_path and Path(audio_path).exists():
        segs, _ = _asr.transcribe(audio_path)
        text = " ".join(s.text for s in segs).strip()
    else:
        text = ""
    return time.time() - t0, text


def measure_voice_pipeline(sample_audio: str | None = None) -> dict:
    asr_s, _ = asr_latency(sample_audio)
    tts_s = tts_latency()
    return {"asr_s": round(asr_s, 3), "tts_s": round(tts_s, 3),
            "voice_turn_s": round(asr_s + tts_s, 3)}
