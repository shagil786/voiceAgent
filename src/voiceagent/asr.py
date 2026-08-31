# src/voiceagent/asr.py
from __future__ import annotations

from pathlib import Path

_asr = None


def get_asr(model: str = "tiny"):
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        _asr = WhisperModel(model, device="cpu", compute_type="int8")
    return _asr


def transcribe_wav(path: str, model: str = "tiny") -> str:
    """Transcribe a 16-bit mono WAV file on CPU. Returns the transcript."""
    segments, _ = get_asr(model).transcribe(str(path))
    return " ".join(s.text for s in segments).strip()


def transcribe_chunks(audio_chunks, model: str = "tiny") -> str:
    """Transcribe concatenated PCM/voice-filtered chunks (writes temp WAV)."""
    data = b"".join(audio_chunks)
    import tempfile
    import wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(data)
        text = transcribe_wav(tmp.name, model)
    Path(tmp.name).unlink(missing_ok=True)
    return text
