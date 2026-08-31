# src/voiceagent/tts.py
from __future__ import annotations

import tempfile
import time
import wave
from pathlib import Path

from voiceagent.voice import _ensure_piper_model

_tts = None


def _get_tts():
    global _tts
    if _tts is None:
        from piper import PiperVoice
        onnx = _ensure_piper_model()
        _tts = PiperVoice.load(onnx)
    return _tts


def synthesize_to_wav(text: str, out_path: str) -> float:
    """Synthesize a full utterance to a WAV file. Returns seconds taken."""
    t0 = time.time()
    tts = _get_tts()
    with wave.open(out_path, "wb") as w:
        tts.synthesize_wav(text, w)
    return time.time() - t0


def synthesize_chunks(text: str, chunk_chars: int = 80) -> list[tuple[str, float]]:
    """Break text into word-boundary chunks and synthesize each. Returns
    [(wav_path, synth_ms)] so the first chunk can stream while later ones
    generate (low perceived latency)."""
    words = text.split()
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= chunk_chars:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append(" ".join(cur))

    out = []
    for chunk in chunks:
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        with wave.open(path, "wb") as w:
            _get_tts().synthesize_wav(chunk, w)
        out.append((path, (time.time() - t0) * 1000))
    return out
