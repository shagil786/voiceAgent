# src/voiceagent/vad.py
from __future__ import annotations

import math
import struct

try:
    import webrtcvad  # optional; falls back to energy-based below
    _vad = webrtcvad.Vad(2)
    _HAS_WEBRTC = True
except Exception:  # pragma: no cover - fallback path
    _vad = None
    _HAS_WEBRTC = False

FRAME_MS = 20


def _rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    n = len(frame) // 2
    if n == 0:
        return 0.0
    vals = struct.unpack(f"<{n}h", frame[: n * 2])
    return math.sqrt(sum(v * v for v in vals) / n) / 32767.0


def is_speech(frame: bytes, sample_rate: int = 16000) -> bool:
    """True if a 20ms 16-bit PCM frame contains speech."""
    if _HAS_WEBRTC:
        try:
            return bool(_vad.is_speech(frame, sample_rate))
        except Exception:
            pass
    # Energy fallback: speech if RMS above a quiet-room threshold.
    return _rms(frame) > 0.02


def split_voice_frames(audio: bytes, sample_rate: int = 16000) -> list[bytes]:
    """Split a PCM byte stream into 20ms frames, keep only speech frames."""
    frame_bytes = int(sample_rate * FRAME_MS / 1000) * 2  # 16-bit mono
    frames = [audio[i:i + frame_bytes]
              for i in range(0, len(audio) - frame_bytes + 1, frame_bytes)]
    return [f for f in frames if is_speech(f, sample_rate)]
