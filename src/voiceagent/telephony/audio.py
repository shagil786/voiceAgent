"""PCM resample + chunk helpers for the LiveKit telephony limb.

16-bit mono PCM only. LiveKit frames arrive at 48 kHz; the pipeline
(ASR/TTS/VAD) runs at 16 kHz — these two linear-interp converters bridge
the gap with exact 3:1 / 1:3 ratios. `chunk_frames` splits PCM into fixed
frame_ms windows, dropping a trailing partial chunk.
"""
from __future__ import annotations

import numpy as np


def _resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if not pcm:
        return b""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_in = x.shape[0]
    n_out = n_in * dst_rate // src_rate
    if n_out == 0:
        return b""
    old_idx = np.arange(n_in, dtype=np.float64)
    new_idx = np.arange(n_out, dtype=np.float64) * (n_in - 1) / max(n_out - 1, 1) if n_out > 1 else np.zeros(1)
    y = np.interp(new_idx, old_idx, x).astype(np.float64)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


def resample_48k_to_16k(pcm: bytes) -> bytes:
    """48 kHz mono int16 PCM -> 16 kHz (exact 1:3 sample ratio)."""
    return _resample(pcm, 48000, 16000)


def resample_16k_to_48k(pcm: bytes) -> bytes:
    """16 kHz mono int16 PCM -> 48 kHz (exact 3:1 sample ratio)."""
    return _resample(pcm, 16000, 48000)


def chunk_frames(pcm: bytes, frame_ms: int, sample_rate: int) -> list[bytes]:
    """Split mono int16 PCM into `frame_ms` chunks; drop trailing partial."""
    samples_per_chunk = sample_rate * frame_ms // 1000
    bytes_per_chunk = samples_per_chunk * 2
    if bytes_per_chunk <= 0:
        return []
    n_full = len(pcm) // bytes_per_chunk
    return [pcm[i * bytes_per_chunk:(i + 1) * bytes_per_chunk] for i in range(n_full)]
