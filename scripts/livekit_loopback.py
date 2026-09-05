#!/usr/bin/env python3
"""Offline loopback drill: caller WAV through BridgeSession -> reply WAV.

Usage:
    .venv/bin/python scripts/livekit_loopback.py [--wav CALLER.wav] [--out REPLY.wav]

Plays a 16k mono int16 caller WAV into a `BridgeSession` in 20ms chunks
(the same shape `telephony/inbound.py` feeds from the room), with a stub
turn fn standing in for the governed ASR/Orchestrator/TTS pipeline, and
writes the session's 48k playback chunks back out as a reply WAV.

No PSTN, no LiveKit connection, no model loads — this proves the bridge
mechanics (VAD endpointing, resample, chunking, playback queue) on a real
file path before any paid call. The first LIVE drill is the manual runbook
(docs/telephony-runbook.md), not this script.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.telephony.livekit_bridge import BridgeSession

_FRAME_BYTES = 640  # 20ms @16k mono int16 (BridgeSession contract)
_SILENCE_FRAMES_TO_ENDPOINT = 25  # 500ms trailing silence, as in tests

_DEFAULT_OUT = Path("data/out/loopback-reply.wav")
REPLY_TEXT = "Loopback reply (stub turn — no model loaded)."


def _synthetic_caller_pcm(seconds: float = 1.0) -> bytes:
    """330Hz sine caller WAV when `--wav` is omitted (VAD-visible energy)."""
    n = int(16000 * seconds)
    return struct.pack(
        f"<{n}h", *[int(6000 * math.sin(2 * math.pi * 330 * i / 16000)) for i in range(n)]
    )


def _read_caller_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(
                f"{path}: expected 16kHz mono 16-bit WAV, got {w.getframerate()}Hz "
                f"{w.getnchannels()}ch {w.getsampwidth() * 8}-bit"
            )
        return w.readframes(w.getnframes())


def _stub_turn(pcm: bytes) -> tuple[str, bytes]:
    """Fixed 16k reply standing in for ASR + governed turn + TTS."""
    del pcm  # content irrelevant to the drill
    m = 6400  # 400ms
    return REPLY_TEXT, struct.pack(
        f"<{m}h", *[int(4000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(m)]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", default=None, type=Path,
                        help="caller WAV (16k mono int16); omit for a synthetic tone")
    parser.add_argument("--out", default=_DEFAULT_OUT, type=Path,
                        help=f"reply WAV path (default: {_DEFAULT_OUT})")
    args = parser.parse_args()

    stage_t0 = time.perf_counter()
    if args.wav is not None:
        caller_pcm = _read_caller_wav(args.wav)
        source = str(args.wav)
    else:
        caller_pcm = _synthetic_caller_pcm()
        source = "synthetic 330Hz tone (1.0s)"
    read_s = time.perf_counter() - stage_t0

    utterances = 0

    def _counting_turn(pcm: bytes) -> tuple[str, bytes]:
        nonlocal utterances
        utterances += 1
        return _stub_turn(pcm)

    session = BridgeSession(on_utterance=_counting_turn)

    feed_t0 = time.perf_counter()
    # Drop a trailing partial frame: feed_pcm16 fail-fasts on non-640-byte input.
    for i in range(0, len(caller_pcm) - _FRAME_BYTES + 1, _FRAME_BYTES):
        session.feed_pcm16(caller_pcm[i:i + _FRAME_BYTES])
    for _ in range(_SILENCE_FRAMES_TO_ENDPOINT):
        session.feed_pcm16(bytes(_FRAME_BYTES))
    feed_s = time.perf_counter() - feed_t0

    drain_t0 = time.perf_counter()
    reply_chunks = []
    while (chunk := session.take_playback()) is not None:
        reply_chunks.append(chunk)
    session.stop()
    drain_s = time.perf_counter() - drain_t0

    if not reply_chunks:
        print("loopback FAILED: no reply chunks produced (VAD never endpointed)")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"".join(reply_chunks))

    reply_frames = sum(len(c) // 2 for c in reply_chunks)
    print(f"loopback OK")
    print(f"  caller   : {source} ({len(caller_pcm) / 2} samples @16k)")
    print(f"  utterances endpointed : {utterances}")
    print(f"  reply    : {args.out} ({reply_frames} samples @48k, "
          f"{reply_frames / 48000:.2f}s)")
    print(f"  timings  : read {read_s * 1000:.1f}ms | feed {feed_s * 1000:.1f}ms | "
          f"drain {drain_s * 1000:.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
