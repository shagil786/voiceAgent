#!/usr/bin/env python3
"""scripts/local_call.py - talk to the agent WITHOUT the phone.

For the agent this IS a call: the exact same utterance pipeline a LiveKit
call runs (voiceagent.telephony.inbound.make_turn_fn - declared greeting,
whisper ASR, governed brain over the real ERP backend, piper TTS, phone-
digit slot, non-speech gate, farewell hangup guard, end_call). Only the
transport differs: your Mac's mic/speakers instead of a SIP trunk.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/local_call.py
        Walkie-talkie: press Enter, speak, stop - the clip ends on ~0.7s of
        silence (or 12s cap). Type 'quit' to leave. Replies are spoken via
        afplay.

    scripts/local_call.py --device ":1"
        Pick a different ffmpeg avfoundation audio input (default ":0").

    scripts/local_call.py --test-wav <file.wav>
        Feed ONE pre-recorded WAV through the same turn and print the reply
        - no mic needed (self-check).

The same INFO lines the worker prints (caller transcript, turn timings,
policy verdicts, reply) appear on stderr, so behaviour is directly
comparable to a phone call.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("local_call")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def record_utterance(device: str, max_s: float = 12.0,
                     silence_s: float = 0.7) -> bytes:
    """Record one utterance from the mic to raw 16k mono s16le PCM.

    ffmpeg runs silencedetect on the live stream; we kill it when a
    trailing silence >= silence_s starts AFTER >= 0.6s of audio (an
    initial silence_start at 0.0 is ignored). Raw PCM has no WAV header,
    so killing ffmpeg mid-stream leaves a readable buffer.
    """
    raw_path = Path(tempfile.mkstemp(suffix=".pcm")[1])
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats",
        "-f", "avfoundation", "-i", device,
        "-ac", "1", "-ar", "16000", "-f", "s16le",
        "-af", "silencedetect=noise=-35dB:d=%s" % silence_s,
        "-y", str(raw_path),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + max_s
    try:
        for line in proc.stderr:
            line = line.strip()
            if "silence_start:" not in line:
                continue
            try:
                at = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                continue
            if at >= 0.6:  # speech happened, then a real trailing pause
                break
            if time.monotonic() > deadline - 1.0:
                break
        if time.monotonic() >= deadline:
            logger.info("record: hit %ss cap", max_s)
    except Exception:
        pass
    finally:
        proc.kill()
        proc.wait(timeout=5)
    pcm = raw_path.read_bytes() if raw_path.exists() else b""
    raw_path.unlink(missing_ok=True)
    logger.info("record: captured %.2fs", len(pcm) / 2 / 16000)
    return pcm


def play_pcm16(pcm16: bytes, tag: str = "reply") -> None:
    """Write 16k mono PCM to a WAV and play it through the speakers."""
    if not pcm16:
        return
    out = ROOT / "data" / "out" / ("local-%s.wav" % tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16)
    subprocess.run(["afplay", str(out)], check=False)


def build_call_env():
    """Same deps the LiveKit worker builds: governed orchestrator over the
    real ERP (VOICEAGENT_ERP_URL required) + warmed ASR/classifier."""
    from voiceagent.runtime import build_orchestrator
    from voiceagent.asr import warmup_asr

    if not os.environ.get("VOICEAGENT_ERP_URL"):
        logger.error("VOICEAGENT_ERP_URL not set - live-call parity requires "
                     "the real ERP backend (scripts/erp_server.py). Refusing "
                     "to talk to the in-memory mock.")
        sys.exit(2)
    orchestrator = build_orchestrator()
    if orchestrator is None:
        logger.error("no frontier brain configured (VOICEAGENT_FRONTIER_URL)")
        sys.exit(2)
    try:  # first-turn parity: intent classifier warm (two ST models ~16s)
        from voiceagent.memory import _sidecar_classifier
        orchestrator._intent_classifier = _sidecar_classifier()
    except Exception:
        logger.warning("classifier warmup failed", exc_info=True)
    try:
        warmup_asr()
    except Exception:
        logger.warning("asr warmup failed", exc_info=True)
    try:
        orchestrator.brain.client.chat(
            [{"role": "user", "content": "Reply with the single word: ready."}],
            tools=None)
    except Exception:
        logger.warning("frontier warmup failed", exc_info=True)
    return orchestrator


def run_utterance(turn_fn, pcm16: bytes) -> bool:
    """Feed one utterance; returns True when the agent ended the call."""
    if not pcm16 or len(pcm16) < 320:
        logger.info("(empty clip - skipping)")
        return False
    t0 = time.monotonic()
    reply_text, wav = turn_fn(pcm16)
    logger.info("local turn took %.1fs", time.monotonic() - t0)
    if reply_text:
        print("\nAGENT: %s\n" % reply_text)
    if wav:
        play_pcm16(wav)
    return bool(getattr(turn_fn, "call_ended", False))


def interactive_loop(orchestrator, language: str, device: str) -> None:
    from voiceagent.telephony.inbound import (
        GREETING_TRANSCRIPT, _default_tts, make_turn_fn)

    session_id = "local-call-%d" % int(time.time())
    turn_fn = make_turn_fn(orchestrator, session_id, language=language)
    greeting = getattr(orchestrator, "greeting", "") or ""
    if greeting.strip():
        logger.info("greeting: declared text (%d chars)", len(greeting))
        greet_pcm = _default_tts(greeting, language)
    else:
        greet_pcm = _default_tts(
            orchestrator.handle_turn(session_id, GREETING_TRANSCRIPT).reply,
            language)
    play_pcm16(greet_pcm, "greeting")
    print("Connected to the agent locally (same pipeline as a phone call).")
    print("Press Enter and speak - clip ends on ~0.7s silence. 'quit' exits.\n")
    while True:
        try:
            line = input("You> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if line in ("quit", "exit", "q"):
            break
        pcm = record_utterance(device)
        try:
            ended = run_utterance(turn_fn, pcm)
        except Exception:
            logger.warning("turn failed", exc_info=True)
            ended = False
        if ended:
            print("(call ended by agent - farewell guard fired)")
            break


def test_wav(orchestrator, language: str, wav_path: str) -> None:
    """One recorded WAV through the real turn (no mic)."""
    from voiceagent.telephony.audio import resample_to_16k
    from voiceagent.telephony.inbound import make_turn_fn

    with wave.open(wav_path, "rb") as w:
        rate = w.getframerate()
        pcm = resample_to_16k(w.readframes(w.getnframes()), rate)
    turn_fn = make_turn_fn(orchestrator, "local-test-%d" % int(time.time()),
                           language=language)
    reply_text, _ = turn_fn(bytes(pcm))
    print("\nREPLY: %s\n" % reply_text)
    print("call_ended: %s" % bool(getattr(turn_fn, "call_ended", False)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=":0",
                        help="ffmpeg avfoundation audio input (default :0)")
    parser.add_argument("--test-wav", metavar="FILE",
                        help="run one WAV through the turn and exit")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    orchestrator = build_call_env()
    language = os.environ.get("VOICEAGENT_DEFAULT_LANG") or "en-US"
    if args.test_wav:
        test_wav(orchestrator, language, args.test_wav)
    else:
        interactive_loop(orchestrator, language, args.device)


if __name__ == "__main__":
    main()
