# src/voiceagent/voice_agent.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from voiceagent.asr import transcribe_wav_auto
from voiceagent.chat import run_turn
from voiceagent.memory import SQLiteMemory
from voiceagent.tts import speak

_MEMORY: SQLiteMemory | None = None


def _default_memory() -> SQLiteMemory:
    """Shared working-memory store for the voice path (M4a). Lives in the
    gitignored data/out/ next to the HTTP server's store."""
    global _MEMORY
    if _MEMORY is None:
        Path("data/out").mkdir(parents=True, exist_ok=True)
        _MEMORY = SQLiteMemory("data/out/memory.db")
    return _MEMORY


@dataclass
class VoiceTurnResult:
    transcript: str
    reply: str
    action: str | None
    decision: str | None
    reasons: list[str]
    tts_path: str | None
    latency_s: float


def voice_turn(agent, audio_path: str, out_audio: str | None = None,
               memory: SQLiteMemory | None = None) -> VoiceTurnResult:
    """Speech in -> speech out. ASR the audio, run the agent, synthesize the
    reply. out_audio defaults to a temp WAV (returned in tts_path). Each call
    is its own conversation (fresh conv id) recorded in working memory; pass
    memory=None to use the shared data/out/memory.db store."""
    t0 = time.time()
    transcript = transcribe_wav_auto(audio_path)
    conv_id = f"voice-{uuid.uuid4().hex[:12]}"
    out = run_turn(agent, transcript, conv_id=conv_id,
                   memory=memory if memory is not None else _default_memory())
    if out_audio is None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_audio = tmp.name
    # M5b-1: reply TTS is multilingual — speak() auto-detects the reply's
    # language (langid) and routes to the matching piper voice. M5b-2: query
    # ASR is language-routed too — whisper small auto-detects; te/ta/native
    # Indic langs re-transcribe with IndicConformer (see voiceagent.asr).
    speak(out["reply"], out_path=out_audio)
    return VoiceTurnResult(
        transcript=transcript,
        reply=out["reply"],
        action=out["action"],
        decision=out["decision"],
        reasons=out["reasons"],
        tts_path=out_audio,
        latency_s=time.time() - t0,
    )
