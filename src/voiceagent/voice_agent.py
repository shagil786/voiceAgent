# src/voiceagent/voice_agent.py
from __future__ import annotations

import time
from dataclasses import dataclass, field

from voiceagent.asr import transcribe_wav
from voiceagent.tts import synthesize_to_wav
from voiceagent.chat import run_turn


@dataclass
class VoiceTurnResult:
    transcript: str
    reply: str
    action: str | None
    decision: str | None
    reasons: list[str]
    tts_path: str | None
    latency_s: float


def voice_turn(agent, audio_path: str, out_audio: str | None = None) -> VoiceTurnResult:
    """Speech in -> speech out. ASR the audio, run the agent, synthesize the
    reply. out_audio defaults to a temp WAV (returned in tts_path)."""
    t0 = time.time()
    transcript = transcribe_wav(audio_path)
    out = run_turn(agent, transcript)
    if out_audio is None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_audio = tmp.name
    synthesize_to_wav(out["reply"], out_audio)
    return VoiceTurnResult(
        transcript=transcript,
        reply=out["reply"],
        action=out["action"],
        decision=out["decision"],
        reasons=out["reasons"],
        tts_path=out_audio,
        latency_s=time.time() - t0,
    )
