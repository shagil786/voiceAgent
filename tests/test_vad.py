# tests/test_vad.py
import wave
from voiceagent.vad import is_speech, split_voice_frames

def _sin_frame(freq=440, sr=16000, dur_s=0.02):
    import math
    n = int(sr * dur_s)
    frames = bytearray()
    for i in range(n):
        v = int(32767 * 0.6 * math.sin(2 * math.pi * freq * i / sr))
        frames += v.to_bytes(2, "little", signed=True)
    return bytes(frames)

def test_silence_is_not_speech():
    silent = b"\x00\x00" * int(16000 * 0.02)  # 20ms of zeros
    assert not is_speech(silent)

def test_tone_is_speech():
    assert is_speech(_sin_frame())

def test_split_voice_frames_keeps_speech_drops_silence():
    speech = _sin_frame()
    silence = b"\x00\x00" * int(16000 * 0.02)
    audio = speech + silence + speech
    kept = split_voice_frames(audio)
    assert len(kept) == 2  # the two speech frames, silence dropped
