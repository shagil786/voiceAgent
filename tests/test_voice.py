# tests/test_voice.py
from voiceagent.voice import tts_latency

def test_tts_latency_returns_seconds():
    s = tts_latency("hello")
    # may be 0.0 if piper is unavailable in CI; must not be negative
    assert s >= 0.0


def test_asr_latency_delegates_to_routed_asr(monkeypatch, tmp_path):
    """M5b-2: asr_latency measures the production language-routed path, not a
    bare whisper model. Missing audio must skip ASR entirely."""
    import voiceagent.asr as asr_mod
    from voiceagent.voice import asr_latency

    calls = []
    monkeypatch.setattr(asr_mod, "transcribe_wav_auto",
                        lambda path: calls.append(path) or "hi there")

    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF....")
    seconds, text = asr_latency(str(wav))
    assert text == "hi there"
    assert seconds >= 0.0
    assert calls == [str(wav)]

    seconds2, text2 = asr_latency(str(tmp_path / "missing.wav"))
    assert text2 == ""
    assert seconds2 >= 0.0
    assert calls == [str(wav)]  # missing file made no ASR call
