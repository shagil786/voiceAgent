# tests/test_tts.py
from voiceagent.tts import synthesize_to_wav

def test_synthesize_to_wav_creates_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "out.wav")
        seconds = synthesize_to_wav("Namaste, aapka order kal deliver ho jayega.", out)
        assert seconds >= 0
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0
