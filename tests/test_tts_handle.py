# tests/test_tts_handle.py — M5b-1 multilingual TTSHandle.
# Unit tests never download piper voices or touch the real PiperVoice:
# the piper layer is stubbed via an injected voice_loader that returns a
# FakeVoice. One real-synthesis test is gated behind VOICEAGENT_TTS_INTEGRATION=1.
import os
import urllib.request
from pathlib import Path

import pytest

from voiceagent import tts as tts_mod
from voiceagent.tts import TTSHandle, VOICE_REGISTRY
from voiceagent.voice import PIPER_CONFIG_URL, PIPER_VOICE_URL, ensure_voice, voice_urls


class FakeVoice:
    """Duck-typed PiperVoice stand-in: records the texts it was asked to
    synthesize and which voice name it belongs to."""

    def __init__(self, voice_name):
        self.voice_name = voice_name
        self.texts = []

    def synthesize_wav(self, text, wav_file, *a, **kw):
        self.texts.append(text)
        # real PiperVoice.synthesize_wav(set_wav_format=True) writes the wav
        # header; replicate that contract or Wave_write.close() raises.
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 220)


def _make_handle(tmp_path, calls, **kw):
    """Handle whose voice_loader records every (voice_name, model_dir) request."""
    def loader(voice_name, model_dir):
        calls.append(voice_name)
        return FakeVoice(voice_name)
    return TTSHandle(model_dir=str(tmp_path), voice_loader=loader, **kw)


def _out(tmp_path, name="o.wav"):
    return str(tmp_path / name)


HI_TEXT = "नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।"
TE_TEXT = "నమస్కారం, మీ ఆర్డర్ రేపు వస్తుంది."
EN_TEXT = "Your order will arrive by tomorrow evening."
TA_TEXT = "உங்கள் ஆர்டர் நாளை வரும்."  # no Tamil voice in piper — must fall back


# ---------------------------------------------------------------- registry

def test_registry_covers_en_hi_te_and_excludes_ta():
    assert {"en", "hi", "te"} <= set(VOICE_REGISTRY)
    assert "ta" not in VOICE_REGISTRY
    assert VOICE_REGISTRY["en"] == "en_US-lessac-medium"
    assert VOICE_REGISTRY["hi"] == "hi_IN-priyamvada-medium"
    assert VOICE_REGISTRY["te"] == "te_IN-maya-medium"


def test_registry_voice_names_are_piper_hf_paths():
    # en mirrors the pre-existing voice.py constants (no double source of truth)
    assert voice_urls(VOICE_REGISTRY["en"]) == (PIPER_VOICE_URL, PIPER_CONFIG_URL)


# ------------------------------------------------------------ language routing

def test_hindi_text_routes_to_hi_voice(tmp_path):
    calls = []
    h = _make_handle(tmp_path, calls)
    h.speak(HI_TEXT, out_path=_out(tmp_path))
    assert calls == ["hi_IN-priyamvada-medium"]


def test_english_text_routes_to_en_voice(tmp_path):
    calls = []
    h = _make_handle(tmp_path, calls)
    h.speak(EN_TEXT, out_path=_out(tmp_path))
    assert calls == ["en_US-lessac-medium"]


def test_telugu_script_routes_to_te_voice(tmp_path):
    calls = []
    h = _make_handle(tmp_path, calls)
    h.speak(TE_TEXT, out_path=_out(tmp_path))
    assert calls == ["te_IN-maya-medium"]


def test_hinglish_routes_to_hi_voice(tmp_path):
    calls = []
    h = _make_handle(tmp_path, calls)
    h.speak("kya hai mera order kab aayega", language="hinglish",
            out_path=_out(tmp_path))
    assert calls == ["hi_IN-priyamvada-medium"]


def test_explicit_language_overrides_detection(tmp_path):
    calls = []
    h = _make_handle(tmp_path, calls)
    # English text forced to Hindi voice: explicit language wins.
    h.speak("hello there", language="hi", out_path=_out(tmp_path))
    assert calls == ["hi_IN-priyamvada-medium"]


# ------------------------------------------------------------------- fallback

def test_tamil_falls_back_to_en_with_warning(tmp_path):
    calls, warns = [], []
    h = _make_handle(tmp_path, calls, warn=warns.append)
    h.speak(TA_TEXT, out_path=_out(tmp_path))
    assert calls == ["en_US-lessac-medium"]          # ta not registered
    assert len(warns) == 1
    assert "ta" in warns[0] and "en" in warns[0]     # names both languages


def test_any_unregistered_language_warns_and_uses_en(tmp_path):
    for bogus in ("bn", "mr", "xx", "kl"):
        calls, warns = [], []
        h = _make_handle(tmp_path, calls, warn=warns.append)
        h.speak("some words", language=bogus, out_path=_out(tmp_path, bogus))
        assert calls == ["en_US-lessac-medium"]
        assert warns, f"{bogus} should have warned"
        assert bogus in warns[0]


def test_fallback_voice_is_injectable(tmp_path):
    calls, warns = [], []
    h = _make_handle(tmp_path, calls, fallback_voice="hi", warn=warns.append)
    h.speak(TA_TEXT, out_path=_out(tmp_path))
    assert calls == ["hi_IN-priyamvada-medium"]
    assert "hi" in warns[0]


def test_fallback_does_not_raise(tmp_path):
    h = _make_handle(tmp_path, [])
    # production path: no exception for an unregistered language
    h.speak(TA_TEXT, language="ta", out_path=_out(tmp_path))


# --------------------------------------------------------- autodetect plumbing

def test_language_none_uses_langid(monkeypatch, tmp_path):
    calls = []
    h = _make_handle(tmp_path, calls)
    monkeypatch.setattr(tts_mod, "detect_language", lambda text: "te")
    h.speak("ignored by detection stub", out_path=_out(tmp_path))
    assert calls == ["te_IN-maya-medium"]


# ------------------------------------------------- voice cache + chunked path

def test_voice_loaded_once_per_language(tmp_path):
    calls = []
    h = _make_handle(tmp_path, calls)
    h.speak(EN_TEXT, out_path=_out(tmp_path, "a.wav"))
    h.speak("Another English order update.", out_path=_out(tmp_path, "b.wav"))
    assert calls == ["en_US-lessac-medium"]          # second call reuses voice


def test_chunked_synthesis_routes_every_chunk_to_language_voice(tmp_path):
    calls = []
    h = _make_handle(tmp_path, calls)
    loader_voices = {}

    def loader(voice_name, model_dir):
        calls.append(voice_name)
        return loader_voices.setdefault(voice_name, FakeVoice(voice_name))

    h = TTSHandle(model_dir=str(tmp_path), voice_loader=loader)
    long_text = " ".join(["order"] * 60)
    out = h.synthesize_chunks(long_text, chunk_chars=80, language="hi")
    assert calls == ["hi_IN-priyamvada-medium"]
    assert len(out) > 1                                # actually chunked
    synth_texts = loader_voices["hi_IN-priyamvada-medium"].texts
    assert len(synth_texts) == len(out)                # one synth per chunk
    assert " ".join(synth_texts).replace(" ", "") == long_text.replace(" ", "")


# ----------------------------------------------------- download-once (network mocked)

def _touch(path, size=8):
    path.write_bytes(b"x" * size)


def test_ensure_voice_skips_download_when_files_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download expected")))
    onnx = tmp_path / "hi_IN-priyamvada-medium.onnx"
    cfg = tmp_path / "hi_IN-priyamvada-medium.onnx.json"
    _touch(onnx)
    _touch(cfg)
    p = ensure_voice("hi_IN-priyamvada-medium", str(tmp_path))
    assert p == onnx and onnx.exists()


def test_ensure_voice_downloads_onnx_and_config_once_each(tmp_path, monkeypatch):
    downloaded = []

    def fake_urlretrieve(url, dest):
        downloaded.append(url)
        Path(dest).write_bytes(b"dummy")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    ensure_voice("hi_IN-priyamvada-medium", str(tmp_path))
    assert len(downloaded) == 2
    assert all("hi_IN-priyamvada-medium" in u for u in downloaded)
    # second call: files now exist -> no new downloads (download-once)
    ensure_voice("hi_IN-priyamvada-medium", str(tmp_path))
    assert len(downloaded) == 2


# ----------------------------------------------- optional real-synthesis test

@pytest.mark.skipif(os.environ.get("VOICEAGENT_TTS_INTEGRATION") != "1",
                    reason="downloads hi/te voices; set VOICEAGENT_TTS_INTEGRATION=1")
def test_real_synthesis_hi_and_te(tmp_path):
    import wave
    h = TTSHandle()  # real piper layer + real data/models downloads
    for lang, text in (("hi", HI_TEXT), ("te", TE_TEXT)):
        out = tmp_path / f"{lang}.wav"
        h.speak(text, language=lang, out_path=str(out))
        assert out.exists() and out.stat().st_size > 10_000
        with wave.open(str(out), "rb") as w:
            assert w.getnframes() > 0 and w.getframerate() == 22050
