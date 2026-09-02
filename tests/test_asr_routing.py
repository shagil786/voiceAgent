# tests/test_asr_routing.py — M5b-2: language-routed ASR.
"""Routing and engine-handle unit tests. All engines are stubbed here —
no model downloads, no GPU/Inference. Real-inference coverage lives behind
VOICEAGENT_ASR_INTEGRATION=1 (see test_asr_integration.py)."""
import wave

import numpy as np
import pytest

from voiceagent import asr as asr_mod
from voiceagent.asr import (
    IndicASRHandle,
    WhisperASRHandle,
    get_asr_for_language,
    transcribe_wav_auto,
)


class StubWhisperEngine:
    """Stands in for a faster-whisper WhisperModel."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None):
        self.calls.append({"audio": audio, "language": language})
        info = type("Info", (), {"language": language or "en"})()
        segs = [type("Seg", (), {"text": "hello "})(), type("Seg", (), {"text": "world"})()]
        return segs, info


class StubIndicModel:
    """Stands in for the IndicConformer remote-code model (callable)."""

    def __init__(self):
        self.calls = []

    def __call__(self, wav, language, decode):
        self.calls.append({"wav": wav, "language": language, "decode": decode})
        return "stub transcript"


def _write_wav(path, sample_rate=16000, dur_s=0.5):
    n = int(sample_rate * dur_s)
    frames = bytearray()
    for i in range(n):
        v = int(32767 * 0.1 * (i % 100) / 100.0)
        frames += v.to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))


# --------------------------------------------------------------------------
# Router: language -> engine kind
# --------------------------------------------------------------------------

def test_router_te_and_ta_use_indic_engine():
    for lang in ("te", "ta"):
        handle = get_asr_for_language(lang, engines=_stub_engines())
        assert isinstance(handle, StubIndicModel), f"{lang} must route to IndicConformer"


def test_router_en_and_hi_use_whisper_engine():
    for lang in ("en", "hi"):
        handle = get_asr_for_language(lang, engines=_stub_engines())
        assert isinstance(handle, StubWhisperEngine), f"{lang} must route to whisper small"


def test_router_other_native_langs_use_indic_engine():
    # All of these are in the IndicConformer-600m model card's 22 languages
    # (as,bn,brx,doi,gu,hi,kn,kok,ks,mai,ml,mni,mr,ne,or,pa,sa,sat,sd,ta,te,ur).
    for lang in ("bn", "mr", "gu", "kn", "ml", "pa"):
        handle = get_asr_for_language(lang, engines=_stub_engines())
        assert isinstance(handle, StubIndicModel), f"{lang} must route to IndicConformer"


def test_router_hinglish_none_and_unknown_use_whisper_engine():
    # hinglish is Romanized Hindi — the proven whisper hi path; None/unknown
    # languages fall back to whisper small (auto-detect inside whisper).
    for lang in ("hinglish", None, "de"):
        handle = get_asr_for_language(lang, engines=_stub_engines())
        assert isinstance(handle, StubWhisperEngine), f"{lang!r} must route to whisper small"


def test_router_falls_back_to_whisper_for_unsupported_native_lang():
    """If a native language ever falls outside the conformer's card list,
    routing must fall back to whisper small WITH a warning (documented)."""
    warnings = []
    engines = {"whisper": StubWhisperEngine, "indic": StubIndicModel}
    # Pretend the conformer only supports 'te' — 'ta' must then fall back.
    handle = get_asr_for_language("ta", engines=engines,
                                  supported=frozenset({"te"}),
                                  warn=warnings.append)
    assert isinstance(handle, StubWhisperEngine)
    assert any("ta" in w for w in warnings), "unsupported native language must warn"


def test_router_returns_singleton_engines_by_default(monkeypatch):
    """By default the router delegates to the module factories; injected
    engines must NOT touch the global cache."""
    whisper_handle, indic_handle = object(), object()
    monkeypatch.setattr(asr_mod, "_get_whisper_small", lambda: whisper_handle)
    monkeypatch.setattr(asr_mod, "_get_indic_asr", lambda: indic_handle)

    assert get_asr_for_language("te") is indic_handle
    assert get_asr_for_language("ta") is indic_handle
    assert get_asr_for_language("en") is whisper_handle


def test_engine_factories_are_lazy_singletons():
    """Default factories cache per engine kind: repeated calls return the same
    handle object. Handles are lazy, so this downloads nothing."""
    assert asr_mod._get_whisper_small() is asr_mod._get_whisper_small()
    assert asr_mod._get_indic_asr() is asr_mod._get_indic_asr()


# --------------------------------------------------------------------------
# IndicASRHandle (real class, stubbed model — exercises the actual logic)
# --------------------------------------------------------------------------

def test_indic_handle_requires_language():
    handle = IndicASRHandle(loader=lambda: StubIndicModel())
    with pytest.raises(ValueError):
        handle.transcribe(np.zeros(8000, dtype=np.float32), language=None)


def test_indic_handle_passes_language_and_decode_to_model():
    model = StubIndicModel()
    handle = IndicASRHandle(loader=lambda: model)
    text = handle.transcribe(np.zeros(8000, dtype=np.float32), language="te")
    assert text == "stub transcript"
    assert model.calls[0]["language"] == "te"
    assert model.calls[0]["decode"] == "ctc"  # CPU default decoding head


def test_indic_handle_accepts_numpy_16k_array_unchanged():
    model = StubIndicModel()
    handle = IndicASRHandle(loader=lambda: model)
    audio = np.zeros(8000, dtype=np.float32)  # 0.5 s @ 16 kHz
    handle.transcribe(audio, language="ta")
    wav = model.calls[0]["wav"]
    assert wav.shape[-1] == 8000  # 16 kHz input passes through unresampled


def test_indic_handle_resamples_wav_path_to_16k():
    """A 22.05 kHz WAV path (piper's output rate) must be resampled to the
    conformer's required 16 kHz before inference."""
    import tempfile
    from pathlib import Path

    model = StubIndicModel()
    handle = IndicASRHandle(loader=lambda: model)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.wav"
        _write_wav(p, sample_rate=22050, dur_s=0.5)
        handle.transcribe(str(p), language="te")
    wav = model.calls[0]["wav"]
    # 0.5 s at 22050 Hz -> 0.5 s at 16000 Hz (±2 samples of polyphase slack)
    assert abs(wav.shape[-1] - 8000) <= 2


def test_indic_handle_model_is_lazy():
    """The 2.4 GB model must not load until the first transcribe call."""
    loaded = []
    handle = IndicASRHandle(loader=lambda: loaded.append(1) or StubIndicModel())
    assert loaded == []
    handle.transcribe(np.zeros(8000, dtype=np.float32), language="hi")
    assert loaded == [1]


# --------------------------------------------------------------------------
# WhisperASRHandle (real class, stubbed engine — exercises the actual logic)
# --------------------------------------------------------------------------

def test_whisper_handle_joins_segment_texts():
    handle = WhisperASRHandle(engine_loader=lambda m, d, c: StubWhisperEngine())
    text = handle.transcribe(np.zeros(8000, dtype=np.float32), language="en")
    assert text == "hello world"


def test_whisper_handle_passes_language_hint():
    engine = StubWhisperEngine()
    handle = WhisperASRHandle(engine_loader=lambda m, d, c: engine)
    handle.transcribe(np.zeros(8000, dtype=np.float32), language="hi")
    assert engine.calls[0]["language"] == "hi"


def test_whisper_handle_no_hint_when_language_none():
    """None must stay None: whisper auto-detects (mirrors the production
    query path, which does not know the language before transcription)."""
    engine = StubWhisperEngine()
    handle = WhisperASRHandle(engine_loader=lambda m, d, c: engine)
    handle.transcribe(np.zeros(8000, dtype=np.float32), language=None)
    assert engine.calls[0]["language"] is None


def test_whisper_handle_hinglish_maps_to_hi_hint():
    engine = StubWhisperEngine()
    handle = WhisperASRHandle(engine_loader=lambda m, d, c: engine)
    handle.transcribe(np.zeros(8000, dtype=np.float32), language="hinglish")
    assert engine.calls[0]["language"] == "hi"


def _raising_segs_engine(detected):
    """Engine whose segments raise if iterated — proves decode was skipped."""

    class RaisingSegsEngine:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language=None):
            self.calls.append({"audio": audio, "language": language})
            info = type("Info", (), {"language": detected})()

            def _segs():
                raise AssertionError("segments must not be decoded when rerouted")
                yield  # pragma: no cover - makes it a generator

            return _segs(), info

    return RaisingSegsEngine()


# --------------------------------------------------------------------------
# transcribe_detected (auto-detect + skip-decode reroute)
# --------------------------------------------------------------------------

def test_transcribe_detected_returns_text_and_detected_language():
    engine = StubWhisperEngine()  # reports the hinted (or default) language
    handle = WhisperASRHandle(engine_loader=lambda m, d, c: engine)
    text, lang = handle.transcribe_detected(np.zeros(8000, dtype=np.float32),
                                            language=None)
    assert text == "hello world"
    assert lang == "en"


def test_transcribe_detected_skips_decode_for_rerouted_autodetect():
    engine = _raising_segs_engine(detected="te")
    handle = WhisperASRHandle(engine_loader=lambda m, d, c: engine)
    text, lang = handle.transcribe_detected(np.zeros(8000, dtype=np.float32),
                                            language=None,
                                            reroute=frozenset({"te", "ta"}))
    assert text is None
    assert lang == "te"
    assert engine.calls[0]["language"] is None  # auto-detect ran, decode didn't


def test_transcribe_detected_explicit_hint_never_reroutes():
    """An explicit language hint means the caller already knows the language;
    reroute only guards the auto-detection path."""
    engine = StubWhisperEngine()
    handle = WhisperASRHandle(engine_loader=lambda m, d, c: engine)
    text, lang = handle.transcribe_detected(np.zeros(8000, dtype=np.float32),
                                            language="te",
                                            reroute=frozenset({"te"}))
    assert text == "hello world"
    assert lang == "te"


# --------------------------------------------------------------------------
# transcribe_wav_auto (the blind production voice-loop entry)
# --------------------------------------------------------------------------

class _StubAutoWhisper:
    def __init__(self, result):
        self.result = result

    def transcribe_detected(self, audio, language=None, reroute=frozenset()):
        self.last = {"audio": audio, "reroute": reroute}
        return self.result


class _StubAutoIndic:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None):
        self.calls.append(language)
        return "indic transcript"


def test_transcribe_wav_auto_reroutes_detected_native_lang_to_indic():
    w = _StubAutoWhisper((None, "te"))
    i = _StubAutoIndic()
    assert transcribe_wav_auto("x.wav", whisper=w, indic=i) == "indic transcript"
    assert i.calls == ["te"]
    assert w.last["reroute"] == asr_mod.INDIC_ROUTE_LANGS


def test_transcribe_wav_auto_keeps_en_on_whisper():
    w = _StubAutoWhisper(("english text", "en"))
    i = _StubAutoIndic()
    assert transcribe_wav_auto("x.wav", whisper=w, indic=i) == "english text"
    assert i.calls == []


def test_transcribe_wav_auto_defaults_to_lazy_singletons(monkeypatch):
    monkeypatch.setattr(asr_mod, "_whisper_small_handle",
                        _StubAutoWhisper((None, "ta")))
    monkeypatch.setattr(asr_mod, "_indic_handle", _StubAutoIndic())
    assert transcribe_wav_auto("x.wav") == "indic transcript"


def _stub_engines():
    return {"whisper": StubWhisperEngine, "indic": StubIndicModel}
