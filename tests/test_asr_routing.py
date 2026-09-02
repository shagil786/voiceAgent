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
    QwenASRHandle,
    WhisperASRHandle,
    get_asr_for_language,
    transcribe_wav_routed,
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


class StubQwenEngine:
    """Stands in for the QwenASRHandle core engine (type-level only — the
    router never calls it, it just returns the factory's instance)."""


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


def test_router_en_hi_hinglish_none_and_unknown_use_qwen_engine():
    """M5b-3: the core slot is Qwen3-ASR-0.6B — en ties whisper at 3x speed,
    hi beats it, and its built-in LID covers None/unknown/hinglish."""
    for lang in ("en", "hi", "hinglish", None, "de"):
        handle = get_asr_for_language(lang, engines=_stub_engines())
        assert isinstance(handle, StubQwenEngine), f"{lang!r} must route to the Qwen core"


def test_router_other_native_langs_use_indic_engine():
    # All of these are in the IndicConformer-600m model card's 22 languages
    # (as,bn,brx,doi,gu,hi,kn,kok,ks,mai,ml,mni,mr,ne,or,pa,sa,sat,sd,ta,te,ur).
    for lang in ("bn", "mr", "gu", "kn", "ml", "pa"):
        handle = get_asr_for_language(lang, engines=_stub_engines())
        assert isinstance(handle, StubIndicModel), f"{lang} must route to IndicConformer"


def test_router_falls_back_to_qwen_for_unsupported_native_lang():
    """If a native language ever falls outside the conformer's card list,
    routing must fall back to the Qwen core WITH a warning (documented)."""
    warnings = []
    engines = {"qwen": StubQwenEngine, "indic": StubIndicModel}
    # Pretend the conformer only supports 'te' — 'ta' must then fall back.
    handle = get_asr_for_language("ta", engines=engines,
                                  supported=frozenset({"te"}),
                                  warn=warnings.append)
    assert isinstance(handle, StubQwenEngine)
    assert any("ta" in w for w in warnings), "unsupported native language must warn"


def test_router_returns_singleton_engines_by_default(monkeypatch):
    """By default the router delegates to the module factories; injected
    engines must NOT touch the global cache."""
    qwen_handle, indic_handle = object(), object()
    monkeypatch.setattr(asr_mod, "_get_qwen_asr", lambda: qwen_handle)
    monkeypatch.setattr(asr_mod, "_get_indic_asr", lambda: indic_handle)

    assert get_asr_for_language("te") is indic_handle
    assert get_asr_for_language("ta") is indic_handle
    assert get_asr_for_language("en") is qwen_handle
    assert get_asr_for_language(None) is qwen_handle


def test_engine_factories_are_lazy_singletons():
    """Default factories cache per engine kind: repeated calls return the same
    handle object. Handles are lazy, so this downloads nothing."""
    assert asr_mod._get_qwen_asr() is asr_mod._get_qwen_asr()
    assert asr_mod._get_indic_asr() is asr_mod._get_indic_asr()
    assert asr_mod._get_whisper_small() is asr_mod._get_whisper_small()


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


def test_transcribe_detected_returns_text_and_detected_language():
    """The blind path gets (text, detected_language) from one whisper pass;
    the detected language is metadata — it never triggers a reroute."""
    engine = StubWhisperEngine()  # reports the hinted (or default) language
    handle = WhisperASRHandle(engine_loader=lambda m, d, c: engine)
    text, lang = handle.transcribe_detected(np.zeros(8000, dtype=np.float32),
                                            language=None)
    assert text == "hello world"
    assert lang == "en"


# --------------------------------------------------------------------------
# transcribe_wav_routed (the voice-loop entry)
# --------------------------------------------------------------------------

class _StubRoutedIndic:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def transcribe(self, audio, language=None):
        self.calls.append(language)
        if self.error:
            raise self.error
        return "indic transcript"


class _StubRoutedWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None):
        self.calls.append(language)
        return "whisper transcript"


class _StubRoutedQwen:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None):
        self.calls.append(language)
        return "qwen transcript"


def test_transcribe_wav_routed_known_te_goes_to_indic(monkeypatch):
    indic = _StubRoutedIndic()
    monkeypatch.setattr(asr_mod, "_indic_handle", indic)
    monkeypatch.setattr(asr_mod, "_whisper_small_handle", _StubRoutedWhisper())
    assert transcribe_wav_routed("x.wav", language="te") == "indic transcript"
    assert indic.calls == ["te"]


def test_transcribe_wav_routed_blind_goes_to_qwen_core(monkeypatch):
    """language=None is the blind path: the Qwen core with its built-in LID,
    never an auto-reroute on detected language (whisper-era probing showed
    detection cannot separate Hinglish from native Indic audio — see
    voiceagent.asr module docstring). Whisper must stay untouched."""
    qwen = _StubRoutedQwen()
    whisper = _StubRoutedWhisper()
    monkeypatch.setattr(asr_mod, "_qwen_handle", qwen)
    monkeypatch.setattr(asr_mod, "_whisper_small_handle", whisper)
    monkeypatch.setattr(asr_mod, "_indic_handle", _StubRoutedIndic())
    assert transcribe_wav_routed("x.wav", language=None) == "qwen transcript"
    assert qwen.calls == [None]
    assert whisper.calls == []


def test_transcribe_wav_routed_falls_back_when_routed_engine_fails(monkeypatch):
    """A routed-engine failure (gated repo, download error) must not kill the
    voice turn mid-call: fall back to whisper small with a warning."""
    warnings = []
    monkeypatch.setattr(asr_mod.logger, "warning",
                        lambda *a, **k: warnings.append(" ".join(map(str, a))))
    monkeypatch.setattr(asr_mod, "_indic_handle",
                        _StubRoutedIndic(error=OSError("gated repo")))
    monkeypatch.setattr(asr_mod, "_whisper_small_handle", _StubRoutedWhisper())
    assert transcribe_wav_routed("x.wav", language="te") == "whisper transcript"
    assert any("gated" in str(w) or "OSError" in str(w) for w in warnings)


def _stub_engines():
    return {"qwen": StubQwenEngine, "indic": StubIndicModel}


# --------------------------------------------------------------------------
# QwenASRHandle (real class, stubbed processor+model — exercises the logic)
# --------------------------------------------------------------------------

class _FakeQwenInputs(dict):
    """Mapping-backed (real code does model.generate(**inputs)) with an
    input_ids tensor-ish attribute and a no-op .to() like BatchDict."""

    def __init__(self, input_len):
        super().__init__(input_ids=np.zeros((1, input_len), dtype=np.int64))
        self.input_ids = self["input_ids"]

    def to(self, *args, **kwargs):
        return self


class _FakeQwenModel:
    device = "cpu"
    dtype = "float32"

    def __init__(self, total_len=9):
        self._out = np.zeros((1, total_len), dtype=np.int64)

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 256
        assert kwargs["do_sample"] is False  # deterministic ASR
        return self._out


class _FakeQwenProcessor:
    """mode='parsed' mimics the documented decode(return_format='parsed')
    dict; mode='plain' breaks the parsed path so the fallback decode runs."""

    def __init__(self, mode="parsed"):
        self.paths = []
        self.decode_calls = []
        self.mode = mode

    def apply_transcription_request(self, audio=None):
        self.paths.append(audio)
        return _FakeQwenInputs(input_len=5)

    def decode(self, ids, **kwargs):
        self.decode_calls.append(kwargs.get("return_format"))
        if self.mode == "parsed":
            return {"transcription": "stub qwen text", "language": "English"}
        return "plain qwen text"


def _qwen_handle_with(processor, model):
    return QwenASRHandle(loader=lambda: (processor, model))


def test_qwen_handle_returns_parsed_transcription():
    proc = _FakeQwenProcessor()
    handle = _qwen_handle_with(proc, _FakeQwenModel())
    assert handle.transcribe("x.wav") == "stub qwen text"
    assert proc.paths == ["x.wav"]


def test_qwen_handle_transcribe_detected_returns_language():
    proc = _FakeQwenProcessor()
    handle = _qwen_handle_with(proc, _FakeQwenModel())
    text, lang = handle.transcribe_detected("x.wav", language="hi")
    assert text == "stub qwen text"
    assert lang == "English"
    # the hint is accepted but never forced into the request
    assert proc.decode_calls == ["parsed"]


def test_qwen_handle_falls_back_to_plain_decode():
    proc = _FakeQwenProcessor(mode="plain")
    handle = _qwen_handle_with(proc, _FakeQwenModel())
    text, lang = handle.transcribe_detected("x.wav")
    assert text == "plain qwen text"
    assert lang is None


def test_qwen_handle_model_is_lazy():
    loaded = []
    handle = QwenASRHandle(
        loader=lambda: loaded.append(1) or (_FakeQwenProcessor(), _FakeQwenModel()))
    assert loaded == []
    handle.transcribe("x.wav")
    assert loaded == [1]


def test_qwen_handle_accepts_float_array_via_temp_wav():
    proc = _FakeQwenProcessor()
    handle = _qwen_handle_with(proc, _FakeQwenModel())
    handle.transcribe(np.zeros(8000, dtype=np.float32))
    assert isinstance(proc.paths[0], str)
    assert proc.paths[0].endswith(".wav")
