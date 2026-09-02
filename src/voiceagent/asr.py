# src/voiceagent/asr.py
"""ASR for the voice path.

M5b-2 adds language-routed ASR on top of the original faster-whisper entry
points (get_asr/transcribe_wav stay backward compatible):

- WhisperASRHandle: uniform wrapper around faster-whisper (small by default —
  the proven M5b-1 size). language=None keeps whisper's auto-detection.
- IndicASRHandle: ai4bharat/indic-conformer-600m-multilingual (CTC conformer
  head, 22 Indic languages, ~2.4 GB download on first use). The conformer
  REQUIRES the target language as input per its recipe — the router supplies
  it from known language context (loopback tests, future telephony routing).
- get_asr_for_language(lang): en/hi/hinglish -> whisper small (proven);
  te/ta/bn/mr/gu/kn/ml/pa -> IndicConformer (all in its model-card language
  list); None/unknown -> whisper small auto-detect. For contexts that already
  know the language (telephony routing, loopback tests).
- transcribe_wav_auto(path): the blind voice-loop entry (voice_agent) — one
  whisper-small pass auto-detects the language; detected te/ta/native langs
  skip whisper's decode (the hallucination part) and re-transcribe with
  IndicConformer.

Loopback measurement (M5b-1) showed whisper te WER >= 1.0 at every size
(small hallucination-loops), hence the dedicated Indic engine.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_asr = None


def get_asr(model: str = "tiny"):
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        _asr = WhisperModel(model, device="cpu", compute_type="int8")
    return _asr


def transcribe_wav(path: str, model: str = "tiny") -> str:
    """Transcribe a 16-bit mono WAV file on CPU. Returns the transcript."""
    segments, _ = get_asr(model).transcribe(str(path))
    return " ".join(s.text for s in segments).strip()


def transcribe_chunks(audio_chunks, model: str = "tiny") -> str:
    """Transcribe concatenated PCM/voice-filtered chunks (writes temp WAV)."""
    data = b"".join(audio_chunks)
    import tempfile
    import wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(data)
        text = transcribe_wav(tmp.name, model)
    Path(tmp.name).unlink(missing_ok=True)
    return text


# ---------------------------------------------------------------------------
# M5b-2: language-routed ASR
# ---------------------------------------------------------------------------

INDIC_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

# CPU default decoding head ("ctc"; the conformer's remote code also offers
# "rnnt"). CTC is deterministic and needs no streaming state.
INDIC_DECODE = "ctc"

# Conformer sample rate + piper's WAV rate (needs resampling to 16 kHz).
INDIC_SAMPLE_RATE = 16000

# The 22 languages from the indic-conformer-600m-multilingual model card.
INDIC_CONFORMER_LANGUAGES = frozenset({
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
})

# Native languages from voiceagent.langid that we route to the conformer
# (subset of the card list, verified 2026-09). Everything else — en, hi,
# hinglish, None, unknown — stays on the proven whisper-small path.
INDIC_ROUTE_LANGS = frozenset({"te", "ta", "bn", "mr", "gu", "kn", "ml", "pa"})


def _real_whisper_engine_loader(model: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model, device=device, compute_type=compute_type)


class WhisperASRHandle:
    """Uniform ASR handle over faster-whisper for the router.

    engine_loader is injectable so tests stub the whisper layer without
    downloading models.
    """

    def __init__(self, model: str = "small", device: str = "cpu",
                 compute_type: str = "int8", engine_loader=None):
        self._model = model
        self._device = device
        self._compute_type = compute_type
        self._engine_loader = engine_loader or _real_whisper_engine_loader
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = self._engine_loader(self._model, self._device,
                                               self._compute_type)
        return self._engine

    def transcribe(self, audio, language: str | None = None) -> str:
        """Transcribe a WAV path or 16 kHz mono float array. language=None
        keeps whisper auto-detection; "hinglish" maps to a "hi" hint (the
        audio is Hindi — whisper has no "hinglish" code)."""
        return self.transcribe_detected(audio, language)[0]

    def transcribe_detected(self, audio, language: str | None = None,
                            reroute: frozenset = frozenset()):
        """One whisper pass returning (text, detected_language). With
        language=None the encoder's language-id result comes back too — and
        if it is in `reroute`, segments are NOT decoded (auto-detection
        happens before any decoder pass, so skipping the iteration skips the
        pathological decode — M5b-1: whisper te hallucination-loops at every
        size) and text is None. An explicit language hint never reroutes:
        the caller already knows better."""
        hint = {"hinglish": "hi"}.get(language, language)
        engine = self._ensure_engine()
        segments, info = engine.transcribe(audio, language=hint)
        lang = getattr(info, "language", None)
        if hint is None and lang in reroute:
            return None, lang
        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        return text, lang


def _real_indic_loader(model_id: str):
    """Load the conformer via its remote-code recipe (model card):
    AutoModel + trust_remote_code; the callable takes (wav, lang, decode)."""
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model


def _wav_to_float16k_array(path: str) -> "np.ndarray":
    """Read a 16-bit mono PCM WAV into a float32 array in [-1, 1], resampled
    to 16 kHz (piper writes 22.05 kHz; the conformer wants 16 kHz)."""
    import numpy as np
    import wave
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        if w.getsampwidth() != 2:
            raise ValueError(f"only 16-bit PCM WAV supported, got sampwidth="
                             f"{w.getsampwidth()} for {path}")
        frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if rate != INDIC_SAMPLE_RATE:
        import math
        from scipy.signal import resample_poly
        g = math.gcd(INDIC_SAMPLE_RATE, rate)
        audio = resample_poly(audio, INDIC_SAMPLE_RATE // g, rate // g)
    return audio


class IndicASRHandle:
    """ai4bharat/indic-conformer-600m-multilingual handle (CPU, eval mode).

    The conformer requires the target language per its recipe, so transcribe
    RAISES on language=None — the router never routes unknown languages here.
    loader is a zero-arg injectable callable returning the model, so tests
    stub the 2.4 GB download."""

    def __init__(self, model_id: str = INDIC_MODEL_ID, decode: str = INDIC_DECODE,
                 loader=None):
        self._model_id = model_id
        self._decode = decode
        self._loader = loader or (lambda: _real_indic_loader(model_id))
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = self._loader()
        return self._model

    def transcribe(self, audio, language: str | None = None) -> str:
        """Transcribe a 16 kHz mono float array (or 16-bit WAV path, which is
        resampled to 16 kHz). Returns the transcript text."""
        if not language:
            raise ValueError("IndicConformer requires an explicit language "
                             "(the router supplies it); got None")
        import torch
        import numpy as np
        if isinstance(audio, (str, Path)):
            audio = _wav_to_float16k_array(audio)
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        model = self._ensure_model()
        with torch.no_grad():
            text = model(wav, language, self._decode)
        return text.strip() if isinstance(text, str) else text


def _get_whisper_small() -> WhisperASRHandle:
    """Lazy singleton for the router's whisper-small path."""
    global _whisper_small_handle
    if _whisper_small_handle is None:
        _whisper_small_handle = WhisperASRHandle(model="small")
    return _whisper_small_handle


def _get_indic_asr() -> IndicASRHandle:
    """Lazy singleton for the router's IndicConformer path (the ~2.4 GB
    model downloads on first te/ta use)."""
    global _indic_handle
    if _indic_handle is None:
        _indic_handle = IndicASRHandle()
    return _indic_handle


_whisper_small_handle: WhisperASRHandle | None = None
_indic_handle: IndicASRHandle | None = None


def get_asr_for_language(lang: str | None, engines=None, supported=None,
                         warn=None):
    """Route a language to an ASR handle with a uniform
    transcribe(audio, language) interface.

    en/hi/hinglish -> whisper small (M5b-1: proven); te/ta/bn/mr/gu/kn/ml/pa
    -> IndicConformer; None/unknown -> whisper small auto-detect. A native
    language outside the conformer's card list falls back to whisper small
    with a warning. `engines`/`supported`/`warn` are injectable for tests;
    default engines are cached per engine kind (process-wide singletons).
    """
    if engines is None:
        engines = {"whisper": _get_whisper_small, "indic": _get_indic_asr}
    if supported is None:
        supported = INDIC_CONFORMER_LANGUAGES
    if warn is None:
        warn = lambda msg: logger.warning(msg)  # noqa: E731
    if lang in INDIC_ROUTE_LANGS:
        if lang in supported:
            return engines["indic"]()
        warn(f"language '{lang}' is not supported by "
             f"{INDIC_MODEL_ID}; falling back to whisper small")
    return engines["whisper"]()


def transcribe_wav_routed(path: str, language: str | None = None) -> str:
    """Transcribe a WAV file through the language router. language=None
    defers to whisper's auto-detection on the small model."""
    return get_asr_for_language(language).transcribe(path, language=language)


def transcribe_wav_auto(path: str, whisper: WhisperASRHandle | None = None,
                        indic: IndicASRHandle | None = None) -> str:
    """Production voice-loop entry (M5b-2): one whisper-small pass with
    auto-detection. A detected te/ta/native-Indic language skips whisper's
    decode (M5b-1: te hallucination-loops at every size) and re-transcribes
    with IndicConformer; en/hi/hinglish/unknown decode on whisper small —
    the proven path (en WER 0.038, hi 0.423). Handles are injectable for
    tests; defaults are the process-wide lazy singletons."""
    whisper = whisper if whisper is not None else _get_whisper_small()
    indic = indic if indic is not None else _get_indic_asr()
    text, lang = whisper.transcribe_detected(path, reroute=INDIC_ROUTE_LANGS)
    if text is None:
        return indic.transcribe(path, language=lang)
    return text
