# src/voiceagent/asr.py
"""ASR for the voice path.

M5b-3 language-routed ASR, engines chosen by loopback bake-off
(data/out/asr-measurement-qwen.json + asr-measurement-indic.json):

- QwenASRHandle (core slot): Qwen/Qwen3-ASR-0.6B-hf, Apache-2.0. Serves
  en/hi/hinglish/unknown. en WER 0.038 ties whisper small at ~3x the CPU
  speed; hi WER 0.247 beats whisper 0.423; natively emits code-switched
  text (Latin-script words inside Hindi) that feeds the hinglish-text
  LLM/intent pipeline. Built-in LID (correct on all bake-off en/hi samples).
- IndicASRHandle (native slot): ai4bharat/indic-conformer-600m-multilingual
  (CTC head, 22 Indic languages, ~2.4 GB download, GATED repo — license
  acceptance + HF token needed once). te WER 0.348 vs whisper 1.067 /
  Qwen 1.315 at ~0.11s warm. The conformer REQUIRES the target language as
  input per its recipe — the router supplies it from known-language context.
- WhisperASRHandle: faster-whisper small — the legacy engine, kept as the
  router's failure fallback (battle-tested, always cached).

Routing (get_asr_for_language / transcribe_wav_routed, the voice-loop entry):
te/ta/bn/mr/gu/kn/ml/pa -> IndicConformer; everything else -> Qwen. The
BLIND path never auto-reroutes on detected language: whisper-era probing
showed language detection cannot separate Hinglish from native Indic audio
(hinglish demo query detected as te/pa @0.74 while a REAL te sample scored
0.69) — native languages must be supplied out of band (telephony trunk
config, per-vertical config).

Loopback caveat throughout: piper TTS audio (clean, no mic/channel noise) —
WER numbers are optimistic; they decide engines relative to each other.
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

# M5b-3 core engine (bake-off winner, data/out/asr-measurement-qwen.json):
# en WER 0.038 ties whisper small at ~3x the speed (0.84s vs 2.77s median);
# hi WER 0.247 beats whisper 0.423; natively emits code-switched text
# (Latin-script English words inside Hindi), which feeds the hinglish-text
# LLM/intent pipeline directly. Apache-2.0. te/ta are NOT covered (detected
# as Hindi) — native languages route to IndicConformer instead.
QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"

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

    def transcribe_detected(self, audio, language: str | None = None):
        """One whisper pass returning (text, detected_language). With
        language=None the encoder's language-id result comes back too — for
        logging and known-language corroboration. The detected language is
        deliberately NOT used to reroute (see module docstring: whisper
        cannot reliably separate Hinglish from Indic-native audio)."""
        hint = {"hinglish": "hi"}.get(language, language)
        engine = self._ensure_engine()
        segments, info = engine.transcribe(audio, language=hint)
        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        return text, getattr(info, "language", None)


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


def _real_qwen_loader(model_id: str):
    """Load Qwen3-ASR via transformers (needs transformers>=5.13; the project
    venv upgraded 4.46.3 -> 5.16.1 with sentence-transformers 5->6 — the
    full suite was re-verified green on the upgrade)."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id,
                                                     dtype=torch.float32)
    model.eval()
    return processor, model


class QwenASRHandle:
    """Qwen/Qwen3-ASR-0.6B-hf handle — the M5b-3 core engine (en/hi/hinglish
    and any unknown language).

    Language hints are accepted for interface uniformity but NOT forced:
    the model's built-in LID was correct on every en/hi bake-off sample
    (including hinglish audio, detected as Hindi), and forcing via the
    assistant-prefill mechanism is deferred until a measured need shows up.
    loader is injectable so tests stub the ~1.6 GB download."""

    def __init__(self, model_id: str = QWEN_ASR_MODEL_ID, loader=None):
        self._model_id = model_id
        self._loader = loader or (lambda: _real_qwen_loader(model_id))
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = self._loader()
        return self._engine

    @staticmethod
    def _array_to_wav_path(audio) -> str:
        """16 kHz mono float array -> temp 16-bit PCM WAV (the processor
        consumes file paths; the voice loop normally passes paths anyway)."""
        import numpy as np
        import tempfile
        import wave
        data = (np.asarray(audio, dtype=np.float32).clip(-1, 1)
                * 32767).astype(np.int16).tobytes()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(data)
            return tmp.name

    def transcribe(self, audio, language: str | None = None) -> str:
        """Transcribe a WAV path (or 16 kHz mono float array) to text.
        Returns the transcription; Qwen's detected language is available via
        transcribe_detected()."""
        return self.transcribe_detected(audio, language)[0]

    def transcribe_detected(self, audio, language: str | None = None):
        """One Qwen pass returning (text, detected_language_or_None)."""
        import torch
        path = audio if isinstance(audio, (str, Path)) \
            else self._array_to_wav_path(audio)
        processor, model = self._ensure_engine()
        inputs = processor.apply_transcription_request(audio=str(path))
        inputs = inputs.to(model.device, model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        try:
            parsed = processor.decode(gen[0], return_format="parsed")
            return parsed.get("transcription", "").strip(), parsed.get("language")
        except Exception:
            text = processor.decode(gen[0], skip_special_tokens=True).strip()
            return text, None


def _get_whisper_small() -> WhisperASRHandle:
    """Lazy singleton for whisper small — the M5b-3 failure fallback engine
    (battle-tested, always available offline once cached)."""
    global _whisper_small_handle
    if _whisper_small_handle is None:
        _whisper_small_handle = WhisperASRHandle(model="small")
    return _whisper_small_handle


def _get_qwen_asr() -> QwenASRHandle:
    """Lazy singleton for the router's core path (~1.6 GB model downloads on
    first use; float32 ~3.1 GB RAM — int8/bf16 quantization is the VPS-tier
    optimization path, deferred until measured)."""
    global _qwen_handle
    if _qwen_handle is None:
        _qwen_handle = QwenASRHandle()
    return _qwen_handle


def _get_indic_asr() -> IndicASRHandle:
    """Lazy singleton for the router's IndicConformer path (the ~2.4 GB
    model downloads on first te/ta use)."""
    global _indic_handle
    if _indic_handle is None:
        _indic_handle = IndicASRHandle()
    return _indic_handle


_whisper_small_handle: WhisperASRHandle | None = None
_qwen_handle: QwenASRHandle | None = None
_indic_handle: IndicASRHandle | None = None


def get_asr_for_language(lang: str | None, engines=None, supported=None,
                         warn=None):
    """Route a language to an ASR handle with a uniform
    transcribe(audio, language) interface.

    M5b-3 routing (bake-off data): te/ta/bn/mr/gu/kn/ml/pa -> IndicConformer
    (te WER 0.348 vs whisper 1.067 / Qwen 1.315, 0.11s warm); everything
    else — en, hi, hinglish, None, unknown — -> Qwen3-ASR-0.6B (en tie with
    whisper at 3x speed, hi 0.247 vs 0.423, code-switch output). A native
    language outside the conformer's card list falls back to the Qwen core
    with a warning. `engines`/`supported`/`warn` are injectable for tests;
    default engines are cached per engine kind (process-wide singletons).
    """
    if engines is None:
        engines = {"qwen": _get_qwen_asr, "indic": _get_indic_asr}
    if supported is None:
        supported = INDIC_CONFORMER_LANGUAGES
    if warn is None:
        warn = lambda msg: logger.warning(msg)  # noqa: E731
    if lang in INDIC_ROUTE_LANGS:
        if lang in supported:
            return engines["indic"]()
        warn(f"language '{lang}' is not supported by "
             f"{INDIC_MODEL_ID}; falling back to the Qwen core engine")
    return engines["qwen"]()


def transcribe_wav_routed(path: str, language: str | None = None) -> str:
    """Transcribe a WAV file through the language router — the voice-loop
    entry when the deployment knows its language (telephony trunk config,
    per-vertical config, loopback tests). language=None defers to whisper's
    auto-detection on the small model (the blind path — never auto-reroutes,
    see module docstring).

    If the routed engine fails (model download/gated repo, load error), the
    turn must not die mid-call: falls back to whisper small with a warning."""
    try:
        return get_asr_for_language(language).transcribe(path, language=language)
    except Exception as e:
        logger.warning("routed ASR failed for language=%s (%s: %s); "
                       "falling back to whisper small",
                       language, type(e).__name__, e)
        return _get_whisper_small().transcribe(path, language=language)
