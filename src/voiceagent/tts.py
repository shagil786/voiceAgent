# src/voiceagent/tts.py
"""M5b-1: multilingual TTS on top of piper.

VOICE_REGISTRY maps a text language (as reported by voiceagent.langid) to a
piper voice name from HF rhasspy/piper-voices. Voices download on demand into
data/models/ (same pattern as voice.ensure_voice, guarded by one shared lock
so concurrent first-uses never double-download).

Language decisions:
- "hinglish" (Romanized Hindi) routes to the hi voice. A Hindi voice reading
  Latin-script Hindi is a known imperfect case (phonemization leans
  English-ish), but speaking Hindi-accented audio is closer to the customer
  than dropping to the English voice. Quality caveat accepted, revisit if a
  dedicated hinglish voice becomes available.
- Languages with no piper voice (ta, bn, mr, gu, kn, ml, pa, ...) fall back
  to the fallback voice with a warning. Production paths never raise.

The M3 chunked/streaming synthesis (first chunk streams while later chunks
generate) is preserved for every registered language.
"""
from __future__ import annotations

import logging
import tempfile
import threading
import time
import wave
from typing import Callable

from voiceagent.langid import detect_language
from voiceagent.voice import ensure_voice

logger = logging.getLogger(__name__)

# Text language -> piper voice name. en: the M3 English voice; hi: pratham
# medium; te: maya medium (verified on HF te/te_IN, no medium Tamil exists).
# Global target set verified on HF at authoring time (HEAD 200): es_MX-ald,
# fr_FR-siwis, de_DE-thorsten, pt_BR-faber — the README's es/fr/de/pt callers
# must not receive the en voice.
VOICE_REGISTRY = {
    "en": "en_US-lessac-medium",
    "hi": "hi_IN-pratham-medium",
    "te": "te_IN-maya-medium",
    "es": "es_MX-ald-medium",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "pt": "pt_BR-faber-medium",
}

# Romanized Hindi -> Hindi voice (see module docstring for the quality caveat).
HINGLISH_VOICE_LANG = "hi"

# One shared lock guarding download+load across handles and threads.
_VOICE_LOCK = threading.Lock()


def _real_voice_loader(voice_name: str, model_dir: str):
    """Default piper voice loader: download if missing, then load."""
    from piper import PiperVoice  # piper-tts
    with _VOICE_LOCK:
        onnx = ensure_voice(voice_name, model_dir)
        return PiperVoice.load(onnx)


class TTSHandle:
    """Multilingual TTS handle: routes text to per-language piper voices.

    voice_loader and warn are injectable so tests can stub the piper layer
    and capture fallback warnings without downloading anything.
    """

    def __init__(self, model_dir: str = "data/models",
                 registry: dict[str, str] | None = None,
                 fallback_voice: str = "en",
                 voice_loader: Callable[[str, str], object] | None = None,
                 warn: Callable[[str], None] | None = None):
        self._model_dir = model_dir
        self._registry = dict(registry if registry is not None else VOICE_REGISTRY)
        self._fallback_voice = fallback_voice
        self._voice_loader = voice_loader or _real_voice_loader
        self._warn = warn or (lambda msg: logger.warning(msg))
        self._voices: dict[str, object] = {}

    def voice_for(self, language: str | None, text: str = "") -> tuple[str, str]:
        """Resolve (lang, voice_name). language=None -> langid auto-detect.
        Unregistered languages fall back (never raise) with a warning."""
        lang = language if language else detect_language(text)
        if lang == "hinglish":
            lang = HINGLISH_VOICE_LANG
        voice_name = self._registry.get(lang)
        if voice_name is None:
            fb_name = self._registry.get(self._fallback_voice, self._fallback_voice)
            self._warn(f"no piper voice registered for language '{lang}'; "
                       f"falling back to '{fb_name}'")
            return self._fallback_voice, fb_name
        return lang, voice_name

    def _get_voice(self, voice_name: str):
        voice = self._voices.get(voice_name)
        if voice is None:
            voice = self._voice_loader(voice_name, self._model_dir)
            self._voices[voice_name] = voice
        return voice

    def speak(self, text: str, language: str | None = None,
              out_path: str | None = None) -> str:
        """Synthesize text to a WAV file (auto-detecting the language when
        language=None). Returns the wav path."""
        if out_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name
        _, voice_name = self.voice_for(language, text)
        voice = self._get_voice(voice_name)
        with wave.open(out_path, "wb") as w:
            voice.synthesize_wav(text, w)  # type: ignore[attr-defined]
        return out_path

    def synthesize_to_wav(self, text: str, out_path: str,
                          language: str | None = None) -> float:
        """Full utterance -> WAV. Returns seconds taken."""
        t0 = time.time()
        self.speak(text, language=language, out_path=out_path)
        return time.time() - t0

    def synthesize_chunks(self, text: str, chunk_chars: int = 80,
                          language: str | None = None) -> list[tuple[str, float]]:
        """Break text into word-boundary chunks and synthesize each with the
        language's voice. Returns [(wav_path, synth_ms)] so the first chunk
        can stream while later ones generate (low perceived latency)."""
        words = text.split()
        chunks, cur, cur_len = [], [], 0
        for w in words:
            cur.append(w)
            cur_len += len(w) + 1
            if cur_len >= chunk_chars:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
        if cur:
            chunks.append(" ".join(cur))

        _, voice_name = self.voice_for(language, text)
        voice = self._get_voice(voice_name)
        out = []
        for chunk in chunks:
            t0 = time.time()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            with wave.open(path, "wb") as w:
                voice.synthesize_wav(chunk, w)  # type: ignore[attr-defined]
            out.append((path, (time.time() - t0) * 1000))
        return out


_default_handle: TTSHandle | None = None


def get_tts_handle() -> TTSHandle:
    """Process-wide default handle (real piper layer, data/models cache)."""
    global _default_handle
    if _default_handle is None:
        _default_handle = TTSHandle()
    return _default_handle


def speak(text: str, language: str | None = None,
          out_path: str | None = None) -> str:
    """Module-level entry: synthesize text to WAV in the customer's language."""
    return get_tts_handle().speak(text, language=language, out_path=out_path)


def synthesize_to_wav(text: str, out_path: str) -> float:
    """M3-compatible wrapper: full utterance -> WAV, returns seconds taken."""
    return get_tts_handle().synthesize_to_wav(text, out_path)


def synthesize_chunks(text: str, chunk_chars: int = 80,
                      language: str | None = None) -> list[tuple[str, float]]:
    """M3-compatible wrapper for the chunked/streaming path, now
    language-aware (language=None auto-detects)."""
    return get_tts_handle().synthesize_chunks(text, chunk_chars, language)
