# src/voiceagent/tts.py
"""M5b-1: multilingual TTS on top of piper.

VOICE_REGISTRY maps a text language (as reported by voiceagent.langid) to a
piper voice name — loaded from data/lang/*.yaml (`tts_voice:`), verified on
HF rhasspy/piper-voices at authoring time. Voices download on demand into
data/models/ (same pattern as voice.ensure_voice, guarded by one shared lock
so concurrent first-uses never double-download).

Language decisions (mechanism, not data):
- alias codes resolve to their concrete voice (`alias_of:` in the lang
  file): Romanized Hindi speaks with the Hindi voice — Hindi-accented
  audio beats dropping to English. Quality caveat accepted; revisit if a
  dedicated voice appears (one data line, no code change).
- Languages with no declared voice fall back to the fallback voice with a
  warning. Production paths never raise.

The M3 chunked/streaming synthesis (first chunk streams while later chunks
generate) is preserved for every registered language.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
import wave
from typing import Callable

from voiceagent.langdata import alias_for as _alias_for
from voiceagent.langdata import voice_registry as _load_registry
from voiceagent.langid import detect_language
from voiceagent.voice import ensure_voice

logger = logging.getLogger(__name__)

# Text language -> piper voice name, from lang files (add-a-voice =
# add-a-line in data/lang/<code>.yaml).
VOICE_REGISTRY = _load_registry()

# Speech rate: >1 slower, <1 faster. Env-overridable so deployments tune the
# perceived pace without code changes (1.0 = the voice's native rate).
DEFAULT_LENGTH_SCALE = 1.0


def voice_overrides_from_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Per-language voice overrides from VOICEAGENT_TTS_VOICES, formatted
    'en=en_US-amy-medium,hi=hi_IN-priyamvada-medium'. Deployment config, not
    code: switching voices never touches the registry default."""
    import os as _os
    e = _os.environ if env is None else env
    raw = e.get("VOICEAGENT_TTS_VOICES") or ""
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" in pair:
            lang, voice = pair.split("=", 1)
            if lang.strip() and voice.strip():
                out[lang.strip()] = voice.strip()
    return out

def resolve_voice_lang(language: str | None) -> str | None:
    """Language code -> voice-resolution code, following the lang file's
    `alias_of:` (Romanized Hindi resolves to the Hindi voice). Detection
    codes pass through untouched; only voice selection follows aliases."""

    if not language:
        return language
    return _alias_for(language) or language

# One shared lock guarding download+load across handles and threads.
_VOICE_LOCK = threading.Lock()


def _real_voice_loader(voice_name: str, model_dir: str):
    """Default piper voice loader: download if missing, then load."""
    from piper import PiperVoice  # piper-tts
    with _VOICE_LOCK:
        onnx = ensure_voice(voice_name, model_dir)
        return PiperVoice.load(onnx)



_RE_DIGIT_RUN = re.compile(r"\d{8,}")
_RE_NONSPEECH = __import__("re").compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2190-\u21FF\u2B00-\u2BFF]")


_RE_MD_BOLD = re.compile(r"\*\*|__|\*|`|#")
_RE_MD_BULLET = re.compile(r"(?m)^\s*[-*+]\s+")
_TYPO = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2011": "-", "\u2013": "-", "\u2014": "-", "\u2026": "...",
    "\u202f": " ", "\u00a0": " ",
}
# Order-ID-shaped tokens: 2-4 letters (+optional separators) then >=3 digits —
# ALWAYS spelled letter-by-letter + digit-by-digit ("ORD-9021" -> "O R D,
# nine zero two one"), never as words/magnitudes.
_RE_ALNUM_ID = re.compile(r"\b([A-Za-z]{2,4})[ -]?((?:(?:[ -]?\d){3,}))\b")


def speech_text(text: str) -> str:
    """Make model text speakable — UNIVERSAL reading mechanics, not domain
    data (domain specifics still arrive via memory/knowledge):
    - markdown syntax (**bold**, bullets, backticks, #) is stripped — the
      brain writes markdown and the phonemizer gurgles on it;
    - typographic characters (curly quotes, non-breaking hyphen, narrow
      spaces, ellipsis) are folded to their plain equivalents;
    - order-ID-shaped tokens are spelled letter-by-letter + digit-by-digit;
    - long digit runs (phone numbers) are read digit-by-digit — never as
      magnitudes ("9 billion");
    - emoji / symbol codepoints are dropped."""
    def _spell_digits(m: "re.Match[str]") -> str:
        return " ".join(m.group(0))
    def _spell_id(m: "re.Match[str]") -> str:
        letters = " ".join(m.group(1).upper())
        digits = " ".join(m.group(2).replace(" ", "").replace("-", ""))
        return f"{letters}, {digits}"
    for bad, good in _TYPO.items():
        text = text.replace(bad, good)
    text = _RE_NONSPEECH.sub(" ", text)
    text = _RE_MD_BOLD.sub("", text)
    text = _RE_MD_BULLET.sub("", text)
    text = _RE_DIGIT_RUN.sub(_spell_digits, text)
    text = _RE_ALNUM_ID.sub(_spell_id, text)
    return re.sub(r"\s{2,}", " ", text).strip()


class TTSHandle:
    """Multilingual TTS handle: routes text to per-language piper voices.

    voice_loader and warn are injectable so tests can stub the piper layer
    and capture fallback warnings without downloading anything.
    """

    def __init__(self, model_dir: str = "data/models",
                 registry: dict[str, str] | None = None,
                 fallback_voice: str = "en",
                 length_scale: float | None = None,
                 voice_loader: Callable[[str, str], object] | None = None,
                 warn: Callable[[str], None] | None = None):
        self._model_dir = model_dir
        self._registry = dict(registry if registry is not None else VOICE_REGISTRY)
        self._registry.update(voice_overrides_from_env())
        self._fallback_voice = fallback_voice
        if length_scale is None:
            length_scale = float(os.environ.get("VOICEAGENT_TTS_LENGTH_SCALE",
                                                DEFAULT_LENGTH_SCALE))
        self._length_scale = length_scale
        self._voice_loader = voice_loader or _real_voice_loader
        self._warn = warn or (lambda msg: logger.warning(msg))
        self._voices: dict[str, object] = {}

    def voice_for(self, language: str | None, text: str = "") -> tuple[str, str]:
        """Resolve (lang, voice_name). language=None -> langid auto-detect.
        Unregistered languages fall back (never raise) with a warning."""
        detected = language if language else detect_language(text)
        lang = resolve_voice_lang(detected) or detected
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

    def warm(self, language: str | None = None) -> str:
        """Preload the voice for a deployment's declared language (download
        + load now, not on the first caller). Returns the voice name.
        Fail-open: callers wrap this."""
        base = (str(language or "en").strip().lower().replace("_", "-")
                .split("-")[0] or "en")
        _, voice_name = self.voice_for(base, "")
        self._get_voice(voice_name)
        return voice_name

    def speak(self, text: str, language: str | None = None,
              out_path: str | None = None) -> str:
        """Synthesize text to a WAV file (auto-detecting the language when
        language=None). Returns the wav path."""
        if out_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name
        text = speech_text(text)
        _, voice_name = self.voice_for(language, text)
        voice = self._get_voice(voice_name)
        with wave.open(out_path, "wb") as w:
            if self._length_scale != DEFAULT_LENGTH_SCALE:
                from piper.config import SynthesisConfig
                voice.synthesize_wav(  # type: ignore[attr-defined]
                    text, w, syn_config=SynthesisConfig(
                        length_scale=self._length_scale))
            else:
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
