# tests/test_tts.py
from voiceagent.tts import TTSHandle, VOICE_REGISTRY, synthesize_to_wav

def test_synthesize_to_wav_creates_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "out.wav")
        seconds = synthesize_to_wav("Namaste, aapka order kal deliver ho jayega.", out)
        assert seconds >= 0
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


# ---------------------------------------------------------------------------
# Global languages (es/fr/de/pt) route to dedicated piper voices. Serving the
# en voice to an es/fr/de/pt customer is wrong-language audio. Names verified
# against rhasspy/piper-voices at authoring time; this test stays offline.
# ---------------------------------------------------------------------------

def test_registry_routes_global_languages():
    assert VOICE_REGISTRY["es"] == "es_MX-ald-medium"
    assert VOICE_REGISTRY["fr"] == "fr_FR-siwis-medium"
    assert VOICE_REGISTRY["de"] == "de_DE-thorsten-medium"
    assert VOICE_REGISTRY["pt"] == "pt_BR-faber-medium"


def test_global_voice_names_match_piper_hf_url_layout():
    # voice_urls() splits "<lang>_<region>-<speaker>-<quality>" — every name
    # must fit the 3-part pattern or the download URL is malformed.
    from voiceagent.voice import voice_urls
    for lang in ("es", "fr", "de", "pt"):
        onnx, cfg = voice_urls(VOICE_REGISTRY[lang])
        name = VOICE_REGISTRY[lang]
        assert onnx.endswith(f"/{name}.onnx")
        assert cfg.endswith(f"/{name}.onnx.json")
        assert f"/{name.split('-')[0]}/" in onnx


def test_handle_voice_for_routes_global_languages():
    # voice_for() only — no download, no piper load.
    h = TTSHandle(model_dir="/nonexistent", voice_loader=lambda *a: None)
    for lang in ("es", "fr", "de", "pt"):
        assert h.voice_for(lang, "") == (lang, VOICE_REGISTRY[lang])


def test_speech_text_spells_long_digit_runs():
    from voiceagent.tts import speech_text

    out = speech_text("Your number 9828379313 is confirmed")
    assert out == "Your number 9 8 2 8 3 7 9 3 1 3 is confirmed"


def test_speech_text_strips_emoji_keeps_money():
    from voiceagent.tts import speech_text

    out = speech_text("Refund of ₹200 approved 😊🌟")
    assert "₹200" in out
    assert "😊" not in out and "🌟" not in out


def test_speech_text_leaves_short_numbers_alone():
    from voiceagent.tts import speech_text

    assert speech_text("Order 7734 is ready") == "Order 7734 is ready"
