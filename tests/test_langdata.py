"""data/lang/ is the source of truth for per-language tables: a language is
a file, never a code branch. These tests pin that posture — the engine
must serve whatever the directory declares, and every language the
platform claims to serve must declare itself here."""
from pathlib import Path

from voiceagent import langdata
from voiceagent.entities import extract_entities, words_to_number
from voiceagent.sentiment import detect_frustration

LANG_DIR = Path("data/lang")


def test_every_served_language_has_a_lang_file():
    # Languages the platform claims (TTS voices + langid script ranges +
    # ASR routing) minus known vocabulary gaps must ship tables.
    from voiceagent.tts import VOICE_REGISTRY
    served = set(VOICE_REGISTRY)
    known_gaps = {"ta", "gu", "kn", "pa", "mr", "ml", "ur", "ar"}
    have = {p.stem for p in LANG_DIR.glob("*.yaml")}
    missing = served - have - known_gaps
    assert not missing, f"served but undeclared: {sorted(missing)}"
    assert {"en", "hi", "te", "ta", "bn", "th"} <= have


def test_add_a_language_is_add_a_file(tmp_path, monkeypatch):
    # A brand-new language file is picked up with zero code changes: the
    # file parses, and the engine (pure table lookups) serves the merged
    # tables — file -> loader -> engine, no branch per language.
    (tmp_path / "xx.yaml").write_text(
        "code: xx\nnumbers:\n  words:\n    zork: 1\n    blip: 2\n"
        "  scales:\n    zillion: 1000\n  hundred: []\n"
        "sentiment:\n  - grr xx\n",
        encoding="utf-8",
    )
    tables = langdata.load_lang_tables(tmp_path)
    assert tables["xx"]["words"] == {"zork": 1, "blip": 2}
    assert tables["xx"]["scales"] == {"zillion": 1000}
    assert tables["xx"]["sentiment"] == ["grr xx"]
    from voiceagent import entities as ent
    monkeypatch.setitem(ent._WORD_VALUES, "zork", 1)
    monkeypatch.setitem(ent._SCALE_VALUES, "zillion", 1000)
    assert ent.words_to_number(["zork", "zillion"]) == 1000


def test_shipped_tables_drive_real_parsers():
    assert extract_entities("ఐదు వేలు రూపాయలు",
                            currency="₹").amount == 5000.0
    assert extract_entities("ห้าพันบาท", currency="฿").amount == 5000.0
    assert detect_frustration("service ta nghỉ", language="te").level == "none"
    assert detect_frustration(" service is โกรธ มาก",
                              language="th").frustrated


def test_companions_are_declared_not_coded():
    import inspect
    from voiceagent import sentiment as s
    src = inspect.getsource(s._phrases_for)
    for code in ("hi", "hinglish", "te", "ta"):
        assert f'"{code}"' not in src and f"'{code}'" not in src, \
            f"language {code!r} named in sentiment code"
    assert langdata.companions_for("hi") == ("hinglish",)
    assert langdata.companions_for("hinglish") == ("hi",)
    assert langdata.companions_for("te") == ()
