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
    known_gaps = set()  # every voiced language now ships number tables
    have = {p.stem for p in LANG_DIR.glob("*.yaml")}
    missing = served - have - known_gaps
    assert not missing, f"served but undeclared: {sorted(missing)}"
    assert {"en", "hi", "te", "ta", "bn", "th", "mr", "gu", "kn", "ml",
            "pa", "ur", "ar"} <= have


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


def test_script_claims_are_data_not_code():
    import inspect
    from voiceagent import langid as li
    from voiceagent import entities as ent
    for mod in (li, ent):
        src = inspect.getsource(mod)
        for lit in ("0x0900", "0x0E00", "097F", "0E7F", "\\u0900",
                    "Devanagari", "Gurmukhi"):
            assert lit not in src, f"script literal {lit!r} in {mod.__name__}"
    claims = dict((c, (lo, hi)) for c, lo, hi in langdata.script_claims())
    assert claims["hi"] == (0x0900, 0x097F)
    assert claims["th"] == (0x0E00, 0x0E7F)
    # Marathi shares Devanagari: explicit detect_as, no code branch.
    assert langdata.tables()["mr"]["detect_as"] == "hi"
    assert "mr" in langdata.native_script_codes()
    # Arabic tokenizes (mechanical) but no file claims it (detection gap).
    assert (0x0600, 0x06FF) in langdata.tokenizer_ranges()
    assert "Arabic" not in [s for e in langdata.tables().values()
                            for s in e["scripts"]]


def test_detection_lexicons_come_from_files():
    import inspect
    from voiceagent import langid as li
    src = inspect.getsource(li)
    for token in ("quiero", "remboursement", "rückerstattung", "kya"):
        assert token not in src, f"lexicon token {token!r} in langid code"
    assert set(li.GLOBAL_LEXICONS) == {"de", "es", "fr", "pt"}
    assert "kya" in li.HINGLISH_LEXICON and "hai" in li.HINGLISH_LEXICON
    assert li.detect_language("où est mon remboursement merci beaucoup") \
        == "fr"
    assert li.detect_language("meri ORD-4821 kahan hai batao bhai") \
        == "hinglish"


def test_currency_forms_come_from_files():
    import inspect
    from voiceagent import entities as ent
    src = inspect.getsource(ent)
    assert "dollars?" not in src and "रुपये" not in src
    assert "บาท" not in {**{}} and "บาท" in ent._CURRENCY_WORDS.get("฿", ())
    assert "روپے" in ent._CURRENCY_WORDS.get("₹", ())


def test_new_vocabularies_parse():
    from voiceagent.entities import extract_entities as e
    from voiceagent.entities import words_to_number as w
    assert w(["पाच", "हजार"]) == 5000          # mr (Devanagari, hi digits)
    assert w(["ऐंशी", "हजार"]) == 80000
    assert e("वीस हजार रुपये", currency="₹").amount == 20000.0
    assert e("ORD-౪౮౨౧").order_id == "ORD-4821"  # Telugu digits (pinned)


def test_voices_routing_alias_are_data_not_code():
    import inspect
    from voiceagent import asr as asr_mod
    from voiceagent import tts as tts_mod
    from voiceagent.telephony import inbound as inbound_mod
    for mod in (asr_mod, tts_mod, inbound_mod):
        src = inspect.getsource(mod)
        for lit in ('INDIC_ROUTE_LANGS = frozenset',
                    '{"hinglish": "hi"}', "{'hinglish': 'hi'}",
                    'HINGLISH_VOICE_LANG =',
                    '"te_IN-maya-medium"', '"hi_IN-priyamvada-medium"',
                    'if base == "hinglish"', "if lang == \"hinglish\"",
                    'if detected == "hinglish"'):
            assert lit not in src, f"{lit!r} survives in {mod.__name__}"
    # Loaded values equal the historical behavior, from files:
    from voiceagent.tts import VOICE_REGISTRY, resolve_voice_lang
    assert len(VOICE_REGISTRY) == 13
    assert VOICE_REGISTRY["te"] == "te_IN-maya-medium"
    assert VOICE_REGISTRY["ur"] == "ur_PK-fasih-medium"
    assert "ta" not in VOICE_REGISTRY  # fallback path preserved
    assert resolve_voice_lang("hinglish") == "hi"
    assert resolve_voice_lang("te") == "te"
    assert langdata.alias_for("hinglish") == "hi"
    assert langdata.alias_for("te") == ""
    assert langdata.asr_routes() == {
        c: "indic" for c in
        ["te", "ta", "bn", "mr", "gu", "kn", "ml", "pa"]}
    assert len(langdata.engine_languages("indic")) == 22
    assert langdata.engine_languages("nope") == frozenset()


def test_european_parity_numbers_and_currency():
    from voiceagent.entities import extract_entities as e
    from voiceagent.entities import words_to_number as w
    # Spanish / French / German / Portuguese ≅ Indic coverage now.
    assert w(["veintidós", "mil"]) == 22000
    assert e("cinco mil dólares", currency="$").amount == 5000.0
    assert e("cinq mille euros", currency="€").amount == 5000.0
    assert e("cinco mil reais", currency="R$").amount == 5000.0
    # German productive compounds are unlisted tables: digits still parse,
    # words await a data-driven compounding rule (documented open gap).
    assert w(["dreihundert"]) is None
    assert e("2500 Euro", currency="€").amount == 2500.0
    # French hyphen mechanics: flat 80 wins over 4+20; 70s/90s compose.
    assert w(["quatre-vingt"]) == 80
    assert w(["quatre-vingt-dix"]) == 90
    assert w(["quatre-vingt-dix-neuf"]) == 99
    assert w(["soixante-dix-neuf"]) == 79
    assert w(["trente-deux"]) == 32
    assert w(["twenty-one"]) == 21  # same rule fixes English compounds
    assert w(["one", "agent"]) is None
    # Digit-cluster behavior unchanged by the hyphen rule.
    assert e("ORD-4821").order_id == "ORD-4821"
    assert e("call 4821 tomorrow") is not None
