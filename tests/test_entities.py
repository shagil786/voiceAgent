# tests/test_entities.py
from voiceagent.entities import extract_entities

def test_extracts_rupee_amount():
    e = extract_entities("I want a refund of ₹25,000 for my order")
    assert e.amount == 25000.0

def test_extracts_plain_amount_words():
    e = extract_entities("refund of 20000 rupees please")
    assert e.amount == 20000.0

def test_extracts_order_id():
    e = extract_entities("Where is my order ORD-55671?")
    assert e.order_id == "ORD-55671"

def test_no_entities():
    e = extract_entities("Someone used my account, block it")
    assert e.amount is None
    assert e.order_id is None

def test_hindi_amount():
    e = extract_entities("मुझे 25000 का रिफंड चाहिए")
    assert e.amount == 25000.0


# --------------------------------------------------------------------------
# M5b-3: number-words normalization (Qwen3-ASR writes numbers as words)
# --------------------------------------------------------------------------

def test_order_id_from_english_number_words():
    # Qwen3-ASR transcript style: "ORD four thousand eight hundred twenty one"
    e = extract_entities("Hello, I want to check the status of my order "
                         "ORD four thousand eight hundred twenty one.")
    assert e.order_id == "ORD-4821"
    assert e.amount is None  # the order id must not double as an amount


def test_order_id_from_digit_list_words():
    e = extract_entities("mera order ORD five five six seven one hai")
    assert e.order_id == "ORD-55671"


def test_order_id_words_span_removed_for_amount_scan():
    # Deliberate pin update: the rupee word forms are currency-scoped, so the
    # rupee-word amount path is pinned with an explicit currency="₹".
    e = extract_entities("refund rupees five hundred for ORD four eight two one",
                         currency="₹")
    assert e.order_id == "ORD-4821"
    assert e.amount == 500.0


def test_digit_form_order_id_still_not_an_amount():
    e = extract_entities("my order ORD-4821 pe ₹500 refund do")
    assert e.order_id == "ORD-4821"
    assert e.amount == 500.0


def test_short_number_words_are_not_order_ids():
    e = extract_entities("order ORD five five please")
    assert e.order_id is None  # 55: wrong shape for an order id


def test_amount_from_currency_number_words():
    # Deliberate pin update: the rupee word forms are currency-scoped — the
    # rupee behaviour is unchanged but needs an explicit currency="₹".
    e = extract_entities("I want a refund of rupees five thousand and two hundred",
                         currency="₹")
    assert e.amount == 5200.0  # "and" swallowed: 5*1000 + 2*100


def test_devanagari_digits_normalized():
    e = extract_entities("मेरा ऑर्डर ४८२१ का स्टेटस चेक करो ORD-4821")
    assert e.order_id == "ORD-4821"


# --------------------------------------------------------------------------
# M5b-4b: Hindi number words (0-99 table), digit+scale combos, observed ASR
# garble aliases — the fresh-caller voice e2e showed hi customers' amounts
# and order ids unextractable from Devanagari transcripts.
# --------------------------------------------------------------------------

def test_hindi_scale_order_id():
    # "ORD-55671" spoken: पचपन हजार छह सौ इकहत्तर
    e = extract_entities("मेरा ऑर्डर ORD पचपन हजार छह सौ इकहत्तर है")
    assert e.order_id == "ORD-55671"

def test_hindi_digit_plus_scale_amount():
    # Observed transcript: "₹6000" spoken as "6 हजार"
    e = extract_entities("मुझे 6 हजार का रिफंड चाहिए")
    assert e.amount == 6000.0

def test_hindi_garble_alias_from_observed_transcript():
    # Real Qwen transcript of the hinglish order query (2026-09-03):
    # 'पचपन हजार छह सौ एकत्र' — एकत्र is the ASR garble of इकहत्तर.
    # With an ORD marker the alias recovers the full id:
    e = extract_entities("मेरा ऑर्डर ORD पचपन हजार छह सौ एकत्र है")
    assert e.order_id == "ORD-55671"
    # Without any ORD marker the same run is order-id-SHAPED: it must NOT
    # read as an amount (₹55,671) and cannot become an order id either.
    e2 = extract_entities("ओड़ा ऐडी ओ ऐडी पचपन हजार छह सौ एकत्र है")
    assert e2.order_id is None
    assert e2.amount is None

def test_hindi_bare_scale_amount_not_too_small():
    # 'सौ' alone without currency must NOT become an amount (ambiguous),
    # but '6 हजार' style phrases always qualify.
    e = extract_entities("मुझे तीन सौ लोग चाहिए")
    assert e.amount is None

def test_english_path_unchanged_regression():
    e = extract_entities("Where is my order ORD-55671?")
    assert e.order_id == "ORD-55671"
    assert e.amount is None


# --------------------------------------------------------------------------
# Currency-scoped money words: a "$" tenant must extract "five thousand
# dollars" / "$5,000"; the rupee words must NOT create amounts for it and
# vice versa (words are scoped to the ACTIVE currency only).
# --------------------------------------------------------------------------

def test_dollar_words_extract_amount_for_dollar_tenant():
    e = extract_entities("I want a refund of five thousand dollars",
                         currency="$")
    assert e.amount == 5000.0

def test_dollar_symbol_extracts_amount():
    e = extract_entities("refund $5,000 please", currency="$")
    assert e.amount == 5000.0

def test_dollar_words_do_not_create_amounts_for_rupee_tenant():
    e = extract_entities("refund of five thousand dollars please",
                         currency="₹")
    assert e.amount is None

def test_rupee_word_path_unchanged_with_rupee_currency():
    e = extract_entities("rupees four thousand eight hundred twenty one",
                         currency="₹")
    assert e.amount == 4821.0


# --------------------------------------------------------------------------
# Additive currency-word coverage: Tamil/Telugu rupee forms, £ and ¥ word
# forms. Existing alternations stay byte-identical (the hi/Devanagari pins
# above are untouched); every new form mints amounts ONLY for its own
# currency (words are scoped to the ACTIVE currency, same as $/€ above).
# --------------------------------------------------------------------------

def test_tamil_rupee_word_forms_extract_amounts():
    e = extract_entities("ரூபாய் five thousand refund", currency="₹")
    assert e.amount == 5000.0
    e = extract_entities("refund of 2000 ரூ. please", currency="₹")
    assert e.amount == 2000.0
    e = extract_entities("ரூ. 2000 refund", currency="₹")
    assert e.amount == 2000.0
    e = extract_entities("five thousand ரூபாய் refund", currency="₹")
    assert e.amount == 5000.0

def test_telugu_rupee_word_forms_extract_amounts():
    e = extract_entities("రూపాయలు five thousand refund", currency="₹")
    assert e.amount == 5000.0
    e = extract_entities("refund of 3000 రూపాయలు", currency="₹")
    assert e.amount == 3000.0
    e = extract_entities("రూ. 3000 refund", currency="₹")
    assert e.amount == 3000.0

def test_pound_word_forms_extract_amounts():
    e = extract_entities("I want a refund of five thousand pounds",
                         currency="£")
    assert e.amount == 5000.0
    e = extract_entities("pounds five thousand refund", currency="£")
    assert e.amount == 5000.0
    e = extract_entities("refund of 2500 sterling", currency="£")
    assert e.amount == 2500.0
    e = extract_entities("refund gbp 2500", currency="£")
    assert e.amount == 2500.0
    e = extract_entities("refund of £2,500 please", currency="£")
    assert e.amount == 2500.0

def test_yen_yuan_word_forms_extract_amounts():
    e = extract_entities("refund of three thousand yuan", currency="¥")
    assert e.amount == 3000.0
    e = extract_entities("renminbi 1500 refund", currency="¥")
    assert e.amount == 1500.0
    e = extract_entities("refund of 1500 yen", currency="¥")
    assert e.amount == 1500.0
    e = extract_entities("yen 1800 refund", currency="¥")
    assert e.amount == 1800.0

def test_new_currency_words_isolated_across_tenants():
    # "pounds"/"sterling" must never mint a ₹/$/¥ amount, "yuan"/"yen" must
    # never mint a ₹/£ amount, and rupee words must not mint £/¥ amounts.
    assert extract_entities("refund of five thousand pounds please",
                            currency="₹").amount is None
    assert extract_entities("refund of three thousand yen please",
                            currency="₹").amount is None
    assert extract_entities("refund of five thousand pounds please",
                            currency="$").amount is None
    assert extract_entities("refund of five thousand pounds please",
                            currency="¥").amount is None
    assert extract_entities("refund of five thousand rupees please",
                            currency="£").amount is None
    assert extract_entities("refund of five thousand rupees please",
                            currency="¥").amount is None
    assert extract_entities("refund of five thousand dollars please",
                            currency="£").amount is None


def test_compositional_number_words_te_ta_bn():
    from voiceagent.entities import extract_entities as e
    from voiceagent.entities import words_to_number as w
    assert w(["ఇరవై", "ఒకటి"]) == 21
    assert w(["ఐదు", "వెయ్యి"]) == 5000
    assert w(["இருபது", "ஒன்று"]) == 21
    assert w(["ஐந்து", "ஆயிரம்"]) == 5000
    assert w(["বিশ", "দুই"]) == 22
    assert w(["পাঁচ", "হাজার"]) == 5000
    assert e("ఐదు వేలు రూపాయలు", currency="₹").amount == 5000.0
    assert e("ஐந்து ஆயிரம் ரூபாய்", currency="₹").amount == 5000.0
    assert e("পাঁচ হাজার টাকা", currency="₹").amount == 5000.0


def test_thai_compounds_split_and_parse():
    from voiceagent.entities import _space_thai_numbers as sp
    from voiceagent.entities import extract_entities as e
    from voiceagent.entities import words_to_number as w
    assert "ยี่สิบ" in sp("ยี่สิบเอ็ด") and "เอ็ด" in sp("ยี่สิบเอ็ด")
    assert "คำสั่งซื้อ" in sp("คำสั่งซื้อยี่สิบเอ็ด")
    assert w(["ยี่สิบ", "เอ็ด"]) == 21
    assert w(["ห้า", "พัน"]) == 5000
    assert e("ห้าพันบาท", currency="฿").amount == 5000.0
    # English/Hindi behavior pinned unchanged
    assert e("ORD-4821").order_id == "ORD-4821"
    assert e("five thousand dollars", currency="$").amount == 5000.0


def test_native_script_digits_normalize():
    from voiceagent.entities import extract_entities as e
    assert e("ORD-౪౮౨౧").order_id == "ORD-4821"
    assert e("ORD-৪৮২১").order_id == "ORD-4821"
    assert e("ORD-๔๘๒๑").order_id == "ORD-4821"


# --------------------------------------------------------------------------
# Scale plurals/inflections are DATA (lang/*.yaml), never code branches:
# zwei millionen, two millions, dos millones, trois millions, ...
# --------------------------------------------------------------------------

def test_scale_plurals_parse_in_six_languages():
    from voiceagent.entities import words_to_number as w
    assert w(["zwei", "millionen"]) == 2000000
    assert w(["zwei", "milliarden"]) == 2000000000
    assert w(["two", "millions"]) == 2000000
    assert w(["three", "thousands"]) == 3000
    assert w(["five", "lakhs"]) == 500000
    assert w(["dos", "millones"]) == 2000000
    assert w(["tres", "miles"]) == 3000
    assert w(["trois", "millions"]) == 3000000
    assert w(["un", "milliard"]) == 1000000000
    assert w(["dois", "milhões"]) == 2000000
    assert w(["tre", "milioni"]) == 3000000
    assert w(["due", "mila"]) == 2000


def test_german_fused_compound_with_plural_scale():
    from voiceagent.entities import _compound_value
    assert _compound_value("zweimillionen") == 2000000
    assert _compound_value("dreimilliarden") == 3000000000
