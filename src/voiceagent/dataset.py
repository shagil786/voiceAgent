from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field

DATASET_SCHEMA = [
    "id", "language", "intent", "user_text", "expected_action",
    "key_facts", "escalate", "authenticated", "amount",
]

INTENTS = [
    "order_status", "refund", "cancel_order", "address_change",
    "payment_declined", "recharge", "billing", "return", "replacement",
    "otp", "fraud", "account_closure", "delivery_delay", "product_info",
    "invoice", "plan_change", "roaming", "network_issue", "complaint",
    "high_value_refund",
]

# (language, intent, user_text, expected_action, key_facts, escalate)
_SEED_TEMPLATES = [
    ("en", "order_status", "Where is my order #ORD-77812?",
     "order_status", ["ORD-77812"], False),
    ("en", "refund", "I need a refund for order #ORD-22109.",
     "refund", ["ORD-22109"], False),
    ("en", "payment_declined", "Why was my payment declined?",
     "payment_declined", ["declined"], False),
    ("hinglish", "order_status", "Bhai mera order abhi tak nahi aaya, order id ORD-55671 hai.",
     "order_status", ["ORD-55671"], False),
    ("hinglish", "refund", "Actually can you refund my order, order #ORD-99032?",
     "refund", ["ORD-99032"], False),
    ("hi", "recharge", "मेरा recharge क्यों fail हुआ?",
     "recharge", ["fail"], False),
    ("hi", "billing", "मुझे अपना bill समझ नहीं आया।",
     "billing", ["bill"], False),
    ("en", "high_value_refund", "I want a refund of ₹25,000 for order #ORD-11223.",
     "high_value_refund", ["ORD-11223"], True),
    ("en", "fraud", "Someone used my account. Block it now.",
     "fraud", ["block"], True),
    ("hinglish", "otp", "OTP nahi aaya mere phone pe, resend karo.",
     "otp", ["otp"], False),
]

def _mutate(text: str, rng: random.Random) -> str:
    """Return the seed text as-is; templates already cover variation.
    Kept as a hook for later augmentation without changing the schema."""
    return text

def generate_eval_set(out_path: str, n: int = 1000, seed: int = 42) -> int:
    rng = random.Random(seed)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_SCHEMA)
        for i in range(n):
            lang, intent, text, action, facts, escalate = rng.choice(_SEED_TEMPLATES)
            order_id = f"ORD-{rng.randint(10000, 99999)}"
            text = text.replace("ORD-77812", order_id)
            facts = [order_id if f == "ORD-77812" else f for f in facts]
            # Auth: a customer quoting their order id is treated as an
            # authenticated session (they're tied to that order).
            authenticated = any(f.startswith("ORD-") for f in facts)
            amount = _template_amount(intent, rng)
            writer.writerow([
                f"conv-{i:04d}", lang, intent, _mutate(text, rng),
                action, "|".join(facts), escalate, authenticated, amount,
            ])
    return n


# ---------------------------------------------------------------------------
# M5a: native-script templates for ta/te/mr/bn/gu. Rows are APPENDED to the
# existing CSV (existing rows are never rewritten). The intent mix mirrors
# _SEED_TEMPLATES: order_status x2, refund x2, payment_declined, recharge,
# billing, high_value_refund, fraud, otp — 10 templates per language, the
# same relative frequencies as the base set. NOTE: the native-script text is
# LLM-authored synthetic phrasing (plausible support language, not real
# transcripts). English keyword facts (fail/otp/bill/block) are omitted:
# the echo guardrail's KEYWORD_FACTS is English-only, so they cannot be
# echoed into a native-language reply and would distort the multilingual
# measurement — order ids (digits) are kept.
# ---------------------------------------------------------------------------

_MULTILINGUAL_TEMPLATES: dict[str, list[tuple]] = {
    "ta": [
        ("order_status", "என் ஆர்டர் ORD-77812 எங்கே உள்ளது?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 ஆர்டருக்கு எனக்கு ரீஃபண்ட் வேண்டும்.",
         "refund", ["ORD-77812"], False),
        ("payment_declined", "என் பேமெண்ட் ஏன் நிராகரிக்கப்பட்டது?",
         "payment_declined", [], False),
        ("order_status", "ORD-77812 ஆர்டர் இன்னும் வரவில்லை, எந்த நிலையில் உள்ளது?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 ஆர்டருக்கு என் பணத்தை திரும்பத் தாருங்கள்.",
         "refund", ["ORD-77812"], False),
        ("recharge", "என் ரீசார்ஜ் ஆகவில்லை, ஏன்?",
         "recharge", [], False),
        ("billing", "என் பில் எனக்குப் புரியவில்லை.",
         "billing", [], False),
        ("high_value_refund", "ORD-77812 ஆர்டருக்கு ₹25000 ரீஃபண்ட் வேண்டும்.",
         "high_value_refund", ["ORD-77812"], True),
        ("fraud", "யாரோ என் கணக்கைப் பயன்படுத்துகிறார்கள். உடனே பிளாக் செய்யுங்கள்.",
         "fraud", [], True),
        ("otp", "எனக்கு ஓடிபி இன்னும் வரவில்லை, மீண்டும் அனுப்பவும்.",
         "otp", [], False),
    ],
    "te": [
        ("order_status", "నా ఆర్డర్ ORD-77812 ఎక్కడ ఉంది?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 ఆర్డర్ డబ్బు నాకు రీఫండ్ చేయండి.",
         "refund", ["ORD-77812"], False),
        ("payment_declined", "నా పేమెంట్ ఎందుకు డిక్లైన్ అయింది?",
         "payment_declined", [], False),
        ("order_status", "ORD-77812 ఆర్డర్ ఇంకా రాలేదు, స్టేటస్ ఏమిటి?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 ఆర్డర్ కోసం నా డబ్బు వాపస్ చేయండి.",
         "refund", ["ORD-77812"], False),
        ("recharge", "నా రీఛార్జ్ కాలేదు, ఎందుకు?",
         "recharge", [], False),
        ("billing", "నా బిల్లు నాకు అర్థం కాలేదు.",
         "billing", [], False),
        ("high_value_refund", "ORD-77812 ఆర్డర్‌కి ₹25000 రీఫండ్ కావాలి.",
         "high_value_refund", ["ORD-77812"], True),
        ("fraud", "ఎవరో నా ఖాతా వాడుతున్నారు. వెంటనే బ్లాక్ చేయండి.",
         "fraud", [], True),
        ("otp", "నాకు ఓటీపీ ఇంకా రాలేదు, మళ్ళీ పంపండి.",
         "otp", [], False),
    ],
    "mr": [
        ("order_status", "माझा ऑर्डर ORD-77812 कुठे आहे?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 ऑर्डरचा रिफंड मिळायला हवा.",
         "refund", ["ORD-77812"], False),
        ("payment_declined", "माझं पेमेंट का फेल झालं?",
         "payment_declined", [], False),
        ("order_status", "ORD-77812 ऑर्डर अजून आलं नाही, स्टेटस काय आहे?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 ऑर्डरसाठी माझं पैसे परत करा.",
         "refund", ["ORD-77812"], False),
        ("recharge", "माझं रिचार्ज झालं नाही, का?",
         "recharge", [], False),
        ("billing", "माझं बिल मला समजत नाही.",
         "billing", [], False),
        ("high_value_refund", "ORD-77812 ऑर्डरसाठी ₹25000 चा रिफंड हवा आहे.",
         "high_value_refund", ["ORD-77812"], True),
        ("fraud", "कोणीतरी माझं खाते वापरतंय. लगेच ब्लॉक करा.",
         "fraud", [], True),
        ("otp", "मला ओटीपी अजून मिळाला नाही, पुन्हा पाठवा.",
         "otp", [], False),
    ],
    "bn": [
        ("order_status", "আমার অর্ডার ORD-77812 কোথায় আছে?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 অর্ডারের টাকা রিফান্ড করে দিন।",
         "refund", ["ORD-77812"], False),
        ("payment_declined", "আমার পেমেন্ট কেন বাতিল হলো?",
         "payment_declined", [], False),
        ("order_status", "ORD-77812 অর্ডার এখনও আসেনি, স্ট্যাটাস কী?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 অর্ডারের জন্য আমার টাকা ফেরত দিন।",
         "refund", ["ORD-77812"], False),
        ("recharge", "আমার রিচার্জ হয়নি, কেন?",
         "recharge", [], False),
        ("billing", "আমার বিলটা বুঝতে পারছি না।",
         "billing", [], False),
        ("high_value_refund", "ORD-77812 অর্ডারের জন্য ₹25000 রিফান্ড চাই।",
         "high_value_refund", ["ORD-77812"], True),
        ("fraud", "কেউ আমার অ্যাকাউন্ট ব্যবহার করছে। এখুনই ব্লক করুন।",
         "fraud", [], True),
        ("otp", "আমার ওটিপি এখনও আসেনি, আবার পাঠান।",
         "otp", [], False),
    ],
    "gu": [
        ("order_status", "મારો ઓર્ડર ORD-77812 ક્યાં છે?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 ઑર્ડરના પૈસા રિફંડ કરી દો.",
         "refund", ["ORD-77812"], False),
        ("payment_declined", "મારું પેમેન્ટ કેમ ફેલ થયું?",
         "payment_declined", [], False),
        ("order_status", "ORD-77812 ઑર્ડર હજી આવ્યો નથી, સ્ટેટસ શું છે?",
         "order_status", ["ORD-77812"], False),
        ("refund", "ORD-77812 ઑર્ડર માટે મારા પૈસા પરત કરો.",
         "refund", ["ORD-77812"], False),
        ("recharge", "મારું રિચાર્જ થયું નથી, કેમ?",
         "recharge", [], False),
        ("billing", "મારું બિલ મને સમજાયું નથી.",
         "billing", [], False),
        ("high_value_refund", "ORD-77812 ઑર્ડર માટે ₹25000 નો રિફંડ જોઈએ.",
         "high_value_refund", ["ORD-77812"], True),
        ("fraud", "કોઈ મારું એકાઉન્ટ વાપરે છે. તરત બ્લોક કરો.",
         "fraud", [], True),
        ("otp", "મને ઓટીપી હજી મળ્યો નથી, ફરી મોકલો.",
         "otp", [], False),
    ],
}

# Base set seed was 42; the appended multilingual rows deliberately use a
# different draw so the two batches are statistically independent.
MULTILINGUAL_SEED = 2000
MULTILINGUAL_START_ID = 2000


def append_multilingual_eval_set(
        out_path: str,
        languages: tuple[str, ...] = ("ta", "te", "mr", "bn", "gu"),
        per_language: int = 30, seed: int = MULTILINGUAL_SEED,
        start_id: int = MULTILINGUAL_START_ID) -> int:
    """Append native-script rows per language to an existing eval CSV and
    return the number of rows written. Ids continue at conv-{start_id}; the
    header and existing rows are untouched. See _MULTILINGUAL_TEMPLATES for
    the intent-mix and key-facts caveats."""
    rng = random.Random(seed)
    # The CSV must end on a newline before appending rows (a bare append
    # would otherwise glue the first new row onto the last existing one).
    needs_newline = False
    try:
        with open(out_path, "rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
    except FileNotFoundError:
        pass
    written = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_newline:
            f.write("\n")
        for lang in languages:
            templates = _MULTILINGUAL_TEMPLATES[lang]
            for _ in range(per_language):
                intent, text, action, facts, escalate = rng.choice(templates)
                order_id = f"ORD-{rng.randint(10000, 99999)}"
                text = text.replace("ORD-77812", order_id)
                facts = [order_id if x == "ORD-77812" else x for x in facts]
                authenticated = any(x.startswith("ORD-") for x in facts)
                amount = _template_amount(intent, rng)
                writer.writerow([
                    f"conv-{start_id + written:04d}", lang, intent, text,
                    action, "|".join(facts), escalate, authenticated, amount,
                ])
                written += 1
    return written


def _template_amount(intent: str, rng: random.Random) -> float | None:
    """Assign a representative amount for amount-relevant intents."""
    if intent == "high_value_refund":
        return float(rng.choice([20000, 25000, 50000]))
    if intent == "refund":
        return float(rng.choice([1000, 2500, 4000]))
    return None


def load_conversations(path: str) -> list["Conversation"]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(Conversation(
                id=row["id"], language=row["language"], intent=row["intent"],
                user_text=row["user_text"], expected_action=row["expected_action"],
                key_facts=[k for k in row["key_facts"].split("|") if k],
                escalate=row["escalate"].lower() == "true",
                authenticated=row.get("authenticated", "false").lower() == "true",
                amount=float(row["amount"]) if row.get("amount") else None,
            ))
    return out


@dataclass
class Conversation:
    id: str
    language: str
    intent: str
    user_text: str
    expected_action: str
    key_facts: list[str] = field(default_factory=list)
    escalate: bool = False
    authenticated: bool = False
    amount: float | None = None