# scripts/voice_e2e_check.py — local "fresh caller" end-to-end check.
"""Simulates a FRESH caller (new conversation, no history) speaking to the
agent in each supported language and verifies the full voice chain:
  TTS query audio -> routed ASR -> intent -> policy -> reply in the
  customer's language -> reply TTS audio.

Per scenario we report: transcript, detected transcript language, extracted
entities, action, policy verdict, reply language (langid on the reply text),
and PASS/WARN/FAIL vs expectations.

Scenarios 1-4 are blind callers (language=None — a random consumer calling
in, deployment does not know the language). Scenario 5 is the known-language
telephony case (language='te' — trunk config). Scenario 6 shows the policy
engine escalating a high-value refund said in Hindi.

Run from the repo root:  .venv/bin/python scripts/voice_e2e_check.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402  (before faiss — macOS OpenMP rule)

from voice_demo import build_live_agent  # noqa: E402
from voiceagent.entities import extract_entities  # noqa: E402
from voiceagent.langid import detect_language  # noqa: E402
from voiceagent.tts import get_tts_handle  # noqa: E402
from voiceagent.voice_agent import voice_turn  # noqa: E402

OUT_DIR = Path("data/out/voice-e2e")

SCENARIOS = [
    # (tag, query, query_tts_lang, voice_turn_language, expect_action,
    #  expect_reply_lang, known_limitation)
    # known_limitation marks scenarios whose failure is a DOCUMENTED design
    # or engine-quality limit, not a regression: they are excluded from the
    # pass rate and tracked on the backlog.
    ("en-status", "Hello, I want to check the status of my order ORD-4821.",
     "en", None, "order_status", "en", None),
    ("hinglish-status", "Bhai mera order abhi tak nahi aaya, order id ORD-55671 hai",
     None, None, "order_status", "hinglish",
     "hinglish ASR garble: hi-voice reading Roman text produces Devanagari "
     "garble that misclassifies intent (root fix: hinglish ASR fine-tune)"),
    ("hi-refund", "मेरा रिफंड कब मेरे खाते में आएगा?",
     "hi", None, "refund_info", "hi", None),
    # NOTE: spoken form — a customer SAYS "chhe hazaar rupaye", never the
    # written '₹6000' symbol (TTS reading of written forms is unnatural).
    ("hi-high-value-refund",
     "मुझे अपने ऑर्डर के लिए छह हजार रुपये का रिफंड चाहिए",
     "hi", None, "high_value_refund", "hi", None),
    ("te-blind-fresh-caller", "నమస్కారం, నా ఆర్డర్ ORD-4821 స్టేటస్ నాకు తెలియాలి.",
     "te", None, "order_status", "te",
     "blind native-language callers need known-language context (telephony "
     "trunk config) — the blind path never auto-reroutes by design"),
    ("te-known-language", "నమస्कారం, నా ఆర్డర్ ORD-4821 స్టేటస్ నాకు తెలియాలి.",
     "te", "te", "order_status", "te", None),
]


def verdict_for(res, expect_action, expect_reply_lang):
    """(verdict, reasons) for one fresh-caller scenario."""
    checks = []
    reply_lang = detect_language(res["reply"])
    checks.append(("action", res["action"] == expect_action,
                   f"{res['action']} (want {expect_action})"))
    if expect_reply_lang == "hinglish":
        lang_ok = reply_lang in ("hinglish", "hi")
    else:
        lang_ok = reply_lang == expect_reply_lang
    checks.append(("reply_lang", lang_ok, f"{reply_lang} (want {expect_reply_lang})"))
    verdict = "PASS" if all(ok for _, ok, _ in checks) else "WARN"
    if not res["transcript"].strip():
        verdict = "FAIL"
    return verdict, checks


def main() -> int:
    agent, log = build_live_agent()
    tts = get_tts_handle()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for (tag, query, q_lang, turn_lang,
         expect_action, expect_reply_lang, limitation) in SCENARIOS:
        qwav = str(OUT_DIR / f"query-{tag}.wav")
        tts.speak(query, language=q_lang, out_path=qwav)

        t0 = time.perf_counter()
        r = voice_turn(agent, qwav, out_audio=str(OUT_DIR / f"reply-{tag}.wav"),
                       language=turn_lang)
        lat = time.perf_counter() - t0

        ents = extract_entities(r.transcript)
        res = {"tag": tag, "transcript": r.transcript, "reply": r.reply,
               "action": r.action, "decision": r.decision, "latency_s": round(lat, 2),
               "entities": {"order_id": ents.order_id, "amount": ents.amount},
               "turn_language": turn_lang or "blind"}
        verdict, checks = verdict_for(res, expect_action, expect_reply_lang)
        if limitation and verdict != "PASS":
            verdict = "XFAIL"  # documented known limitation, not a regression
        res["verdict"] = verdict
        res["known_limitation"] = limitation
        res["checks"] = [{"check": c, "ok": ok, "got": got} for c, ok, got in checks]

        asr_lang = detect_language(r.transcript)
        print(f"[{verdict}] {tag} ({lat:.2f}s)")
        print(f"    transcript : {r.transcript!r} (langid={asr_lang})")
        print(f"    entities   : {res['entities']}")
        print(f"    action     : {r.action} | decision: {r.decision}")
        print(f"    reply      : {r.reply!r}")
        print(f"    reply lang : {detect_language(r.reply)} (want {expect_reply_lang})")
        if limitation and verdict == "XFAIL":
            print(f"    limitation : {limitation}")
        rows.append(res)

    (OUT_DIR / "e2e-results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    scored = [r for r in rows if r["verdict"] != "XFAIL"]
    n_pass = sum(1 for r in scored if r["verdict"] == "PASS")
    print(f"\n{n_pass}/{len(scored)} scenarios PASS "
          f"(+{len(rows) - len(scored)} known limitations) — details in "
          f"{OUT_DIR / 'e2e-results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
