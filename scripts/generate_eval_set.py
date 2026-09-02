import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.dataset import (MULTILINGUAL_SEED, MULTILINGUAL_START_ID,
                                append_multilingual_eval_set,
                                generate_eval_set)

if __name__ == "__main__":
    out = "data/eval/conversations.csv"
    ap = argparse.ArgumentParser(description="Generate the eval CSV.")
    ap.add_argument("n", nargs="?", type=int, default=1000,
                    help="rows for a full base regeneration (seed 42); "
                         "REWRITES the file, dropping appended rows")
    ap.add_argument("--append-multilingual", action="store_true",
                    help="append native-script rows (ta/te/mr/bn/gu, "
                         f"{MULTILINGUAL_SEED=} rows each) at "
                         f"conv-{MULTILINGUAL_START_ID}..; append-only, "
                         "existing rows untouched")
    ap.add_argument("--langs", default="ta,te,mr,bn,gu",
                    help="comma-separated languages for --append-multilingual")
    ap.add_argument("--per-lang", type=int, default=30,
                    help="rows per language for --append-multilingual")
    ap.add_argument("--seed", type=int, default=MULTILINGUAL_SEED,
                    help=f"seed for --append-multilingual (default {MULTILINGUAL_SEED}; "
                         "the base set used 42)")
    ap.add_argument("--start-id", type=int, default=MULTILINGUAL_START_ID)
    args = ap.parse_args()

    if args.append_multilingual:
        langs = tuple(s.strip() for s in args.langs.split(",") if s.strip())
        added = append_multilingual_eval_set(out, languages=langs,
                                             per_language=args.per_lang,
                                             seed=args.seed,
                                             start_id=args.start_id)
        print(f"appended {added} rows ({len(langs)} langs x "
              f"{args.per_lang}, seed={args.seed}, start conv-{args.start_id}) "
              f"to {out}")
    else:
        print(f"wrote {generate_eval_set(out, args.n)} rows to {out}")
