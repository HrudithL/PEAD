"""CLI: score a single ``(ticker, anndate)`` or a batch CSV of events.

Single-event flow:

  python run_predict_drift.py --ticker AAPL --anndate 2025-01-30

If the IBES cache already covers the announcement, the CLI prints the cached
surprise fields and asks *"use cached values or supply your own?"*. Pass
``--no-prompt --use-cache`` (or ``--no-prompt --override-cache --actual ...``)
to run non-interactively.

Batch flow:

  python run_predict_drift.py --events events.csv --out predictions.csv
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from pead.sub_sampling_ml.config import build_parser, config_from_args
from pead.sub_sampling_ml.serving.featurize_one import (EventInputs,
                                                        find_in_ibes_cache)
from pead.sub_sampling_ml.serving.predict import (COVERAGE_WARN_THRESHOLD,
                                                  load_model, predict_events,
                                                  predict_one)


def _add_serving_args(p) -> None:
    p.add_argument("--models-root", default="models",
                   help="Root of trained bundles (default: models/).")
    p.add_argument("--version", default=None,
                   help="Bundle subdir to load; default = latest.")
    p.add_argument("--out", default=None,
                   help="Write predictions to this CSV (batch mode).")
    p.add_argument("--events", default=None, help="CSV of events (batch mode).")
    # single-event flags
    p.add_argument("--ticker")
    p.add_argument("--anndate", help="YYYY-MM-DD announcement date.")
    p.add_argument("--anntime", default=None)
    p.add_argument("--fpedats", default=None)
    p.add_argument("--statpers", default=None)
    p.add_argument("--cname", default=None)
    p.add_argument("--actual", type=float, default=None)
    p.add_argument("--meanest", type=float, default=None)
    p.add_argument("--medest", type=float, default=None)
    p.add_argument("--stdev", type=float, default=None)
    p.add_argument("--numest", type=float, default=None)
    p.add_argument("--numup", type=float, default=None)
    p.add_argument("--numdown", type=float, default=None)
    # prompting
    p.add_argument("--no-prompt", action="store_true",
                   help="Do not ask interactively when a cache hit exists.")
    p.add_argument("--use-cache", action="store_true",
                   help="Force use of cached IBES values (with --no-prompt).")
    p.add_argument("--override-cache", action="store_true",
                   help="Ignore cached IBES values, use CLI-supplied only.")


def _prompt_cache_decision(cache_row: dict) -> bool:
    print("[drift-serve] Cached IBES row found for this (ticker, anndate):")
    keep = ("actual", "meanest", "stdev", "numest", "numup", "numdown",
            "statpers", "anntims", "fpedats", "cname")
    for k in keep:
        if k in cache_row and cache_row[k] is not None:
            print(f"    {k:>10} = {cache_row[k]}")
    while True:
        ans = input("Use cached values? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def _run_single(args, cfg) -> int:
    if not args.ticker or not args.anndate:
        print("ERROR: --ticker and --anndate are required for single-event mode.",
              file=sys.stderr)
        return 2

    prefer_cache = True
    if args.override_cache:
        prefer_cache = False
    elif not args.no_prompt:
        cache_row = find_in_ibes_cache(args.ticker, args.anndate, cfg)
        if cache_row is not None:
            prefer_cache = _prompt_cache_decision(cache_row)

    user = EventInputs(
        ticker=args.ticker, anndate=args.anndate, anntime=args.anntime,
        fpedats=args.fpedats, statpers=args.statpers, cname=args.cname,
        actual=args.actual, meanest=args.meanest, medest=args.medest,
        stdev=args.stdev, numest=args.numest, numup=args.numup, numdown=args.numdown,
    )
    model = load_model(args.models_root, args.version)
    record = predict_one(user, cfg, model=model, prefer_cache=prefer_cache)

    print(json.dumps(record, indent=2, default=str))
    if record["coverage"] < COVERAGE_WARN_THRESHOLD:
        print(f"\n[drift-serve] WARNING: feature coverage "
              f"{record['coverage']:.0%} is below the "
              f"{COVERAGE_WARN_THRESHOLD:.0%} threshold; treat with caution.",
              file=sys.stderr)
    return 0


def _run_batch(args, cfg) -> int:
    events = pd.read_csv(args.events)
    model = load_model(args.models_root, args.version)
    prefer_cache = not args.override_cache
    preds = predict_events(events, cfg, model=model, prefer_cache=prefer_cache)
    out = args.out or "predictions.csv"
    preds.to_csv(out, index=False)
    print(f"[drift-serve] Wrote {len(preds):,} predictions -> {out}")
    return 0


def main() -> None:
    p = build_parser(prog="run_predict_drift",
                     description="Score earnings announcements with the frozen "
                                 "drift model.")
    _add_serving_args(p)
    args = p.parse_args()
    cfg = config_from_args(args)

    if args.events:
        sys.exit(_run_batch(args, cfg))
    sys.exit(_run_single(args, cfg))


if __name__ == "__main__":
    main()
