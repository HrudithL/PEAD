"""Command-line entry point for the options-market PEAD pipeline.

Stages: build events (IBES + secid) -> DuckDB extract -> compute engine ->
bucket by surprise -> write summary.

Examples
--------
  python run_pead_options.py --tickers AAPL,MSFT --start-year 2008 --end-year 2010
  python run_pead_options.py --start-year 2009 --end-year 2009 --window-post 45
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pead.io import resolver
from pead.options import events as events_mod
from pead.options import extract, engine, analysis, report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pead_options",
        description="Post-Earnings Announcement Drift in the options market.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tickers", default=None,
                   help="Comma-separated tickers (default: full universe).")
    p.add_argument("--start-year", type=int, default=2008)
    p.add_argument("--end-year", type=int, default=2010)
    p.add_argument("--window-pre", type=int, default=5)
    p.add_argument("--window-post", type=int, default=60)
    p.add_argument("--buckets", type=int, default=10)
    p.add_argument("--min-numest", type=int, default=1)
    p.add_argument("--output-dir", default=str(Path("outputs")))
    p.add_argument("--no-native", action="store_true",
                   help="Force the pandas fallback instead of the native engine.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    om_dir = resolver.require_optionmetrics()
    derived = resolver.DERIVED_DIR

    cfg = events_mod.EventConfig(
        start_year=args.start_year, end_year=args.end_year,
        tickers=tickers, min_numest=args.min_numest,
    )
    print("[1/4] Building events from IBES + secnmd ...")
    ev = events_mod.build_events(cfg, om_dir=om_dir)
    print(f"      {len(ev):,} (ticker, secid, date) events")

    print("[2/4] Extracting option panel via DuckDB ...")
    panel_path = extract.extract_event_panel(
        ev, derived / "event_panel.parquet", om_dir=om_dir,
        window_pre=args.window_pre, window_post=args.window_post,
    )
    print(f"      panel -> {panel_path}")

    print("[3/4] Computing per-event features ...")
    results_path = engine.run(panel_path, derived / "event_results.parquet",
                              use_native=not args.no_native)
    import pandas as pd
    results = pd.read_parquet(results_path)

    print("[4/4] Bucketing by surprise + writing summary ...")
    summary = analysis.bucket_drift(results, ev, buckets=args.buckets)
    paths = report.write_summary(summary, args.output_dir, tag="options")
    spread = analysis.long_short_spread(summary)
    print(f"      long/short IV-drift spread: {spread:.4f}")
    print(f"      summary -> {paths['csv']}")


if __name__ == "__main__":
    main()
