"""Configuration and command-line / interactive input handling for the PEAD study."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

from . import ticker_groups
from .io import resolver

# Equities data lives OUTSIDE the repo and is resolved via BEI_DATA_DIR (see
# .env.example). The resolver returns a best-guess path even when the data is
# absent, so importing this module never fails.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PEAD_DIR = os.path.dirname(_PKG_DIR)

DEFAULT_IBES = str(resolver.ibes_path("IBES_Summary_2015_2024.csv"))
DEFAULT_STOCK = str(resolver.master_stock_path("master_stock.csv"))
DEFAULT_OUTPUT_DIR = os.path.join(_PEAD_DIR, "outputs")
DEFAULT_CACHE_DIR = os.path.join(_PEAD_DIR, "cache")


@dataclass
class Config:
    """All knobs controlling a single PEAD run."""

    ibes_path: str = DEFAULT_IBES
    stock_path: str = DEFAULT_STOCK
    output_dir: str = DEFAULT_OUTPUT_DIR
    cache_dir: str = DEFAULT_CACHE_DIR

    start_year: int = 2015
    end_year: int = 2024

    # Optional universe restriction (already expanded from any group names).
    tickers: Optional[list[str]] = None

    # The raw, unexpanded --tickers argument (e.g. "SP500,AAPL"), kept so the
    # report and reproduce-command can show the group name instead of hundreds
    # of expanded tickers.
    ticker_spec: Optional[str] = None

    # Event window in trading days relative to the first post-announcement
    # trading day (day 0).
    window_pre: int = 5
    window_post: int = 60

    # Number of surprise buckets (e.g. 10 = deciles, 5 = quintiles).
    buckets: int = 10

    # Minimum number of analyst estimates in the consensus to keep an event.
    min_numest: int = 1

    # 'spy'  -> market-adjusted (stock return minus SPY return)
    # 'raw'  -> raw returns, no benchmark
    benchmark: str = "spy"

    # Primary surprise measure used for bucketing:
    # 'std'   -> (ACTUAL - MEANEST) / STDEV of estimates  (standardized surprise)
    # 'price' -> (ACTUAL - MEANEST) / pre-announcement price
    measure: str = "std"

    # Use a cached parquet of parsed events / aligned prices when available.
    use_cache: bool = True

    benchmark_ticker: str = "SPY"

    def label_measure(self) -> str:
        return {
            "std": "Standardized surprise  (ACTUAL - MEANEST) / STDEV",
            "price": "Price-scaled surprise  (ACTUAL - MEANEST) / price",
        }[self.measure]

    def label_benchmark(self) -> str:
        return {
            "spy": f"Market-adjusted (return minus {self.benchmark_ticker})",
            "raw": "Raw returns (no benchmark)",
        }[self.benchmark]

    def as_cli_command(self) -> str:
        """Reconstruct the equivalent run_pead.py invocation for reproducibility."""
        parts = [
            "python run_pead.py",
            f"--start-year {self.start_year}",
            f"--end-year {self.end_year}",
            f"--buckets {self.buckets}",
            f"--window-pre {self.window_pre}",
            f"--window-post {self.window_post}",
            f"--min-numest {self.min_numest}",
            f"--benchmark {self.benchmark}",
            f"--measure {self.measure}",
        ]
        if self.ticker_spec:
            parts.append("--tickers " + self.ticker_spec)
        elif self.tickers:
            parts.append("--tickers " + ",".join(self.tickers))
        return " ".join(parts)


def _prompt(text: str, default: str) -> str:
    raw = input(f"{text} [{default}]: ").strip()
    return raw if raw else default


def parse_args(argv: Optional[list[str]] = None) -> Config:
    """Build a Config from CLI flags, optionally falling back to interactive prompts."""

    p = argparse.ArgumentParser(
        prog="run_pead",
        description="Post-Earnings Announcement Drift (PEAD) analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ibes", dest="ibes_path", default=DEFAULT_IBES,
                   help="Path to the IBES summary CSV.")
    p.add_argument("--stock", dest="stock_path", default=DEFAULT_STOCK,
                   help="Path to the master stock-price CSV.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory for the output PDF.")
    p.add_argument("--start-year", type=int, default=2015,
                   help="First announcement year to include.")
    p.add_argument("--end-year", type=int, default=2024,
                   help="Last announcement year to include.")
    p.add_argument("--tickers", default=None,
                   help="Comma-separated tickers and/or group names, or a path to "
                        "a text file with one entry per line. Group names expand to "
                        "their members and overlaps are de-duplicated, so "
                        "'SP500,AAPL,MAG7' counts each firm once. Available groups: "
                        + ticker_groups.describe_groups()
                        + ". Default: full universe.")
    p.add_argument("--window-pre", type=int, default=5,
                   help="Trading days before day 0 to include.")
    p.add_argument("--window-post", type=int, default=60,
                   help="Trading days after day 0 to include.")
    p.add_argument("--buckets", type=int, default=10,
                   help="Number of surprise buckets (10=deciles, 5=quintiles).")
    p.add_argument("--min-numest", type=int, default=1,
                   help="Minimum analyst estimates (NUMEST) per event.")
    p.add_argument("--benchmark", choices=["spy", "raw"], default="spy",
                   help="Return-adjustment benchmark.")
    p.add_argument("--measure", choices=["std", "price"], default="std",
                   help="Primary surprise measure for bucketing.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable reading/writing the parsed-data cache.")
    p.add_argument("--interactive", action="store_true",
                   help="Prompt for the main parameters in the terminal.")

    args = p.parse_args(argv)

    if args.interactive:
        print("\n=== PEAD analysis — interactive setup (press Enter to accept defaults) ===")
        args.start_year = int(_prompt("Start year", str(args.start_year)))
        args.end_year = int(_prompt("End year", str(args.end_year)))
        tk = _prompt("Ticker subset (comma list / file path / blank=all)",
                     args.tickers or "")
        args.tickers = tk if tk else None
        args.window_pre = int(_prompt("Window pre (trading days)", str(args.window_pre)))
        args.window_post = int(_prompt("Window post (trading days)", str(args.window_post)))
        args.buckets = int(_prompt("Number of buckets", str(args.buckets)))
        args.min_numest = int(_prompt("Min analyst estimates", str(args.min_numest)))
        args.benchmark = _prompt("Benchmark (spy/raw)", args.benchmark)
        args.measure = _prompt("Surprise measure (std/price)", args.measure)
        print("=" * 72 + "\n")

    tickers = _parse_tickers(args.tickers)
    ticker_spec = args.tickers if (args.tickers and tickers) else None

    return Config(
        ibes_path=args.ibes_path,
        stock_path=args.stock_path,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        tickers=tickers,
        ticker_spec=ticker_spec,
        window_pre=abs(args.window_pre),
        window_post=abs(args.window_post),
        buckets=args.buckets,
        min_numest=args.min_numest,
        benchmark=args.benchmark,
        measure=args.measure,
        use_cache=not args.no_cache,
    )


def _parse_tickers(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    if os.path.isfile(raw):
        with open(raw) as fh:
            items = [ln.strip() for ln in fh if ln.strip()]
    else:
        items = [t.strip() for t in raw.split(",") if t.strip()]
    items = [t.upper() for t in items]
    # Expand any group names (SP500, MAG7, ...) and de-duplicate across overlaps.
    expanded = ticker_groups.expand(items)
    return expanded or None
