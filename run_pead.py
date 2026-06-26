"""Command-line entry point for the PEAD analysis.

Examples
--------
  python run_pead.py                              # full run, sensible defaults
  python run_pead.py --interactive                # prompt for parameters
  python run_pead.py --start-year 2018 --buckets 5 --benchmark spy
  python run_pead.py --tickers AAPL,MSFT,NVDA --window-post 90
  python run_pead.py --tickers SP500                  # a named group
  python run_pead.py --tickers MAG7,DOW30,AAPL        # groups + ticker, deduped

Named ticker groups (case-insensitive) expand to their members and overlaps
are counted once: MAG7, FAANG, DOW30, NASDAQ100, SP500, RUSSELL2000,
RUSSELL3000. See pead/ticker_groups.py.
"""

from __future__ import annotations

import sys

from pead.config import parse_args
from pead.analysis import run


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
