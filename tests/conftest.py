"""Shared fixtures: deterministic synthetic IBES + stock-price datasets.

The synthetic data deliberately embeds a post-announcement drift that is
monotonic in the earnings surprise, so the full pipeline should recover a
positive long/short spread and ordered buckets.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the `pead` package importable regardless of pytest's rootdir.
_PEAD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PEAD_DIR not in sys.path:
    sys.path.insert(0, _PEAD_DIR)


# Full IBES header so a usecols-based reader finds every column it expects.
_IBES_HEADER = [
    "TICKER", "CUSIP", "OFTIC", "CNAME", "STATPERS", "MEASURE", "FISCALP",
    "FPI", "ESTFLAG", "CURCODE", "NUMEST", "NUMUP", "NUMDOWN", "MEDEST",
    "MEANEST", "STDEV", "HIGHEST", "LOWEST", "USFIRM", "FPEDATS", "ACTUAL",
    "ANNDATS_ACT", "ANNTIMS_ACT", "CURR_ACT",
]


@pytest.fixture
def synthetic(tmp_path):
    n = 30
    n_days = 160
    pos0 = 91                 # day-0 index on the trading calendar
    win_post = 10
    drift = 0.01              # daily abnormal return per unit of surprise

    dates = pd.bdate_range("2020-01-02", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n)]
    surprises = np.linspace(-3.0, 3.0, n)   # becomes sue_std exactly

    ann_date = dates[90]                    # announced after close -> day 0 = next session

    # ---- stock prices (long format) ----
    rows = []
    for tic, s in zip(tickers, surprises):
        r = np.zeros(n_days)
        r[pos0:pos0 + win_post] = drift * s          # injected drift after day 0
        price = 100.0 * np.cumprod(1.0 + r)
        for d, p in zip(dates, price):
            rows.append((tic, d.strftime("%Y-%m-%d"), p))
    # Flat benchmark -> benchmark return is ~0, so market-adjusted == raw.
    for d in dates:
        rows.append(("SPY", d.strftime("%Y-%m-%d"), 400.0))
    stock = pd.DataFrame(rows, columns=["Tic", "Date", "Price"])
    stock_path = tmp_path / "stock.csv"
    stock.to_csv(stock_path, index=False)

    # ---- IBES events (two consensus snapshots each to exercise dedup) ----
    irows = []
    for tic, s in zip(tickers, surprises):
        actual = 1.0 + s
        for stat_offset, mean in ((20, 0.9), (10, 1.0)):   # latest (10d) should win
            statpers = (ann_date - pd.Timedelta(days=stat_offset)).strftime("%Y-%m-%d")
            irows.append({
                "TICKER": tic, "CUSIP": "00000000", "OFTIC": tic,
                "CNAME": f"{tic} CORP", "STATPERS": statpers, "MEASURE": "EPS",
                "FISCALP": "QTR", "FPI": "6", "ESTFLAG": "P", "CURCODE": "USD",
                "NUMEST": 5, "NUMUP": 1, "NUMDOWN": 0, "MEDEST": mean,
                "MEANEST": mean, "STDEV": 1.0, "HIGHEST": mean + 0.1,
                "LOWEST": mean - 0.1, "USFIRM": 1,
                "FPEDATS": (ann_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                "ACTUAL": actual,
                "ANNDATS_ACT": ann_date.strftime("%Y-%m-%d"),
                "ANNTIMS_ACT": "16:30:00", "CURR_ACT": "USD",
            })
    ibes = pd.DataFrame(irows, columns=_IBES_HEADER)
    ibes_path = tmp_path / "ibes.csv"
    ibes.to_csv(ibes_path, index=False)

    return {
        "ibes_path": str(ibes_path),
        "stock_path": str(stock_path),
        "output_dir": str(tmp_path / "outputs"),
        "n": n,
        "pos0": pos0,
        "win_post": win_post,
        "drift": drift,
        "tickers": tickers,
        "surprises": surprises,
    }
