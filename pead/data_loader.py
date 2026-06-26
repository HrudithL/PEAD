"""Memory-safe loading of IBES earnings announcements and stock prices."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .config import Config

# ---------------------------------------------------------------------------
# IBES earnings announcements
# ---------------------------------------------------------------------------

_IBES_COLS = [
    "OFTIC", "CNAME", "STATPERS", "MEASURE", "FISCALP", "FPI",
    "NUMEST", "NUMUP", "NUMDOWN", "MEDEST", "MEANEST", "STDEV",
    "FPEDATS", "ACTUAL", "ANNDATS_ACT", "ANNTIMS_ACT",
]


def load_events(cfg: Config) -> pd.DataFrame:
    """Return one row per quarterly EPS announcement with the last pre-announcement consensus.

    Columns out: oftic, cname, anndats, anntims, fpedats, statpers, actual,
    meanest, medest, stdev, numest.
    """
    df = pd.read_csv(
        cfg.ibes_path,
        usecols=_IBES_COLS,
        dtype={"OFTIC": "string", "CNAME": "string", "MEASURE": "string",
               "FISCALP": "string", "FPI": "string", "ANNTIMS_ACT": "string"},
        low_memory=False,
    )

    df = df[(df["MEASURE"] == "EPS") & (df["FISCALP"] == "QTR")].copy()

    for col in ("STATPERS", "FPEDATS", "ANNDATS_ACT"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ("ACTUAL", "MEANEST", "MEDEST", "STDEV", "NUMEST"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Need a real announcement date and a real actual to be an "event".
    df = df.dropna(subset=["OFTIC", "ANNDATS_ACT", "FPEDATS", "ACTUAL"])

    # Restrict to the requested announcement years.
    yr = df["ANNDATS_ACT"].dt.year
    df = df[(yr >= cfg.start_year) & (yr <= cfg.end_year)]

    if cfg.tickers:
        df = df[df["OFTIC"].str.upper().isin(set(cfg.tickers))]

    # The summary file carries many forecast snapshots (STATPERS) per fiscal
    # period. Keep the latest consensus STRICTLY BEFORE the announcement so the
    # surprise reflects what the market expected going in.
    df = df[df["STATPERS"] < df["ANNDATS_ACT"]]

    df = df.sort_values(["OFTIC", "FPEDATS", "ANNDATS_ACT", "STATPERS"])
    last = df.groupby(["OFTIC", "FPEDATS", "ANNDATS_ACT"], as_index=False).tail(1)

    out = last.rename(columns={
        "OFTIC": "oftic", "CNAME": "cname", "STATPERS": "statpers",
        "FPEDATS": "fpedats", "ACTUAL": "actual", "MEANEST": "meanest",
        "MEDEST": "medest", "STDEV": "stdev", "NUMEST": "numest",
        "ANNDATS_ACT": "anndats", "ANNTIMS_ACT": "anntims",
    })[[
        "oftic", "cname", "statpers", "fpedats", "anndats", "anntims",
        "actual", "meanest", "medest", "stdev", "numest",
    ]]

    out["oftic"] = out["oftic"].str.upper()
    out = out[out["numest"].fillna(0) >= cfg.min_numest]
    out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Stock prices
# ---------------------------------------------------------------------------

def load_prices(cfg: Config, needed: set[str]) -> pd.DataFrame:
    """Chunk-read master_stock, keeping only needed tickers (+ benchmark).

    Returns long DataFrame: columns tic, date, price (sorted, deduped).
    """
    keep = {t.upper() for t in needed}
    keep.add(cfg.benchmark_ticker)

    pieces: list[pd.DataFrame] = []
    reader = pd.read_csv(
        cfg.stock_path,
        usecols=["Tic", "Date", "Price"],
        dtype={"Tic": "string"},
        chunksize=500_000,
        low_memory=False,
    )
    for chunk in reader:
        chunk["Tic"] = chunk["Tic"].str.upper()
        sub = chunk[chunk["Tic"].isin(keep)]
        if not sub.empty:
            pieces.append(sub)

    if not pieces:
        return pd.DataFrame(columns=["tic", "date", "price"])

    px = pd.concat(pieces, ignore_index=True)
    px.columns = ["tic", "date", "price"]
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["price"] = pd.to_numeric(px["price"], errors="coerce")
    px = px.dropna(subset=["date", "price"])
    px = px[px["price"] > 0]
    px = px.drop_duplicates(subset=["tic", "date"]).sort_values(["tic", "date"])
    return px.reset_index(drop=True)


def build_return_panel(px: pd.DataFrame, cfg: Config):
    """Pivot prices to a wide daily return panel aligned on a common trading calendar.

    Returns (returns_df, calendar, benchmark_returns, prices_wide):
      returns_df : index = trading dates, columns = tickers, values = simple daily returns
      calendar   : sorted DatetimeIndex of trading days (from the benchmark, or union)
      benchmark_returns : Series of benchmark daily returns (zeros if raw mode)
      prices_wide : index = trading dates, columns = tickers, values = close prices
    """
    wide = px.pivot(index="date", columns="tic", values="price").sort_index()

    bench = cfg.benchmark_ticker
    if bench in wide.columns:
        calendar = wide[bench].dropna().index
    else:
        calendar = wide.index

    wide = wide.reindex(calendar)
    rets = wide.pct_change(fill_method=None)

    if cfg.benchmark == "spy" and bench in rets.columns:
        bench_ret = rets[bench].fillna(0.0)
    else:
        bench_ret = pd.Series(0.0, index=calendar)

    return rets, calendar, bench_ret, wide
