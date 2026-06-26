"""Build the earnings-event table that drives the options extract.

An *event* is a single earnings announcement: ``(ticker, secid, ann_date)`` plus
the surprise inputs (actual, mean estimate, dispersion). Tickers and actuals come
from IBES; ``secid`` is the OptionMetrics security id resolved through ``secnmd``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..io import om_schema, resolver

# IBES summary columns we need (others are ignored to keep the read cheap).
_IBES_USECOLS = ["TICKER", "ANNDATS_ACT", "ACTUAL", "MEANEST", "STDEV", "NUMEST"]


@dataclass
class EventConfig:
    start_year: int = 1996
    end_year: int = 2013
    tickers: Optional[list[str]] = None
    min_numest: int = 1


def load_ibes_events(ibes_path, cfg: EventConfig) -> pd.DataFrame:
    """Read IBES actuals and reduce to one row per (ticker, announcement)."""
    df = pd.read_csv(
        ibes_path,
        usecols=lambda c: c in _IBES_USECOLS,
        dtype={"TICKER": "string"},
        low_memory=False,
    )
    df = df.rename(columns=str.lower).rename(columns={"anndats_act": "ann_date"})
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df = df.dropna(subset=["ticker", "ann_date", "actual"])

    df = df[(df["ann_date"].dt.year >= cfg.start_year)
            & (df["ann_date"].dt.year <= cfg.end_year)]
    if cfg.min_numest > 1 and "numest" in df:
        df = df[df["numest"].fillna(0) >= cfg.min_numest]
    if cfg.tickers:
        wanted = {t.upper() for t in cfg.tickers}
        df["ticker"] = df["ticker"].str.upper()
        df = df[df["ticker"].isin(wanted)]

    # Standardized unexpected earnings (SUE); guard against zero dispersion.
    if "stdev" in df:
        df["sue_std"] = (df["actual"] - df["meanest"]) / df["stdev"].replace(0, pd.NA)

    # One announcement per (ticker, date): keep the last consensus snapshot.
    df = (df.sort_values(["ticker", "ann_date"])
            .drop_duplicates(["ticker", "ann_date"], keep="last")
            .reset_index(drop=True))
    return df


def map_secids(events_df: pd.DataFrame, om_dir) -> pd.DataFrame:
    """Attach OptionMetrics ``secid`` to each event via the ``secnmd`` map.

    A ticker can map to several secids over time (share-class / relisting). We
    keep every match; the extract de-duplicates by actual option activity.
    """
    om_dir = Path(om_dir)
    secnmd = om_dir / om_schema.SECNMD
    con = duckdb.connect()
    names = con.execute(
        f"SELECT DISTINCT secid, upper(ticker) AS ticker "
        f"FROM read_parquet('{secnmd.as_posix()}') WHERE ticker IS NOT NULL"
    ).df()
    con.close()

    ev = events_df.copy()
    ev["ticker"] = ev["ticker"].str.upper()
    merged = ev.merge(names, on="ticker", how="inner")
    merged["secid"] = merged["secid"].astype("int64")
    return merged.reset_index(drop=True)


def build_events(cfg: EventConfig,
                 ibes_path=None,
                 om_dir=None) -> pd.DataFrame:
    """Convenience: IBES events joined to secids, ready for ``extract``."""
    ibes_path = ibes_path or resolver.ibes_path()
    om_dir = om_dir or resolver.require_optionmetrics()
    events = load_ibes_events(ibes_path, cfg)
    return map_secids(events, om_dir)
