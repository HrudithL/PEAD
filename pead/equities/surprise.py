"""Earnings-surprise computation and trading-day anchoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

# Below this absolute STDEV the standardized surprise blows up; treat as invalid.
_MIN_STDEV = 1e-6


def _parse_hour(anntims: pd.Series) -> pd.Series:
    """Extract announcement hour from strings like '16:30:00'. NaN -> -1."""
    s = anntims.astype("string").str.split(":").str[0]
    h = pd.to_numeric(s, errors="coerce")
    return h.fillna(-1).astype(int)


def attach_anchor_date(events: pd.DataFrame) -> pd.DataFrame:
    """Add `anchor` = the calendar date whose first trading session is event day 0.

    Convention (Livnat & Mendenhall 2006): an announcement released at/after the
    market close (>=16:00) is first tradable the NEXT day, so anchor = anndats + 1.
    Otherwise the announcement is tradable the same day, anchor = anndats.
    Missing time -> assume after close (conservative).
    """
    ev = events.copy()
    hour = _parse_hour(ev["anntims"])
    after_close = (hour >= 16) | (hour < 0)
    ev["anchor"] = ev["anndats"] + pd.to_timedelta(after_close.astype(int), unit="D")
    return ev


def compute_surprise(events: pd.DataFrame, anchor_price: pd.Series,
                     cfg: Config) -> pd.DataFrame:
    """Add surprise measures and the chosen `surprise` column used for bucketing.

    anchor_price: Series aligned to events.index giving the price just before day 0.
    """
    ev = events.copy()
    ev["pre_price"] = anchor_price.values

    err = ev["actual"] - ev["meanest"]

    std = ev["stdev"].where(ev["stdev"] > _MIN_STDEV)
    ev["sue_std"] = err / std

    price = ev["pre_price"].where(ev["pre_price"] > 0)
    ev["sue_price"] = err / price

    ev["surprise_raw"] = err

    if cfg.measure == "std":
        ev["surprise"] = ev["sue_std"]
    else:
        ev["surprise"] = ev["sue_price"]

    # Winsorize the chosen measure at 1/99% to stop a few outliers from
    # dominating the decile cut points.
    ev["surprise"] = _winsorize(ev["surprise"], 0.01)
    return ev


def _winsorize(s: pd.Series, p: float) -> pd.Series:
    valid = s.dropna()
    if valid.empty:
        return s
    lo, hi = valid.quantile(p), valid.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def assign_buckets(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign each event to a surprise bucket (1=most negative ... N=most positive).

    Buckets are formed per calendar quarter so that cut points adapt over time
    and each quarter contributes proportionally to every bucket.
    """
    ev = events.dropna(subset=["surprise"]).copy()
    ev["cal_q"] = ev["anndats"].dt.to_period("Q")

    n = cfg.buckets

    def _qcut(group: pd.Series) -> pd.Series:
        # Need enough distinct values to form n buckets.
        try:
            return pd.qcut(group.rank(method="first"), n, labels=False) + 1
        except ValueError:
            return pd.Series(np.nan, index=group.index)

    ev["bucket"] = (
        ev.groupby("cal_q")["surprise"].transform(_qcut)
    )
    ev = ev.dropna(subset=["bucket"])
    ev["bucket"] = ev["bucket"].astype(int)
    return ev
