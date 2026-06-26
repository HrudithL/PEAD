"""Bucket per-event option drift by earnings surprise.

Joins the engine's per-event features back to the surprise measure and reports
how option-implied drift (post- minus pre-announcement ATM IV) varies across
surprise buckets - the options-market analogue of equity PEAD.
"""

from __future__ import annotations

import pandas as pd


def bucket_drift(
    results: pd.DataFrame,
    events: pd.DataFrame,
    buckets: int = 10,
    measure: str = "sue_std",
) -> pd.DataFrame:
    """Return a per-bucket summary of IV drift ordered by ``measure``.

    ``results`` comes from ``engine.run`` (per-event features);
    ``events`` carries the surprise ``measure`` per ``(secid, ann_date)``.
    """
    ev = events.copy()
    ev["ann_date"] = pd.to_datetime(ev["ann_date"]).dt.date
    res = results.copy()
    res["ann_date"] = pd.to_datetime(res["ann_date"]).dt.date

    merged = res.merge(
        ev[["secid", "ann_date", measure]], on=["secid", "ann_date"], how="inner"
    ).dropna(subset=[measure, "iv_drift"])

    if merged.empty:
        return pd.DataFrame()

    merged["bucket"] = pd.qcut(
        merged[measure].rank(method="first"), q=buckets, labels=False
    )

    summary = (
        merged.groupby("bucket")
        .agg(
            n=("iv_drift", "size"),
            mean_surprise=(measure, "mean"),
            mean_iv_drift=("iv_drift", "mean"),
            median_iv_drift=("iv_drift", "median"),
            mean_total_volume=("total_volume", "mean"),
        )
        .reset_index()
    )
    return summary


def long_short_spread(summary: pd.DataFrame) -> float:
    """Top-minus-bottom bucket spread in mean IV drift."""
    if summary.empty:
        return float("nan")
    return float(summary["mean_iv_drift"].iloc[-1] - summary["mean_iv_drift"].iloc[0])
