"""Drift labels: market-adjusted CAR[+1, +H] per event, with regime-neutral encodings.

The label is the *realized post-announcement drift*, computed directly from the
abnormal-return matrix produced by ``pead.equities.event_study.compute_ar_matrix``
(``offsets`` runs from ``-window_pre`` to ``+window_post``). For each horizon H we
sum the abnormal returns over trading-day offsets ``1..H`` -- i.e. excluding the
day-0 announcement jump -- so the label isolates the *drift*, not the reaction.

Encodings produced per horizon (Section 3):

==============  ==================================================  ===================
Column          Definition                                          Used for
==============  ==================================================  ===================
``drift_raw``   market-adjusted CAR[+1, +H]                         regression target
``drift_z``     cross-sectional z-score within calendar quarter     regression (regime-neutral)
``drift_decile``per-quarter decile (1..n) of ``drift_raw``          descriptive / ranking
``drift_class`` top decile = 1, bottom decile = 0 (middle = NaN)    classifier target
==============  ==================================================  ===================

Columns are suffixed ``_h{H}`` per horizon (e.g. ``drift_raw_h60``); the primary
horizon is additionally exposed under the bare names (``drift_raw`` ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DriftMLConfig

# Bare-named label aliases for the primary horizon, plus the per-horizon set.
_ENCODINGS = ("drift_raw", "drift_z", "drift_decile", "drift_class")


def car_window(offsets: np.ndarray, ar_mat: np.ndarray,
               start: int = 1, end: int = 60) -> np.ndarray:
    """Per-event cumulative abnormal return over trading-day offsets [start, end].

    NaN abnormal returns are treated as zero so a single missing day does not
    wipe out an event's whole label (mirrors the equities aggregation).
    """
    offsets = np.asarray(offsets)
    cols = np.where((offsets >= start) & (offsets <= end))[0]
    if cols.size == 0:
        return np.full(ar_mat.shape[0], np.nan)
    block = np.nan_to_num(ar_mat[:, cols], nan=0.0)
    return block.sum(axis=1)


def cross_sectional_z(values: pd.Series, quarter: pd.Series) -> pd.Series:
    """Z-score ``values`` within each calendar quarter (regime-neutral target).

    Quarters with fewer than two valid observations (no dispersion) yield NaN.
    """
    df = pd.DataFrame({"v": values.astype(float), "q": quarter})

    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean()) / sd

    return df.groupby("q")["v"].transform(_z)


def per_quarter_decile(values: pd.Series, quarter: pd.Series,
                       n: int = 10) -> pd.Series:
    """Assign a 1..n decile within each calendar quarter (NaN where too few obs)."""
    df = pd.DataFrame({"v": values.astype(float), "q": quarter})

    def _qcut(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if valid.nunique() < n:
            return pd.Series(np.nan, index=s.index)
        try:
            # Rank first so ties spread across adjacent deciles deterministically.
            return pd.qcut(s.rank(method="first"), n, labels=False) + 1
        except ValueError:
            return pd.Series(np.nan, index=s.index)

    return df.groupby("q")["v"].transform(_qcut)


def decile_to_class(decile: pd.Series, n: int = 10) -> pd.Series:
    """Top decile -> 1, bottom decile -> 0, everything in between -> NaN."""
    out = pd.Series(np.nan, index=decile.index, dtype="float64")
    out[decile == n] = 1.0
    out[decile == 1] = 0.0
    return out


def compute_labels(ev: pd.DataFrame, offsets: np.ndarray, ar_mat: np.ndarray,
                   cfg: DriftMLConfig) -> pd.DataFrame:
    """Build the full label frame (one row per event, aligned to ``ev.index``).

    Returns a DataFrame with, for every horizon H in ``cfg.horizons``, the four
    encodings suffixed ``_h{H}``, plus bare-named aliases for the primary
    horizon and a ``cal_q`` column (calendar quarter of the announcement).
    """
    out = pd.DataFrame(index=ev.index)
    quarter = ev["anndats"].dt.to_period("Q")
    out["cal_q"] = quarter.astype(str)

    for h in cfg.horizons:
        raw = pd.Series(car_window(offsets, ar_mat, start=1, end=h), index=ev.index)
        z = cross_sectional_z(raw, quarter)
        dec = per_quarter_decile(raw, quarter, n=cfg.n_deciles)
        cls = decile_to_class(dec, n=cfg.n_deciles)
        out[f"drift_raw_h{h}"] = raw
        out[f"drift_z_h{h}"] = z
        out[f"drift_decile_h{h}"] = dec
        out[f"drift_class_h{h}"] = cls

    ph = cfg.primary_horizon
    for enc in _ENCODINGS:
        out[enc] = out[f"{enc}_h{ph}"]

    return out
