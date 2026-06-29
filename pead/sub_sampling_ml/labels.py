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


def car_window(offsets: np.ndarray, ar_mat: np.ndarray,
               start: int = 1, end: int = 60) -> np.ndarray:
    """Per-event cumulative abnormal return over trading-day offsets [start, end].

    NaN abnormal returns are treated as zero so a single missing day does not
    wipe out an event's whole label (mirrors the equities aggregation).
    """
    raise NotImplementedError


def cross_sectional_z(values: pd.Series, quarter: pd.Series) -> pd.Series:
    """Z-score ``values`` within each calendar quarter (regime-neutral target)."""
    raise NotImplementedError


def per_quarter_decile(values: pd.Series, quarter: pd.Series,
                       n: int = 10) -> pd.Series:
    """Assign a 1..n decile within each calendar quarter (NaN where too few obs)."""
    raise NotImplementedError


def decile_to_class(decile: pd.Series, n: int = 10) -> pd.Series:
    """Top decile -> 1, bottom decile -> 0, everything in between -> NaN."""
    raise NotImplementedError


def compute_labels(ev: pd.DataFrame, offsets: np.ndarray, ar_mat: np.ndarray,
                   cfg: DriftMLConfig) -> pd.DataFrame:
    """Build the full label frame (one row per event, aligned to ``ev.index``).

    Returns a DataFrame with, for every horizon H in ``cfg.horizons``, the four
    encodings suffixed ``_h{H}``, plus bare-named aliases for the primary
    horizon and a ``cal_q`` column (calendar quarter of the announcement).
    """
    raise NotImplementedError
