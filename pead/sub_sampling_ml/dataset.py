"""Assemble the event x feature x label table (``event_features.parquet``).

This is the orchestration seam: it reuses ``pead.equities`` to construct events
and the abnormal-return matrix, calls :mod:`labels` and :mod:`features`, performs
the point-in-time as-of joins onto the WRDS panels, and writes the compact
parquet under ``data/derived/`` (gitignored).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import DriftMLConfig


def as_of_merge(left: pd.DataFrame, right: pd.DataFrame, *,
                by: str, left_time: str, right_time: str,
                suffix: str = "") -> pd.DataFrame:
    """Point-in-time backward as-of merge.

    For each ``left`` row, attach the most recent ``right`` row for the same
    ``by`` key whose ``right_time`` is strictly on/before ``left_time`` -- the
    core no-look-ahead join used for fundamentals and CRSP daily snapshots.
    """
    raise NotImplementedError


def build_event_panel(cfg: DriftMLConfig) -> dict:
    """Run the equities front-end and return the pieces needed downstream.

    Returns a dict with: ``ev`` (located, surprise-tagged events), ``offsets``,
    ``ar_mat``, ``rets``, ``prices_wide``, ``calendar``, ``bench_ret``.
    """
    raise NotImplementedError


def build_event_features(cfg: DriftMLConfig, *, write: bool = True) -> pd.DataFrame:
    """Full assembly: events -> labels + features (+WRDS) -> as-of joins -> parquet.

    Returns the event x (features + labels) DataFrame and, when ``write`` is
    True, caches it to :meth:`DriftMLConfig.derived_path`.
    """
    raise NotImplementedError


def load_event_features(cfg: DriftMLConfig) -> Optional[pd.DataFrame]:
    """Load the cached ``event_features.parquet`` if present, else None."""
    raise NotImplementedError


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The modelling feature columns (everything that is not a label or key)."""
    raise NotImplementedError
