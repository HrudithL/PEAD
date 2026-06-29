"""Assemble the event x feature x label table (``event_features.parquet``).

This is the orchestration seam: it reuses ``pead.equities`` to construct events
and the abnormal-return matrix, calls :mod:`labels` and :mod:`features`, performs
the point-in-time as-of joins onto the WRDS panels, and writes the compact
parquet under ``data/derived/`` (gitignored).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..equities import data_loader, surprise, event_study
from . import labels as labels_mod
from . import features as features_mod
from . import wrds_extract
from .config import DriftMLConfig

# Columns carried through from the event table (identifiers / timing), kept
# distinct from the feature and label columns so feature_columns() can split
# them out cleanly. Surprise/SUE live in the feature matrix, not here.
_META_COLS = ("oftic", "cname", "anndats", "anntims", "fpedats", "pos0", "pre_price")
_LABEL_PREFIXES = ("drift_raw", "drift_z", "drift_decile", "drift_class")
_NON_FEATURE = set(_META_COLS) | {"cal_q"}


def as_of_merge(left: pd.DataFrame, right: pd.DataFrame, *,
                by: str, left_time: str, right_time: str,
                suffix: str = "") -> pd.DataFrame:
    """Point-in-time backward as-of merge.

    For each ``left`` row, attach the most recent ``right`` row for the same
    ``by`` key whose ``right_time`` is on/before ``left_time`` -- the core
    no-look-ahead join used for fundamentals and CRSP daily snapshots. The
    original ``left`` row order/index is preserved.
    """
    left = left.reset_index(drop=True).copy()
    left["_order"] = np.arange(len(left))
    lsorted = left.sort_values(left_time, kind="mergesort")
    rsorted = right.sort_values(right_time, kind="mergesort")

    merged = pd.merge_asof(
        lsorted, rsorted,
        left_on=left_time, right_on=right_time,
        by=by, direction="backward",
        suffixes=("", suffix or "_y"),
    )
    merged = (merged.sort_values("_order", kind="mergesort")
              .drop(columns="_order")
              .reset_index(drop=True))
    return merged


def build_event_panel(cfg: DriftMLConfig) -> dict:
    """Run the equities front-end and return the pieces needed downstream.

    Returns a dict with: ``ev`` (located, surprise-tagged events), ``offsets``,
    ``ar_mat``, ``rets``, ``prices_wide``, ``calendar``, ``bench_ret``.
    """
    eq = cfg.to_equities_config()

    events = data_loader.load_events(eq)
    if events.empty:
        raise SystemExit("No earnings events matched the filters.")
    events = surprise.attach_anchor_date(events)

    needed = set(events["oftic"].unique())
    px = data_loader.load_prices(eq, needed)
    if px.empty:
        raise SystemExit("No matching price data found.")

    rets, calendar, bench_ret, prices_wide = data_loader.build_return_panel(px, eq)

    ev = event_study.locate_events(events, rets, calendar, eq)
    pre_price = event_study.attach_pre_price(ev, prices_wide, calendar)
    ev = surprise.compute_surprise(ev, pre_price, eq).reset_index(drop=True)
    if ev.empty:
        raise SystemExit("No events could be located on the trading calendar.")

    offsets, ar_mat = event_study.compute_ar_matrix(ev, rets, bench_ret, eq)
    return {
        "ev": ev,
        "offsets": offsets,
        "ar_mat": ar_mat,
        "rets": rets,
        "prices_wide": prices_wide,
        "calendar": calendar,
        "bench_ret": bench_ret,
    }


def build_event_features(cfg: DriftMLConfig, *, write: bool = True) -> pd.DataFrame:
    """Full assembly: events -> labels + features (+WRDS) -> as-of joins -> parquet.

    Returns the event x (features + labels) DataFrame and, when ``write`` is
    True, caches it to :meth:`DriftMLConfig.derived_path`.
    """
    panel = build_event_panel(cfg)
    ev = panel["ev"]

    wrds_panels = wrds_extract.build_wrds_panels(ev, cfg)

    label_df = labels_mod.compute_labels(ev, panel["offsets"], panel["ar_mat"], cfg)
    feat_df = features_mod.build_features(
        ev, panel["rets"], panel["prices_wide"], panel["calendar"],
        panel["bench_ret"], cfg, wrds_panels=wrds_panels or None,
    )

    meta = ev[[c for c in _META_COLS if c in ev.columns]].copy()
    out = pd.concat([meta, feat_df, label_df], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    if write:
        _save(out, cfg.derived_path())
    return out


def _save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_event_features(cfg: DriftMLConfig) -> Optional[pd.DataFrame]:
    """Load the cached ``event_features.parquet`` if present, else None."""
    path = cfg.derived_path()
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The modelling feature columns (everything that is not a label or key)."""
    return [
        c for c in df.columns
        if c not in _NON_FEATURE and not c.startswith(_LABEL_PREFIXES)
    ]
