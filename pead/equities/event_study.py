"""Event-study engine: align announcements to trading days and accumulate CARs."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import Config


def locate_events(events: pd.DataFrame, rets: pd.DataFrame, calendar: pd.DatetimeIndex,
                  cfg: Config) -> pd.DataFrame:
    """Attach pos0 (calendar index of day 0) and pre_price to each event.

    Drops events whose ticker is absent from the price panel or that lack enough
    trading history to fill the window.
    """
    ev = events.copy()
    cols = set(rets.columns)
    ev = ev[ev["oftic"].isin(cols)]

    cal_values = calendar.values  # datetime64[ns], sorted
    anchors = ev["anchor"].values.astype("datetime64[ns]")

    # First trading day on or after the anchor date.
    pos0 = np.searchsorted(cal_values, anchors, side="left")
    ev = ev.assign(pos0=pos0)

    n = len(calendar)
    lo = ev["pos0"] - cfg.window_pre
    hi = ev["pos0"] + cfg.window_post
    ev = ev[(lo >= 1) & (hi < n)]  # need pos0-1 for pre_price and full window

    # Pre-announcement close = price on the trading day before day 0.
    # rets is returns; recover price not needed — use the close on pos0-1 from
    # the price panel passed via attribute. Instead we read it from the wide
    # price frame attached on rets (see build below).
    return ev.reset_index(drop=True)


def attach_pre_price(ev: pd.DataFrame, prices_wide: pd.DataFrame,
                     calendar: pd.DatetimeIndex) -> pd.Series:
    """Close price on the trading day immediately before day 0, per event."""
    px = prices_wide.reindex(calendar)
    px_np = px.to_numpy()
    col_idx = {c: i for i, c in enumerate(px.columns)}
    rows = (ev["pos0"] - 1).to_numpy()
    cols = ev["oftic"].map(col_idx).to_numpy()
    vals = px_np[rows, cols]
    return pd.Series(vals, index=ev.index)


def compute_ar_matrix(ev: pd.DataFrame, rets: pd.DataFrame, bench_ret: pd.Series,
                      cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (offsets, AR matrix) where AR[i, j] is the abnormal return of event i
    at trading-day offset offsets[j] relative to day 0."""
    if cfg.benchmark == "spy":
        ar = rets.sub(bench_ret, axis=0)
    else:
        ar = rets

    ar_np = ar.to_numpy()
    col_idx = {c: i for i, c in enumerate(ar.columns)}

    offsets = np.arange(-cfg.window_pre, cfg.window_post + 1)
    pos0 = ev["pos0"].to_numpy()
    cols = ev["oftic"].map(col_idx).to_numpy()

    # row indices: [n_events, n_offsets]
    row_mat = pos0[:, None] + offsets[None, :]
    col_mat = np.broadcast_to(cols[:, None], row_mat.shape)
    ar_mat = ar_np[row_mat, col_mat]
    return offsets, ar_mat


def aggregate_by_bucket(ev: pd.DataFrame, offsets: np.ndarray, ar_mat: np.ndarray,
                        cfg: Config) -> dict:
    """Compute mean CAR paths per bucket plus the long/short spread.

    Returns dict with:
      offsets, buckets (sorted list), car_by_bucket {b: array}, n_by_bucket,
      car_ls (top-bottom), ar_se_by_bucket, summary (DataFrame).
    """
    buckets = sorted(ev["bucket"].unique())
    bucket_arr = ev["bucket"].to_numpy()

    car_by_bucket: dict[int, np.ndarray] = {}
    ar_mean_by_bucket: dict[int, np.ndarray] = {}
    n_by_bucket: dict[int, int] = {}

    for b in buckets:
        mask = bucket_arr == b
        block = ar_mat[mask]
        with warnings.catch_warnings():
            # Offsets where every event in the bucket is missing -> empty slice.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_ar = np.nanmean(block, axis=0)
        mean_ar = np.nan_to_num(mean_ar, nan=0.0)
        car_by_bucket[b] = np.cumsum(mean_ar)
        ar_mean_by_bucket[b] = mean_ar
        n_by_bucket[b] = int(mask.sum())

    top, bottom = buckets[-1], buckets[0]
    car_ls = car_by_bucket[top] - car_by_bucket[bottom]

    # Per-event terminal CAR for t-stats of the long/short spread.
    car_event = np.cumsum(np.nan_to_num(ar_mat, nan=0.0), axis=1)
    term_idx = -1
    top_term = car_event[bucket_arr == top, term_idx]
    bot_term = car_event[bucket_arr == bottom, term_idx]
    ls_mean = np.nanmean(top_term) - np.nanmean(bot_term)
    ls_se = np.sqrt(np.nanvar(top_term) / len(top_term)
                    + np.nanvar(bot_term) / len(bot_term))
    ls_t = ls_mean / ls_se if ls_se > 0 else np.nan

    summary = _build_summary(ev, offsets, car_by_bucket, car_event, buckets, cfg)

    return {
        "offsets": offsets,
        "buckets": buckets,
        "car_by_bucket": car_by_bucket,
        "ar_mean_by_bucket": ar_mean_by_bucket,
        "n_by_bucket": n_by_bucket,
        "car_ls": car_ls,
        "car_event": car_event,
        "bucket_arr": bucket_arr,
        "ls_mean": ls_mean,
        "ls_t": ls_t,
        "summary": summary,
    }


def _idx_of(offsets: np.ndarray, target: int) -> int:
    """Index of the offset closest to target (clamped within range)."""
    target = min(max(target, offsets[0]), offsets[-1])
    return int(np.where(offsets == target)[0][0])


def _build_summary(ev, offsets, car_by_bucket, car_event, buckets, cfg) -> pd.DataFrame:
    i0 = _idx_of(offsets, 0)
    i20 = _idx_of(offsets, 20)
    iend = len(offsets) - 1
    bucket_arr = ev["bucket"].to_numpy()

    rows = []
    for b in buckets:
        mask = bucket_arr == b
        term = car_event[mask, iend]
        se = np.nanstd(term) / np.sqrt(mask.sum()) if mask.sum() > 0 else np.nan
        t = np.nanmean(term) / se if se and se > 0 else np.nan
        rows.append({
            "Bucket": b,
            "N": int(mask.sum()),
            "Mean surprise": float(np.nanmean(ev.loc[mask, "surprise"])),
            "CAR day 0 (%)": car_by_bucket[b][i0] * 100,
            "CAR +20 (%)": car_by_bucket[b][i20] * 100,
            f"CAR +{cfg.window_post} (%)": car_by_bucket[b][iend] * 100,
            "t-stat": t,
        })
    return pd.DataFrame(rows)
