"""Fit the frozen deployable models on all events at/before a cutoff.

Public entry point: :func:`train_final`. Given a :class:`DriftMLConfig` and a
cutoff date (default: today, so *all* history participates), it

1. Builds (or loads) the event x feature x label parquet.
2. Restricts to the training universe (default S&P 500) and to
   ``anndats <= cutoff_date``.
3. Fits standardization stats and category levels once.
4. Trains, on the same feature matrix:
     * LightGBM **quantile boosters** on ``drift_raw_h{H}`` at
       :data:`DEFAULT_QUANTILES` (§4 / §0.1),
     * A LightGBM regressor on ``drift_z_h{H}`` for research/ranking,
     * A LightGBM classifier on ``drift_class_h{H}`` for ``prob_up``.
5. Packages everything, plus per-industry closed-event history, into a
   :class:`DriftModel` bundle and writes it under ``models/<version>/``.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DriftMLConfig
from ..dataset import build_event_features, feature_columns, load_event_features
from ..features import CATEGORICAL_FEATURES
from ...ticker_groups import expand
from .artifact import (DEFAULT_QUANTILES, DriftModel, FeatureSchema,
                       _quantile_name, make_metadata)


# LightGBM hyperparameters -- mirror the research pipeline so the quantile
# heads behave like the existing regressor, only with a different objective.
_BASE_PARAMS: dict = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,   # analogue of sklearn colsample_bytree
    "bagging_fraction": 0.8,   # analogue of subsample
    "bagging_freq": 1,
    "min_data_in_leaf": 20,
    "verbose": -1,
}
_NUM_BOOST_ROUND = 400
_CLF_BOOST_ROUND = 300


def _log(msg: str) -> None:
    print(f"[drift-serve] {msg}", flush=True)


# ---------------------------------------------------------- data prep


def _load_or_build(cfg: DriftMLConfig) -> pd.DataFrame:
    df = load_event_features(cfg) if cfg.use_cache else None
    if df is None:
        _log("Building event_features.parquet ...")
        df = build_event_features(cfg, write=True)
    return df


def _restrict_universe(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    if universe.lower() in {"all", "any", "*"}:
        return df
    tickers = {t.upper() for t in (expand([universe]) or [])}
    if not tickers:
        _log(f"Universe '{universe}' expanded to nothing; keeping all rows.")
        return df
    return df[df["oftic"].astype(str).str.upper().isin(tickers)].copy()


def _restrict_cutoff(df: pd.DataFrame, cutoff_date: str, horizon: int) -> pd.DataFrame:
    """Keep only events whose label window has *closed* on or before cutoff.

    Filtering on ``anndats <= cutoff`` alone is not enough: the label is
    CAR[+1, +horizon], so an event dated right before the cutoff can still
    have its label realized using prices *after* the cutoff whenever the
    parquet was built with a longer price history. That leaks post-cutoff
    returns into an as-of fit. Purge the final horizon window using the same
    conservative trading-day -> calendar-day approximation as
    :func:`_industry_history`.
    """
    cutoff = pd.Timestamp(cutoff_date)
    anndats = pd.to_datetime(df["anndats"])
    close_date = anndats + pd.to_timedelta(int(round(horizon * 7 / 5)), unit="D")
    return df[(anndats <= cutoff) & (close_date <= cutoff)].copy()


# ---------------------------------------------------------- schema


def _numeric_and_cat(df: pd.DataFrame,
                     feature_cols: list[str]) -> tuple[list[str], list[str]]:
    cats = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    nums = [c for c in feature_cols
            if c not in cats and pd.api.types.is_numeric_dtype(df[c])]
    # Guard: if any "categorical" col is missing entirely, keep it out.
    cats = [c for c in cats if c in df.columns]
    return nums, cats


def _fit_schema(df: pd.DataFrame, feature_cols: list[str]) -> FeatureSchema:
    num_cols, cat_cols = _numeric_and_cat(df, feature_cols)
    mean = {c: float(df[c].mean(skipna=True)) if df[c].notna().any() else 0.0
            for c in num_cols}
    std = {}
    for c in num_cols:
        sd = float(df[c].std(skipna=True, ddof=0)) if df[c].notna().any() else 1.0
        std[c] = sd if np.isfinite(sd) and sd > 0 else 1.0

    category_levels: dict[str, list] = {}
    for c in cat_cols:
        col = df[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            levels = list(col.cat.categories)
        else:
            levels = sorted(v for v in col.dropna().unique().tolist())
        category_levels[c] = [str(v) if not isinstance(v, (int, float)) else v
                              for v in levels]
    return FeatureSchema(
        feature_cols=list(feature_cols),
        cat_cols=cat_cols,
        numeric_cols=num_cols,
        category_levels=category_levels,
        mean=mean,
        std=std,
    )


def _apply_schema(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Build the train X matrix using the schema's persisted stats."""
    X = pd.DataFrame(index=df.index)
    for col in schema.numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        mu = schema.mean.get(col, 0.0)
        sd = schema.std.get(col, 1.0) or 1.0
        X[col] = (s - mu) / sd
    for col in schema.cat_cols:
        X[col] = pd.Categorical(df[col], categories=schema.category_levels[col])
    return X[schema.feature_cols]


# ---------------------------------------------------------- fitting


def _fit_booster(X: pd.DataFrame, y: pd.Series, params: dict,
                 cat_cols: list[str], num_boost_round: int):
    import lightgbm as lgb

    mask = y.notna()
    Xf, yf = X.loc[mask], y.loc[mask].astype(float).values
    dset = lgb.Dataset(
        Xf, label=yf,
        categorical_feature=cat_cols or "auto",
        free_raw_data=False,
    )
    booster = lgb.train(params, dset, num_boost_round=num_boost_round)
    return booster


def _fit_quantile_boosters(X: pd.DataFrame, y: pd.Series, cat_cols: list[str],
                           quantiles: tuple[float, ...],
                           random_state: int) -> dict[str, object]:
    boosters = {}
    for q in quantiles:
        params = {**_BASE_PARAMS,
                  "objective": "quantile",
                  "alpha": float(q),
                  "seed": int(random_state)}
        boosters[_quantile_name(q)] = _fit_booster(
            X, y, params, cat_cols, _NUM_BOOST_ROUND)
    return boosters


def _fit_z_booster(X: pd.DataFrame, y: pd.Series, cat_cols: list[str],
                   random_state: int):
    params = {**_BASE_PARAMS,
              "objective": "regression",
              "metric": "l2",
              "seed": int(random_state)}
    return _fit_booster(X, y, params, cat_cols, _NUM_BOOST_ROUND)


def _fit_class_booster(X: pd.DataFrame, y: pd.Series, cat_cols: list[str],
                       random_state: int):
    params = {**_BASE_PARAMS,
              "objective": "binary",
              "metric": "binary_logloss",
              "seed": int(random_state)}
    return _fit_booster(X, y, params, cat_cols, _CLF_BOOST_ROUND)


# ---------------------------------------------------------- history table


def _industry_history(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per-event closed-drift record for use at inference."""
    needed = {"ff12", "pos0", f"drift_raw_h{horizon}", "anndats"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=["ff12", "close_date", "drift_raw"])
    dropcols = [c for c in ("ff12", "pos0", "anndats", f"drift_raw_h{horizon}")]
    sub = df.dropna(subset=dropcols).copy()
    if sub.empty:
        return pd.DataFrame(columns=["ff12", "close_date", "drift_raw"])
    # close_date = anndats + horizon trading days (approx with 7/5 factor -- we
    # only need a *conservative* PIT filter at inference; a slight overshoot is
    # fine because the effect is small and only skips events on the boundary).
    sub["close_date"] = pd.to_datetime(sub["anndats"]) + pd.to_timedelta(
        int(round(horizon * 7 / 5)), unit="D")
    return sub.rename(columns={f"drift_raw_h{horizon}": "drift_raw"})[
        ["ff12", "close_date", "drift_raw"]].reset_index(drop=True)


# ---------------------------------------------------------- public API


def train_final(cfg: DriftMLConfig, *,
                cutoff_date: Optional[str] = None,
                universe: str = "SP500",
                out_root: str = "models",
                oos_metrics: Optional[dict] = None) -> DriftModel:
    """Fit and persist the frozen bundle. Returns the in-memory model."""
    if cutoff_date is None:
        cutoff_date = date.today().isoformat()

    df_all = _load_or_build(cfg)
    _log(f"Full parquet: {len(df_all):,} events x {df_all.shape[1]:,} cols.")

    horizon = cfg.primary_horizon
    df = _restrict_universe(df_all, universe)
    df = _restrict_cutoff(df, cutoff_date, horizon)
    _log(f"After universe='{universe}' and cutoff={cutoff_date}: {len(df):,} events.")

    if df.empty:
        raise SystemExit("Training set is empty -- widen universe or cutoff.")

    raw_col = f"drift_raw_h{horizon}"
    z_col = f"drift_z_h{horizon}"
    cls_col = f"drift_class_h{horizon}"
    for c in (raw_col, z_col):
        if c not in df.columns:
            raise SystemExit(f"Required label column '{c}' missing from parquet.")

    feature_cols = feature_columns(df)
    schema = _fit_schema(df, feature_cols)
    X = _apply_schema(df, schema)

    _log(f"Fitting quantile boosters at {DEFAULT_QUANTILES} on {raw_col} ...")
    quantile_boosters = _fit_quantile_boosters(
        X, df[raw_col], schema.cat_cols, DEFAULT_QUANTILES, cfg.random_state)

    _log(f"Fitting z regressor on {z_col} ...")
    z_booster = _fit_z_booster(X, df[z_col], schema.cat_cols, cfg.random_state)

    class_booster = None
    if cls_col in df.columns and df[cls_col].notna().sum() >= 100:
        _log(f"Fitting classifier on {cls_col} (n={int(df[cls_col].notna().sum())}) ...")
        class_booster = _fit_class_booster(
            X, df[cls_col], schema.cat_cols, cfg.random_state)
    else:
        _log(f"Skipping classifier (need >=100 non-NaN in {cls_col}).")

    metadata = make_metadata(
        cutoff_date=cutoff_date,
        universe=universe,
        horizon=horizon,
        quantiles=DEFAULT_QUANTILES,
        n_events=int(len(df)),
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        embargo_months=cfg.embargo_months,
        oos_metrics=oos_metrics,
    )
    hist = _industry_history(df, horizon)

    model = DriftModel(
        schema=schema, metadata=metadata,
        quantile_boosters=quantile_boosters,
        z_booster=z_booster, class_booster=class_booster,
        industry_history=hist,
    )
    out_dir = _version_dir(out_root, metadata)
    model.save(out_dir)
    _refresh_latest_pointer(out_root, out_dir)
    _log(f"Saved model -> {out_dir}")
    return model


def _version_dir(out_root: str, metadata) -> Path:
    tag = f"{metadata.cutoff_date}_{metadata.git_sha}"
    return Path(out_root) / tag


def _refresh_latest_pointer(out_root: str, out_dir: Path) -> None:
    """Write ``models/latest.txt`` pointing to the most recently trained bundle.

    A plain text pointer file (not a symlink) so Windows works without special
    perms.
    """
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        (root / "latest.txt").write_text(str(out_dir.resolve()))
    except Exception:
        pass


def latest_model_dir(out_root: str = "models") -> Optional[Path]:
    """Return the directory of the most recently trained bundle, if any."""
    ptr = Path(out_root) / "latest.txt"
    if ptr.is_file():
        p = Path(ptr.read_text().strip())
        if p.is_dir():
            return p
    # Fall back to the newest subdirectory containing manifest.json.
    if not Path(out_root).is_dir():
        return None
    candidates = [p for p in Path(out_root).iterdir()
                  if p.is_dir() and (p / "manifest.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
