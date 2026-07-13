"""Hyperparameter tuning for the LightGBM quantile boosters (doc S3.2 / S3.4).

Optuna TPE search over the shared LightGBM param set used by every quantile
head, scored via the SAME leakage-safe protocol used everywhere else in this
package: purged, embargoed walk-forward CV
(:func:`pead.sub_sampling_ml.model.purged_walk_forward_splits`), with a
per-fold time-ordered holdout (:func:`time_ordered_holdout`, never random)
for LightGBM early stopping. Nothing here ever sees the held-out TEST
quarters carved off by :func:`split_tuning_test` -- callers pass only the
tuning-pool frame into :func:`tune_hyperparameters`.

This module is a leaf at import time: it must not import
``pead.sub_sampling_ml.serving.train_final`` at module scope. A later change
adds ``from .tune import time_ordered_holdout`` to ``train_final`` at module
top, and a top-level import back here would create an import cycle. The two
helpers this module borrows from ``train_final`` (``_fit_schema`` /
``_apply_schema``) are therefore imported lazily, inside
:func:`_evaluate_params`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd

from ..config import DriftMLConfig
from ..model import QUARTER_COL, purged_walk_forward_splits
from .artifact import DEFAULT_QUANTILES


# Mirror of train_final._BASE_PARAMS -- today's hardcoded defaults. Duplicated
# (rather than imported) so this module carries no top-level dependency on
# train_final; see the module docstring.
OLD_BASE_PARAMS: dict = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 20,
    "verbose": -1,
}


def _as_quarter_period(quarters) -> pd.PeriodIndex:
    """Same parsing as ``model._as_quarter_period`` -- duplicated (2 lines)
    rather than imported, to avoid reaching into another module's private
    helper."""
    return pd.PeriodIndex(pd.Series(quarters).astype(str).values, freq="Q")


# ---------------------------------------------------------- quarter splits


def time_ordered_holdout(quarters, n_quarters: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Positional ``(fit_idx, val_idx)`` time-ordered holdout for early stopping.

    ``val_idx`` selects the rows whose quarter is among the most-recent
    ``n_quarters`` DISTINCT quarters in ``quarters`` (time order, never
    random); ``fit_idx`` is everything else. Positions are 0-based and
    relative to the row order of ``quarters`` (i.e. valid for ``.iloc`` on
    the frame ``quarters`` was drawn from).

    If there are fewer than ``n_quarters + 1`` distinct quarters, returns
    ``(all_positions, empty)`` so callers can skip early stopping rather than
    train on nothing. Shared early-stop splitter (doc S3.2 level-C
    validation).
    """
    arr = np.asarray(_as_quarter_period(quarters))
    n = arr.shape[0]
    all_pos = np.arange(n)

    uniq = sorted(pd.unique(arr))
    if len(uniq) < n_quarters + 1:
        return all_pos, np.array([], dtype=int)

    val_quarters = set(uniq[-n_quarters:])
    val_mask = np.fromiter((q in val_quarters for q in arr), dtype=bool, count=n)
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def split_tuning_test(quarters, test_frac: float = 0.25) -> tuple[list, list]:
    """Chronological ``(tuning_quarters, test_quarters)`` split of distinct quarters.

    ``test_quarters`` is the most-recent ``round(test_frac * n_distinct)``
    whole event-quarters, where ``n_distinct`` is computed at runtime from
    however many distinct quarters ``quarters`` actually contains (never
    hardcoded); ``tuning_quarters`` is everything earlier. Guarantees at
    least one quarter on each side whenever there are >= 2 distinct quarters.
    Returns the original quarter labels as strings (e.g. ``"2018Q1"``) so
    callers can filter a frame with ``df[QUARTER_COL].isin(...)``. (doc S3.4)
    """
    uniq = sorted(pd.unique(np.asarray(_as_quarter_period(quarters))))
    n = len(uniq)
    if n < 2:
        return [str(q) for q in uniq], []

    n_test = int(round(test_frac * n))
    n_test = min(max(n_test, 1), n - 1)  # at least 1 test AND at least 1 tuning quarter
    tuning_quarters = uniq[: n - n_test]
    test_quarters = uniq[n - n_test:]
    return [str(q) for q in tuning_quarters], [str(q) for q in test_quarters]


# ---------------------------------------------------------- objective


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss at level ``alpha``, over finite rows."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    diff = y_true[m] - y_pred[m]
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(loss.mean())


def _evaluate_params(params: dict, df: pd.DataFrame,
                     splits: list[tuple[np.ndarray, np.ndarray]],
                     feature_cols: list[str], cat_cols: list[str],
                     cfg: DriftMLConfig) -> float:
    """Mean pinball loss (across the 5 quantiles, across folds) for one shared
    LightGBM param set, under purged walk-forward CV.

    Leakage-safe: within each fold, the schema (standardization stats +
    category levels) is fit on the fold's FIT rows only -- that fold's
    training window minus its own most-recent quarter, which is held out
    (:func:`time_ordered_holdout`) purely for LightGBM early stopping -- and
    applied unchanged to that fold's val/test rows. The test quarter is only
    ever scored with ``.predict()``.
    """
    import lightgbm as lgb

    from .train_final import _apply_schema, _fit_schema

    label_col = f"drift_raw_h{cfg.primary_horizon}"
    fold_scores: list[float] = []

    for tr_idx, te_idx in splits:
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        if len(train) < 50 or len(test) == 0:
            continue

        fit_pos, val_pos = time_ordered_holdout(train[QUARTER_COL], 1)
        fit, val = train.iloc[fit_pos], train.iloc[val_pos]

        schema = _fit_schema(fit, feature_cols)

        def _xy(part: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
            X = _apply_schema(part, schema)
            y = part[label_col].astype(float)
            mask = y.notna()
            return X.loc[mask], y.loc[mask].to_numpy()

        X_fit, y_fit = _xy(fit)
        X_test, y_test = _xy(test)
        X_val, y_val = _xy(val) if len(val) else (None, None)
        if X_val is not None and len(X_val) == 0:
            X_val, y_val = None, None

        if len(X_fit) < 20 or len(X_test) == 0:
            continue

        q_losses = []
        for q in DEFAULT_QUANTILES:
            p = {**params, "objective": "quantile", "alpha": float(q),
                "seed": int(cfg.random_state)}
            dtrain = lgb.Dataset(X_fit, label=y_fit,
                                 categorical_feature=cat_cols or "auto",
                                 free_raw_data=False)
            if X_val is not None:
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                                   categorical_feature=cat_cols or "auto",
                                   free_raw_data=False)
                booster = lgb.train(
                    p, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(0)])
            else:
                booster = lgb.train(p, dtrain, num_boost_round=400)

            y_pred = booster.predict(X_test)
            q_losses.append(_pinball(y_test, y_pred, q))

        fold_scores.append(float(np.mean(q_losses)))

    return float(np.mean(fold_scores)) if fold_scores else float("nan")


# ---------------------------------------------------------- public API


@dataclass
class TuneResult:
    """Outcome of one Optuna hyperparameter search (doc S3.2 / S3.4)."""

    best_params: dict      # winning shared LightGBM params -- NO objective/alpha/seed keys
    best_value: float      # mean pinball of the best trial
    baseline_value: float  # SAME evaluation on the OLD hardcoded params (for PDF #1, doc S4)
    n_trials: int
    trials_dataframe: pd.DataFrame


def tune_hyperparameters(df: pd.DataFrame, feature_cols: list[str],
                         cat_cols: list[str], cfg: DriftMLConfig, *,
                         n_trials: int = 400, timeout: int = 8 * 3600,
                         study_path: Optional[str] = None,
                         snapshot_every: Optional[int] = None,
                         snapshot_cb: Optional[Callable[[optuna.Study], None]] = None,
                         ) -> TuneResult:
    """Optuna TPE search over LightGBM hyperparameters on the TUNING POOL only.

    ``df`` must already be restricted to the tuning pool -- the held-out test
    quarters from :func:`split_tuning_test` must never be passed in. Each
    trial's objective is :func:`_evaluate_params`, evaluated under the same
    purged/embargoed walk-forward CV used everywhere else in this package.
    Also scores :data:`OLD_BASE_PARAMS` (today's hardcoded defaults) under the
    identical protocol as ``baseline_value``, so a report can show the tuned
    model against the status quo (doc S4).

    ``study_path``, when given, backs the Optuna study with a SQLite file
    (``load_if_exists=True``) so a killed run can resume. ``snapshot_cb``,
    when given together with ``snapshot_every``, is called with the live
    ``optuna.Study`` every ``snapshot_every`` completed trials (doc S3.3
    resumability) -- e.g. to persist intermediate best-params to disk.
    """
    splits = purged_walk_forward_splits(df[QUARTER_COL], cfg.primary_horizon, cfg)
    if not splits:
        raise ValueError(
            "No walk-forward splits available on the tuning pool -- check "
            "min_train_quarters / embargo_months against the number of "
            "distinct quarters passed in.")

    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    storage = f"sqlite:///{study_path}" if study_path else None
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                storage=storage, study_name="drift_tune",
                                load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_categorical(
                "max_depth", [-1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            "bagging_freq": 1,
            "verbose": -1,
        }
        return _evaluate_params(params, df, splits, feature_cols, cat_cols, cfg)

    callbacks = None
    if snapshot_cb is not None and snapshot_every:
        completed = {"n": 0}

        def _snapshot(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                completed["n"] += 1
                if completed["n"] % snapshot_every == 0:
                    snapshot_cb(study)

        callbacks = [_snapshot]

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)

    baseline_value = _evaluate_params(OLD_BASE_PARAMS, df, splits, feature_cols,
                                      cat_cols, cfg)
    best_params = {**study.best_params, "bagging_freq": 1, "verbose": -1}

    return TuneResult(
        best_params=best_params,
        best_value=float(study.best_value),
        baseline_value=float(baseline_value),
        n_trials=len(study.trials),
        trials_dataframe=study.trials_dataframe(),
    )


def save_best_params(result: TuneResult, path: str | Path) -> None:
    """Persist the winner as JSON: best_params + best_value + baseline_value + n_trials."""
    payload = {
        "best_params": result.best_params,
        "best_value": result.best_value,
        "baseline_value": result.baseline_value,
        "n_trials": result.n_trials,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_best_params(path: str | Path) -> dict:
    """Return just the ``best_params`` dict written by :func:`save_best_params`."""
    payload = json.loads(Path(path).read_text())
    return payload["best_params"]
