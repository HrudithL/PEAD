"""Tests for the hyperparameter-tuning layer (:mod:`pead.sub_sampling_ml.serving.tune`).

Self-contained: a synthetic 12-quarter event-features frame is fabricated so
the quarter-split helpers and the Optuna search itself can be exercised
without touching WRDS or the real IBES/price files. ``n_trials=5`` and an
in-memory study (``study_path=None``) keep the Optuna search fast; a single
walk-forward fold is engineered (via ``min_train_quarters``) so the whole
file runs in a couple of seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pead.sub_sampling_ml.config import DriftMLConfig
from pead.sub_sampling_ml.model import QUARTER_COL, purged_walk_forward_splits
from pead.sub_sampling_ml.serving import tune as tune_mod
from pead.sub_sampling_ml.serving.tune import TuneResult

_FEATURE_COLS = ["sue_std", "ear", "mom_1m", "gics_sector", "ff12"]
_CAT_COLS = ["gics_sector", "ff12"]


# ---------------------------------------------------------- fixtures


def _synthetic_tune_frame(n_per_q: int = 18, seed: int = 0) -> pd.DataFrame:
    """12 quarters of a synthetic event table with a linear drift DGP."""
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2018Q1", periods=12, freq="Q")

    rows = []
    idx = 0
    for q in quarters:
        base_date = q.start_time + pd.Timedelta(days=30)
        for j in range(n_per_q):
            sue = rng.normal(0, 1)
            ear = rng.normal(0, 0.03)
            mom = rng.normal(0, 0.1)
            ff12 = rng.choice(["Manuf", "Tech", "Fin", "Utils"])
            drift_raw = 0.02 * sue + 0.5 * ear + rng.normal(0, 0.03)
            rows.append({
                "oftic": f"T{idx:04d}",
                "anndats": base_date + pd.Timedelta(days=j % 20),
                "cal_q": str(q),
                "pos0": idx,
                "ff12": ff12,
                "gics_sector": ff12,
                "sue_std": sue,
                "ear": ear,
                "mom_1m": mom,
                "drift_raw_h60": drift_raw,
            })
            idx += 1
    df = pd.DataFrame(rows)
    df["gics_sector"] = df["gics_sector"].astype("category")
    return df


def _tuning_pool_cfg(min_train_quarters: int = 6) -> DriftMLConfig:
    """A single walk-forward fold on the 9-quarter tuning pool (fast tests)."""
    return DriftMLConfig(
        horizons=(60,), window_pre=5, min_train_quarters=min_train_quarters,
        embargo_months=3, random_state=0, use_wrds=False, use_cache=False,
    )


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return _synthetic_tune_frame()


@pytest.fixture(scope="module")
def tuning_pool_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    tuning_quarters, _ = tune_mod.split_tuning_test(
        synthetic_df[QUARTER_COL], test_frac=0.25)
    return synthetic_df[synthetic_df[QUARTER_COL].isin(tuning_quarters)].reset_index(drop=True)


@pytest.fixture(scope="module")
def tune_result(tuning_pool_df: pd.DataFrame) -> TuneResult:
    """One shared, cheap Optuna search (n_trials=5, in-memory study, 1 fold)."""
    cfg = _tuning_pool_cfg()
    return tune_mod.tune_hyperparameters(
        tuning_pool_df, _FEATURE_COLS, _CAT_COLS, cfg,
        n_trials=5, study_path=None)


# ---------------------------------------------------------- split_tuning_test


def test_split_tuning_test_last_quarters_form_test_set(synthetic_df):
    quarters = synthetic_df[QUARTER_COL]
    assert quarters.nunique() == 12

    tuning_quarters, test_quarters = tune_mod.split_tuning_test(quarters, test_frac=0.25)

    assert len(test_quarters) == 3
    assert len(tuning_quarters) == 9
    all_quarters = sorted(quarters.unique())
    assert test_quarters == all_quarters[-3:]
    assert tuning_quarters == all_quarters[:9]
    # disjoint and covers every distinct quarter exactly once
    assert set(tuning_quarters).isdisjoint(test_quarters)
    assert set(tuning_quarters) | set(test_quarters) == set(all_quarters)


# ---------------------------------------------------------- time_ordered_holdout


def test_time_ordered_holdout_val_is_single_most_recent_quarter(tuning_pool_df):
    quarters = tuning_pool_df[QUARTER_COL]
    fit_idx, val_idx = tune_mod.time_ordered_holdout(quarters, 1)

    most_recent = sorted(quarters.unique())[-1]
    val_quarters = set(quarters.iloc[val_idx].unique())
    assert val_quarters == {most_recent}
    assert not (quarters.iloc[fit_idx] == most_recent).any()

    # fit and val partition every row exactly once
    n = len(quarters)
    assert set(fit_idx.tolist()) & set(val_idx.tolist()) == set()
    assert set(fit_idx.tolist()) | set(val_idx.tolist()) == set(range(n))


def test_time_ordered_holdout_too_few_quarters_returns_all_as_fit():
    # Single distinct quarter -> fewer than n_quarters(1) + 1 -> no split possible.
    quarters = pd.Series(["2020Q1"] * 10)
    fit_idx, val_idx = tune_mod.time_ordered_holdout(quarters, 1)
    assert val_idx.size == 0
    assert fit_idx.tolist() == list(range(10))


# ---------------------------------------------------------- _evaluate_params


def test_evaluate_params_ignores_cat_cols_outside_feature_cols(tuning_pool_df):
    """cat_cols may legitimately be wider than feature_cols (e.g. a caller's
    full categorical roster vs. the subset actually being tuned). X_fit only
    ever has feature_cols' columns (via _apply_schema), so passing the raw,
    unfiltered cat_cols straight into lgb.Dataset(categorical_feature=...)
    would raise once a name is absent from the frame. _evaluate_params must
    intersect against the frame it actually built and still score fine."""
    cfg = _tuning_pool_cfg()
    splits = purged_walk_forward_splits(tuning_pool_df[QUARTER_COL], cfg.primary_horizon, cfg)
    wide_cat_cols = [*_CAT_COLS, "not_a_tuned_feature"]

    score = tune_mod._evaluate_params(
        tune_mod.OLD_BASE_PARAMS, tuning_pool_df, splits, _FEATURE_COLS, wide_cat_cols, cfg)

    assert np.isfinite(score)


# ---------------------------------------------------------- tune_hyperparameters


def test_tune_hyperparameters_raises_clear_error_when_every_fold_is_too_thin():
    """purged_walk_forward_splits can hand back a non-empty split list (its
    own quarter-count guard is satisfied) while every fold is still too thin
    to survive _evaluate_params's own row-count guards -- e.g. a sparse
    tuning pool. Previously this meant baseline_value/every trial's value was
    silently NaN, so Optuna would only fail later at study.best_params with
    an opaque error. Must instead fail fast with a clear ValueError, before
    study.optimize is ever called."""
    thin_df = _synthetic_tune_frame(n_per_q=5, seed=1)
    tuning_quarters, _ = tune_mod.split_tuning_test(thin_df[QUARTER_COL], test_frac=0.25)
    thin_pool = thin_df[thin_df[QUARTER_COL].isin(tuning_quarters)].reset_index(drop=True)
    cfg = _tuning_pool_cfg(min_train_quarters=6)

    # Sanity check: splits DO exist (the pre-existing "no splits" guard is
    # not what's under test here) -- each one is just too small.
    splits = purged_walk_forward_splits(thin_pool[QUARTER_COL], cfg.primary_horizon, cfg)
    assert len(splits) >= 1
    assert all(len(tr_idx) < 50 for tr_idx, _ in splits)

    with pytest.raises(ValueError, match="walk-forward"):
        tune_mod.tune_hyperparameters(
            thin_pool, _FEATURE_COLS, _CAT_COLS, cfg, n_trials=2, study_path=None)


def test_tune_hyperparameters_resume_caps_to_total_trial_budget(tmp_path, tuning_pool_df):
    """n_trials is the TOTAL trial budget across resumes of a study_path-backed
    study (load_if_exists=True), not additional trials stacked on top of
    whatever the resumed study already has."""
    cfg = _tuning_pool_cfg()
    study_path = str(tmp_path / "resume_study.db")

    first = tune_mod.tune_hyperparameters(
        tuning_pool_df, _FEATURE_COLS, _CAT_COLS, cfg,
        n_trials=3, study_path=study_path)
    assert first.n_trials == 3

    second = tune_mod.tune_hyperparameters(
        tuning_pool_df, _FEATURE_COLS, _CAT_COLS, cfg,
        n_trials=5, study_path=study_path)
    assert second.n_trials == 5


def test_tune_hyperparameters_returns_valid_result(tune_result):
    assert isinstance(tune_result, TuneResult)
    assert tune_result.n_trials == 5
    assert np.isfinite(tune_result.best_value)
    assert np.isfinite(tune_result.baseline_value)

    expected_keys = {"num_leaves", "learning_rate", "min_data_in_leaf",
                     "feature_fraction", "bagging_fraction", "lambda_l1",
                     "lambda_l2", "max_depth"}
    assert expected_keys <= set(tune_result.best_params.keys())
    assert "alpha" not in tune_result.best_params
    assert "objective" not in tune_result.best_params

    assert isinstance(tune_result.trials_dataframe, pd.DataFrame)
    assert len(tune_result.trials_dataframe) == 5


# ---------------------------------------------------------- save / load


def test_save_and_load_best_params_round_trip(tmp_path, tune_result):
    path = tmp_path / "best_params.json"
    tune_mod.save_best_params(tune_result, path)
    assert path.is_file()

    loaded = tune_mod.load_best_params(path)
    assert loaded == tune_result.best_params
