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
from pead.sub_sampling_ml.model import QUARTER_COL
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


# ---------------------------------------------------------- tune_hyperparameters


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
