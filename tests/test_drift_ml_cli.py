"""Tests for the ``run_train_drift_model`` CLI dispatch (doc S3.1 / S3.4 Jobs 1-3).

Self-contained: a synthetic event-features frame plus lightweight monkeypatch
stubs for the tune / train_final / backtest entry points, so the CLI's control
flow -- reuse-vs-search params plumbing, and the tag_dir -> bundle move -- can
be exercised without touching WRDS, LightGBM, or Optuna (mirrors the synthetic
fixture style in ``tests/test_drift_ml_serving.py``).
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

import run_train_drift_model as cli
from pead.sub_sampling_ml.model import QUARTER_COL
from pead.sub_sampling_ml.serving import backtest as backtest_mod
from pead.sub_sampling_ml.serving import train_final as train_mod
from pead.sub_sampling_ml.serving import tune as tune_mod
from pead.sub_sampling_ml.serving.tune import TuneResult


# ---------------------------------------------------------- fixtures


def _synthetic_features(n: int = 320, periods: int = 8, seed: int = 0) -> pd.DataFrame:
    """An event x feature x label frame shaped like ``event_features.parquet``
    (mirrors ``tests/test_drift_ml_serving.py::_synthetic_features``)."""
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2018Q1", periods=periods, freq="Q")
    per_q = n // len(quarters)

    tickers = [f"T{i:03d}" for i in range(n)]
    rows = []
    for qi, q in enumerate(quarters):
        base_date = q.start_time + pd.Timedelta(days=30)
        for j in range(per_q):
            idx = qi * per_q + j
            sue = rng.normal(0, 1)
            ear = rng.normal(0, 0.03)
            mom = rng.normal(0, 0.1)
            ff12 = rng.choice(["Manuf", "Tech", "Fin", "Utils"])
            drift_raw = 0.02 * sue + 0.5 * ear + rng.normal(0, 0.03)
            rows.append({
                "oftic": tickers[idx],
                "anndats": base_date + pd.Timedelta(days=j % 20),
                "cal_q": str(q),
                "pos0": idx,
                "ff12": ff12,
                "gics_sector": ff12,
                "sue_std": sue,
                "ear": ear,
                "mom_1m": mom,
                "drift_raw_h60": drift_raw,
                "drift_z_h60": drift_raw / 0.03,
                "drift_class_h60": (1.0 if drift_raw > 0.02 else
                                    (0.0 if drift_raw < -0.02 else np.nan)),
            })
    df = pd.DataFrame(rows)
    df["gics_sector"] = df["gics_sector"].astype("category")
    return df


# A fixed, JSON/`_version_dir`-friendly stand-in for the real TrainingMetadata
# dataclass -- only `.cutoff_date` / `.git_sha` are used by `_version_dir`.
_FAKE_METADATA = types.SimpleNamespace(cutoff_date="2100-01-01", git_sha="testsha")


class _FakeModel:
    def __init__(self, metadata):
        self.metadata = metadata


def _install_train_and_backtest_stubs(monkeypatch, train_calls: dict, backtest_calls: dict):
    """Record kwargs; skip LightGBM entirely.

    The train stub also creates `_version_dir(out_root, _FAKE_METADATA)` on
    disk, mirroring the real `train_final`'s side effect of `model.save()`
    creating the bundle dir -- the CLI's post-hoc `shutil.move` of
    backtest/best-params artifacts into that dir requires it to exist.
    """

    def _train_stub(cfg, *, cutoff_date, universe, out_root, oos_metrics, params):
        train_calls["called"] = True
        train_calls["cutoff_date"] = cutoff_date
        train_calls["universe"] = universe
        train_calls["out_root"] = out_root
        train_calls["oos_metrics"] = oos_metrics
        train_calls["params"] = params
        final_dir = train_mod._version_dir(out_root, _FAKE_METADATA)
        final_dir.mkdir(parents=True, exist_ok=True)
        return _FakeModel(_FAKE_METADATA)

    def _backtest_stub(cfg, *, universe, out_dir, params=None, tune_result=None,
                       test_quarters=None):
        backtest_calls["called"] = True
        backtest_calls["universe"] = universe
        backtest_calls["out_dir"] = out_dir
        backtest_calls["params"] = params
        backtest_calls["tune_result"] = tune_result
        backtest_calls["test_quarters"] = test_quarters
        return {"n_events": 0}

    monkeypatch.setattr(train_mod, "train_final", _train_stub)
    monkeypatch.setattr(backtest_mod, "run_backtest", _backtest_stub)


# ---------------------------------------------------------- parser defaults


def test_parser_defaults_match_doc_s3_4():
    args = cli.build_cli_parser().parse_args([])
    assert args.universe == "ALL"
    assert args.n_trials == 400
    assert args.timeout_hours == 8.0
    assert args.test_frac == 0.25
    assert args.no_tune is False
    assert args.no_backtest is False


# ---------------------------------------------------------- reuse path (--best-params)


def test_best_params_reuse_skips_search_and_reaches_train_final(tmp_path, monkeypatch):
    loaded_best_params = {
        "num_leaves": 55, "learning_rate": 0.02, "min_data_in_leaf": 30,
        "feature_fraction": 0.6, "bagging_fraction": 0.65,
        "lambda_l1": 0.001, "lambda_l2": 0.002, "max_depth": 5,
        "bagging_freq": 1, "verbose": -1,
    }
    prior = types.SimpleNamespace(best_params=loaded_best_params, best_value=0.1,
                                  baseline_value=0.2, n_trials=5)
    best_params_path = tmp_path / "prior_best_params.json"
    tune_mod.save_best_params(prior, best_params_path)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("tune_hyperparameters must not be called when "
                             "--best-params is given")
    monkeypatch.setattr(tune_mod, "tune_hyperparameters", _must_not_be_called)

    synthetic = _synthetic_features()
    monkeypatch.setattr(train_mod, "_load_or_build", lambda cfg: synthetic)

    train_calls: dict = {}
    backtest_calls: dict = {}
    _install_train_and_backtest_stubs(monkeypatch, train_calls, backtest_calls)

    out_dir = tmp_path / "out"
    cli.main([
        "--best-params", str(best_params_path),
        "--no-wrds", "--no-backtest",
        "--universe", "all", "--out", str(out_dir), "--cutoff", "2100-01-01",
    ])

    assert train_calls.get("called") is True
    assert train_calls["params"] == loaded_best_params
    assert train_calls["oos_metrics"] is None
    assert not backtest_calls  # --no-backtest -> run_backtest never invoked


# ---------------------------------------------------------- search path (default tuning)


def test_default_tuning_calls_search_and_reaches_train_final(tmp_path, monkeypatch):
    fake_result = TuneResult(
        best_params={
            "num_leaves": 40, "learning_rate": 0.05, "min_data_in_leaf": 15,
            "feature_fraction": 0.7, "bagging_fraction": 0.75,
            "lambda_l1": 0.01, "lambda_l2": 0.02, "max_depth": 7,
            "bagging_freq": 1, "verbose": -1,
        },
        best_value=0.0123, baseline_value=0.0150, n_trials=7,
        trials_dataframe=pd.DataFrame(),
    )
    tune_calls: dict = {}

    def _tune_stub(df, feature_cols, cat_cols, cfg, **kwargs):
        tune_calls["called"] = True
        tune_calls["n_trials"] = kwargs.get("n_trials")
        tune_calls["timeout"] = kwargs.get("timeout")
        tune_calls["study_path"] = kwargs.get("study_path")
        tune_calls["quarters"] = sorted(df[QUARTER_COL].astype(str).unique().tolist())
        return fake_result
    monkeypatch.setattr(tune_mod, "tune_hyperparameters", _tune_stub)

    synthetic = _synthetic_features()
    monkeypatch.setattr(train_mod, "_load_or_build", lambda cfg: synthetic)

    train_calls: dict = {}
    backtest_calls: dict = {}
    _install_train_and_backtest_stubs(monkeypatch, train_calls, backtest_calls)

    out_dir = tmp_path / "out"
    cli.main([
        "--no-wrds",
        "--universe", "all", "--out", str(out_dir), "--cutoff", "2100-01-01",
        "--n-trials", "9", "--timeout-hours", "0.5",
    ])

    assert tune_calls.get("called") is True
    assert tune_calls["n_trials"] == 9
    assert tune_calls["timeout"] == int(0.5 * 3600)
    # The tuning pool handed to the search must exclude the held-out test quarters.
    all_quarters = sorted(synthetic[QUARTER_COL].unique().tolist())
    _, test_quarters = tune_mod.split_tuning_test(synthetic[QUARTER_COL], 0.25)
    assert set(tune_calls["quarters"]) == set(all_quarters) - set(test_quarters)

    assert train_calls.get("called") is True
    assert train_calls["params"] == fake_result.best_params
    assert train_calls["oos_metrics"] == {"n_events": 0}

    assert backtest_calls.get("called") is True
    assert backtest_calls["params"] == fake_result.best_params
    assert backtest_calls["tune_result"] is fake_result
    assert backtest_calls["test_quarters"] == test_quarters

    # best_params.json is written under the tag dir and then moved into the
    # final bundle dir (doc S3.4: PDF #1 artifacts ship next to the model).
    final_dir = train_mod._version_dir(str(out_dir), _FAKE_METADATA)
    assert (final_dir / "best_params.json").is_file()
