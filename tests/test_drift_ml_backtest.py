"""Tests for the drift walk-forward backtest layer
(:mod:`pead.sub_sampling_ml.serving.backtest`).

Self-contained: synthetic event-features / backtest-results frames are
fabricated so ``summarize``, the model-card PDF, the hyperparameter-summary
text, and the ``test_quarters``-restricted walk-forward can all be exercised
without touching WRDS or the real IBES/price files (mirrors the synthetic
fixture style in ``tests/test_drift_ml_serving.py``).
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from pead.sub_sampling_ml.config import DriftMLConfig
from pead.sub_sampling_ml.serving import backtest
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


def _cfg() -> DriftMLConfig:
    return DriftMLConfig(
        horizons=(60,), window_pre=5, min_train_quarters=2,
        embargo_months=3, random_state=0, use_wrds=False, use_cache=False,
    )


def _synthetic_backtest_results(n: int = 300, seed: int = 0,
                                with_prob_up: bool = True) -> pd.DataFrame:
    """A ``_predict_fold``-shaped results frame, fabricated directly (no
    LightGBM) so ``summarize`` / ``write_model_card`` / ``_reliability_table``
    can be tested fast and deterministically."""
    rng = np.random.default_rng(seed)
    drift_raw = rng.normal(0, 0.05, size=n)
    noise = rng.normal(0, 0.01, size=n)
    p50 = drift_raw * 0.8 + noise  # correlated but imperfect predictor
    spread = np.abs(rng.normal(0.02, 0.005, size=n)) + 0.005
    quarters = pd.period_range("2020Q1", periods=6, freq="Q")
    cal_q = [str(quarters[i % len(quarters)]) for i in range(n)]

    df = pd.DataFrame({
        "oftic": [f"T{i:04d}" for i in range(n)],
        "anndats": pd.Timestamp("2020-01-15") + pd.to_timedelta(np.arange(n), unit="D"),
        "cal_q": cal_q,
        "drift_raw": drift_raw,
        "pred_q10": p50 - 1.5 * spread,
        "pred_q25": p50 - 0.7 * spread,
        "pred_q50": p50,
        "pred_q75": p50 + 0.7 * spread,
        "pred_q90": p50 + 1.5 * spread,
        "drift_z_pred": p50 / 0.03,
    })
    if with_prob_up:
        df["prob_up"] = 1.0 / (1.0 + np.exp(-p50 / 0.02))
    return df


_FAKE_BEST_PARAMS = {
    "num_leaves": 40, "learning_rate": 0.05, "min_data_in_leaf": 15,
    "feature_fraction": 0.7, "bagging_fraction": 0.75,
    "lambda_l1": 0.01, "lambda_l2": 0.02, "max_depth": 7,
    "bagging_freq": 1, "verbose": -1,
}


def _fake_tune_result(best_value: float = 0.0123, baseline_value: float = 0.0150,
                      n_trials: int = 50) -> TuneResult:
    return TuneResult(
        best_params=dict(_FAKE_BEST_PARAMS),
        best_value=best_value,
        baseline_value=baseline_value,
        n_trials=n_trials,
        trials_dataframe=pd.DataFrame(),
    )


# ---------------------------------------------------------- summarize


def test_summarize_returns_expected_keys_and_sane_values():
    results = _synthetic_backtest_results(n=300)
    summary = backtest.summarize(results)

    expected_keys = {"n_events", "spearman_ic_p50", "r2_p50",
                     "decile_spread_p50", "interval_80_coverage", "prob_up_auc"}
    assert expected_keys <= set(summary.keys())
    assert summary["n_events"] == 300
    # Predictions were constructed to correlate positively with the outcome.
    assert summary["spearman_ic_p50"] > 0
    assert 0.0 <= summary["interval_80_coverage"] <= 1.0


def test_summarize_without_prob_up_omits_auc_key():
    results = _synthetic_backtest_results(n=200, with_prob_up=False)
    summary = backtest.summarize(results)
    assert "prob_up_auc" not in summary
    assert summary["n_events"] == 200


# ---------------------------------------------------------- _reliability_table (fix b)


def test_reliability_table_falls_back_to_fewer_bins_instead_of_blanking():
    # Only 5 rows -- the old hard `n>=10` blanking used to return an
    # all-empty table; the fix should still produce a usable (fewer-bin) curve.
    df = pd.DataFrame({
        "drift_raw": [0.05, -0.03, 0.02, -0.01, 0.04],
        "prob_up": [0.9, 0.2, 0.6, 0.4, 0.8],
    })
    table = backtest._reliability_table(df, n=10)
    assert not table.empty
    assert table["n"].sum() == 5


def test_reliability_table_missing_column_is_empty():
    df = pd.DataFrame({"drift_raw": [0.05, -0.03, 0.02]})
    table = backtest._reliability_table(df, n=10)
    assert table.empty


# ---------------------------------------------------------- _hyperparameter_summary_text (fix d)


def test_hyperparameter_summary_text_none_says_not_run():
    text = backtest._hyperparameter_summary_text(None)
    assert "not run" in text.lower()
    assert "default" in text.lower()


def test_hyperparameter_summary_text_with_tune_result():
    best_value, baseline_value, n_trials = 0.0123, 0.0150, 50
    tune_result = _fake_tune_result(best_value=best_value, baseline_value=baseline_value,
                                    n_trials=n_trials)
    text = backtest._hyperparameter_summary_text(tune_result)

    # Winning params are all listed.
    for k in _FAKE_BEST_PARAMS:
        assert k in text

    # Tuned vs. default mean pinball, plus absolute + percent improvement,
    # computed the same way the source does (robust to float formatting).
    improvement = baseline_value - best_value
    pct = improvement / baseline_value * 100.0
    assert f"{best_value:.4f}" in text
    assert f"{baseline_value:.4f}" in text
    assert f"{improvement:.4f}" in text
    assert f"{pct:.1f}%" in text
    assert str(n_trials) in text


def test_hyperparameter_summary_text_with_simple_namespace():
    """Duck-typed: a plain namespace with the same attributes works too."""
    ns = types.SimpleNamespace(best_params={"max_depth": 7, "num_leaves": 63},
                               best_value=0.02, baseline_value=0.025, n_trials=10)
    text = backtest._hyperparameter_summary_text(ns)
    assert "max_depth" in text
    assert "num_leaves" in text
    assert "63" in text


def test_hyperparameter_summary_text_handles_missing_attrs_gracefully():
    ns = types.SimpleNamespace(best_params={"num_leaves": 10})  # no value/trial attrs
    text = backtest._hyperparameter_summary_text(ns)
    assert "num_leaves" in text
    assert "n/a" in text


# ---------------------------------------------------------- write_model_card


def test_write_model_card_with_tune_result_produces_nonempty_pdf(tmp_path):
    results = _synthetic_backtest_results(n=300)
    summary = backtest.summarize(results)
    out = tmp_path / "card.pdf"

    result_path = backtest.write_model_card(
        results, summary, out_path=out, title="Test model card",
        tune_result=_fake_tune_result())

    assert result_path == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_write_model_card_tune_result_none_renders(tmp_path):
    """``tune_result=None`` (the default) must still render, and so must the
    'no prob_up' fallback panel (fix b) when the classifier wasn't trained."""
    results = _synthetic_backtest_results(n=150, with_prob_up=False)
    summary = backtest.summarize(results)
    out = tmp_path / "card_no_tune.pdf"

    result_path = backtest.write_model_card(results, summary, out_path=out, title="No tune")

    assert result_path == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_write_model_card_handles_tiny_results_without_raising(tmp_path):
    """Small samples (below the old decile / reliability bin minimums)
    shouldn't crash the PDF -- degrade gracefully instead."""
    results = _synthetic_backtest_results(n=4)
    summary = backtest.summarize(results)
    out = tmp_path / "card_tiny.pdf"

    backtest.write_model_card(results, summary, out_path=out)

    assert out.is_file()
    assert out.stat().st_size > 0


# ---------------------------------------------------------- run_walk_forward test_quarters


def test_run_walk_forward_test_quarters_restricts_scored_folds(monkeypatch, tmp_path):
    synthetic = _synthetic_features(n=320, periods=8)
    monkeypatch.setattr(backtest, "_load_or_build", lambda cfg: synthetic)
    cfg = _cfg()

    # embargo_months=3 (embargo_q=1) + horizon=60 (purge_q=1) -> gap=2; with
    # min_train_quarters=2 the scoreable test quarters are indices 4..7 of
    # the 8 synthetic quarters -- the last 2 are comfortably inside that range.
    all_quarters = sorted(synthetic["cal_q"].unique())
    requested = all_quarters[-2:]

    results_subset = backtest.run_walk_forward(
        cfg, universe="all", out_dir=tmp_path / "subset", test_quarters=requested)
    results_all = backtest.run_walk_forward(
        cfg, universe="all", out_dir=tmp_path / "all")

    assert not results_subset.empty
    scored_quarters = set(results_subset["cal_q"].astype(str).unique())
    assert scored_quarters and scored_quarters <= set(requested)
    # The unrestricted run covers strictly more (all 4 scoreable folds).
    assert len(results_subset) < len(results_all)
    assert (tmp_path / "subset" / "backtest_results.csv").is_file()


def test_run_walk_forward_test_quarters_no_match_raises(monkeypatch, tmp_path):
    synthetic = _synthetic_features(n=320, periods=8)
    monkeypatch.setattr(backtest, "_load_or_build", lambda cfg: synthetic)
    cfg = _cfg()

    with pytest.raises(SystemExit):
        backtest.run_walk_forward(
            cfg, universe="all", out_dir=tmp_path / "none_match",
            test_quarters=["2099Q1"])


def test_reliability_table_drops_missing_outcomes():
    """Events with a missing realized label are dropped BEFORE deriving up/down
    -- NaN>0 would otherwise silently count a missing outcome as 'down'."""
    res = pd.DataFrame({
        "drift_raw": [0.01, -0.02, np.nan, 0.03, np.nan],
        "prob_up": [0.6, 0.4, 0.9, 0.7, 0.1],
    })
    tbl = backtest._reliability_table(res, n=2)
    assert int(tbl["n"].sum()) == 3  # only the 3 finite-drift rows contribute


def test_run_backtest_derives_params_from_tune_result(tmp_path, monkeypatch):
    """Passing tune_result but not params must still score with the tuned params
    so the model card and the fitted folds agree."""
    captured: dict = {}

    def _fake_wf(cfg, *, universe, out_dir, params=None, test_quarters=None):
        captured["params"] = params
        return pd.DataFrame({"drift_raw": [0.0], "pred_q50": [0.0]})

    monkeypatch.setattr(backtest, "run_walk_forward", _fake_wf)
    monkeypatch.setattr(backtest, "summarize", lambda results: {})
    monkeypatch.setattr(backtest, "write_model_card", lambda *a, **k: tmp_path / "x.pdf")
    tr = types.SimpleNamespace(best_params={"num_leaves": 7}, best_value=0.1,
                               baseline_value=0.2, n_trials=3)
    backtest.run_backtest(_cfg(), universe="all", out_dir=tmp_path,
                          params=None, tune_result=tr)
    assert captured["params"] == {"num_leaves": 7}
