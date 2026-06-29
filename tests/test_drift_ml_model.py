"""Tests for models + leakage-aware validation (purged CV, Fama-MacBeth, LGBM)."""

import numpy as np
import pandas as pd

from pead.sub_sampling_ml import model
from pead.sub_sampling_ml.config import DriftMLConfig


def _make_panel(n_quarters=10, per_q=60, seed=0):
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2018Q1", periods=n_quarters, freq="Q")
    rows = []
    for q in quarters:
        x1 = rng.normal(0, 1, per_q)
        x2 = rng.normal(0, 1, per_q)
        noise = rng.normal(0, 0.3, per_q)
        target = 1.5 * x1 + noise           # x1 informative, x2 irrelevant
        sector = rng.choice(["Tech", "Fin", "Energy"], per_q)
        for a, b, t, s in zip(x1, x2, target, sector):
            rows.append((str(q), a, b, s, t))
    df = pd.DataFrame(rows, columns=["cal_q", "x1", "x2", "sector", "drift_z"])
    df["sector"] = df["sector"].astype("category")
    # Classifier label: top/bottom 30% -> 1/0, middle dropped.
    lo, hi = df["drift_z"].quantile([0.3, 0.7])
    cls = pd.Series(np.nan, index=df.index)
    cls[df["drift_z"] >= hi] = 1.0
    cls[df["drift_z"] <= lo] = 0.0
    df["drift_class"] = cls
    return df


def _cfg():
    return DriftMLConfig(min_train_quarters=2, embargo_months=3, random_state=0)


def test_purged_walk_forward_no_overlap_and_gap():
    df = _make_panel()
    splits = model.purged_walk_forward_splits(df["cal_q"], horizon=5, cfg=_cfg())
    assert len(splits) > 0
    q = model._as_quarter_period(df["cal_q"])
    arr = np.asarray(q)
    for tr, te in splits:
        assert set(tr).isdisjoint(set(te))            # no shared events
        max_train_q = max(arr[tr])
        test_q = arr[te][0]
        assert (arr[te] == test_q).all()              # one test quarter
        # gap = ceil(5/63)+ceil(3/3) = 2 quarters between train end and test.
        assert (test_q - max_train_q).n >= 2


def test_purged_respects_min_train_quarters():
    df = _make_panel(n_quarters=6)
    cfg = DriftMLConfig(min_train_quarters=3, embargo_months=3)
    splits = model.purged_walk_forward_splits(df["cal_q"], horizon=5, cfg=cfg)
    q = np.asarray(model._as_quarter_period(df["cal_q"]))
    for tr, _ in splits:
        assert len(set(q[tr])) >= 3


def test_standardize_uses_train_stats_only():
    train = pd.DataFrame({"a": [0.0, 10.0, 20.0]})
    test = pd.DataFrame({"a": [10.0]})
    tr, te = model.standardize(train, test, ["a"])
    assert abs(tr["a"].mean()) < 1e-9
    # test point at train mean -> standardized ~0.
    assert abs(te["a"].iloc[0]) < 1e-9


def test_fama_macbeth_recovers_signed_significant_driver():
    df = _make_panel()
    fm = model.fama_macbeth(df, ["x1", "x2"], target="drift_z")
    assert fm.loc["x1", "coef"] > 1.0
    assert fm.loc["x1", "t_stat"] > 3.0
    assert fm.loc["x1", "frac_positive"] > 0.9
    assert abs(fm.loc["x2", "coef"]) < 0.5


def test_lightgbm_cv_regression_ranks_informative_feature():
    df = _make_panel()
    folds = model.fit_lightgbm_cv(
        df, ["x1", "x2", "sector"], ["sector"], _cfg(),
        target="drift_z", horizon=5, task="regression")
    assert len(folds) > 0
    f = folds[0]
    assert f.y_pred.shape == f.y_true.shape
    assert f.shap_importance.index[0] == "x1"        # most important
    oos = model.aggregate_oos(folds)
    assert oos["spearman_ic"] > 0.3


def test_lightgbm_cv_classification_auc():
    df = _make_panel()
    folds = model.fit_lightgbm_cv(
        df, ["x1", "x2", "sector"], ["sector"], _cfg(),
        target="drift_class", horizon=5, task="classification")
    assert len(folds) > 0
    assert "roc_auc" in folds[0].metrics
    oos = model.aggregate_oos(folds)
    assert oos["roc_auc"] > 0.6


def test_regression_metrics_values():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    m = model.regression_metrics(y, y)
    assert np.isclose(m["spearman_ic"], 1.0)
    assert np.isclose(m["r2"], 1.0)


def test_classification_metrics_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = model.classification_metrics(y, s)
    assert np.isclose(m["roc_auc"], 1.0)
