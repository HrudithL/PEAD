"""Tests for the drift-serving layer (artifact bundle, train, predict).

These are self-contained: a synthetic event-features frame is fabricated to
exercise the train -> save -> load -> predict round-trip without touching WRDS
or the real IBES/price files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pead.sub_sampling_ml.config import DriftMLConfig
from pead.sub_sampling_ml.serving import artifact as artifact_mod
from pead.sub_sampling_ml.serving.artifact import DriftModel
from pead.sub_sampling_ml.serving.featurize_one import (
    EventInputs, industry_drift_from_history, merge_inputs)
from pead.sub_sampling_ml.serving import train_final


# ---------------------------------------------------------- fixtures


def _synthetic_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2018Q1", periods=8, freq="Q")
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
            sector = ff12  # keep aligned for test simplicity
            # True drift depends linearly on sue + ear with noise.
            drift_raw = 0.02 * sue + 0.5 * ear + rng.normal(0, 0.03)
            rows.append({
                "oftic": tickers[idx],
                "anndats": base_date + pd.Timedelta(days=j % 20),
                "cal_q": str(q),
                "pos0": idx,
                "ff12": ff12,
                "gics_sector": sector,
                "sue_std": sue,
                "ear": ear,
                "mom_1m": mom,
                "drift_raw_h60": drift_raw,
                "drift_z_h60": drift_raw / 0.03,  # rough within-panel z
                "drift_class_h60": 1.0 if drift_raw > 0.02 else (0.0 if drift_raw < -0.02 else np.nan),
                "drift_raw": drift_raw,
                "drift_z": drift_raw / 0.03,
                "drift_class": 1.0 if drift_raw > 0.02 else (0.0 if drift_raw < -0.02 else np.nan),
            })
    df = pd.DataFrame(rows)
    df["gics_sector"] = df["gics_sector"].astype("category")
    return df


def _cfg() -> DriftMLConfig:
    return DriftMLConfig(
        horizons=(60,), window_pre=5, min_train_quarters=2,
        embargo_months=3, random_state=0, use_wrds=False, use_cache=False,
    )


# ---------------------------------------------------------- schema + apply


def test_schema_and_apply_align_columns_and_categories():
    df = _synthetic_features(n=200)
    schema = train_final._fit_schema(df, ["sue_std", "ear", "mom_1m",
                                          "gics_sector", "ff12"])
    X = train_final._apply_schema(df, schema)

    assert list(X.columns) == schema.feature_cols
    assert set(schema.numeric_cols) == {"sue_std", "ear", "mom_1m"}
    # standardization mean/std recovered from train window
    assert abs(X["sue_std"].mean()) < 1e-6
    assert abs(X["sue_std"].std(ddof=0) - 1.0) < 1e-6
    # categoricals present as pandas category dtype
    assert isinstance(X["gics_sector"].dtype, pd.CategoricalDtype)


# ---------------------------------------------------------- fit + save + load


@pytest.fixture
def trained_bundle(tmp_path):
    df = _synthetic_features(n=400)
    cfg = _cfg()
    feature_cols = ["sue_std", "ear", "mom_1m", "gics_sector", "ff12"]
    schema = train_final._fit_schema(df, feature_cols)
    X = train_final._apply_schema(df, schema)

    q_boosters = train_final._fit_quantile_boosters(
        X, df["drift_raw_h60"], schema.cat_cols,
        artifact_mod.DEFAULT_QUANTILES, cfg.random_state)
    z_booster = train_final._fit_z_booster(
        X, df["drift_z_h60"], schema.cat_cols, cfg.random_state)
    class_booster = train_final._fit_class_booster(
        X, df["drift_class_h60"], schema.cat_cols, cfg.random_state)

    meta = artifact_mod.make_metadata(
        cutoff_date="2020-01-01", universe="TEST", horizon=60,
        quantiles=artifact_mod.DEFAULT_QUANTILES, n_events=len(df),
        start_year=cfg.start_year, end_year=cfg.end_year,
        embargo_months=cfg.embargo_months,
    )
    hist = train_final._industry_history(df, horizon=60)

    model = DriftModel(schema=schema, metadata=meta,
                       quantile_boosters=q_boosters, z_booster=z_booster,
                       class_booster=class_booster, industry_history=hist)
    out = tmp_path / "bundle"
    model.save(out)
    return out, df


def test_bundle_round_trip_predict_shape_and_monotonic_quantiles(trained_bundle):
    out, df = trained_bundle

    reloaded = DriftModel.load(out)
    assert reloaded.quantile_boosters.keys() == {"q10", "q25", "q50", "q75", "q90"}
    assert reloaded.z_booster is not None
    assert reloaded.class_booster is not None

    # Score the tail of df (unseen by the schema's mean/std only in the trivial
    # sense; we just verify the pipeline runs and shapes are right).
    tail = df.tail(20).reset_index(drop=True)
    preds = reloaded.predict(tail)

    for expected in ("expected_drift", "interval_low", "interval_high",
                     "prob_up", "drift_z_pred", "coverage",
                     "n_features_present", "n_features_total", "horizon"):
        assert expected in preds.columns

    q_cols = [f"drift_raw_pred_q{q}" for q in (10, 25, 50, 75, 90)]
    assert all(c in preds.columns for c in q_cols)
    # Monotonicity: q10 <= q25 <= q50 <= q75 <= q90 per row (after enforced sort).
    q_matrix = preds[q_cols].to_numpy()
    assert (np.diff(q_matrix, axis=1) >= -1e-9).all()

    # Expected drift is p50; interval is [p10, p90].
    assert np.allclose(preds["expected_drift"], preds["drift_raw_pred_q50"])
    assert np.allclose(preds["interval_low"], preds["drift_raw_pred_q10"])
    assert np.allclose(preds["interval_high"], preds["drift_raw_pred_q90"])

    # prob_up is in [0, 1].
    assert ((preds["prob_up"] >= 0) & (preds["prob_up"] <= 1)).all()


def test_predict_signal_direction(trained_bundle):
    """With a linear DGP, higher sue_std should on average push predictions up."""
    out, _ = trained_bundle
    model = DriftModel.load(out)

    lo = pd.DataFrame([{"sue_std": -2.0, "ear": 0.0, "mom_1m": 0.0,
                        "gics_sector": "Tech", "ff12": "Tech"}])
    hi = pd.DataFrame([{"sue_std": 2.0, "ear": 0.0, "mom_1m": 0.0,
                        "gics_sector": "Tech", "ff12": "Tech"}])
    p_lo = model.predict(lo)["expected_drift"].iloc[0]
    p_hi = model.predict(hi)["expected_drift"].iloc[0]
    assert p_hi > p_lo


def test_missing_features_produce_low_coverage(trained_bundle):
    out, _ = trained_bundle
    model = DriftModel.load(out)
    # Only supply one feature out of five -> coverage = 1/5 = 0.2.
    sparse = pd.DataFrame([{"sue_std": 0.5}])
    preds = model.predict(sparse)
    assert preds["coverage"].iloc[0] == pytest.approx(0.2)
    assert preds["n_features_present"].iloc[0] == 1
    assert preds["n_features_total"].iloc[0] == 5


# ---------------------------------------------------------- industry history


def test_industry_history_causal_filter():
    hist = pd.DataFrame({
        "ff12": ["Tech", "Tech", "Tech", "Fin"],
        "close_date": pd.to_datetime(["2020-03-01", "2020-05-01",
                                      "2021-01-01", "2020-03-01"]),
        "drift_raw": [0.02, 0.04, -0.10, 0.05],
    })
    # As of 2020-06-01: two Tech events have closed; mean = 0.03.
    v = industry_drift_from_history(hist, "Tech", pd.Timestamp("2020-06-01"))
    assert v == pytest.approx(0.03)
    # As of 2020-01-01: none closed yet -> NaN.
    v = industry_drift_from_history(hist, "Tech", pd.Timestamp("2020-01-01"))
    assert np.isnan(v)


# ---------------------------------------------------------- merge inputs


def test_merge_inputs_prefer_cache_uses_cached_when_user_missing():
    user = EventInputs(ticker="AAPL", anndate="2025-01-30")
    cached = {"anntims": "16:30:00", "actual": 2.34, "meanest": 2.10, "stdev": 0.05,
              "numest": 30, "fpedats": pd.Timestamp("2024-12-31"),
              "statpers": pd.Timestamp("2025-01-25"), "numup": 5, "numdown": 2,
              "medest": 2.11, "cname": "APPLE INC"}
    resolved = merge_inputs(user, cached, prefer_cache=True)
    assert resolved["actual"] == 2.34
    assert resolved["numest"] == 30


def test_merge_inputs_user_override_wins():
    user = EventInputs(ticker="AAPL", anndate="2025-01-30", actual=9.99)
    cached = {"actual": 2.34, "meanest": 2.10}
    resolved = merge_inputs(user, cached, prefer_cache=True)
    assert resolved["actual"] == 9.99  # explicit user value wins
    assert resolved["meanest"] == 2.10


def test_merge_inputs_no_cache():
    user = EventInputs(ticker="AAPL", anndate="2025-01-30", actual=2.5, meanest=2.4)
    resolved = merge_inputs(user, None, prefer_cache=False)
    assert resolved["actual"] == 2.5
    assert resolved["meanest"] == 2.4


# ---------------------------------------------------------- artifact metadata


def test_artifact_version_tag_and_manifest(tmp_path):
    df = _synthetic_features(n=200)
    schema = train_final._fit_schema(df, ["sue_std", "ear", "gics_sector"])
    X = train_final._apply_schema(df, schema)
    boosters = train_final._fit_quantile_boosters(
        X, df["drift_raw_h60"], schema.cat_cols,
        (0.5,), random_state=0)
    meta = artifact_mod.make_metadata(
        cutoff_date="2024-06-30", universe="TEST", horizon=60,
        quantiles=(0.5,), n_events=len(df),
        start_year=2015, end_year=2024, embargo_months=3,
    )
    model = DriftModel(schema=schema, metadata=meta,
                       quantile_boosters=boosters)
    out = tmp_path / "bundle"
    model.save(out)
    assert (out / "manifest.json").is_file()
    assert (out / "booster_q50.txt").is_file()

    reloaded = DriftModel.load(out)
    assert reloaded.metadata.cutoff_date == "2024-06-30"
    assert reloaded.metadata.horizon == 60
    assert "q50" in reloaded.quantile_boosters
