"""Tests for dataset assembly: as-of joins, feature/label split, end-to-end build."""

import numpy as np
import pandas as pd

from pead.sub_sampling_ml import dataset
from pead.sub_sampling_ml.config import DriftMLConfig


def test_as_of_merge_backward_pit_and_order_preserved():
    left = pd.DataFrame({
        "key": ["A", "A", "B"],
        "t": pd.to_datetime(["2020-03-15", "2020-01-10", "2020-06-01"]),
        "x": [1, 2, 3],
    })
    right = pd.DataFrame({
        "key": ["A", "A", "B"],
        "rt": pd.to_datetime(["2020-01-01", "2020-03-01", "2020-05-01"]),
        "val": [10, 20, 30],
    })
    out = dataset.as_of_merge(left, right, by="key", left_time="t", right_time="rt")
    # Order matches the original left rows.
    assert list(out["x"]) == [1, 2, 3]
    # A@2020-03-15 -> most recent right on/before = 2020-03-01 (val 20).
    assert out.loc[out["x"] == 1, "val"].iloc[0] == 20
    # A@2020-01-10 -> 2020-01-01 (val 10).
    assert out.loc[out["x"] == 2, "val"].iloc[0] == 10
    # B@2020-06-01 -> 2020-05-01 (val 30).
    assert out.loc[out["x"] == 3, "val"].iloc[0] == 30


def test_as_of_merge_returns_nan_before_first_right():
    left = pd.DataFrame({"key": ["A"], "t": pd.to_datetime(["2019-12-01"]), "x": [1]})
    right = pd.DataFrame({"key": ["A"], "rt": pd.to_datetime(["2020-01-01"]), "val": [9]})
    out = dataset.as_of_merge(left, right, by="key", left_time="t", right_time="rt")
    assert pd.isna(out["val"].iloc[0])


def test_feature_columns_excludes_labels_and_meta():
    df = pd.DataFrame(columns=[
        "oftic", "anndats", "cal_q",
        "sue_std", "ear", "mktcap",
        "drift_raw", "drift_z_h60", "drift_decile", "drift_class_h20",
    ])
    cols = dataset.feature_columns(df)
    assert set(cols) == {"sue_std", "ear", "mktcap"}


def test_build_event_features_end_to_end_synthetic(synthetic):
    cfg = DriftMLConfig(
        ibes_path=synthetic["ibes_path"],
        stock_path=synthetic["stock_path"],
        output_dir=synthetic["output_dir"],
        start_year=2019, end_year=2021,
        horizons=(5,),               # synthetic injects drift over ~10 post days
        n_deciles=5,
        use_wrds=False,
    )
    df = dataset.build_event_features(cfg, write=False)

    assert len(df) == synthetic["n"]
    # Labels present and finite where computable.
    assert "drift_raw" in df.columns and "drift_raw_h5" in df.columns
    assert df["drift_raw"].notna().any()

    fcols = dataset.feature_columns(df)
    assert "ear" in fcols and "sue_std" in fcols
    assert not any(c.startswith("drift_") for c in fcols)
    assert "anndats" not in fcols

    # Synthetic drift is monotonic in surprise, so realized drift should
    # correlate positively with the standardized surprise feature.
    sub = df[["sue_std", "drift_raw_h5"]].dropna()
    assert sub["sue_std"].corr(sub["drift_raw_h5"]) > 0.5
