"""Tests for the drift-label encodings (CAR windows, z/decile/class, horizons)."""

import numpy as np
import pandas as pd

from pead.sub_sampling_ml import labels
from pead.sub_sampling_ml.config import DriftMLConfig


def test_car_window_excludes_day0_and_sums_post_window():
    offsets = np.array([-1, 0, 1, 2, 3])
    ar = np.array([
        [0.10, 0.50, 0.01, 0.02, 0.03],   # [+1,+3] -> 0.06 (day0 0.50 excluded)
        [0.00, 0.00, 0.10, 0.00, 0.00],   # -> 0.10
    ])
    car3 = labels.car_window(offsets, ar, start=1, end=3)
    np.testing.assert_allclose(car3, [0.06, 0.10])
    car1 = labels.car_window(offsets, ar, start=1, end=1)
    np.testing.assert_allclose(car1, [0.01, 0.10])


def test_car_window_treats_nan_as_zero():
    offsets = np.array([0, 1, 2])
    ar = np.array([[0.5, np.nan, 0.02]])
    np.testing.assert_allclose(labels.car_window(offsets, ar, 1, 2), [0.02])


def test_car_window_returns_nan_when_horizon_out_of_range():
    offsets = np.array([-1, 0, 1])
    ar = np.array([[0.0, 0.0, 0.01]])
    out = labels.car_window(offsets, ar, start=1, end=60)
    # offsets up to +1 exist, so [1,60] still picks the +1 column.
    np.testing.assert_allclose(out, [0.01])
    out2 = labels.car_window(offsets, ar, start=5, end=60)
    assert np.isnan(out2).all()


def test_cross_sectional_z_is_per_quarter():
    vals = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    q = pd.Series(["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"])
    z = labels.cross_sectional_z(vals, q)
    # Each quarter standardized independently -> mean 0 within quarter.
    assert abs(z[:3].mean()) < 1e-9
    assert abs(z[3:].mean()) < 1e-9
    np.testing.assert_allclose(z[:3].values, z[3:].values)


def test_decile_and_class_encoding():
    vals = pd.Series(np.arange(20, dtype=float))
    q = pd.Series(["Q1"] * 20)
    dec = labels.per_quarter_decile(vals, q, n=10)
    assert dec.min() == 1 and dec.max() == 10
    cls = labels.decile_to_class(dec, n=10)
    assert set(cls.dropna().unique()) == {0.0, 1.0}
    # bottom two values -> decile 1 -> class 0; top two -> decile 10 -> class 1.
    assert cls.iloc[0] == 0.0 and cls.iloc[-1] == 1.0
    assert cls.isna().sum() == 16


def test_per_quarter_decile_nan_when_too_few():
    vals = pd.Series([1.0, 2.0, 3.0])
    q = pd.Series(["Q1", "Q1", "Q1"])
    dec = labels.per_quarter_decile(vals, q, n=10)
    assert dec.isna().all()


def test_compute_labels_horizons_and_primary_alias():
    n = 40
    offsets = np.arange(-5, 61)
    rng = np.random.default_rng(0)
    ar = rng.normal(0, 0.01, size=(n, offsets.size))
    ev = pd.DataFrame({
        "anndats": pd.to_datetime(["2020-02-10"] * 20 + ["2020-05-10"] * 20),
        "oftic": [f"T{i}" for i in range(n)],
    })
    cfg = DriftMLConfig(horizons=(60, 20, 5), n_deciles=10)
    out = labels.compute_labels(ev, offsets, ar, cfg)

    for h in (60, 20, 5):
        for enc in ("drift_raw", "drift_z", "drift_decile", "drift_class"):
            assert f"{enc}_h{h}" in out.columns
    assert "cal_q" in out.columns
    assert out.index.equals(ev.index)

    # Primary-horizon bare aliases mirror the suffixed columns.
    np.testing.assert_allclose(out["drift_raw"], out["drift_raw_h60"])
    pd.testing.assert_series_equal(
        out["drift_decile"], out["drift_decile_h60"], check_names=False
    )

    # raw_h60 should equal sum of AR over offsets 1..60.
    expected = ar[:, (offsets >= 1) & (offsets <= 60)].sum(axis=1)
    np.testing.assert_allclose(out["drift_raw_h60"].values, expected)

    # Two quarters of 20 -> deciles populated within each quarter.
    assert out["drift_decile_h60"].notna().all()
