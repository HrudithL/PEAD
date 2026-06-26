"""Tests for the event-study engine (alignment, AR matrix, aggregation)."""

import numpy as np
import pandas as pd

from pead.config import Config
from pead import event_study


def _calendar(n=12):
    return pd.bdate_range("2020-01-01", periods=n)


def test_locate_events_assigns_pos0_and_filters():
    cal = _calendar(12)
    rets = pd.DataFrame(0.0, index=cal, columns=["A", "B"])
    events = pd.DataFrame({
        "oftic": ["A", "B", "A", "C"],
        # anchors map onto calendar positions 5, 0 (too early), 11 (too late), n/a ticker
        "anchor": [cal[5], cal[0], cal[11], cal[5]],
    })
    cfg = Config(window_pre=1, window_post=2)
    ev = event_study.locate_events(events, rets, cal, cfg)

    # 'C' dropped (not in panel); pos0=0 dropped (needs pos0-1>=1);
    # pos0=11 dropped (hi=13 >= n). Only the A@pos5 survives.
    assert list(ev["oftic"]) == ["A"]
    assert int(ev["pos0"].iloc[0]) == 5


def test_attach_pre_price_reads_prior_close():
    cal = _calendar(12)
    prices = pd.DataFrame({"A": np.arange(100.0, 112.0)}, index=cal)
    ev = pd.DataFrame({"oftic": ["A"], "pos0": [5]})
    pre = event_study.attach_pre_price(ev, prices, cal)
    assert pre.iloc[0] == 104.0          # price at pos0-1 = index 4


def test_compute_ar_matrix_raw_and_market_adjusted():
    cal = _calendar(12)
    a = np.zeros(12)
    a[4:8] = [0.00, 0.02, 0.03, 0.01]
    spy = np.zeros(12)
    spy[4:8] = [0.00, 0.01, 0.01, 0.00]
    rets = pd.DataFrame({"A": a, "SPY": spy}, index=cal)
    bench = rets["SPY"]
    ev = pd.DataFrame({"oftic": ["A"], "pos0": [5]})

    cfg_raw = Config(window_pre=1, window_post=2, benchmark="raw")
    offsets, ar = event_study.compute_ar_matrix(ev, rets, bench, cfg_raw)
    assert list(offsets) == [-1, 0, 1, 2]
    np.testing.assert_allclose(ar[0], [0.00, 0.02, 0.03, 0.01])

    cfg_spy = Config(window_pre=1, window_post=2, benchmark="spy")
    _, ar2 = event_study.compute_ar_matrix(ev, rets, bench, cfg_spy)
    np.testing.assert_allclose(ar2[0], [0.00, 0.01, 0.02, 0.01])


def test_aggregate_by_bucket_car_and_long_short():
    offsets = np.array([-1, 0, 1, 2])
    ar = np.array([
        [0.00, -0.01, -0.02, -0.01],   # bucket 1
        [0.00,  0.02,  0.03,  0.01],   # bucket 2
    ])
    ev = pd.DataFrame({
        "bucket": [1, 2],
        "surprise": [-2.0, 2.0],
        "anndats": [pd.Timestamp("2020-02-10")] * 2,
    })
    cfg = Config(window_pre=1, window_post=2, buckets=2)
    agg = event_study.aggregate_by_bucket(ev, offsets, ar, cfg)

    np.testing.assert_allclose(agg["car_by_bucket"][1], [0.0, -0.01, -0.03, -0.04])
    np.testing.assert_allclose(agg["car_by_bucket"][2], [0.0, 0.02, 0.05, 0.06])
    np.testing.assert_allclose(agg["car_ls"], [0.0, 0.03, 0.08, 0.10])
    assert np.isclose(agg["ls_mean"], 0.06 - (-0.04))
    assert agg["n_by_bucket"] == {1: 1, 2: 1}
    assert list(agg["summary"]["Bucket"]) == [1, 2]


def test_compute_ar_matrix_handles_nan_returns():
    cal = _calendar(12)
    a = np.full(12, np.nan)
    a[4:8] = [0.0, 0.05, np.nan, 0.02]      # a gap mid-window
    rets = pd.DataFrame({"A": a}, index=cal)
    bench = pd.Series(0.0, index=cal)
    ev = pd.DataFrame({
        "oftic": ["A"], "pos0": [5], "bucket": [1], "surprise": [1.0],
        "anndats": [pd.Timestamp("2020-02-10")],
    })
    cfg = Config(window_pre=1, window_post=2, buckets=1, benchmark="raw")
    offsets, ar = event_study.compute_ar_matrix(ev, rets, bench, cfg)
    agg = event_study.aggregate_by_bucket(ev, offsets, ar, cfg)
    # NaN AR is treated as 0 in the cumulative path -> no propagation of NaN.
    assert np.isfinite(agg["car_by_bucket"][1]).all()
