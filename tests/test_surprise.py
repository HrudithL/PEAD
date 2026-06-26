"""Tests for surprise measures, trading-day anchoring and bucketing."""

import numpy as np
import pandas as pd

from pead.config import Config
from pead.equities import surprise


def _events(anntims):
    return pd.DataFrame({
        "oftic": ["AAA"] * len(anntims),
        "anndats": [pd.Timestamp("2020-02-10")] * len(anntims),
        "anntims": anntims,
    })


def test_anchor_after_close_rolls_to_next_day():
    ev = surprise.attach_anchor_date(_events(["16:30:00"]))
    assert ev["anchor"].iloc[0] == pd.Timestamp("2020-02-11")


def test_anchor_pre_market_is_same_day():
    ev = surprise.attach_anchor_date(_events(["06:00:00"]))
    assert ev["anchor"].iloc[0] == pd.Timestamp("2020-02-10")


def test_anchor_intraday_is_same_day():
    ev = surprise.attach_anchor_date(_events(["10:00:00"]))
    assert ev["anchor"].iloc[0] == pd.Timestamp("2020-02-10")


def test_anchor_missing_time_is_conservative_next_day():
    ev = surprise.attach_anchor_date(_events([None]))
    assert ev["anchor"].iloc[0] == pd.Timestamp("2020-02-11")


def test_compute_surprise_std_and_price():
    ev = pd.DataFrame({
        "actual": [1.5, 2.0, 0.5],
        "meanest": [1.0, 1.0, 1.0],
        "stdev": [0.5, 0.0, 0.25],   # second has invalid (zero) stdev
        "anndats": [pd.Timestamp("2020-02-10")] * 3,
    })
    pre_price = pd.Series([100.0, 50.0, 25.0], index=ev.index)

    cfg = Config(measure="std")
    out = surprise.compute_surprise(ev, pre_price, cfg)

    # sue_std = (actual-meanest)/stdev ; zero-stdev -> NaN
    assert np.isclose(out["sue_std"].iloc[0], 1.0)
    assert pd.isna(out["sue_std"].iloc[1])
    assert np.isclose(out["sue_std"].iloc[2], -2.0)

    # sue_price = (actual-meanest)/price
    assert np.isclose(out["sue_price"].iloc[0], 0.5 / 100.0)

    # chosen measure is std-based: invalid-stdev row stays NaN, and the
    # (winsorized) surprise preserves the sign/order of sue_std.
    assert pd.isna(out["surprise"].iloc[1])
    assert out["surprise"].iloc[0] > 0 > out["surprise"].iloc[2]


def test_compute_surprise_price_measure_selected():
    ev = pd.DataFrame({
        "actual": [1.5], "meanest": [1.0], "stdev": [0.5],
        "anndats": [pd.Timestamp("2020-02-10")],
    })
    out = surprise.compute_surprise(ev, pd.Series([10.0]), Config(measure="price"))
    assert np.isclose(out["surprise"].iloc[0], 0.5 / 10.0)


def test_assign_buckets_balanced_and_ordered():
    n = 30
    ev = pd.DataFrame({
        "surprise": np.linspace(-5, 5, n),
        "anndats": [pd.Timestamp("2020-02-10")] * n,
    })
    out = surprise.assign_buckets(ev, Config(buckets=3))

    counts = out["bucket"].value_counts()
    assert set(counts.index) == {1, 2, 3}
    assert counts.min() == counts.max() == 10           # perfectly balanced

    means = out.groupby("bucket")["surprise"].mean()
    assert means[1] < means[2] < means[3]               # ordered low -> high
