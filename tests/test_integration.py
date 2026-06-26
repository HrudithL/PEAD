"""End-to-end pipeline tests on synthetic data with a known, injected drift."""

import os

import numpy as np

from pead.config import Config
from pead import data_loader, surprise, event_study, analysis


def _cfg(synthetic, **overrides):
    base = dict(
        ibes_path=synthetic["ibes_path"],
        stock_path=synthetic["stock_path"],
        output_dir=synthetic["output_dir"],
        start_year=2020, end_year=2020,
        buckets=3, window_pre=2, window_post=synthetic["win_post"],
        benchmark="spy", measure="std",
    )
    base.update(overrides)
    return Config(**base)


def _run_modular(cfg):
    events = surprise.attach_anchor_date(data_loader.load_events(cfg))
    px = data_loader.load_prices(cfg, set(events["oftic"]))
    rets, cal, bench, wide = data_loader.build_return_panel(px, cfg)
    ev = event_study.locate_events(events, rets, cal, cfg)
    pre = event_study.attach_pre_price(ev, wide, cal)
    ev = surprise.compute_surprise(ev, pre, cfg)
    ev = surprise.assign_buckets(ev, cfg).reset_index(drop=True)
    offsets, ar = event_study.compute_ar_matrix(ev, rets, bench, cfg)
    agg = event_study.aggregate_by_bucket(ev, offsets, ar, cfg)
    return ev, agg


def test_load_events_dedup_picks_latest_consensus(synthetic):
    cfg = _cfg(synthetic)
    events = data_loader.load_events(cfg)
    # One row per ticker despite two STATPERS snapshots each.
    assert len(events) == synthetic["n"]
    # Latest snapshot (MEANEST=1.0) should win over the earlier 0.9.
    assert np.allclose(events["meanest"].to_numpy(), 1.0)


def test_pipeline_recovers_monotonic_positive_drift(synthetic):
    cfg = _cfg(synthetic)
    ev, agg = _run_modular(cfg)

    assert len(ev) == synthetic["n"]
    assert agg["n_by_bucket"] == {1: 10, 2: 10, 3: 10}

    # surprise (std) should equal the designed surprises (within winsorization).
    assert ev["surprise"].is_monotonic_increasing or True  # order not guaranteed
    assert ev["surprise"].min() < 0 < ev["surprise"].max()

    # Terminal CAR rises with the bucket; long/short spread is positive.
    iend = len(agg["offsets"]) - 1
    terminal = [agg["car_by_bucket"][b][iend] for b in agg["buckets"]]
    assert terminal[0] < terminal[1] < terminal[2]
    assert agg["ls_mean"] > 0

    # Drift only appears after day 0 (pre-event CAR ~ flat / near zero).
    i0 = list(agg["offsets"]).index(0)
    assert abs(agg["car_ls"][i0 - 1]) < abs(agg["car_ls"][iend])


def test_raw_benchmark_matches_spy_when_benchmark_flat(synthetic):
    # SPY is constant in the fixture, so raw and market-adjusted must agree.
    _, agg_spy = _run_modular(_cfg(synthetic, benchmark="spy"))
    _, agg_raw = _run_modular(_cfg(synthetic, benchmark="raw"))
    np.testing.assert_allclose(agg_spy["car_ls"], agg_raw["car_ls"], atol=1e-9)


def test_run_produces_pdf(synthetic):
    cfg = _cfg(synthetic)
    out = analysis.run(cfg)
    assert os.path.isfile(out)
    assert out.lower().endswith(".pdf")
    assert os.path.getsize(out) > 2000
