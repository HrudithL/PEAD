"""Tests for point-in-time drift-ML feature engineering (Section 5)."""

import numpy as np
import pandas as pd

from pead.sub_sampling_ml.config import DriftMLConfig
from pead.sub_sampling_ml import features as F

_POS0_FULL = 260      # full 252-day history available (260 - 252 = 8 >= 0)
_POS0_SHORT = 30      # short history: no 252d window, but a 21d window exists

def _make_data():
    rng = np.random.default_rng(0)
    n_days = 300
    cal = pd.bdate_range("2018-01-01", periods=n_days)
    cols = ["AAA", "BBB", "SPY"]
    R = rng.normal(0.0005, 0.01, size=(n_days, len(cols)))
    rets = pd.DataFrame(R, index=cal, columns=cols)
    prices = 100.0 * (1.0 + rets).cumprod()
    bench = rets["SPY"]
    return cal, rets, prices, bench, R

def _make_events(cal, prices):
    ev = pd.DataFrame({
        "oftic": ["AAA", "BBB"],
        "anndats": [cal[_POS0_FULL], cal[_POS0_SHORT]],
        "anntims": ["16:30:00", "09:00:00"],   # AMC vs BMO
        "pos0": [_POS0_FULL, _POS0_SHORT],
        "pre_price": [prices["AAA"].iloc[_POS0_FULL - 1],
                      prices["BBB"].iloc[_POS0_SHORT - 1]],
        "actual": [1.2, 0.8],
        "meanest": [1.0, 1.0],
        "medest": [1.0, 1.0],
        "stdev": [0.5, 0.4],
        "numest": [8, 5],
        "numup": [6, 1],
        "numdown": [1, 3],
    })
    return ev

def _build():
    cal, rets, prices, bench, R = _make_data()
    ev = _make_events(cal, prices)
    cfg = DriftMLConfig()
    feats = F.build_features(ev, rets, prices, cal, bench, cfg, wrds_panels=None)
    return ev, feats, R

def test_output_aligned_to_event_index():
    ev, feats, _ = _build()
    assert feats.index.equals(ev.index)
    assert len(feats) == 2                      # short-history event still appears

def test_ear_matches_hand_computed_day0_abnormal_return():
    _, feats, R = _build()
    expected = R[_POS0_FULL, 0] - R[_POS0_FULL, 2]   # AAA - SPY at day 0
    assert np.isclose(feats.loc[0, "ear"], expected)

def test_momentum_and_reversal_match_hand_computed():
    _, feats, R = _build()
    aaa = R[:, 0]
    mom_1m = np.prod(1.0 + aaa[_POS0_FULL - 21:_POS0_FULL]) - 1.0     # offsets [-21,-1]
    rev_1w = np.prod(1.0 + aaa[_POS0_FULL - 5:_POS0_FULL]) - 1.0      # offsets [-5,-1]
    assert np.isclose(feats.loc[0, "mom_1m"], mom_1m)
    assert np.isclose(feats.loc[0, "reversal_1w"], rev_1w)

def test_short_history_yields_nan_long_momentum_but_row_survives():
    _, feats, _ = _build()
    # pos0=30 -> no 252d window -> mom_12_1, beta_252, rvol_60 are NaN ...
    assert np.isnan(feats.loc[1, "mom_12_1"])
    assert np.isnan(feats.loc[1, "beta_252"])
    assert np.isnan(feats.loc[1, "rvol_60"])
    # ... but the 21d momentum is computable (30 - 21 = 9 >= 0).
    assert not np.isnan(feats.loc[1, "mom_1m"])

def test_repo_only_signal_definitions():
    _, feats, R = _build()
    assert np.isclose(feats.loc[0, "sue_std"], 0.2 / 0.5)      # (1.2-1.0)/0.5
    assert int(feats.loc[0, "beat_flag"]) == 1                 # beat
    assert int(feats.loc[1, "beat_flag"]) == -1                # miss
    assert feats.loc[0, "report_time"] == "AMC"
    assert feats.loc[1, "report_time"] == "BMO"

def test_wrds_columns_present_and_all_nan_without_panels():
    _, feats, _ = _build()
    wrds_cols = list(F._LIQUIDITY_WRDS_COLS) + list(F._FUNDAMENTAL_COLS) + [
        "industry_mom", "industry_drift_base", "gics_sector", "ff12", "ff48",
        "idx_member",
    ]
    for c in wrds_cols:
        assert c in feats.columns
        assert feats[c].isna().all(), c

def test_categorical_columns_have_category_dtype():
    _, feats, _ = _build()
    for c in F.CATEGORICAL_FEATURES:
        assert c in feats.columns
        assert str(feats[c].dtype) == "category", c

def test_revision_net_uses_numup_numdown():
    _, feats, _ = _build()
    # AAA: (6 - 1) / 8 ; BBB: (1 - 3) / 5
    assert np.isclose(feats.loc[0, "revision_net"], 5 / 8)
    assert np.isclose(feats.loc[1, "revision_net"], -2 / 5)


def _make_wrds_panels(cal):
    """Minimal WRDS panels for two same-industry firms with 5+ fundq quarters."""
    link = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "permno": [101, 102],
        "gvkey": ["1001", "1002"],
    })
    company = pd.DataFrame({          # same SIC -> same FF12 industry
        "gvkey": ["1001", "1002"],
        "gsector": [45, 45],
        "sic": [3674, 3674],
    })
    rdqs = pd.bdate_range("2017-01-15", periods=8, freq="63D")
    rows = []
    for gv, atq0 in (("1001", 1000.0), ("1002", 2000.0)):
        for k, rdq in enumerate(rdqs):
            rows.append({
                "gvkey": gv, "rdq": rdq,
                "atq": atq0 + 50 * k, "ceqq": 400 + 10 * k, "niq": 50 + k,
                "revtq": 300 + 5 * k, "cogsq": 180 + 2 * k,
                "saleq": 300 + 5 * k, "dlttq": 100 + 3 * k, "dlcq": 20 + k,
                "xrdq": 15.0, "emp": 5.0,
                "actq": 500 + 8 * k, "lctq": 200 + 4 * k, "cheq": 60 + k,
                "txpq": 10 + k, "dpq": 12.0,
            })
    fundq = pd.DataFrame(rows)
    return {"link": link, "company": company, "fundq": fundq}


def _events_for_panels(cal, prices):
    ev = _make_events(cal, prices)
    # Both events late enough that >=5 prior fundq quarters exist (rdq < anndats).
    ev["anndats"] = [cal[_POS0_FULL], cal[_POS0_FULL]]
    ev["pos0"] = [_POS0_FULL, _POS0_FULL]
    ev["oftic"] = ["AAA", "BBB"]
    return ev


def test_leverage_nan_when_debt_fields_absent():
    cal, rets, prices, bench, _ = _make_data()
    ev = _events_for_panels(cal, prices)
    panels = _make_wrds_panels(cal)
    panels["fundq"] = panels["fundq"].drop(columns=["dlttq", "dlcq"])
    cfg = DriftMLConfig()
    feats = F.build_features(ev, rets, prices, cal, bench, cfg, wrds_panels=panels)
    assert feats["leverage"].isna().all()        # no 0-instead-of-NaN


def test_leverage_and_accruals_computed_with_panels():
    cal, rets, prices, bench, _ = _make_data()
    ev = _events_for_panels(cal, prices)
    panels = _make_wrds_panels(cal)
    cfg = DriftMLConfig()
    feats = F.build_features(ev, rets, prices, cal, bench, cfg, wrds_panels=panels)
    assert feats["leverage"].notna().all() and (feats["leverage"] > 0).all()
    assert feats["accruals"].notna().all()       # balance-sheet accruals filled
    assert feats["gics_sector"].notna().all()


def test_industry_mom_is_peer_average_excluding_self():
    cal, rets, prices, bench, R = _make_data()
    ev = _events_for_panels(cal, prices)
    panels = _make_wrds_panels(cal)
    cfg = DriftMLConfig()
    feats = F.build_features(ev, rets, prices, cal, bench, cfg, wrds_panels=panels)
    # Two firms in one FF12 group -> each one's industry_mom is the OTHER firm's
    # trailing [-21,-1] return (self excluded).
    p0 = _POS0_FULL
    bbb_tr = np.prod(1.0 + R[p0 - 21:p0, 1]) - 1.0
    aaa_tr = np.prod(1.0 + R[p0 - 21:p0, 0]) - 1.0
    assert np.isclose(feats.loc[0, "industry_mom"], bbb_tr)
    assert np.isclose(feats.loc[1, "industry_mom"], aaa_tr)


def test_build_features_returns_union_of_all_families():
    _, feats, _ = _build()
    expected = {
        # 5.1
        "sue_std", "sue_price", "surprise_raw", "analyst_disp", "n_analysts",
        "revision_net", "forecast_staleness", "beat_flag",
        # 5.2
        "mom_12_1", "mom_1m", "reversal_1w", "rvol_60", "beta_252", "ivol",
        "high52_prox", "pre_run", "ear",
        # 5.3
        "price_level", "mktcap", "dollar_vol", "turnover", "amihud", "idx_member",
        # 5.4
        "gics_sector", "ff12", "ff48", "industry_mom", "industry_drift_base",
        # 5.5
        "book_to_market", "roa", "gross_margin", "leverage", "asset_growth",
        "sales_growth", "accruals", "n_employees", "earn_vol", "rd_intensity",
        # 5.7
        "fiscal_q", "report_time", "mkt_ret_pre", "vix_level",
    }
    assert expected.issubset(set(feats.columns))
