"""Point-in-time feature engineering per family (Section 5).

Every feature must be knowable strictly before trading day +1 (Section 4) or it
is leakage. Repo-only families (earnings/analyst, price/return & risk, calendar)
are computed from IBES + the price panel. Families needing CRSP/Compustat
(full liquidity, sector/GICS, fundamentals) are filled from the WRDS panels
assembled by :mod:`pead.sub_sampling_ml.wrds_extract`; when those panels are
absent the columns are still emitted (as NaN) so the schema is stable.

The public entry point is :func:`build_features`, which returns one row per
event aligned to ``ev.index``. The day-0 announcement abnormal return (``ear``)
ends before the [+1, +H] drift window and is therefore a legitimate -- and
likely powerful -- point-in-time feature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import DriftMLConfig

# Categorical features that downstream models must treat as categories, not floats.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "beat_flag", "fiscal_q", "report_time", "gics_sector", "ff12", "ff48",
    "idx_member",
)


def earnings_features(ev: pd.DataFrame, cfg: DriftMLConfig) -> pd.DataFrame:
    """5.1 Earnings / analyst signal: sue_std, sue_price, surprise_raw,
    analyst_disp, n_analysts, revision_net, forecast_staleness, beat_flag.

    (``ear`` is added in :func:`price_risk_features` since it needs returns.)
    """
    raise NotImplementedError


def price_risk_features(ev: pd.DataFrame, rets: pd.DataFrame,
                        prices_wide: pd.DataFrame, calendar: pd.DatetimeIndex,
                        bench_ret: pd.Series, cfg: DriftMLConfig) -> pd.DataFrame:
    """5.2 Price/return & risk (pre-event) plus ``ear``: mom_12_1, mom_1m,
    reversal_1w, rvol_60, beta_252, ivol, high52_prox, pre_run, ear."""
    raise NotImplementedError


def liquidity_features(ev: pd.DataFrame, prices_wide: pd.DataFrame,
                       calendar: pd.DatetimeIndex, cfg: DriftMLConfig,
                       wrds_panels: Optional[dict] = None) -> pd.DataFrame:
    """5.3 Liquidity / size: price_level (px) + mktcap, dollar_vol, turnover,
    amihud, idx_member (CRSP/ref via ``wrds_panels``; NaN when unavailable)."""
    raise NotImplementedError


def sector_features(ev: pd.DataFrame, rets: pd.DataFrame,
                    calendar: pd.DatetimeIndex, cfg: DriftMLConfig,
                    wrds_panels: Optional[dict] = None) -> pd.DataFrame:
    """5.4 Sector / industry: gics_sector, ff12, ff48, industry_mom,
    industry_drift_base (PIT expanding mean). Needs SIC/GICS from ``wrds_panels``."""
    raise NotImplementedError


def fundamental_features(ev: pd.DataFrame, cfg: DriftMLConfig,
                         wrds_panels: Optional[dict] = None) -> pd.DataFrame:
    """5.5 Fundamentals / firm quality (Compustat, as-of joined PIT):
    book_to_market, roa, gross_margin, leverage, asset_growth, sales_growth,
    accruals, n_employees, earn_vol, rd_intensity."""
    raise NotImplementedError


def calendar_features(ev: pd.DataFrame, rets: pd.DataFrame,
                      calendar: pd.DatetimeIndex, bench_ret: pd.Series,
                      cfg: DriftMLConfig) -> pd.DataFrame:
    """5.7 Context / calendar: fiscal_q, report_time, mkt_ret_pre, vix_level."""
    raise NotImplementedError


def build_features(ev: pd.DataFrame, rets: pd.DataFrame,
                   prices_wide: pd.DataFrame, calendar: pd.DatetimeIndex,
                   bench_ret: pd.Series, cfg: DriftMLConfig,
                   wrds_panels: Optional[dict] = None) -> pd.DataFrame:
    """Assemble all feature families into one event-indexed matrix.

    Returns a DataFrame aligned to ``ev.index`` whose columns are the union of
    every family's features. Categorical columns are dtype ``category``.
    """
    raise NotImplementedError
