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

# Column schemas per WRDS-backed family, so the output is stable when panels are
# absent (every column still appears, filled with NaN).
_LIQUIDITY_WRDS_COLS = ("mktcap", "dollar_vol", "turnover", "amihud")
_SECTOR_COLS = ("gics_sector", "ff12", "ff48", "industry_mom", "industry_drift_base")
_FUNDAMENTAL_COLS = (
    "book_to_market", "roa", "gross_margin", "leverage", "asset_growth",
    "sales_growth", "accruals", "n_employees", "earn_vol", "rd_intensity",
)

# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _cumret(series: np.ndarray, lo: int, hi: int) -> float:
    """Compounded simple return over inclusive index range [lo, hi].

    Returns NaN when the window runs off the start/end of the calendar (i.e.
    insufficient history); missing daily returns inside the window are treated
    as a flat day so a single gap does not void the whole window.
    """
    if lo < 0 or hi >= series.shape[0] or hi < lo:
        return np.nan
    seg = series[lo:hi + 1]
    return float(np.prod(1.0 + np.nan_to_num(seg, nan=0.0)) - 1.0)

def _safe_log(x: float) -> float:
    return float(np.log(x)) if (x is not None and np.isfinite(x) and x > 0) else np.nan

def _beta_ivol(stock: np.ndarray, mkt: np.ndarray) -> tuple[float, float]:
    """Market-model slope of ``stock`` on ``mkt`` and residual std (idiosyncratic vol)."""
    m = np.isfinite(stock) & np.isfinite(mkt)
    if m.sum() < 2:
        return np.nan, np.nan
    x, y = mkt[m], stock[m]
    vx = np.var(x)
    if vx <= 0:
        return np.nan, np.nan
    beta = float(np.cov(x, y, bias=True)[0, 1] / vx)
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    return beta, float(np.std(resid))

def _hour_series(anntims: pd.Series) -> pd.Series:
    """Announcement hour from strings like '16:30:00' (mirrors surprise._parse_hour)."""
    s = anntims.astype("string").str.split(":").str[0]
    return pd.to_numeric(s, errors="coerce")

def _wrds_ok(wrds_panels: Optional[dict]) -> bool:
    return bool(wrds_panels) and isinstance(wrds_panels, dict)

def _colmap(df: pd.DataFrame) -> dict:
    return {str(c).lower(): c for c in df.columns}

# ---------------------------------------------------------------------------
# 5.1 Earnings / analyst signal
# ---------------------------------------------------------------------------

def earnings_features(ev: pd.DataFrame, cfg: DriftMLConfig) -> pd.DataFrame:
    """5.1 Earnings / analyst signal: sue_std, sue_price, surprise_raw,
    analyst_disp, n_analysts, revision_net, forecast_staleness, beat_flag.

    (``ear`` is added in :func:`price_risk_features` since it needs returns.)
    """
    idx = ev.index
    err = ev["actual"] - ev["meanest"]
    out = pd.DataFrame(index=idx)

    if "sue_std" in ev.columns:
        out["sue_std"] = ev["sue_std"]
    else:
        out["sue_std"] = err / ev["stdev"].where(ev["stdev"] > 0)

    if "sue_price" in ev.columns:
        out["sue_price"] = ev["sue_price"]
    else:
        out["sue_price"] = err / ev["pre_price"].where(ev.get("pre_price", pd.Series(np.nan, index=idx)) > 0)

    out["surprise_raw"] = ev["surprise_raw"] if "surprise_raw" in ev.columns else err

    disp = ev["stdev"] / ev["meanest"].abs()
    out["analyst_disp"] = disp.replace([np.inf, -np.inf], np.nan)

    out["n_analysts"] = ev["numest"]

    if {"numup", "numdown"}.issubset(ev.columns):
        out["revision_net"] = (ev["numup"] - ev["numdown"]) / ev["numest"].where(ev["numest"] > 0)
    else:
        out["revision_net"] = np.nan

    if "statpers" in ev.columns:
        out["forecast_staleness"] = (ev["anndats"] - ev["statpers"]).dt.days
    else:
        out["forecast_staleness"] = np.nan

    # sign(surprise): miss/-1, meet/0, beat/+1 (categorised in build_features).
    out["beat_flag"] = np.sign(out["surprise_raw"])
    return out

# ---------------------------------------------------------------------------
# 5.2 Price / return & risk (+ ear)
# ---------------------------------------------------------------------------

def price_risk_features(ev: pd.DataFrame, rets: pd.DataFrame,
                        prices_wide: pd.DataFrame, calendar: pd.DatetimeIndex,
                        bench_ret: pd.Series, cfg: DriftMLConfig) -> pd.DataFrame:
    """5.2 Price/return & risk (pre-event) plus ``ear``: mom_12_1, mom_1m,
    reversal_1w, rvol_60, beta_252, ivol, high52_prox, pre_run, ear."""
    ret_cols = {c: i for i, c in enumerate(rets.columns)}
    px_cols = {c: i for i, c in enumerate(prices_wide.columns)}
    ret_np = rets.to_numpy(dtype=float)
    px_np = prices_wide.to_numpy(dtype=float)
    bench_np = bench_ret.reindex(calendar).to_numpy(dtype=float)
    n = len(calendar)

    names = ("mom_12_1", "mom_1m", "reversal_1w", "rvol_60", "beta_252",
             "ivol", "high52_prox", "pre_run", "ear")
    rows = {k: [] for k in names}

    for i in ev.index:
        tic = ev.at[i, "oftic"]
        p0 = int(ev.at[i, "pos0"])
        ci = ret_cols.get(tic)
        pj = px_cols.get(tic)
        if ci is None or not (0 <= p0 < n):
            for k in names:
                rows[k].append(np.nan)
            continue

        stock = ret_np[:, ci]
        rows["mom_12_1"].append(_cumret(stock, p0 - 252, p0 - 21))
        rows["mom_1m"].append(_cumret(stock, p0 - 21, p0 - 1))
        rows["reversal_1w"].append(_cumret(stock, p0 - 5, p0 - 1))

        lo = p0 - 60
        rows["rvol_60"].append(float(np.nanstd(stock[lo:p0])) if lo >= 0 else np.nan)

        lo = p0 - 252
        if lo >= 0:
            beta, ivol = _beta_ivol(stock[lo:p0], bench_np[lo:p0])
        else:
            beta, ivol = np.nan, np.nan
        rows["beta_252"].append(beta)
        rows["ivol"].append(ivol)

        if pj is not None and (p0 - 252) >= 0:
            seg = px_np[p0 - 252:p0 + 1, pj]
            hi52 = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
            cur = px_np[p0, pj]
            rows["high52_prox"].append(float(cur / hi52) if (hi52 and hi52 > 0) else np.nan)
        else:
            rows["high52_prox"].append(np.nan)

        lo = p0 - 5
        if lo >= 0:
            ab = stock[lo:p0] - bench_np[lo:p0]
            rows["pre_run"].append(float(np.nansum(ab)))
        else:
            rows["pre_run"].append(np.nan)

        ear = stock[p0] - bench_np[p0]
        rows["ear"].append(float(ear) if np.isfinite(ear) else np.nan)

    return pd.DataFrame(rows, index=ev.index)

# ---------------------------------------------------------------------------
# 5.3 Liquidity / size
# ---------------------------------------------------------------------------

def liquidity_features(ev: pd.DataFrame, prices_wide: pd.DataFrame,
                       calendar: pd.DatetimeIndex, cfg: DriftMLConfig,
                       wrds_panels: Optional[dict] = None) -> pd.DataFrame:
    """5.3 Liquidity / size: price_level (px) + mktcap, dollar_vol, turnover,
    amihud, idx_member (CRSP/ref via ``wrds_panels``; NaN when unavailable)."""
    px_cols = {c: i for i, c in enumerate(prices_wide.columns)}
    px_np = prices_wide.to_numpy(dtype=float)
    n = len(calendar)

    out = pd.DataFrame(index=ev.index)
    price_level = []
    for i in ev.index:
        pj = px_cols.get(ev.at[i, "oftic"])
        p0 = int(ev.at[i, "pos0"])
        price_level.append(_safe_log(px_np[p0, pj]) if (pj is not None and 0 <= p0 < n) else np.nan)
    out["price_level"] = price_level

    crsp = _crsp_liquidity(ev, wrds_panels)
    for c in _LIQUIDITY_WRDS_COLS:
        out[c] = crsp[c]

    # Index membership comes from constituent/ref lists, which are not part of
    # the supplied panels -> emitted as a missing category for schema stability.
    out["idx_member"] = pd.Series(np.nan, index=ev.index, dtype="object")
    return out

def _crsp_liquidity(ev: pd.DataFrame, wrds_panels: Optional[dict]) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=ev.index, columns=list(_LIQUIDITY_WRDS_COLS))
    if not _wrds_ok(wrds_panels):
        return out
    try:
        link = wrds_panels.get("link")
        crsp = wrds_panels.get("crsp_daily")
        if link is None or crsp is None:
            return out
        t2p, _ = _link_maps(link)
        groups = _crsp_groups(crsp)
        for i in ev.index:
            permno = t2p.get(str(ev.at[i, "oftic"]).upper())
            g = groups.get(permno)
            if g is None:
                continue
            cutoff = ev.at[i, "anndats"]
            g = g[g["date"] <= cutoff]
            if g.empty:
                continue
            last = g.iloc[-1]
            prc, shr = abs(float(last["prc"])), float(last["shrout"])
            if prc > 0 and shr > 0:
                out.at[i, "mktcap"] = np.log(prc * shr)
            w = g.tail(60)
            dvol = (w["prc"].abs() * w["vol"]).mean()
            out.at[i, "dollar_vol"] = np.log(dvol) if dvol and dvol > 0 else np.nan
            to = (w["vol"] / w["shrout"].where(w["shrout"] > 0))
            out.at[i, "turnover"] = float(to.replace([np.inf, -np.inf], np.nan).mean())
            denom = (w["prc"].abs() * w["vol"]).replace(0, np.nan)
            am = (w["ret"].abs() / denom).replace([np.inf, -np.inf], np.nan)
            out.at[i, "amihud"] = float(am.mean())
    except Exception:
        return out
    return out

# ---------------------------------------------------------------------------
# 5.4 Sector / industry
# ---------------------------------------------------------------------------

def sector_features(ev: pd.DataFrame, rets: pd.DataFrame,
                    calendar: pd.DatetimeIndex, cfg: DriftMLConfig,
                    wrds_panels: Optional[dict] = None) -> pd.DataFrame:
    """5.4 Sector / industry: gics_sector, ff12, ff48, industry_mom,
    industry_drift_base (PIT expanding mean). Needs SIC/GICS from ``wrds_panels``."""
    out = pd.DataFrame(index=ev.index)
    out["gics_sector"] = pd.Series(np.nan, index=ev.index, dtype="object")
    out["ff12"] = pd.Series(np.nan, index=ev.index, dtype="object")
    out["ff48"] = pd.Series(np.nan, index=ev.index, dtype="object")
    out["industry_mom"] = np.nan
    # PIT trailing industry drift needs realized labels, which features cannot
    # see -> emitted as NaN here and filled causally at dataset assembly
    # (dataset.build_event_features), using only prior events whose drift window
    # has already closed.
    out["industry_drift_base"] = np.nan

    if not _wrds_ok(wrds_panels):
        return out
    try:
        link = wrds_panels.get("link")
        company = wrds_panels.get("company")
        if link is None or company is None:
            return out
        _, t2g = _link_maps(link)
        g2sec, g2sic = _company_maps(company)

        tic_ff12: dict[str, object] = {}
        for i in ev.index:
            tic = str(ev.at[i, "oftic"]).upper()
            g = t2g.get(tic)
            sic = g2sic.get(g)
            out.at[i, "gics_sector"] = g2sec.get(g, np.nan)
            ff12 = _ff12_from_sic(sic)
            out.at[i, "ff12"] = ff12
            out.at[i, "ff48"] = _ff48_from_sic(sic)
            tic_ff12[tic] = ff12

        out["industry_mom"] = _industry_mom(ev, rets, calendar, link, g2sic)
    except Exception:
        return out
    return out

def _industry_mom(ev, rets, calendar, link, g2sic) -> pd.Series:
    """Equal-weighted peer-industry [-21,-1] return as of each event (FF12 group).

    Vectorized: each ticker's trailing 21-day compounded return is computed for
    every calendar day at once via cumulative log-returns, then averaged within
    each FF12 group per day. The own-firm leg is removed from its group mean so
    the feature is a true *peer* return. O(days x tickers), not O(events x
    universe).
    """
    res = pd.Series(np.nan, index=ev.index)
    try:
        _, t2g = _link_maps(link)
        cols = list(rets.columns)
        col_idx = {c: i for i, c in enumerate(cols)}
        ff_of_col = np.array([_ff12_from_sic(g2sic.get(t2g.get(str(c).upper())))
                              for c in cols], dtype=object)

        ret_np = rets.to_numpy(dtype=float)
        n_days = ret_np.shape[0]
        if n_days < 22:
            return res

        # trailing[t] over offsets [t-21, t-1] = prod(1+r[t-21..t-1]) - 1,
        # treating missing daily returns as flat (mirrors _cumret's nan->0).
        # S is a zero-prepended prefix sum so sum(log1p[a..b]) = S[b+1] - S[a];
        # the window [t-21, t-1] is therefore S[t] - S[t-21].
        log1p = np.log1p(np.nan_to_num(ret_np, nan=0.0))
        S = np.concatenate([np.zeros((1, ret_np.shape[1])), np.cumsum(log1p, axis=0)])
        trailing = np.full_like(ret_np, np.nan)
        trailing[21:] = np.exp(S[21:n_days] - S[0:n_days - 21]) - 1.0

        # Per-FF12-group, per-day finite sum and count for an equal-weight mean.
        finite = np.isfinite(trailing)
        groups = {ff for ff in ff_of_col if ff is not None and ff == ff}  # drop NaN
        grp_sum: dict[object, np.ndarray] = {}
        grp_cnt: dict[object, np.ndarray] = {}
        for ff in groups:
            mask = (ff_of_col == ff)
            sub = np.where(finite[:, mask], trailing[:, mask], 0.0)
            grp_sum[ff] = sub.sum(axis=1)
            grp_cnt[ff] = finite[:, mask].sum(axis=1)

        for i in ev.index:
            p0 = int(ev.at[i, "pos0"])
            own = str(ev.at[i, "oftic"]).upper()
            ff = _ff12_from_sic(g2sic.get(t2g.get(own)))
            if ff is None or ff != ff or ff not in grp_sum or not (0 <= p0 < n_days):
                continue
            s = grp_sum[ff][p0]
            c = grp_cnt[ff][p0]
            oc = col_idx.get(own)
            if oc is not None and finite[p0, oc]:   # remove own firm from the mean
                s -= trailing[p0, oc]
                c -= 1
            if c > 0:
                res.at[i] = float(s / c)
    except Exception:
        return res
    return res

# ---------------------------------------------------------------------------
# 5.5 Fundamentals / firm quality
# ---------------------------------------------------------------------------

def fundamental_features(ev: pd.DataFrame, cfg: DriftMLConfig,
                         wrds_panels: Optional[dict] = None) -> pd.DataFrame:
    """5.5 Fundamentals / firm quality (Compustat, as-of joined PIT):
    book_to_market, roa, gross_margin, leverage, asset_growth, sales_growth,
    accruals, n_employees, earn_vol, rd_intensity."""
    out = pd.DataFrame(np.nan, index=ev.index, columns=list(_FUNDAMENTAL_COLS))
    if not _wrds_ok(wrds_panels):
        return out
    try:
        link = wrds_panels.get("link")
        fundq = wrds_panels.get("fundq")
        if link is None or fundq is None:
            return out
        _, t2g = _link_maps(link)
        cm = _colmap(fundq)
        gcol, rdq = cm.get("gvkey"), cm.get("rdq")
        if gcol is None or rdq is None:
            return out
        fq = fundq.copy()
        fq[rdq] = pd.to_datetime(fq[rdq], errors="coerce")
        fq = fq.dropna(subset=[rdq]).sort_values([gcol, rdq])
        fgroups = dict(tuple(fq.groupby(gcol)))

        crsp_groups, t2p = None, None
        crsp = wrds_panels.get("crsp_daily")
        if crsp is not None:
            crsp_groups = _crsp_groups(crsp)
            t2p, _ = _link_maps(link)

        def col(name):
            return cm.get(name)

        for i in ev.index:
            g = t2g.get(str(ev.at[i, "oftic"]).upper())
            grp = fgroups.get(g)
            if grp is None:
                continue
            grp = grp[grp[rdq] < ev.at[i, "anndats"]]   # strict PIT on report date
            if grp.empty:
                continue
            cur = grp.iloc[-1]

            def v(name):
                c = col(name)
                return float(cur[c]) if (c is not None and pd.notna(cur[c])) else np.nan

            atq, ceqq, niq = v("atq"), v("ceqq"), v("niq")
            revtq, cogsq = v("revtq"), v("cogsq")
            dlttq, dlcq = v("dlttq"), v("dlcq")
            saleq, xrdq = v("saleq"), v("xrdq")
            emp = v("emp") if col("emp") else v("empq")
            sales = saleq if np.isfinite(saleq) else revtq

            out.at[i, "roa"] = niq / atq if atq else np.nan
            out.at[i, "gross_margin"] = (revtq - cogsq) / revtq if revtq else np.nan
            # Missing debt fields -> NaN leverage (not 0): nansum of all-NaN is 0,
            # which would falsely read as a debt-free firm.
            if np.isfinite(dlttq) or np.isfinite(dlcq):
                debt = float(np.nansum([dlttq, dlcq]))
                out.at[i, "leverage"] = debt / atq if atq else np.nan
            out.at[i, "rd_intensity"] = xrdq / sales if sales else np.nan
            out.at[i, "n_employees"] = _safe_log(emp)

            if len(grp) >= 5:
                prev = grp.iloc[-5]

                def pv(name):
                    c = col(name)
                    return float(prev[c]) if (c is not None and pd.notna(prev[c])) else np.nan

                atq_p, sales_p = pv("atq"), (pv("saleq") if col("saleq") else pv("revtq"))
                out.at[i, "asset_growth"] = atq / atq_p - 1.0 if atq_p else np.nan
                out.at[i, "sales_growth"] = sales / sales_p - 1.0 if sales_p else np.nan

                # 5.5 balance-sheet accruals (Sloan 1996), year-over-year change in
                # non-cash working capital net of depreciation, scaled by avg assets.
                # Requires current assets + current liabilities (and their lags);
                # cash / current-debt / taxes-payable / depreciation default to 0
                # when a field is absent (standard practice), but the two core
                # working-capital legs must be present or accruals stays NaN.
                actq, lctq = v("actq"), v("lctq")
                actq_p, lctq_p = pv("actq"), pv("lctq")
                if np.isfinite(actq) and np.isfinite(lctq) and \
                        np.isfinite(actq_p) and np.isfinite(lctq_p) and atq and atq_p:
                    d_ca = actq - actq_p
                    d_cash = np.nansum([v("cheq"), -pv("cheq")])
                    d_cl = lctq - lctq_p
                    d_std = np.nansum([v("dlcq"), -pv("dlcq")])
                    d_tp = np.nansum([v("txpq"), -pv("txpq")])
                    dep_ttm = pd.to_numeric(grp[col("dpq")], errors="coerce").tail(4).sum() \
                        if col("dpq") else 0.0
                    delta_wc = (d_ca - d_cash) - (d_cl - d_std - d_tp)
                    avg_at = (atq + atq_p) / 2.0
                    out.at[i, "accruals"] = (delta_wc - dep_ttm) / avg_at if avg_at else np.nan

            nic = col("niq")
            if nic is not None:
                hist = pd.to_numeric(grp[nic], errors="coerce").dropna().tail(8)
                if len(hist) >= 2:
                    out.at[i, "earn_vol"] = float(hist.std())

            if crsp_groups is not None and np.isfinite(ceqq):
                permno = t2p.get(str(ev.at[i, "oftic"]).upper())
                cg = crsp_groups.get(permno)
                if cg is not None:
                    cg = cg[cg["date"] <= ev.at[i, "anndats"]]
                    if not cg.empty:
                        last = cg.iloc[-1]
                        mc = abs(float(last["prc"])) * float(last["shrout"])
                        out.at[i, "book_to_market"] = ceqq / mc if mc > 0 else np.nan
            # accruals: left NaN unless explicit balance-sheet fields are present.
    except Exception:
        return out
    return out

# ---------------------------------------------------------------------------
# 5.7 Context / calendar
# ---------------------------------------------------------------------------

def calendar_features(ev: pd.DataFrame, rets: pd.DataFrame,
                      calendar: pd.DatetimeIndex, bench_ret: pd.Series,
                      cfg: DriftMLConfig) -> pd.DataFrame:
    """5.7 Context / calendar: fiscal_q, report_time, mkt_ret_pre, vix_level."""
    out = pd.DataFrame(index=ev.index)
    out["fiscal_q"] = ev["anndats"].dt.quarter

    hour = _hour_series(ev["anntims"])
    out["report_time"] = np.where((hour >= 16) | (hour.isna()), "AMC", "BMO")

    bench_np = bench_ret.reindex(calendar).to_numpy(dtype=float)
    n = len(calendar)
    pre = []
    for i in ev.index:
        p0 = int(ev.at[i, "pos0"])
        pre.append(_cumret(bench_np, p0 - 21, p0 - 1) if 0 <= p0 < n else np.nan)
    out["mkt_ret_pre"] = pre

    out["vix_level"] = np.nan
    return out

# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build_features(ev: pd.DataFrame, rets: pd.DataFrame,
                   prices_wide: pd.DataFrame, calendar: pd.DatetimeIndex,
                   bench_ret: pd.Series, cfg: DriftMLConfig,
                   wrds_panels: Optional[dict] = None) -> pd.DataFrame:
    """Assemble all feature families into one event-indexed matrix.

    Returns a DataFrame aligned to ``ev.index`` whose columns are the union of
    every family's features. Categorical columns are dtype ``category``.
    """
    frames = [
        earnings_features(ev, cfg),
        price_risk_features(ev, rets, prices_wide, calendar, bench_ret, cfg),
        liquidity_features(ev, prices_wide, calendar, cfg, wrds_panels),
        sector_features(ev, rets, calendar, cfg, wrds_panels),
        fundamental_features(ev, cfg, wrds_panels),
        calendar_features(ev, rets, calendar, bench_ret, cfg),
    ]
    feats = pd.concat([f.reindex(ev.index) for f in frames], axis=1)

    for col in CATEGORICAL_FEATURES:
        if col in feats.columns:
            feats[col] = feats[col].astype("category")
    return feats

# ---------------------------------------------------------------------------
# WRDS identifier / panel helpers
# ---------------------------------------------------------------------------

def _link_maps(link: pd.DataFrame) -> tuple[dict, dict]:
    cm = _colmap(link)
    tcol = cm.get("ticker") or cm.get("oftic")
    pcol, gcol = cm.get("permno"), cm.get("gvkey")
    t2p, t2g = {}, {}
    if tcol is None:
        return t2p, t2g
    for _, r in link.iterrows():
        t = str(r[tcol]).upper()
        if pcol is not None and pd.notna(r[pcol]):
            t2p[t] = r[pcol]
        if gcol is not None and pd.notna(r[gcol]):
            t2g[t] = r[gcol]
    return t2p, t2g

def _company_maps(company: pd.DataFrame) -> tuple[dict, dict]:
    cm = _colmap(company)
    gcol, sec, sic = cm.get("gvkey"), cm.get("gsector"), cm.get("sic")
    g2sec, g2sic = {}, {}
    if gcol is None:
        return g2sec, g2sic
    for _, r in company.iterrows():
        g = r[gcol]
        if sec is not None:
            g2sec[g] = r[sec]
        if sic is not None:
            g2sic[g] = r[sic]
    return g2sec, g2sic

def _crsp_groups(crsp: pd.DataFrame) -> dict:
    cm = _colmap(crsp)
    ren = {cm[k]: k for k in ("permno", "date", "ret", "prc", "vol", "shrout") if k in cm}
    c = crsp.rename(columns=ren).copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    c = c.dropna(subset=["date"]).sort_values(["permno", "date"])
    return dict(tuple(c.groupby("permno")))

# ---------------------------------------------------------------------------
# Fama-French industry classification from SIC
# ---------------------------------------------------------------------------

def _ff12_from_sic(sic) -> object:
    try:
        s = int(float(sic))
    except (TypeError, ValueError):
        return np.nan

    def r(a, b):
        return a <= s <= b

    if r(100, 999) or r(2000, 2399) or r(2700, 2749) or r(2770, 2799) or r(3100, 3199) or r(3940, 3989):
        return "NoDur"
    if (r(2500, 2519) or r(2590, 2599) or r(3630, 3659) or r(3710, 3711) or s in (3714, 3716)
            or r(3750, 3751) or s == 3792 or r(3900, 3939) or r(3990, 3999)):
        return "Durbl"
    if r(1200, 1399) or r(2900, 2999):
        return "Enrgy"
    if r(2800, 2829) or r(2840, 2899):
        return "Chems"
    if r(3570, 3579) or r(3660, 3692) or r(3694, 3699) or r(3810, 3829) or r(7370, 7379):
        return "BusEq"
    if r(4800, 4899):
        return "Telcm"
    if r(4900, 4949):
        return "Utils"
    if r(5000, 5999) or r(7200, 7299) or r(7600, 7699):
        return "Shops"
    if r(2830, 2839) or s == 3693 or r(3840, 3859) or r(8000, 8099):
        return "Hlth"
    if r(6000, 6999):
        return "Money"
    if (r(2520, 2589) or r(2600, 2699) or r(2750, 2769) or r(3000, 3099) or r(3200, 3569)
            or r(3580, 3629) or r(3700, 3709) or r(3712, 3713) or s == 3715 or r(3717, 3749)
            or r(3752, 3791) or r(3793, 3799) or r(3830, 3839) or r(3860, 3899)):
        return "Manuf"
    return "Other"

# Representative (not exhaustive) FF48 mapping keyed by major SIC groups.
_FF48_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("Agric", ((100, 299), (700, 799), (910, 919), (2048, 2048))),
    ("Food", ((2000, 2046), (2050, 2063), (2070, 2079), (2090, 2095), (2098, 2099))),
    ("Soda", ((2064, 2068), (2086, 2087), (2096, 2097))),
    ("Beer", ((2080, 2085),)),
    ("Smoke", ((2100, 2199),)),
    ("Toys", ((920, 999), (3650, 3652), (3732, 3732), (3930, 3949))),
    ("Books", ((2700, 2749), (2770, 2771), (2780, 2799))),
    ("Hshld", ((2840, 2844), (3160, 3199), (3260, 3260), (3630, 3639), (3910, 3919))),
    ("Clths", ((2300, 2390), (3020, 3021), (3100, 3111), (3130, 3159), (3965, 3965))),
    ("Hlth", ((8000, 8099),)),
    ("MedEq", ((3693, 3693), (3840, 3851))),
    ("Drugs", ((2830, 2836),)),
    ("Chems", ((2800, 2829), (2850, 2899))),
    ("Rubbr", ((3000, 3099),)),
    ("Txtls", ((2200, 2284), (2290, 2295), (2297, 2299))),
    ("BldMt", ((2400, 2459), (2490, 2499), (3200, 3259), (3261, 3299))),
    ("Cnstr", ((1500, 1799),)),
    ("Steel", ((3300, 3369), (3390, 3399))),
    ("Mach", ((3510, 3536), (3540, 3569), (3580, 3599))),
    ("ElcEq", ((3600, 3621), (3623, 3629), (3640, 3646), (3648, 3649), (3660, 3699))),
    ("Autos", ((3710, 3711), (3713, 3716), (3792, 3792), (3790, 3791), (3799, 3799))),
    ("Aero", ((3720, 3729),)),
    ("Ships", ((3730, 3731),)),
    ("Mines", ((1000, 1119), (1400, 1499))),
    ("Coal", ((1200, 1299),)),
    ("Oil", ((1300, 1399), (2900, 2912), (2990, 2999))),
    ("Util", ((4900, 4961),)),
    ("Telcm", ((4800, 4899),)),
    ("PerSv", ((7020, 7021), (7200, 7299), (7395, 7395), (7500, 7549))),
    ("BusSv", ((7300, 7372), (7374, 7385), (7389, 7394), (7397, 7397))),
    ("Comps", ((3570, 3579), (3680, 3689), (3695, 3695), (7373, 7373))),
    ("Chips", ((3622, 3622), (3660, 3679), (3810, 3829))),
    ("LabEq", ((3811, 3827), (3829, 3829))),
    ("Boxes", ((2440, 2449), (2640, 2659), (3220, 3221), (3410, 3412))),
    ("Trans", ((4000, 4013), (4040, 4049), (4100, 4173), (4190, 4299), (4400, 4789))),
    ("Whlsl", ((5000, 5199),)),
    ("Rtail", ((5200, 5999),)),
    ("Meals", ((5800, 5899), (7000, 7019))),
    ("Banks", ((6000, 6199),)),
    ("Insur", ((6300, 6411),)),
    ("RlEst", ((6500, 6611),)),
    ("Fin", ((6200, 6299), (6700, 6799))),
)

def _ff48_from_sic(sic) -> object:
    try:
        s = int(float(sic))
    except (TypeError, ValueError):
        return np.nan
    for label, ranges in _FF48_RANGES:
        for lo, hi in ranges:
            if lo <= s <= hi:
                return label
    return "Other"
