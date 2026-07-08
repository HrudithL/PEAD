"""Per-ticker incremental append to the shared ``Data Source/wrds_cache/*.csv``.

The research pipeline (:mod:`pead.sub_sampling_ml.wrds_extract`) treats the
CSV cache as *"the whole universe was pulled once"*: reading it returns the
entire file, and refreshing means re-pulling the whole thing. For live serving
we want to add just the missing ticker without invalidating anything else.

Functions here open the CSVs in place, check whether the requested ticker /
permno / gvkey is already present, and, only if not, run the same WRDS SQL as
the research module for that one identifier and append the rows.

Everything is best-effort: if WRDS is unreachable, functions return whatever
was already cached and the caller decides whether the coverage is good enough
to score (see :mod:`predict`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import DriftMLConfig
from .. import wrds_extract as wex


# ---------------------------------------------------------------- readers


def _read_full_cache(name: str, cfg: DriftMLConfig) -> Optional[pd.DataFrame]:
    """Read a cache CSV regardless of the ``refresh_wrds`` flag."""
    path = wex.cache_path(name, cfg)
    if not path.is_file():
        return None
    try:
        return wex._parse_dates(pd.read_csv(path))
    except Exception:
        return None


def _append_and_write(existing: Optional[pd.DataFrame], new_rows: pd.DataFrame,
                      name: str, cfg: DriftMLConfig,
                      dedupe_on: Optional[list[str]] = None) -> pd.DataFrame:
    if new_rows is None or new_rows.empty:
        return existing if existing is not None else pd.DataFrame()
    combined = (pd.concat([existing, new_rows], ignore_index=True)
                if existing is not None else new_rows.reset_index(drop=True))
    if dedupe_on:
        keep = [c for c in dedupe_on if c in combined.columns]
        if keep:
            combined = combined.drop_duplicates(subset=keep, keep="last")
    path = wex.cache_path(name, cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)
    except Exception:
        pass
    return combined


# ---------------------------------------------------------------- link tables


def ensure_ticker_permno(ticker: str, cfg: DriftMLConfig) -> Optional[int]:
    """Return the (most recent) PERMNO for ``ticker``, pulling & appending if new."""
    ticker = str(ticker).upper()
    link = _read_full_cache("ibes_permno_link", cfg)
    if link is not None and not link.empty:
        hit = link[link["ticker"].astype(str).str.upper() == ticker]
        if not hit.empty:
            return _pick_permno(hit)

    # Not cached -- pull just this ticker from WRDS.
    try:
        conn = wex.get_connection(cfg)
        df = conn.raw_sql(
            "select ticker, permno, sdate, edate "
            f"from wrdsapps.ibcrsphist where upper(ticker) = '{ticker}'"
        )
        df = wex._parse_dates(df)
    except Exception as exc:
        print(f"[drift-serve] WRDS ibes->permno pull for {ticker} failed: {exc}")
        return None

    _append_and_write(link, df, "ibes_permno_link", cfg,
                      dedupe_on=["ticker", "permno", "sdate"])
    if df.empty:
        return None
    return _pick_permno(df)


def ensure_permno_gvkey(permno: int, cfg: DriftMLConfig) -> Optional[str]:
    permno = int(permno)
    link = _read_full_cache("permno_gvkey_link", cfg)
    if link is not None and not link.empty and "permno" in link.columns:
        hit = link[pd.to_numeric(link["permno"], errors="coerce") == permno]
        if not hit.empty:
            return _pick_gvkey(hit)

    try:
        conn = wex.get_connection(cfg)
        df = conn.raw_sql(
            "select lpermno as permno, gvkey, linkdt, linkenddt "
            "from crsp.ccmxpf_lnkhist "
            "where linktype in ('LU', 'LC') and linkprim in ('P', 'C') "
            f"and lpermno = {permno}"
        )
        df = wex._parse_dates(df)
    except Exception as exc:
        print(f"[drift-serve] WRDS permno->gvkey pull for {permno} failed: {exc}")
        return None
    _append_and_write(link, df, "permno_gvkey_link", cfg,
                      dedupe_on=["permno", "gvkey", "linkdt"])
    if df.empty:
        return None
    return _pick_gvkey(df)


def _pick_permno(df: pd.DataFrame) -> Optional[int]:
    order = df.sort_values("edate", ascending=False) if "edate" in df.columns else df
    try:
        return int(order["permno"].iloc[0])
    except Exception:
        return None


def _pick_gvkey(df: pd.DataFrame) -> Optional[str]:
    order = df.sort_values("linkenddt", ascending=False) if "linkenddt" in df.columns else df
    try:
        return str(order["gvkey"].iloc[0])
    except Exception:
        return None


# ---------------------------------------------------------------- CRSP daily


def ensure_crsp_daily(permno: int, cfg: DriftMLConfig) -> pd.DataFrame:
    """Return CRSP daily rows for ``permno``, pulling/appending if missing or stale.

    A cache hit only short-circuits once it actually covers through the
    prediction/config date range -- otherwise a cache built through 2024
    would be reused as-is (stale liquidity/book-to-market inputs) to score a
    2025 announcement. When stale, pull just the missing tail and append.
    """
    permno = int(permno)
    cache = _read_full_cache("crsp_daily", cfg)
    hit = pd.DataFrame()
    if cache is not None and "permno" in cache.columns:
        hit = cache[pd.to_numeric(cache["permno"], errors="coerce") == permno]

    target_end = max(pd.Timestamp.today().normalize(),
                     pd.Timestamp(f"{cfg.end_year}-12-31"))
    if not hit.empty and "date" in hit.columns:
        cached_through = pd.to_datetime(hit["date"]).max()
        # CRSP typically lags a day or two; a small buffer avoids re-pulling
        # on every call just because today's print hasn't posted yet.
        if cached_through >= target_end - pd.Timedelta(days=5):
            return hit.reset_index(drop=True)
        pull_start = (cached_through + pd.Timedelta(days=1)).date().isoformat()
    else:
        pull_start = f"{cfg.start_year - 2}-01-01"

    try:
        conn = wex.get_connection(cfg)
        df = conn.raw_sql(
            "select permno, date, ret, abs(prc) as prc, vol, shrout "
            f"from crsp.dsf where permno = {permno} "
            f"and date between '{pull_start}' and '{target_end.date().isoformat()}'"
        )
        df = wex._parse_dates(df)
    except Exception as exc:
        print(f"[drift-serve] WRDS crsp.dsf pull for permno={permno} failed: {exc}")
        if not hit.empty:
            return hit.reset_index(drop=True)
        return pd.DataFrame(columns=["permno", "date", "ret", "prc", "vol", "shrout"])

    combined = _append_and_write(cache, df, "crsp_daily", cfg,
                                 dedupe_on=["permno", "date"])
    if "permno" in combined.columns:
        out = combined[pd.to_numeric(combined["permno"], errors="coerce") == permno]
    else:
        out = df
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- Compustat


def ensure_fundq(gvkey: str, cfg: DriftMLConfig) -> pd.DataFrame:
    gvkey = str(gvkey)
    cache = _read_full_cache("compustat_fundq", cfg)
    if cache is not None and "gvkey" in cache.columns:
        hit = cache[cache["gvkey"].astype(str) == gvkey]
        if not hit.empty:
            return hit.reset_index(drop=True)

    base_cols = ("gvkey, datadate, rdq, fqtr, fyearq, atq, ceqq, niq, "
                 "revtq, cogsq, saleq, dlttq, dlcq, xrdq, actq, lctq, "
                 "cheq, txpq, dpq")
    where = (f"where gvkey = '{gvkey}' and consol = 'C' and indfmt = 'INDL' "
             "and datafmt = 'STD' and popsrc = 'D'")
    try:
        conn = wex.get_connection(cfg)
        try:
            df = conn.raw_sql(f"select {base_cols}, emp from comp.fundq {where}")
        except Exception:
            df = conn.raw_sql(f"select {base_cols} from comp.fundq {where}")
        df = wex._parse_dates(df)
    except Exception as exc:
        print(f"[drift-serve] WRDS comp.fundq pull for gvkey={gvkey} failed: {exc}")
        return pd.DataFrame()
    _append_and_write(cache, df, "compustat_fundq", cfg,
                      dedupe_on=["gvkey", "datadate"])
    return df.reset_index(drop=True)


def ensure_company(gvkey: str, cfg: DriftMLConfig) -> pd.DataFrame:
    gvkey = str(gvkey)
    cache = _read_full_cache("compustat_company", cfg)
    if cache is not None and "gvkey" in cache.columns:
        hit = cache[cache["gvkey"].astype(str) == gvkey]
        if not hit.empty:
            return hit.reset_index(drop=True)
    try:
        conn = wex.get_connection(cfg)
        df = conn.raw_sql(
            f"select gvkey, gsector, sic from comp.company where gvkey = '{gvkey}'")
        df = wex._parse_dates(df)
    except Exception as exc:
        print(f"[drift-serve] WRDS comp.company pull for gvkey={gvkey} failed: {exc}")
        return pd.DataFrame()
    _append_and_write(cache, df, "compustat_company", cfg,
                      dedupe_on=["gvkey"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- combined


def ensure_wrds_for_ticker(ticker: str, cfg: DriftMLConfig) -> dict:
    """One call: make sure every WRDS panel covers ``ticker`` and return them.

    The returned dict matches the shape of :func:`wrds_extract.build_wrds_panels`
    so it drops straight into :func:`features.build_features`.
    """
    permno = ensure_ticker_permno(ticker, cfg)
    if permno is None:
        return {}
    gvkey = ensure_permno_gvkey(permno, cfg)

    crsp = ensure_crsp_daily(permno, cfg)
    fundq = ensure_fundq(gvkey, cfg) if gvkey else pd.DataFrame()
    company = ensure_company(gvkey, cfg) if gvkey else pd.DataFrame()

    # Also read back the full link tables so features._link_maps sees them.
    link_tp = _read_full_cache("ibes_permno_link", cfg)
    link_pg = _read_full_cache("permno_gvkey_link", cfg)
    # Match wrds_extract.build_wrds_panels's "link" merged shape: (ticker,
    # permno, gvkey, linkdt, linkenddt).
    link = _merge_links(link_tp, link_pg)

    return {"link": link, "crsp_daily": crsp,
            "fundq": fundq, "company": company}


def _merge_links(tp: Optional[pd.DataFrame],
                 pg: Optional[pd.DataFrame]) -> pd.DataFrame:
    if tp is None or tp.empty:
        return pd.DataFrame(columns=["ticker", "permno", "gvkey"])
    tp = tp.copy()
    if "permno" in tp.columns:
        tp["permno"] = pd.to_numeric(tp["permno"], errors="coerce")
    if pg is not None and not pg.empty and "permno" in pg.columns:
        pg = pg.copy()
        pg["permno"] = pd.to_numeric(pg["permno"], errors="coerce")
        return tp.merge(pg, on="permno", how="left")
    tp["gvkey"] = pd.NA
    return tp
