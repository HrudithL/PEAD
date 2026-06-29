"""CRSP / Compustat / IBES extraction via the ``wrds`` package, with CSV caching.

Strategy (Section 6): pull each WRDS table once through a live ``wrds.Connection``
and cache the result as CSV under :func:`pead.io.resolver.wrds_cache_dir` (the
external Data Source dir, never committed). Subsequent runs read the CSV instead
of re-querying the API. Pass ``cfg.refresh_wrds=True`` to force a re-pull.

Identifier linking is the main data-engineering task because IBES is keyed by
ticker while CRSP/Compustat are keyed by PERMNO/GVKEY:

1. IBES ticker -> CRSP PERMNO via ``wrdsapps.ibcrsphist``.
2. PERMNO -> Compustat GVKEY via ``crsp.ccmxpf_lnkhist`` (date-valid CCM link).
3. SIC / GICS carried from ``comp.company``.

Heavy / optional imports (``wrds``) are done lazily inside functions so importing
this module never requires the package or a live connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DriftMLConfig

# Columns that should be coerced to datetime whenever they appear in a frame.
_DATE_COLUMNS = (
    "date", "rdq", "datadate", "sdate", "edate", "linkdt", "linkenddt",
)

# Sentinel used for open-ended link / validity windows so as-of merges work.
_FAR_FUTURE = pd.Timestamp("2099-12-31")

# Chunk size for the CRSP permno IN-list so the generated SQL stays reasonable.
_PERMNO_CHUNK = 1000

def cache_path(name: str, cfg: DriftMLConfig) -> Path:
    """Absolute path of a cached WRDS CSV (``name`` without extension)."""
    return Path(cfg.wrds_cache_dir) / f"{name}.csv"

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known date-like columns to datetime; open-ended -> far future."""
    for col in df.columns:
        if col.lower() in _DATE_COLUMNS:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if col.lower() in ("edate", "linkenddt"):
                parsed = parsed.fillna(_FAR_FUTURE)
            df[col] = parsed
    return df

def _read_cache(name: str, cfg: DriftMLConfig) -> Optional[pd.DataFrame]:
    """Return the cached CSV for ``name`` if present and not being refreshed."""
    if cfg.refresh_wrds:
        return None
    path = cache_path(name, cfg)
    try:
        if not path.is_file():
            return None
        return _parse_dates(pd.read_csv(path))
    except Exception:
        return None

def _write_cache(df: pd.DataFrame, name: str, cfg: DriftMLConfig) -> None:
    """Persist ``df`` to the WRDS CSV cache (creates the dir if needed)."""
    try:
        path = cache_path(name, cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        pass

def get_connection(cfg: DriftMLConfig):
    """Open a live ``wrds.Connection`` using ``cfg.wrds_username`` / env creds."""
    import wrds  # lazy: importing this module must not require the package

    return wrds.Connection(wrds_username=cfg.wrds_username)

def link_ibes_to_permno(tickers: list[str], cfg: DriftMLConfig) -> pd.DataFrame:
    """IBES ticker -> PERMNO with validity dates (``wrdsapps.ibcrsphist``)."""
    cols = ["ticker", "permno", "sdate", "edate"]
    cached = _read_cache("ibes_permno_link", cfg)
    if cached is None:
        conn = get_connection(cfg)
        df = conn.raw_sql(
            "select ticker, permno, sdate, edate from wrdsapps.ibcrsphist"
        )
        df = _parse_dates(df)
        _write_cache(df, "ibes_permno_link", cfg)
    else:
        df = cached

    wanted = {str(t).upper() for t in (tickers or [])}
    if wanted:
        df = df[df["ticker"].astype(str).str.upper().isin(wanted)]
    return df[cols].reset_index(drop=True)

def link_permno_to_gvkey(permnos: list[int], cfg: DriftMLConfig) -> pd.DataFrame:
    """PERMNO -> GVKEY date-valid link (``crsp.ccmxpf_lnkhist``)."""
    cols = ["permno", "gvkey", "linkdt", "linkenddt"]
    cached = _read_cache("permno_gvkey_link", cfg)
    if cached is None:
        conn = get_connection(cfg)
        df = conn.raw_sql(
            "select lpermno as permno, gvkey, linkdt, linkenddt "
            "from crsp.ccmxpf_lnkhist "
            "where linktype in ('LU', 'LC') and linkprim in ('P', 'C')"
        )
        df = _parse_dates(df)
        _write_cache(df, "permno_gvkey_link", cfg)
    else:
        df = cached

    wanted = {int(p) for p in (permnos or [])}
    if wanted and "permno" in df.columns:
        df = df[pd.to_numeric(df["permno"], errors="coerce").isin(wanted)]
    return df[cols].reset_index(drop=True)

def extract_crsp_daily(permnos: list[int], cfg: DriftMLConfig) -> pd.DataFrame:
    """Daily stock file (``crsp.dsf``): ret, prc, vol, shrout for the universe."""
    cols = ["permno", "date", "ret", "prc", "vol", "shrout"]
    cached = _read_cache("crsp_daily", cfg)
    if cached is not None:
        return cached[[c for c in cols if c in cached.columns]].reset_index(drop=True)

    uniq = sorted({int(p) for p in (permnos or [])})
    if not uniq:
        return pd.DataFrame(columns=cols)

    start = f"{cfg.start_year - 2}-01-01"   # 2y lookback for momentum / beta
    end = f"{cfg.end_year}-12-31"
    conn = get_connection(cfg)
    frames = []
    for i in range(0, len(uniq), _PERMNO_CHUNK):
        chunk = uniq[i:i + _PERMNO_CHUNK]
        in_list = ", ".join(str(p) for p in chunk)
        sql = (
            "select permno, date, ret, abs(prc) as prc, vol, shrout "
            "from crsp.dsf "
            f"where permno in ({in_list}) "
            f"and date between '{start}' and '{end}'"
        )
        frames.append(conn.raw_sql(sql))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    df = _parse_dates(df)
    _write_cache(df, "crsp_daily", cfg)
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

def extract_compustat_fundq(gvkeys: list[str], cfg: DriftMLConfig) -> pd.DataFrame:
    """Quarterly fundamentals (``comp.fundq``) with ``rdq`` for point-in-time use."""
    base_cols = [
        "gvkey", "datadate", "rdq", "fqtr", "fyearq", "atq", "ceqq", "niq",
        "revtq", "cogsq", "saleq", "dlttq", "dlcq", "xrdq",
    ]
    cached = _read_cache("compustat_fundq", cfg)
    if cached is not None:
        return cached.reset_index(drop=True)

    uniq = sorted({str(g) for g in (gvkeys or [])})
    if not uniq:
        return pd.DataFrame(columns=base_cols)

    in_list = ", ".join(f"'{g}'" for g in uniq)
    conn = get_connection(cfg)
    select = ", ".join(base_cols)
    where = (
        f"where gvkey in ({in_list}) "
        "and consol = 'C' and indfmt = 'INDL' "
        "and datafmt = 'STD' and popsrc = 'D'"
    )
    try:
        df = conn.raw_sql(f"select {select}, emp from comp.fundq {where}")
    except Exception:
        # ``emp`` is not guaranteed on fundq across vintages; fall back without it.
        df = conn.raw_sql(f"select {select} from comp.fundq {where}")

    df = _parse_dates(df)
    _write_cache(df, "compustat_fundq", cfg)
    return df.reset_index(drop=True)

def extract_company(gvkeys: list[str], cfg: DriftMLConfig) -> pd.DataFrame:
    """Static firm attributes (``comp.company``): gsector (GICS), sic."""
    cols = ["gvkey", "gsector", "sic"]
    cached = _read_cache("compustat_company", cfg)
    if cached is not None:
        return cached[[c for c in cols if c in cached.columns]].reset_index(drop=True)

    uniq = sorted({str(g) for g in (gvkeys or [])})
    if not uniq:
        return pd.DataFrame(columns=cols)

    in_list = ", ".join(f"'{g}'" for g in uniq)
    conn = get_connection(cfg)
    df = conn.raw_sql(
        f"select gvkey, gsector, sic from comp.company where gvkey in ({in_list})"
    )
    _write_cache(df, "compustat_company", cfg)
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

def build_wrds_panels(ev: pd.DataFrame, cfg: DriftMLConfig) -> dict:
    """Assemble every WRDS panel needed by :mod:`features`, keyed for joining.

    Returns a dict with keys ``link`` (ticker/permno/gvkey), ``crsp_daily``,
    ``fundq``, and ``company``. On any failure (no creds, package missing,
    ``cfg.use_wrds=False``) returns an empty dict so the pipeline degrades to
    repo-only features rather than crashing.
    """
    if not cfg.use_wrds:
        return {}

    try:
        if ev is None or "oftic" not in ev.columns:
            tickers: list[str] = []
        else:
            tickers = sorted(
                {
                    str(t).upper()
                    for t in ev["oftic"].dropna().tolist()
                    if str(t).strip()
                }
            )

        ibes_link = link_ibes_to_permno(tickers, cfg)
        permnos = sorted(
            {int(p) for p in pd.to_numeric(ibes_link["permno"], errors="coerce").dropna()}
        )

        gvkey_link = link_permno_to_gvkey(permnos, cfg)
        link = ibes_link.merge(gvkey_link, on="permno", how="left")
        gvkeys = sorted({str(g) for g in gvkey_link["gvkey"].dropna().tolist()})

        crsp_daily = extract_crsp_daily(permnos, cfg)
        fundq = extract_compustat_fundq(gvkeys, cfg)
        company = extract_company(gvkeys, cfg)

        return {
            "link": link,
            "crsp_daily": crsp_daily,
            "fundq": fundq,
            "company": company,
        }
    except Exception as exc:  # no creds, package missing, network, bad SQL...
        print(
            f"[drift-ml] WRDS unavailable: {exc} -> falling back to repo-only features"
        )
        return {}
