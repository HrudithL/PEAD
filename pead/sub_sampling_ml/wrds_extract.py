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


def cache_path(name: str, cfg: DriftMLConfig) -> Path:
    """Absolute path of a cached WRDS CSV (``name`` without extension)."""
    raise NotImplementedError


def _read_cache(name: str, cfg: DriftMLConfig) -> Optional[pd.DataFrame]:
    """Return the cached CSV for ``name`` if present and not being refreshed."""
    raise NotImplementedError


def _write_cache(df: pd.DataFrame, name: str, cfg: DriftMLConfig) -> None:
    """Persist ``df`` to the WRDS CSV cache (creates the dir if needed)."""
    raise NotImplementedError


def get_connection(cfg: DriftMLConfig):
    """Open a live ``wrds.Connection`` using ``cfg.wrds_username`` / env creds."""
    raise NotImplementedError


def link_ibes_to_permno(tickers: list[str], cfg: DriftMLConfig) -> pd.DataFrame:
    """IBES ticker -> PERMNO with validity dates (``wrdsapps.ibcrsphist``)."""
    raise NotImplementedError


def link_permno_to_gvkey(permnos: list[int], cfg: DriftMLConfig) -> pd.DataFrame:
    """PERMNO -> GVKEY date-valid link (``crsp.ccmxpf_lnkhist``)."""
    raise NotImplementedError


def extract_crsp_daily(permnos: list[int], cfg: DriftMLConfig) -> pd.DataFrame:
    """Daily stock file (``crsp.dsf``): ret, prc, vol, shrout for the universe."""
    raise NotImplementedError


def extract_compustat_fundq(gvkeys: list[str], cfg: DriftMLConfig) -> pd.DataFrame:
    """Quarterly fundamentals (``comp.fundq``) with ``rdq`` for point-in-time use."""
    raise NotImplementedError


def extract_company(gvkeys: list[str], cfg: DriftMLConfig) -> pd.DataFrame:
    """Static firm attributes (``comp.company``): gsector (GICS), sic."""
    raise NotImplementedError


def build_wrds_panels(ev: pd.DataFrame, cfg: DriftMLConfig) -> dict:
    """Assemble every WRDS panel needed by :mod:`features`, keyed for joining.

    Returns a dict with keys ``link`` (ticker/permno/gvkey), ``crsp_daily``,
    ``fundq``, and ``company``. On any failure (no creds, package missing,
    ``cfg.use_wrds=False``) returns an empty dict so the pipeline degrades to
    repo-only features rather than crashing.
    """
    raise NotImplementedError
