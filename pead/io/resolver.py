"""Locate data that lives OUTSIDE the repository.

Equities CSVs (``master_stock``, ``IBES_Summary_*``) and the ~1 TB OptionMetrics
parquet are never committed. This module resolves their on-disk locations from
environment variables first, then a list of known candidate paths, so the same
code runs on the Windows workstation, the Linux GPU box, and CI.

Small, stable constituent lists ARE committed under ``data/reference/`` and are
preferred over the external Data Source copy.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root: pead/io/resolver.py -> pead/io -> pead -> <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Committed, in-repo reference data (constituent lists).
REFERENCE_DIR = _REPO_ROOT / "data" / "reference"
# Local cache for extracted event panels (gitignored).
DERIVED_DIR = _REPO_ROOT / "data" / "derived"


# --------------------------------------------------------------------------- #
# Equities data source (master_stock, IBES)
# --------------------------------------------------------------------------- #
_DATA_SOURCE_CANDIDATES = [
    _REPO_ROOT.parent / "Data Source",   # BEI/Data Source, sibling of the repo
    _REPO_ROOT / "data" / "source",
]


def data_source_dir() -> Path:
    """Directory holding the big equities CSVs (env ``BEI_DATA_DIR`` wins)."""
    env = os.environ.get("BEI_DATA_DIR")
    if env:
        return Path(env)
    for cand in _DATA_SOURCE_CANDIDATES:
        if cand.is_dir():
            return cand
    return _DATA_SOURCE_CANDIDATES[0]  # best guess; caller checks existence


def ibes_path(filename: str = "IBES_Summary_2015_2024.csv") -> Path:
    return data_source_dir() / filename


def master_stock_path(filename: str = "master_stock.csv") -> Path:
    return data_source_dir() / filename


def constituent_csv(filename: str) -> Path:
    """Prefer the committed in-repo reference copy; fall back to Data Source."""
    ref = REFERENCE_DIR / filename
    if ref.is_file():
        return ref
    return data_source_dir() / filename


def wrds_cache_dir() -> Path:
    """Directory for cached WRDS pulls (CRSP/Compustat CSVs).

    Lives under the external Data Source dir (``BEI_DATA_DIR``/``WRDS_CACHE_DIR``)
    so heavy WRDS extracts are pulled once and re-read locally instead of being
    re-queried from the API on every run. Never committed (outside the repo).
    """
    env = os.environ.get("WRDS_CACHE_DIR")
    base = Path(env) if env else (data_source_dir() / "wrds_cache")
    return base


# --------------------------------------------------------------------------- #
# OptionMetrics parquet (year-partitioned)
# --------------------------------------------------------------------------- #
_OPTIONMETRICS_CANDIDATES = [
    r"D:\OptionMetrics\parquet",                            # Windows workstation
    "/Volumes/Rishabh(OM SAS)/OptionMetrics/parquet",      # macOS
    "/Volumes/copyrishabh/OptionMetrics/parquet",
    "/media/swirl/New Volume/OptionMetrics/parquet",        # Linux GPU box
    "/media/swirl/Rishabh(OM SAS)/OptionMetrics/parquet",
    "/home/swirl/Option Metrics(CSV+Parquet+SAS)/parquet",
]

# Always-present small table used to confirm a directory really is the OM root.
_OM_SENTINEL = "secnmd.parquet"


def optionmetrics_dir() -> Path:
    """OptionMetrics parquet root (env ``OPTIONMETRICS_DIR`` wins)."""
    env = os.environ.get("OPTIONMETRICS_DIR")
    if env:
        return Path(env)
    for cand in _OPTIONMETRICS_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / _OM_SENTINEL).exists():
            return p
    return Path(_OPTIONMETRICS_CANDIDATES[0])  # best guess; caller checks


def require_optionmetrics() -> Path:
    """Return the OM parquet root or raise a clear error if it is not mounted."""
    p = optionmetrics_dir()
    if not (p.is_dir() and (p / _OM_SENTINEL).exists()):
        raise FileNotFoundError(
            f"OptionMetrics parquet not found at '{p}'. "
            "Set OPTIONMETRICS_DIR or mount the data drive (see .env.example)."
        )
    return p
