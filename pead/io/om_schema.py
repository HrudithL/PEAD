"""Schema knowledge for the OptionMetrics parquet tables PEAD consumes.

Only the columns PEAD actually needs are listed; DuckDB uses these for
projection pushdown so scans never read more than required. Year-partitioned
tables follow the ``f"{stem}{year}.parquet"`` naming on disk.
"""

from __future__ import annotations

from pathlib import Path

# --- ticker <-> secid dimension (small, unpartitioned) --------------------- #
SECNMD = "secnmd.parquet"
SECNMD_COLUMNS = ["secid", "ticker", "cusip", "issuer"]

# --- year-partitioned fact tables ----------------------------------------- #
OPPRCD_STEM = "opprcd"   # option prices + greeks (core fact table)
SECPRD_STEM = "secprd"   # underlying security daily prices

# Columns PEAD reads from opprcd{year}. NOTE: strike_price is in 1/1000 dollars
# in OptionMetrics and must be divided by 1000 downstream.
OPPRCD_COLUMNS = [
    "secid", "date", "exdate", "cp_flag", "strike_price",
    "best_bid", "best_offer", "volume", "open_interest",
    "impl_volatility", "delta", "gamma", "vega", "theta",
]
SECPRD_COLUMNS = ["secid", "date", "close", "open", "low", "high", "volume", "return"]

# --- unpartitioned support tables ----------------------------------------- #
ZEROCD = "zerocd.parquet"          # zero-coupon yield curve (rates for Black-Scholes)
ZEROCD_COLUMNS = ["date", "days", "rate"]
DISTRD = "distrd.parquet"          # distributions / dividends


def partitioned_path(om_dir: Path, stem: str, year: int) -> Path:
    return Path(om_dir) / f"{stem}{year}.parquet"


def opprcd_paths(om_dir: Path, start_year: int, end_year: int) -> list[Path]:
    return [partitioned_path(om_dir, OPPRCD_STEM, y)
            for y in range(start_year, end_year + 1)]


def secprd_paths(om_dir: Path, start_year: int, end_year: int) -> list[Path]:
    return [partitioned_path(om_dir, SECPRD_STEM, y)
            for y in range(start_year, end_year + 1)]
