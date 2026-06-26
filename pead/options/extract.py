"""Reduce the ~1 TB OptionMetrics parquet to a compact per-event option panel.

Strategy
--------
For each event ``(secid, ann_date)`` keep ``opprcd`` rows whose quote date falls
in ``[ann_date - pre, ann_date + post]``. DuckDB does the heavy lifting:

* **partition pruning** - we hand it only the ``opprcd{year}.parquet`` files that
  the event windows can touch, never the full dataset;
* **projection pushdown** - only the columns in ``om_schema.OPPRCD_COLUMNS``;
* **predicate pushdown** - the per-event date-window join is evaluated as the
  files stream, so the 1 TB is read once and reduced to a few GB (or less).

The result is written as a single parquet panel under ``data/derived/`` that the
native compute engine consumes via Arrow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..io import om_schema, resolver


def _years_spanned(events_df: pd.DataFrame, window_pre: int, window_post: int) -> list[int]:
    lo = (events_df["ann_date"] - pd.Timedelta(days=window_pre)).dt.year.min()
    hi = (events_df["ann_date"] + pd.Timedelta(days=window_post)).dt.year.max()
    return list(range(int(lo), int(hi) + 1))


def _existing_opprcd_files(om_dir: Path, years: list[int]) -> list[str]:
    files = []
    for y in years:
        p = om_schema.partitioned_path(om_dir, om_schema.OPPRCD_STEM, y)
        if p.exists():
            files.append(p.as_posix())
    return files


def extract_event_panel(
    events_df: pd.DataFrame,
    out_path,
    om_dir=None,
    window_pre: int = 5,
    window_post: int = 60,
) -> Path:
    """Extract the windowed option panel for ``events_df`` to ``out_path``.

    ``events_df`` must contain at least ``secid`` and ``ann_date`` (datetime).
    Returns the path to the written parquet panel.
    """
    if events_df.empty:
        raise ValueError("No events to extract (events_df is empty).")

    om_dir = Path(om_dir) if om_dir else resolver.require_optionmetrics()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    years = _years_spanned(events_df, window_pre, window_post)
    files = _existing_opprcd_files(om_dir, years)
    if not files:
        raise FileNotFoundError(
            f"No opprcd parquet partitions found in {om_dir} for years {years}."
        )

    ev = events_df[["secid", "ann_date"]].dropna().copy()
    ev["secid"] = ev["secid"].astype("int64")
    ev["ann_date"] = pd.to_datetime(ev["ann_date"]).dt.date

    cols = ", ".join(f"o.{c}" for c in om_schema.OPPRCD_COLUMNS)
    file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"

    con = duckdb.connect()
    try:
        con.register("events", ev)
        query = f"""
            SELECT
                {cols},
                o.strike_price / 1000.0 AS strike,
                e.ann_date,
                date_diff('day', e.ann_date, o.date) AS rel_day
            FROM read_parquet({file_list}) o
            JOIN events e
              ON o.secid = e.secid
             AND o.date BETWEEN (e.ann_date - INTERVAL {int(window_pre)} DAY)
                            AND (e.ann_date + INTERVAL {int(window_post)} DAY)
        """
        con.execute(
            f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        con.close()

    return out_path
