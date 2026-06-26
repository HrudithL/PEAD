"""Derived event-panel schema and parquet IO.

The panel is the contract between the DuckDB reduce stage and the native compute
engine. Keeping the column set explicit here means both sides agree on layout
and the Arrow buffers can be handed to C++/CUDA zero-copy.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Columns produced by extract.extract_event_panel (order is the on-disk order).
PANEL_COLUMNS = [
    "secid", "date", "exdate", "cp_flag", "strike_price",
    "best_bid", "best_offer", "volume", "open_interest",
    "impl_volatility", "delta", "gamma", "vega", "theta",
    "strike", "ann_date", "rel_day",
]


def read_panel(path) -> pa.Table:
    """Load a derived panel as an Arrow table (zero-copy friendly)."""
    return pq.read_table(Path(path))


def read_panel_df(path):
    """Load a derived panel as a pandas DataFrame (convenience)."""
    return read_panel(path).to_pandas()


def write_panel(table_or_df, path) -> Path:
    """Write an Arrow table or pandas DataFrame as a ZSTD parquet panel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = table_or_df if isinstance(table_or_df, pa.Table) else pa.Table.from_pandas(table_or_df)
    pq.write_table(table, path, compression="zstd")
    return path
