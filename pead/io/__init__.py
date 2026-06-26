"""Shared data-access layer: locate and describe out-of-repo datasets."""

from .resolver import (
    REFERENCE_DIR,
    DERIVED_DIR,
    data_source_dir,
    ibes_path,
    master_stock_path,
    constituent_csv,
    optionmetrics_dir,
    require_optionmetrics,
)
from .manifest import Manifest

__all__ = [
    "REFERENCE_DIR",
    "DERIVED_DIR",
    "data_source_dir",
    "ibes_path",
    "master_stock_path",
    "constituent_csv",
    "optionmetrics_dir",
    "require_optionmetrics",
    "Manifest",
]
