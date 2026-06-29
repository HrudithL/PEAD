"""The PDF deliverable (Section 10), matching the repo's reportlab convention.

Assembles fit metrics, the SHAP summary & dependence plots, the Fama-MacBeth
coefficient table, the top/bottom descriptive comparison, and the per-fold
consistency table into ``outputs/drift_ml_report_*.pdf``. Also writes the ranked
driver map to ``outputs/feature_importance.csv``.

Matplotlib figure builders are kept here (like ``pead.equities.plots``) and
embedded as PNGs; the document itself is built with reportlab.
"""

from __future__ import annotations

import pandas as pd

from .config import DriftMLConfig


def fig_oos_metrics(results: dict, cfg: DriftMLConfig):
    """Bar/line summary of out-of-sample IC, R^2, decile spread, AUC per horizon."""
    raise NotImplementedError


def fig_shap_summary(results: dict, cfg: DriftMLConfig):
    """SHAP global importance (signed) for the primary-horizon regressor."""
    raise NotImplementedError


def fig_consistency(results: dict, cfg: DriftMLConfig):
    """Per-fold rank-stability heatmap / dispersion of the top drivers."""
    raise NotImplementedError


def fig_descriptive(results: dict, cfg: DriftMLConfig):
    """Top-decile vs bottom-decile feature distributions with effect sizes."""
    raise NotImplementedError


def write_feature_importance_csv(table: pd.DataFrame, cfg: DriftMLConfig) -> str:
    """Persist the ranked driver map to ``outputs/feature_importance.csv``."""
    raise NotImplementedError


def build_pdf(cfg: DriftMLConfig, results: dict) -> str:
    """Render the full PDF report and return its path.

    ``results`` is the bundle produced by the pipeline (model folds, attribution
    tables, Fama-MacBeth coefficients, descriptive comparison, metrics).
    """
    raise NotImplementedError
