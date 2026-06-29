"""Models + leakage-aware validation (Sections 7 & 8).

Two complementary models on the same event-feature-label table:

A. **Fama-MacBeth** -- per-calendar-quarter cross-sectional OLS of ``drift_z`` on
   standardized features; coefficients averaged across quarters with Newey-West
   t-stats. The per-quarter coefficient series *is* the consistency evidence for
   linear effects.

B. **LightGBM** -- gradient-boosted regressor on ``drift_z`` (primary) and an
   optional classifier on ``drift_class``, with native categorical handling and
   SHAP-ready models returned per fold.

Both are evaluated under **purged, embargoed walk-forward CV** (Lopez de Prado):
train on past quarters, test on the next; purge any training event whose
[+1, +H] window overlaps the test period; embargo ``cfg.embargo_months`` after
each test block. Cross-sectional standardization and any time-dependent encoding
are fit on the training window only.

Heavy imports (lightgbm, statsmodels, sklearn, scipy) are lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import DriftMLConfig


@dataclass
class FoldResult:
    """Per-fold predictions, importances, and metrics for one walk-forward split."""

    fold: int
    test_index: np.ndarray = field(default_factory=lambda: np.array([]))
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: dict = field(default_factory=dict)
    shap_importance: Optional[pd.Series] = None
    perm_importance: Optional[pd.Series] = None


def purged_walk_forward_splits(quarters: pd.Series, horizon: int,
                               cfg: DriftMLConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) per expanding walk-forward step.

    ``quarters`` is the per-event calendar quarter. Each step tests one quarter
    using all prior quarters for training, purges training events whose label
    window overlaps the test quarter, and embargoes ``cfg.embargo_months`` after.
    """
    raise NotImplementedError


def standardize(train: pd.DataFrame, test: pd.DataFrame,
                cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score ``cols`` using train-window mean/std only; apply to both."""
    raise NotImplementedError


def fama_macbeth(df: pd.DataFrame, feature_cols: list[str], *,
                 target: str = "drift_z", quarter_col: str = "cal_q",
                 nw_lags: int = 4) -> pd.DataFrame:
    """Fama-MacBeth coefficients with Newey-West t-stats.

    Returns a table indexed by feature with columns ``coef``, ``t_stat``,
    ``n_quarters``, ``frac_positive`` (the linear-consistency signal).
    """
    raise NotImplementedError


def fit_lightgbm_cv(df: pd.DataFrame, feature_cols: list[str],
                    cat_cols: list[str], cfg: DriftMLConfig, *,
                    target: str = "drift_z", horizon: Optional[int] = None,
                    task: str = "regression") -> list[FoldResult]:
    """Train LightGBM under purged walk-forward CV; return per-fold results.

    ``task`` is ``"regression"`` (target ``drift_z``) or ``"classification"``
    (target ``drift_class``). SHAP importances are attached per fold.
    """
    raise NotImplementedError


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Spearman rank IC, OOS R^2, and top-minus-bottom decile spread."""
    raise NotImplementedError


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """ROC-AUC and PR-AUC for the strong-up vs down classifier."""
    raise NotImplementedError


def aggregate_oos(folds: list[FoldResult]) -> dict:
    """Pool out-of-sample predictions across folds into headline metrics."""
    raise NotImplementedError
