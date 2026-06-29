"""Feature attribution & the consistency / robustness protocol (Section 9).

Turns the per-fold model outputs into the headline deliverable: a ranked,
sign-annotated, stability-scored map of **feature -> direction of effect on
drift -> strength -> consistency**.

Consistency is a property of a *feature*: it is consistent if it stays in the
top-k importance across most walk-forward folds, holds up when the model is
re-fit within GICS sectors / size terciles / surprise-sign groups, and agrees in
sign between the Fama-MacBeth (linear) and SHAP (non-linear) views.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import DriftMLConfig
from .model import FoldResult


def shap_signed_importance(folds: list[FoldResult]) -> pd.DataFrame:
    """Mean |SHAP| importance and signed direction per feature, pooled over folds."""
    raise NotImplementedError


def rank_stability(folds: list[FoldResult], top_k: int = 10) -> pd.DataFrame:
    """Per-feature rank stability across folds: mean rank, dispersion, top-k hit rate."""
    raise NotImplementedError


def permutation_consistency(folds: list[FoldResult]) -> pd.DataFrame:
    """Aggregate permutation importance across folds as a cross-check on SHAP."""
    raise NotImplementedError


def subgroup_refits(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str],
                    cfg: DriftMLConfig, *, target: str = "drift_z") -> pd.DataFrame:
    """Re-fit within GICS sector / size tercile / surprise-sign groups.

    Returns top-driver importances per subgroup so a driver can be confirmed
    broad rather than a one-sector artifact.
    """
    raise NotImplementedError


def consistency_table(shap_imp: pd.DataFrame, stability: pd.DataFrame,
                      fm_coefs: pd.DataFrame,
                      perm: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The headline driver map.

    Joins SHAP strength/sign, rank stability, Fama-MacBeth sign/t-stat, and
    (optionally) permutation importance into one ranked, sign-annotated,
    stability-scored table, with a flag where linear and SHAP signs disagree.
    """
    raise NotImplementedError
