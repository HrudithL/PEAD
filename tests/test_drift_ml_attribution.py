"""Unit tests for the drift-ML attribution / consistency layer (no training)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pead.sub_sampling_ml.model import FoldResult
from pead.sub_sampling_ml import attribution as attr

def _make_folds() -> list[FoldResult]:
    """3 folds over features a,b,c where 'a' is consistently most important."""
    importances = [
        {"a": 1.0, "b": 0.5, "c": 0.2},
        {"a": 0.9, "b": 0.4, "c": 0.3},
        {"a": 1.2, "b": 0.6, "c": 0.1},
    ]
    signed = [
        {"a": 1.0, "b": -0.5, "c": 0.2},
        {"a": 0.8, "b": -0.4, "c": 0.1},
        {"a": 1.1, "b": -0.6, "c": 0.3},
    ]
    folds = []
    for i, (imp, sgn) in enumerate(zip(importances, signed)):
        folds.append(FoldResult(
            fold=i,
            shap_importance=pd.Series(imp, dtype="float64"),
            metrics={"shap_signed": pd.Series(sgn, dtype="float64")},
        ))
    return folds

def test_shap_signed_importance_ranks_a_first():
    out = attr.shap_signed_importance(_make_folds())
    assert list(out.columns) == ["importance", "importance_std", "direction", "n_folds"]
    assert out.index[0] == "a"
    assert int(out.loc["a", "n_folds"]) == 3
    assert out.loc["a", "importance"] == out["importance"].max()
    assert out.loc["a", "direction"] > 0
    assert out.loc["b", "direction"] < 0

def test_shap_signed_importance_empty():
    out = attr.shap_signed_importance([])
    assert out.empty
    assert list(out.columns) == ["importance", "importance_std", "direction", "n_folds"]

def test_rank_stability_a_most_consistent():
    out = attr.rank_stability(_make_folds(), top_k=2)
    assert out["mean_rank"].idxmin() == "a"
    assert out.index[0] == "a"
    assert out.loc["a", "mean_rank"] == 1.0
    assert out.loc["a", "rank_std"] == 0.0
    assert out.loc["a", "top_k_hit_rate"] == 1.0
    assert int(out.loc["a", "n_folds"]) == 3

def test_consistency_table_joins_signs_and_score():
    shap_imp = pd.DataFrame(
        {"importance": [1.0, 0.5], "direction": [1.0, -0.5]},
        index=pd.Index(["a", "b"], name="feature"),
    )
    stability = pd.DataFrame(
        {"mean_rank": [1.0, 2.0], "rank_std": [0.0, 0.5], "top_k_hit_rate": [1.0, 0.5]},
        index=pd.Index(["a", "b"], name="feature"),
    )
    fm = pd.DataFrame(
        {"coef": [0.3, 0.4], "t_stat": [3.0, 2.0], "frac_positive": [0.9, 0.8]},
        index=pd.Index(["a", "b"], name="feature"),
    )

    out = attr.consistency_table(shap_imp, stability, fm)

    assert "coef" in out.columns and "mean_rank" in out.columns
    assert out.loc["a", "sign_agreement"] is True or out.loc["a", "sign_agreement"] == True  # +/+ match
    assert out.loc["b", "sign_agreement"] == False  # -/+ differ
    scores = out["consistency_score"].astype(float)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    assert out.index[0] == "a"

def test_consistency_table_sign_agreement_nan_when_missing():
    shap_imp = pd.DataFrame(
        {"importance": [1.0], "direction": [np.nan]},
        index=pd.Index(["a"], name="feature"),
    )
    stability = pd.DataFrame(
        {"mean_rank": [1.0], "rank_std": [0.0], "top_k_hit_rate": [1.0]},
        index=pd.Index(["a"], name="feature"),
    )
    fm = pd.DataFrame(
        {"coef": [0.3], "t_stat": [3.0], "frac_positive": [0.9]},
        index=pd.Index(["a"], name="feature"),
    )
    out = attr.consistency_table(shap_imp, stability, fm)
    assert pd.isna(out.loc["a", "sign_agreement"])

def test_permutation_consistency_empty_when_all_none():
    folds = [FoldResult(fold=i, perm_importance=None) for i in range(3)]
    out = attr.permutation_consistency(folds)
    assert out.empty
    assert list(out.columns) == ["mean", "std", "n_folds"]

def test_permutation_consistency_aggregates():
    folds = [
        FoldResult(fold=0, perm_importance=pd.Series({"a": 0.4, "b": 0.1})),
        FoldResult(fold=1, perm_importance=pd.Series({"a": 0.6, "b": 0.2})),
        FoldResult(fold=2, perm_importance=None),
    ]
    out = attr.permutation_consistency(folds)
    assert list(out.columns) == ["mean", "std", "n_folds"]
    assert out.index[0] == "a"
    assert int(out.loc["a", "n_folds"]) == 2
    assert np.isclose(out.loc["a", "mean"], 0.5)
