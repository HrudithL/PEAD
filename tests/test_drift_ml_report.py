"""Tests for the report layer and the end-to-end pipeline glue.

The synthetic fixture puts every event in one calendar quarter, so walk-forward
CV produces no folds; the pipeline must still run cleanly and emit a PDF. The
figure/table code paths are exercised directly with a fabricated, fully populated
results bundle.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pead.sub_sampling_ml import report, pipeline, dataset
from pead.sub_sampling_ml.config import DriftMLConfig


def _fabricated_results() -> dict:
    feats = ["sue_std", "mom_12_1", "ear", "mktcap", "amihud"]
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({f: rng.normal(size=n) for f in feats})
    df["drift_decile"] = rng.integers(1, 11, size=n)
    fidx = pd.Index(feats, name="feature")

    shap_imp = pd.DataFrame({
        "importance": [1.0, 0.7, 0.5, 0.3, 0.1],
        "importance_std": [0.1, 0.1, 0.1, 0.1, 0.1],
        "direction": [1.0, -1.0, 0.5, -0.2, 0.05],
        "n_folds": [3, 3, 3, 3, 3],
    }, index=fidx)
    stability = pd.DataFrame({
        "mean_rank": [1.0, 2.0, 3.0, 4.0, 5.0],
        "rank_std": [0.0, 0.3, 0.5, 0.8, 1.0],
        "top_k_hit_rate": [1.0, 0.9, 0.6, 0.3, 0.1],
        "n_folds": [3, 3, 3, 3, 3],
    }, index=fidx)
    fm = pd.DataFrame({
        "coef": [0.30, -0.20, 0.10, -0.05, 0.01],
        "t_stat": [3.5, -2.4, 1.2, -0.6, 0.2],
        "n_quarters": [12, 12, 12, 12, 12],
        "frac_positive": [0.92, 0.10, 0.6, 0.4, 0.55],
    }, index=fidx)
    perm = pd.DataFrame({"mean": [0.4, 0.3, 0.2, 0.1, 0.05],
                         "std": [0.05] * 5, "n_folds": [3] * 5}, index=fidx)
    from pead.sub_sampling_ml import attribution as attr
    consistency = attr.consistency_table(shap_imp, stability, fm, perm)
    descriptive = pipeline.descriptive_comparison(df, feats, n_deciles=10)

    return {
        "df": df,
        "feature_cols": feats,
        "cat_cols": [],
        "horizons": (60, 20, 5),
        "primary_horizon": 60,
        "fm": fm,
        "folds_reg": {},
        "oos_reg": {
            60: {"spearman_ic": 0.05, "r2": 0.01, "decile_spread": 0.4, "n_folds": 3},
            20: {"spearman_ic": 0.04, "r2": 0.00, "decile_spread": 0.2, "n_folds": 3},
            5: {"spearman_ic": 0.02, "r2": -0.01, "decile_spread": 0.1, "n_folds": 3},
        },
        "oos_clf": {"roc_auc": 0.58, "pr_auc": 0.40, "n_folds": 3},
        "shap_imp": shap_imp,
        "stability": stability,
        "perm": perm,
        "consistency": consistency,
        "subgroups": pd.DataFrame(),
        "descriptive": descriptive,
    }


def test_descriptive_comparison_signs_and_effect_size():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "good": np.r_[rng.normal(2, 1, n // 2), rng.normal(-2, 1, n // 2)],
        "noise": rng.normal(0, 1, n),
        "drift_decile": np.r_[np.full(n // 2, 10), np.full(n // 2, 1)],
    })
    out = pipeline.descriptive_comparison(df, ["good", "noise"], n_deciles=10)
    assert out.index[0] == "good"                 # largest |effect|
    assert out.loc["good", "cohens_d"] > 1.0      # strong separation
    assert abs(out.loc["noise", "cohens_d"]) < abs(out.loc["good", "cohens_d"])


def test_write_feature_importance_csv(tmp_path):
    cfg = DriftMLConfig(output_dir=str(tmp_path), use_wrds=False)
    table = _fabricated_results()["consistency"]
    path = report.write_feature_importance_csv(table, cfg)
    assert os.path.exists(path)
    back = pd.read_csv(path, index_col=0)
    assert "consistency_score" in back.columns
    assert len(back) == len(table)


def test_build_pdf_produces_file(tmp_path):
    cfg = DriftMLConfig(output_dir=str(tmp_path), use_wrds=False)
    path = report.build_pdf(cfg, _fabricated_results())
    assert os.path.exists(path)
    assert path.endswith(".pdf")
    assert os.path.getsize(path) > 5_000          # a real multi-page document


def test_build_pdf_handles_empty_results(tmp_path):
    cfg = DriftMLConfig(output_dir=str(tmp_path), use_wrds=False)
    empty = {
        "df": pd.DataFrame(), "feature_cols": [], "cat_cols": [],
        "horizons": (60,), "primary_horizon": 60,
        "fm": pd.DataFrame(), "oos_reg": {}, "oos_clf": None,
        "shap_imp": pd.DataFrame(), "stability": pd.DataFrame(),
        "perm": pd.DataFrame(), "consistency": pd.DataFrame(),
        "subgroups": pd.DataFrame(), "descriptive": pd.DataFrame(),
    }
    path = report.build_pdf(cfg, empty)
    assert os.path.exists(path)


def test_pipeline_run_end_to_end_synthetic(synthetic):
    cfg = DriftMLConfig(
        ibes_path=synthetic["ibes_path"],
        stock_path=synthetic["stock_path"],
        output_dir=synthetic["output_dir"],
        start_year=2019, end_year=2021,
        horizons=(5,),
        n_deciles=5,
        use_wrds=False,
        use_cache=False,           # deterministic: always rebuild in tests
        fit_classifier=False,
    )
    pdf_path = pipeline.run(cfg)
    assert os.path.exists(pdf_path)
    assert os.path.exists(os.path.join(cfg.output_dir, "feature_importance.csv"))


def test_pipeline_uses_cache_when_available(synthetic, monkeypatch):
    cfg = DriftMLConfig(
        ibes_path=synthetic["ibes_path"],
        stock_path=synthetic["stock_path"],
        output_dir=synthetic["output_dir"],
        start_year=2019, end_year=2021,
        horizons=(5,), n_deciles=5,
        use_wrds=False, use_cache=True, fit_classifier=False,
    )
    cached = dataset.build_event_features(cfg, write=False)

    monkeypatch.setattr(dataset, "load_event_features", lambda c: cached)

    def _must_not_build(*a, **k):
        raise AssertionError("build_event_features called despite a warm cache")

    monkeypatch.setattr(dataset, "build_event_features", _must_not_build)
    pdf_path = pipeline.run(cfg)
    assert os.path.exists(pdf_path)
