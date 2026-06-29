"""End-to-end orchestration for the drift-ML study.

Wires the stages together: build (or load) the event-feature-label table, fit
the Fama-MacBeth baseline and the LightGBM models under purged walk-forward CV,
run the attribution / consistency protocol, and render the PDF report. Kept thin
so each stage stays independently testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import dataset, model, attribution, report
from .features import CATEGORICAL_FEATURES
from .config import DriftMLConfig


def _log(msg: str) -> None:
    print(f"[drift-ml] {msg}", flush=True)


def descriptive_comparison(df: pd.DataFrame, feature_cols: list[str],
                           decile_col: str = "drift_decile",
                           n_deciles: int = 10) -> pd.DataFrame:
    """Top-decile vs bottom-decile feature means with a Cohen's d effect size.

    A plain-language "strong upward-drift firms look like ___; downward-drift
    firms look like ___" table over the numeric features.
    """
    if decile_col not in df.columns:
        return pd.DataFrame(columns=["top_mean", "bottom_mean", "diff", "cohens_d"])

    top = df[df[decile_col] == n_deciles]
    bot = df[df[decile_col] == 1]
    rows = []
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        a, b = top[c].dropna(), bot[c].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        d = (a.mean() - b.mean()) / pooled if pooled and np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append((c, float(a.mean()), float(b.mean()),
                     float(a.mean() - b.mean()), float(d)))
    out = pd.DataFrame(rows, columns=["feature", "top_mean", "bottom_mean", "diff", "cohens_d"])
    if out.empty:
        return out.set_index("feature") if "feature" in out.columns else out
    return out.set_index("feature").sort_values("cohens_d", key=lambda s: s.abs(),
                                                 ascending=False)


def run(cfg: DriftMLConfig) -> str:
    """Run the full pipeline and return the path to the generated PDF report."""
    _log(f"Building event-feature-label table {cfg.start_year}-{cfg.end_year} "
         f"(WRDS={'on' if cfg.use_wrds else 'off'}) ...")
    df = dataset.build_event_features(cfg, write=True)
    _log(f"  {len(df):,} events x {df.shape[1]:,} columns.")

    feature_cols = dataset.feature_columns(df)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    _log("Fitting Fama-MacBeth baseline ...")
    fm = model.fama_macbeth(df, feature_cols, target="drift_z")

    folds_reg: dict[int, list] = {}
    oos_reg: dict[int, dict] = {}
    for h in cfg.horizons:
        target = f"drift_z_h{h}"
        if target not in df.columns:
            continue
        _log(f"LightGBM regression on {target} (purged walk-forward) ...")
        folds = model.fit_lightgbm_cv(df, feature_cols, cat_cols, cfg,
                                      target=target, horizon=h, task="regression")
        folds_reg[h] = folds
        oos_reg[h] = model.aggregate_oos(folds)
        _log(f"  OOS: {oos_reg[h]}")

    folds_clf, oos_clf = None, None
    if cfg.fit_classifier and "drift_class" in df.columns \
            and df["drift_class"].notna().sum() >= 100:
        _log("LightGBM classifier on drift_class ...")
        folds_clf = model.fit_lightgbm_cv(df, feature_cols, cat_cols, cfg,
                                          target="drift_class",
                                          horizon=cfg.primary_horizon,
                                          task="classification")
        oos_clf = model.aggregate_oos(folds_clf)
        _log(f"  OOS: {oos_clf}")

    ph = cfg.primary_horizon
    primary_folds = folds_reg.get(ph, [])
    _log("Aggregating SHAP / permutation / consistency ...")
    shap_imp = attribution.shap_signed_importance(primary_folds)
    stability = attribution.rank_stability(primary_folds)
    perm = attribution.permutation_consistency(primary_folds)
    consistency = attribution.consistency_table(shap_imp, stability, fm, perm)

    _log("Subgroup robustness re-fits ...")
    subgroups = attribution.subgroup_refits(df, feature_cols, cat_cols, cfg)

    descriptive = descriptive_comparison(df, feature_cols, n_deciles=cfg.n_deciles)

    results = {
        "df": df,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "horizons": tuple(cfg.horizons),
        "primary_horizon": ph,
        "fm": fm,
        "folds_reg": folds_reg,
        "oos_reg": oos_reg,
        "folds_clf": folds_clf,
        "oos_clf": oos_clf,
        "shap_imp": shap_imp,
        "stability": stability,
        "perm": perm,
        "consistency": consistency,
        "subgroups": subgroups,
        "descriptive": descriptive,
    }

    csv_path = report.write_feature_importance_csv(consistency, cfg)
    _log(f"Wrote {csv_path}")
    pdf_path = report.build_pdf(cfg, results)
    _log(f"Done -> {pdf_path}")
    return pdf_path
