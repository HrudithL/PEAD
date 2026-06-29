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

_SHAP_COLS = ["importance", "importance_std", "direction", "n_folds"]
_STAB_COLS = ["mean_rank", "rank_std", "top_k_hit_rate", "n_folds"]
_PERM_COLS = ["mean", "std", "n_folds"]
_SUBGROUP_COLS = ["grouping", "subgroup", "feature", "importance", "rank"]
_TABLE_COLS = [
    "importance", "direction", "mean_rank", "rank_std", "top_k_hit_rate",
    "coef", "t_stat", "frac_positive", "sign_agreement", "consistency_score",
]

def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=cols).rename_axis("feature")

def _as_float_series(obj) -> Optional[pd.Series]:
    if obj is None:
        return None
    try:
        s = pd.Series(obj, dtype="float64")
    except Exception:
        return None
    return s if not s.empty else None

def shap_signed_importance(folds: list[FoldResult]) -> pd.DataFrame:
    """Mean |SHAP| importance and signed direction per feature, pooled over folds.

    Strength is ``importance`` = mean over folds of each fold's mean(|shap|)
    (``FoldResult.shap_importance``, always >= 0). Direction is the mean of the
    per-fold signed-shap Series found at ``FoldResult.metrics['shap_signed']``
    when present, else NaN. Sorted by ``importance`` descending.
    """
    if not folds:
        return _empty(_SHAP_COLS)

    imp_series: list[pd.Series] = []
    sign_series: list[pd.Series] = []
    for f in folds:
        s = _as_float_series(getattr(f, "shap_importance", None))
        if s is None:
            continue
        imp_series.append(s)
        metrics = getattr(f, "metrics", {}) or {}
        signed = _as_float_series(metrics.get("shap_signed"))
        if signed is not None:
            sign_series.append(signed)

    if not imp_series:
        return _empty(_SHAP_COLS)

    imp_df = pd.concat(imp_series, axis=1)
    importance = imp_df.mean(axis=1)
    if sign_series:
        direction = pd.concat(sign_series, axis=1).mean(axis=1).reindex(importance.index)
    else:
        direction = pd.Series(np.nan, index=importance.index)

    out = pd.DataFrame({
        "importance": importance,
        "importance_std": imp_df.std(axis=1),
        "direction": direction,
        "n_folds": imp_df.notna().sum(axis=1).astype(int),
    })
    out = out.sort_values("importance", ascending=False)
    out.index.name = "feature"
    return out

def rank_stability(folds: list[FoldResult], top_k: int = 10) -> pd.DataFrame:
    """Per-feature rank stability across folds: mean rank, dispersion, top-k hit rate.

    Within each fold features are ranked by ``shap_importance`` (rank 1 = most
    important). ``top_k_hit_rate`` is the fraction of all folds in which the
    feature lands in the top ``top_k``. A feature is "consistent" when its
    ``top_k_hit_rate`` is high and ``rank_std`` low. Sorted by ``mean_rank`` asc.
    """
    if not folds:
        return _empty(_STAB_COLS)

    rank_series: list[pd.Series] = []
    for f in folds:
        s = _as_float_series(getattr(f, "shap_importance", None))
        if s is None:
            continue
        rank_series.append(s.dropna().rank(ascending=False, method="min"))

    if not rank_series:
        return _empty(_STAB_COLS)

    rank_df = pd.concat(rank_series, axis=1)
    n_total = rank_df.shape[1]
    out = pd.DataFrame({
        "mean_rank": rank_df.mean(axis=1),
        "rank_std": rank_df.std(axis=1, ddof=0),
        "top_k_hit_rate": (rank_df <= top_k).sum(axis=1) / float(n_total),
        "n_folds": rank_df.notna().sum(axis=1).astype(int),
    })
    out = out.sort_values("mean_rank", ascending=True)
    out.index.name = "feature"
    return out

def permutation_consistency(folds: list[FoldResult]) -> pd.DataFrame:
    """Aggregate permutation importance across folds as a cross-check on SHAP.

    Folds whose ``perm_importance`` is None are skipped. Returns an empty frame
    with columns ``[mean, std, n_folds]`` when no fold carries permutation
    importance. Sorted by ``mean`` descending.
    """
    series: list[pd.Series] = []
    for f in (folds or []):
        s = _as_float_series(getattr(f, "perm_importance", None))
        if s is not None:
            series.append(s)

    if not series:
        return _empty(_PERM_COLS)

    perm_df = pd.concat(series, axis=1)
    out = pd.DataFrame({
        "mean": perm_df.mean(axis=1),
        "std": perm_df.std(axis=1),
        "n_folds": perm_df.notna().sum(axis=1).astype(int),
    })
    out = out.sort_values("mean", ascending=False)
    out.index.name = "feature"
    return out

def subgroup_refits(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str],
                    cfg: DriftMLConfig, *, target: str = "drift_z") -> pd.DataFrame:
    """Re-fit within GICS sector / size tercile / surprise-sign groups.

    Returns a long DataFrame ``[grouping, subgroup, feature, importance, rank]``
    holding the top ~15 SHAP drivers per subgroup, so a driver can be confirmed
    broad rather than a one-sector artifact. Each subgroup fit is wrapped in
    try/except and skipped on failure; only successful subgroups are returned.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_SUBGROUP_COLS)

    from .model import fit_lightgbm_cv

    min_quarters = getattr(cfg, "min_train_quarters", 0)
    rows: list[dict] = []

    def _quarters_ok(sub: pd.DataFrame) -> bool:
        if "cal_q" in sub.columns:
            return int(sub["cal_q"].nunique()) >= min_quarters
        return True

    def _run(grouping: str, label: str, sub: pd.DataFrame) -> None:
        if len(sub) < 200 or not _quarters_ok(sub):
            return
        try:
            folds = fit_lightgbm_cv(sub, feature_cols, cat_cols, cfg, target=target)
            imp = shap_signed_importance(folds)
        except Exception:
            return
        if imp is None or imp.empty:
            return
        for rank, (feat, row) in enumerate(imp.head(15).iterrows(), start=1):
            rows.append({
                "grouping": grouping,
                "subgroup": label,
                "feature": feat,
                "importance": float(row["importance"]),
                "rank": rank,
            })

    if "gics_sector" in df.columns:
        for val, sub in df.groupby("gics_sector", observed=True):
            _run("gics_sector", str(val), sub)

    if "mktcap" in df.columns:
        try:
            terciles = pd.qcut(df["mktcap"], 3, labels=["small", "mid", "large"])
            for val, sub in df.groupby(terciles, observed=True):
                _run("size_tercile", str(val), sub)
        except Exception:
            pass

    surprise_col = next((c for c in ("surprise_raw", "sue_std") if c in df.columns), None)
    if surprise_col is not None:
        label_map = {-1.0: "negative", 0.0: "zero", 1.0: "positive"}
        signs = np.sign(df[surprise_col]).map(label_map).fillna("unknown")
        for val, sub in df.groupby(signs, observed=True):
            _run("surprise_sign", str(val), sub)

    return pd.DataFrame(rows, columns=_SUBGROUP_COLS)

def consistency_table(shap_imp: pd.DataFrame, stability: pd.DataFrame,
                      fm_coefs: pd.DataFrame,
                      perm: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The headline driver map: feature -> direction -> strength -> consistency.

    LEFT-joins (on ``feature``) SHAP strength/sign (``importance``,
    ``direction``), rank stability (``mean_rank``, ``rank_std``,
    ``top_k_hit_rate``), and Fama-MacBeth (``coef``, ``t_stat``,
    ``frac_positive``); permutation ``mean`` is attached as ``perm_importance``
    when supplied.

    ``sign_agreement`` is True/False from ``sign(direction) == sign(coef)``, or
    NaN when either is missing.

    ``consistency_score`` in [0, 1] is a weighted blend of three normalized
    consistency signals::

        score = clip(0.5 * hit + 0.3 * stab + 0.2 * sig, 0, 1)

    where ``hit`` = ``top_k_hit_rate`` (fraction of folds in the top-k),
    ``stab`` = ``1 - rank_std / max(rank_std)`` (low rank dispersion => high
    stability), and ``sig`` = ``min(|t_stat| / 2, 1)`` (linear significance, so a
    |t| of 2 saturates). Missing components contribute 0. Sorted by
    ``importance`` descending.
    """
    if shap_imp is None or shap_imp.empty:
        return _empty(_TABLE_COLS)

    keep = shap_imp.reindex(columns=["importance", "direction"]).copy()

    if stability is not None and not stability.empty:
        keep = keep.join(
            stability.reindex(columns=["mean_rank", "rank_std", "top_k_hit_rate"]),
            how="left",
        )
    if fm_coefs is not None and not fm_coefs.empty:
        keep = keep.join(
            fm_coefs.reindex(columns=["coef", "t_stat", "frac_positive"]),
            how="left",
        )
    if perm is not None and not perm.empty and "mean" in perm.columns:
        keep = keep.join(perm[["mean"]].rename(columns={"mean": "perm_importance"}), how="left")

    for col in ("mean_rank", "rank_std", "top_k_hit_rate", "coef", "t_stat", "frac_positive"):
        if col not in keep.columns:
            keep[col] = np.nan

    direction = keep["direction"]
    coef = keep["coef"]
    have_both = direction.notna() & coef.notna()
    agreement = (np.sign(direction) == np.sign(coef))
    keep["sign_agreement"] = agreement.where(have_both, other=np.nan).astype("object")
    keep.loc[~have_both, "sign_agreement"] = np.nan

    hit = keep["top_k_hit_rate"].astype(float)
    rank_std = keep["rank_std"].astype(float)
    max_rs = rank_std.max(skipna=True)
    norm_rs = rank_std / max_rs if (pd.notna(max_rs) and max_rs > 0) else rank_std * 0.0
    stab = 1.0 - norm_rs
    sig = (keep["t_stat"].astype(float).abs() / 2.0).clip(upper=1.0)

    score = 0.5 * hit.fillna(0.0) + 0.3 * stab.fillna(0.0) + 0.2 * sig.fillna(0.0)
    keep["consistency_score"] = score.clip(lower=0.0, upper=1.0)

    keep = keep.sort_values("importance", ascending=False)
    keep.index.name = "feature"
    return keep
