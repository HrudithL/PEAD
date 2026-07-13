"""Models + leakage-aware validation (Sections 7 & 8).

Two complementary models on the same event-feature-label table:

A. **Fama-MacBeth** -- per-calendar-quarter cross-sectional OLS of ``drift_z`` on
   cross-sectionally standardized features; coefficients averaged across quarters
   with Newey-West t-stats. The per-quarter coefficient series *is* the
   consistency evidence for linear effects.

B. **LightGBM** -- gradient-boosted regressor on ``drift_z`` (primary) and an
   optional classifier on ``drift_class``, with native categorical handling and
   SHAP-ready models returned per fold.

Both are evaluated under **purged, embargoed walk-forward CV** (Lopez de Prado):
train on past quarters, test on the next; purge any training quarter whose
[+1, +H] label window overlaps the test quarter; embargo ``cfg.embargo_months``
after each test block. Cross-sectional standardization is fit on the training
window only.

Heavy imports (lightgbm, statsmodels, sklearn, scipy) are lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import DriftMLConfig

# Per-event calendar-quarter column produced by labels.compute_labels.
QUARTER_COL = "cal_q"

# Approximate trading days per calendar quarter, used to translate a label
# horizon (trading days) into a quarter-level purge gap.
_TRADING_DAYS_PER_Q = 63


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


def _as_quarter_period(quarters: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.Series(quarters).astype(str).values, freq="Q")


def purged_walk_forward_splits(quarters: pd.Series, horizon: int,
                               cfg: DriftMLConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) per expanding walk-forward step.

    ``quarters`` is the per-event calendar quarter (positional, length n). Each
    step tests one calendar quarter using all *earlier* quarters for training,
    after removing a gap of quarters immediately before the test quarter so that
    no training event's [+1, +horizon] label window overlaps the test period
    (purge), plus the embargo. The gap is
    ``ceil(horizon / 63) + ceil(embargo_months / 3)`` quarters.
    """
    q = _as_quarter_period(quarters)
    arr = np.asarray(q)
    uniq = sorted(pd.unique(arr))

    purge_q = max(1, int(np.ceil(horizon / _TRADING_DAYS_PER_Q)))
    embargo_q = max(0, int(np.ceil(cfg.embargo_months / 3)))
    gap = purge_q + embargo_q

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(len(uniq)):
        last_train_pos = k - 1 - gap
        if last_train_pos < cfg.min_train_quarters - 1:
            continue
        train_quarters = set(uniq[:last_train_pos + 1])
        test_q = uniq[k]
        train_mask = np.fromiter((qq in train_quarters for qq in arr),
                                 dtype=bool, count=len(arr))
        test_mask = arr == test_q
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if train_idx.size and test_idx.size:
            splits.append((train_idx, test_idx))
    return splits


def standardize(train: pd.DataFrame, test: pd.DataFrame,
                cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score ``cols`` using train-window mean/std only; apply to both."""
    train, test = train.copy(), test.copy()
    for c in cols:
        mu = train[c].mean(skipna=True)
        sd = train[c].std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        train[c] = (train[c] - mu) / sd
        test[c] = (test[c] - mu) / sd
    return train, test


def _numeric_features(df: pd.DataFrame, feature_cols: list[str],
                      cat_cols: list[str]) -> list[str]:
    cats = set(cat_cols)
    return [c for c in feature_cols
            if c not in cats and pd.api.types.is_numeric_dtype(df[c])]


def fama_macbeth(df: pd.DataFrame, feature_cols: list[str], *,
                 target: str = "drift_z", quarter_col: str = QUARTER_COL,
                 nw_lags: int = 4) -> pd.DataFrame:
    """Fama-MacBeth coefficients with Newey-West t-stats.

    Returns a table indexed by feature with columns ``coef``, ``t_stat``,
    ``n_quarters``, ``frac_positive`` (the linear-consistency signal). Features
    are cross-sectionally standardized within each quarter before the OLS.
    """
    import statsmodels.api as sm

    num_cols = _numeric_features(df, feature_cols, [])
    per_quarter: dict[str, list[float]] = {c: [] for c in num_cols}

    for _, g in df.groupby(quarter_col):
        sub = g[num_cols + [target]].copy()
        sub = sub.dropna(subset=[target])
        if len(sub) < max(10, len(num_cols) + 2):
            continue
        X = sub[num_cols]
        # Cross-sectional standardization within the quarter; drop dead columns.
        X = (X - X.mean()) / X.std(ddof=0)
        keep = [c for c in num_cols if X[c].notna().any() and X[c].std(ddof=0) > 0]
        if not keep:
            continue
        X = X[keep].fillna(0.0)
        X = sm.add_constant(X, has_constant="add")
        y = sub[target].astype(float).values
        try:
            res = sm.OLS(y, X).fit()
        except Exception:
            continue
        for c in keep:
            per_quarter[c].append(float(res.params.get(c, np.nan)))

    rows = []
    for c in num_cols:
        series = np.array([v for v in per_quarter[c] if np.isfinite(v)])
        if series.size == 0:
            rows.append((c, np.nan, np.nan, 0, np.nan))
            continue
        coef = float(series.mean())
        frac_pos = float((series > 0).mean())
        t_stat = _nw_tstat(series, nw_lags)
        rows.append((c, coef, t_stat, series.size, frac_pos))

    out = pd.DataFrame(rows, columns=["feature", "coef", "t_stat",
                                      "n_quarters", "frac_positive"])
    return out.set_index("feature").sort_values("coef", key=lambda s: s.abs(),
                                                 ascending=False)


def _nw_tstat(series: np.ndarray, nw_lags: int) -> float:
    """Newey-West t-stat for the mean of a coefficient time series."""
    import statsmodels.api as sm

    if series.size < 2:
        return np.nan
    try:
        res = sm.OLS(series, np.ones(series.size)).fit(
            cov_type="HAC", cov_kwds={"maxlags": min(nw_lags, series.size - 1)})
        return float(res.tvalues[0])
    except Exception:
        sd = series.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return np.nan
        return float(series.mean() / (sd / np.sqrt(series.size)))


def fit_lightgbm_cv(df: pd.DataFrame, feature_cols: list[str],
                    cat_cols: list[str], cfg: DriftMLConfig, *,
                    target: str = "drift_z", horizon: Optional[int] = None,
                    task: str = "regression") -> list[FoldResult]:
    """Train LightGBM under purged walk-forward CV; return per-fold results.

    ``task`` is ``"regression"`` (target ``drift_z``) or ``"classification"``
    (target ``drift_class``). SHAP importances (mean |value| per feature) are
    attached per fold; the mean *signed* SHAP value is stored in
    ``metrics['shap_signed']`` for the attribution layer.
    """
    import lightgbm as lgb

    horizon = horizon or cfg.primary_horizon
    cats = [c for c in cat_cols if c in feature_cols]
    num_cols = _numeric_features(df, feature_cols, cats)
    use_cols = num_cols + cats

    work = df.dropna(subset=[target]).copy()
    for c in cats:
        if not isinstance(work[c].dtype, pd.CategoricalDtype):
            work[c] = work[c].astype("category")

    splits = purged_walk_forward_splits(work[QUARTER_COL], horizon, cfg)
    results: list[FoldResult] = []

    for fold, (tr, te) in enumerate(splits):
        train, test = work.iloc[tr], work.iloc[te]
        train, test = standardize(train, test, num_cols)

        X_tr, y_tr = train[use_cols], train[target].astype(float).values
        X_te, y_te = test[use_cols], test[target].astype(float).values
        if len(X_tr) < 50 or len(X_te) == 0:
            continue

        if task == "classification":
            model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=cfg.random_state,
                n_jobs=-1, verbose=-1)
            model.fit(X_tr, y_tr, categorical_feature=cats or "auto")
            y_pred = model.predict_proba(X_te)[:, 1]
            metrics = classification_metrics(y_te, y_pred)
        else:
            model = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=cfg.random_state,
                n_jobs=-1, verbose=-1)
            model.fit(X_tr, y_tr, categorical_feature=cats or "auto")
            y_pred = model.predict(X_te)
            metrics = regression_metrics(y_te, y_pred)

        shap_abs, shap_signed = _shap_importances(model, X_te, use_cols, task)
        metrics["shap_signed"] = shap_signed
        perm = _perm_importance(model, X_te, y_te, use_cols, cfg)

        results.append(FoldResult(
            fold=fold,
            test_index=test.index.to_numpy(),
            y_true=y_te,
            y_pred=np.asarray(y_pred),
            metrics=metrics,
            shap_importance=shap_abs,
            perm_importance=perm,
        ))
    return results


def _shap_importances(model, X, cols, task) -> tuple[pd.Series, pd.Series]:
    """Mean |SHAP| (importance) and mean signed SHAP (direction) per feature."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):  # classifier -> per-class; take positive class
            vals = vals[-1]
        vals = np.asarray(vals)
        abs_imp = pd.Series(np.abs(vals).mean(axis=0), index=cols).sort_values(
            ascending=False)
        signed = pd.Series(vals.mean(axis=0), index=cols)
        return abs_imp, signed
    except Exception:
        # Fall back to the model's split-gain importance (unsigned).
        try:
            imp = pd.Series(model.feature_importances_, index=cols).sort_values(
                ascending=False)
            return imp, pd.Series(np.nan, index=cols)
        except Exception:
            return pd.Series(dtype=float), pd.Series(dtype=float)


def _perm_importance(model, X, y, cols, cfg) -> Optional[pd.Series]:
    try:
        from sklearn.inspection import permutation_importance

        r = permutation_importance(model, X, y, n_repeats=5,
                                   random_state=cfg.random_state, n_jobs=1)
        return pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)
    except Exception:
        return None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Spearman rank IC, OOS R^2, and top-minus-bottom decile spread."""
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size < 3:
        return {"spearman_ic": np.nan, "r2": np.nan, "decile_spread": np.nan, "n": int(y_true.size)}

    ic = float(spearmanr(y_pred, y_true).correlation)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"spearman_ic": ic, "r2": r2,
            "decile_spread": _decile_spread(y_true, y_pred), "n": int(y_true.size)}


def _decile_spread(y_true: np.ndarray, y_pred: np.ndarray, n: int = 10) -> float:
    if y_pred.size < n:
        return np.nan
    try:
        ranks = pd.qcut(pd.Series(y_pred).rank(method="first"), n, labels=False)
    except ValueError:
        return np.nan
    top = y_true[ranks.values == n - 1].mean()
    bot = y_true[ranks.values == 0].mean()
    return float(top - bot)


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """ROC-AUC and PR-AUC for the strong-up vs down classifier."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[m], y_score[m]
    out = {"roc_auc": np.nan, "pr_auc": np.nan, "n": int(y_true.size)}
    if y_true.size >= 3 and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def aggregate_oos(folds: list[FoldResult]) -> dict:
    """Pool out-of-sample predictions across folds into headline metrics."""
    if not folds:
        return {}
    y_true = np.concatenate([f.y_true for f in folds]) if folds else np.array([])
    y_pred = np.concatenate([f.y_pred for f in folds]) if folds else np.array([])
    is_clf = any("roc_auc" in f.metrics for f in folds)
    pooled = classification_metrics(y_true, y_pred) if is_clf else regression_metrics(y_true, y_pred)
    pooled["n_folds"] = len(folds)
    return pooled
