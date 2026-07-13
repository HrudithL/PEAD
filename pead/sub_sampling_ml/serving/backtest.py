"""Honest walk-forward backtest of the frozen serving pipeline (§8).

For each test quarter, we refit the full artifact -- quantile boosters, the
``drift_z`` regressor, and the ``drift_class`` classifier -- on all *earlier*
quarters, respecting the same purged/embargoed splits as the research pipeline
(:func:`pead.sub_sampling_ml.model.purged_walk_forward_splits`), and predict
raw drift on the test quarter. Nothing about the fit peeks at the test window,
so the calibration and coverage numbers are the ones a live deployment would
actually see. Each fold can optionally fit with the tuned hyperparameters and
early stopping (``params=``, mirroring
:func:`~pead.sub_sampling_ml.serving.train_final.train_final`) instead of the
legacy fixed-round defaults, and the walk-forward can be restricted to a
specific set of TEST quarters (``test_quarters=``) -- e.g. the held-out test
set from :func:`~pead.sub_sampling_ml.serving.tune.split_tuning_test` (doc
S3.4 Job 2).

Emits two artifacts next to the model bundle:

* ``backtest_results.csv`` -- per-event: predicted quantiles, realized drift,
  prob_up, calendar quarter.
* ``model_card.pdf``       -- one-page visual: IC, calibration, prediction
  interval coverage, ``prob_up`` reliability curve, headline metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DriftMLConfig
from ..dataset import build_event_features, feature_columns, load_event_features
from ..features import CATEGORICAL_FEATURES
from ..model import QUARTER_COL, purged_walk_forward_splits, regression_metrics
from .artifact import DEFAULT_QUANTILES, _quantile_name
from .train_final import (_apply_schema, _fit_class_booster, _fit_quantile_boosters,
                          _fit_schema, _fit_z_booster, _log, _restrict_universe)


def _load_or_build(cfg: DriftMLConfig) -> pd.DataFrame:
    df = load_event_features(cfg) if cfg.use_cache else None
    if df is None:
        df = build_event_features(cfg, write=True)
    return df


def _predict_fold(train: pd.DataFrame, test: pd.DataFrame,
                  horizon: int, cfg: DriftMLConfig,
                  params: Optional[dict] = None) -> pd.DataFrame:
    """Fit one walk-forward fold and score its test quarter.

    ``params``, when given, is the shared tuned LightGBM parameter set (doc
    S3.2/S3.4). Passing ``quarters=train[QUARTER_COL]`` unconditionally (same
    as :func:`~pead.sub_sampling_ml.serving.train_final.train_final`) routes
    every head through :func:`~pead.sub_sampling_ml.serving.train_final._fit_head`:
    it early-stops on this fold's own most-recent TRAINING quarter (never the
    test quarter) and then refits on the fold's full training window at the
    discovered round count, so the backtest mirrors the discipline used for
    the final shipped bundle.
    """
    feature_cols = feature_columns(train)
    schema = _fit_schema(train, feature_cols)
    X_tr = _apply_schema(train, schema)
    X_te = _apply_schema(test, schema)

    raw_col = f"drift_raw_h{horizon}"
    z_col = f"drift_z_h{horizon}"
    cls_col = f"drift_class_h{horizon}"
    quarters = train[QUARTER_COL]

    q_boosters = _fit_quantile_boosters(
        X_tr, train[raw_col], schema.cat_cols, DEFAULT_QUANTILES, cfg.random_state,
        params=params, quarters=quarters)
    z_booster = _fit_z_booster(X_tr, train[z_col], schema.cat_cols, cfg.random_state,
                               params=params, quarters=quarters)
    class_booster = None
    if cls_col in train.columns and train[cls_col].notna().sum() >= 100:
        class_booster = _fit_class_booster(
            X_tr, train[cls_col], schema.cat_cols, cfg.random_state,
            params=params, quarters=quarters)

    out = test[["oftic", "anndats", QUARTER_COL, raw_col]].copy()
    out = out.rename(columns={raw_col: "drift_raw"})
    for name, b in q_boosters.items():
        out[f"pred_{name}"] = b.predict(X_te)
    out["drift_z_pred"] = z_booster.predict(X_te)
    if class_booster is not None:
        out["prob_up"] = class_booster.predict(X_te)
    return out


def _quarter_str(q) -> str:
    """Canonical ``"YYYYQn"`` rendering, robust to Period vs plain-string input."""
    return str(pd.Period(str(q), freq="Q"))


def run_walk_forward(cfg: DriftMLConfig, *,
                     universe: str = "SP500",
                     out_dir: str | Path,
                     params: Optional[dict] = None,
                     test_quarters: Optional[list[str]] = None) -> pd.DataFrame:
    """Return the per-event backtest table and write ``backtest_results.csv``.

    ``params``, when given, is the shared tuned LightGBM parameter set
    threaded into every fold's :func:`_predict_fold` (early-stopping on each
    fold's own most-recent training quarter, doc S3.2/S3.4); ``None`` keeps
    the legacy hardcoded-params/fixed-round behaviour.

    ``test_quarters``, when given, restricts the walk-forward to the folds
    whose TEST quarter is in that set (e.g. ``["2022Q3", "2022Q4", ...]``,
    the held-out test set from
    :func:`~pead.sub_sampling_ml.serving.tune.split_tuning_test`) -- training
    still uses every earlier quarter as usual; only which quarters get
    SCORED is restricted (doc S3.4 Job 2). ``None`` scores every available
    fold, same as before this option existed.
    """
    df_all = _load_or_build(cfg)
    df = _restrict_universe(df_all, universe).reset_index(drop=True)
    _log(f"Backtest on {len(df):,} events (universe='{universe}').")

    horizon = cfg.primary_horizon
    splits = purged_walk_forward_splits(df[QUARTER_COL], horizon, cfg)
    if not splits:
        raise SystemExit(
            "No walk-forward splits available (not enough training quarters).")

    if test_quarters is not None:
        wanted = {_quarter_str(q) for q in test_quarters}
        splits = [(tr, te) for tr, te in splits
                 if _quarter_str(df[QUARTER_COL].iloc[te[0]]) in wanted]
        if not splits:
            raise SystemExit(
                "No walk-forward folds fall within the requested test_quarters.")

    fold_results = []
    for i, (tr, te) in enumerate(splits, start=1):
        train, test = df.iloc[tr], df.iloc[te]
        _log(f"  fold {i}/{len(splits)}  train={len(train):,} test={len(test):,} "
             f"(test q={test[QUARTER_COL].iloc[0]})")
        fold_results.append(_predict_fold(train, test, horizon, cfg, params=params))

    results = pd.concat(fold_results, ignore_index=True)
    # Enforce quantile monotonicity per row.
    qcols = sorted(c for c in results.columns if c.startswith("pred_q"))
    if qcols:
        results[qcols] = np.sort(results[qcols].to_numpy(), axis=1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "backtest_results.csv", index=False)
    return results


# ---------------------------------------------------------- metrics


def summarize(results: pd.DataFrame) -> dict:
    """Headline OOS metrics for the model card."""
    y = pd.to_numeric(results["drift_raw"], errors="coerce").to_numpy()
    p50 = pd.to_numeric(results.get("pred_q50"), errors="coerce").to_numpy()

    reg = regression_metrics(y, p50)
    lo = pd.to_numeric(results.get("pred_q10"), errors="coerce").to_numpy()
    hi = pd.to_numeric(results.get("pred_q90"), errors="coerce").to_numpy()
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    coverage = float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()) if m.any() else float("nan")

    out = {
        "n_events": int(len(results)),
        "spearman_ic_p50": reg.get("spearman_ic"),
        "r2_p50": reg.get("r2"),
        "decile_spread_p50": reg.get("decile_spread"),
        "interval_80_coverage": coverage,
    }
    if "prob_up" in results.columns:
        pu = pd.to_numeric(results["prob_up"], errors="coerce").to_numpy()
        m2 = np.isfinite(pu) & np.isfinite(y)
        if m2.any():
            from sklearn.metrics import roc_auc_score
            try:
                out["prob_up_auc"] = float(roc_auc_score((y[m2] > 0).astype(int), pu[m2]))
            except Exception:
                out["prob_up_auc"] = float("nan")
    return out


def _calibration_table(results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    y = pd.to_numeric(results["drift_raw"], errors="coerce")
    p = pd.to_numeric(results.get("pred_q50"), errors="coerce")
    df = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(df) < n:
        return pd.DataFrame(columns=["bin", "pred_mean", "realized_mean", "n"])
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n, labels=False) + 1
    return (df.groupby("bin")
              .agg(pred_mean=("p", "mean"), realized_mean=("y", "mean"), n=("y", "size"))
              .reset_index())


def _reliability_table(results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if "prob_up" not in results.columns:
        return pd.DataFrame()
    y = (pd.to_numeric(results["drift_raw"], errors="coerce") > 0).astype(float)
    p = pd.to_numeric(results["prob_up"], errors="coerce")
    df = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(df) < n:
        return pd.DataFrame(columns=["bin", "pred_prob", "empirical_prob", "n"])
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n, labels=False,
                        duplicates="drop") + 1
    return (df.groupby("bin")
              .agg(pred_prob=("p", "mean"), empirical_prob=("y", "mean"), n=("y", "size"))
              .reset_index())


# ---------------------------------------------------------- model card PDF


def write_model_card(results: pd.DataFrame, summary: dict, *,
                     out_path: str | Path, title: str = "Drift model card") -> Path:
    """One-page PDF with headline metrics, calibration, and reliability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    calib = _calibration_table(results)
    rel = _reliability_table(results)

    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(title, fontsize=14, y=0.98)

        # (0,0) headline metrics as text
        ax = axes[0, 0]; ax.axis("off")
        lines = [f"{k}: {_fmt(v)}" for k, v in summary.items()]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")
        ax.set_title("Headline OOS metrics")

        # (0,1) calibration: pred vs realized decile means
        ax = axes[0, 1]
        if not calib.empty:
            ax.plot(calib["pred_mean"], calib["realized_mean"], "o-")
            lo = float(min(calib["pred_mean"].min(), calib["realized_mean"].min()))
            hi = float(max(calib["pred_mean"].max(), calib["realized_mean"].max()))
            ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
        ax.set_xlabel("Predicted p50 (bin mean)")
        ax.set_ylabel("Realized drift (bin mean)")
        ax.set_title("Calibration (p50 vs realized)")

        # (1,0) prediction interval coverage: quantile-by-quantile empirical
        ax = axes[1, 0]
        y = pd.to_numeric(results["drift_raw"], errors="coerce").to_numpy()
        qs, emp = [], []
        for q in DEFAULT_QUANTILES:
            col = f"pred_{_quantile_name(q)}"
            if col not in results.columns:
                continue
            p = pd.to_numeric(results[col], errors="coerce").to_numpy()
            m = np.isfinite(p) & np.isfinite(y)
            qs.append(q)
            emp.append(float((y[m] <= p[m]).mean()) if m.any() else float("nan"))
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.plot(qs, emp, "o-")
        ax.set_xlabel("Nominal quantile")
        ax.set_ylabel("Empirical P(y <= pred_q)")
        ax.set_title("Quantile calibration")

        # (1,1) prob_up reliability
        ax = axes[1, 1]
        if not rel.empty:
            ax.plot(rel["pred_prob"], rel["empirical_prob"], "o-")
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted P(up)")
        ax.set_ylabel("Empirical P(up)")
        ax.set_title("prob_up reliability")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

    return out_path


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    try:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


def run_backtest(cfg: DriftMLConfig, *, universe: str = "SP500",
                 out_dir: str | Path) -> dict:
    """End-to-end: walk-forward + summarise + write PDF; return summary dict."""
    results = run_walk_forward(cfg, universe=universe, out_dir=out_dir)
    summary = summarize(results)
    write_model_card(results, summary, out_path=Path(out_dir) / "model_card.pdf",
                     title=f"Drift model card ({universe}, h={cfg.primary_horizon})")
    _log(f"Backtest complete. Summary: {summary}")
    return summary
