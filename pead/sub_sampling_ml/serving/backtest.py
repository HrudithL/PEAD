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
* ``model_card.pdf``       -- one-page visual: headline metrics (each with a
  plain-English caption), a predicted-vs-realized scatter, quantile-interval
  coverage, ``prob_up`` reliability, and a hyperparameter-search summary
  (winning params vs. the old defaults) when ``tune_result`` is given.
"""

from __future__ import annotations

import textwrap
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
    """Binned ``prob_up`` reliability curve, robust to small samples.

    Falls back to progressively fewer bins (down to a single bin) instead of
    blanking the whole table whenever there isn't enough data for ``n`` bins
    -- only truly empty input (no ``prob_up`` column, or zero finite
    (pred, outcome) pairs) returns an empty frame.
    """
    if "prob_up" not in results.columns:
        return pd.DataFrame(columns=["bin", "pred_prob", "empirical_prob", "n"])
    y = (pd.to_numeric(results["drift_raw"], errors="coerce") > 0).astype(float)
    p = pd.to_numeric(results["prob_up"], errors="coerce")
    df = pd.DataFrame({"y": y, "p": p}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "pred_prob", "empirical_prob", "n"])

    bins = max(1, min(n, len(df)))
    if bins == 1:
        df["bin"] = 1
    else:
        df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False,
                            duplicates="drop") + 1
    return (df.groupby("bin")
              .agg(pred_prob=("p", "mean"), empirical_prob=("y", "mean"), n=("y", "size"))
              .reset_index())


# ---------------------------------------------------------- model card PDF


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    try:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


def _wrap(text: str, width: int = 44) -> str:
    """Manual word-wrap (deterministic; doesn't depend on matplotlib's
    renderer-dependent ``wrap=True`` text layout)."""
    return "\n".join(textwrap.wrap(text, width=width))


# Short plain-English caption for each headline metric (doc S4 fix c) --
# unknown keys (e.g. future metrics) simply get no caption line.
_METRIC_CAPTIONS: dict[str, str] = {
    "n_events": "events scored out-of-sample in this backtest",
    "spearman_ic_p50": "rank correlation, p50 pred vs realized drift (-1..1, higher=better)",
    "r2_p50": "OOS R^2 of p50 vs realized (0=predicts the mean, <0=worse than that)",
    "decile_spread_p50": "mean realized drift: top predicted decile minus bottom decile",
    "interval_80_coverage": "share of events with realized drift inside [q10,q90] (target ~0.80)",
    "prob_up_auc": "ROC-AUC of prob_up predicting realized drift > 0 (0.5=chance, 1=perfect)",
}


def _metric_lines(summary: dict) -> str:
    """Headline metric lines, each followed by a one-line plain-English caption."""
    lines = []
    for k, v in summary.items():
        lines.append(f"{k}: {_fmt(v)}")
        caption = _METRIC_CAPTIONS.get(k)
        if caption:
            lines.append(f"  ({caption})")
    return "\n".join(lines)


def _add_caption(ax, text: str, width: int = 40, fontsize: float = 6.5) -> None:
    """Short italic how-to-read-this caption placed under a panel (doc S4 fix c)."""
    ax.text(0.5, -0.30, _wrap(text, width=width), transform=ax.transAxes,
           ha="center", va="top", fontsize=fontsize, style="italic")


def _hyperparameter_summary_text(tune_result) -> str:
    """Plain-English hyperparameter-search summary for PDF #1 (doc S4).

    Duck-typed on ``.best_params`` / ``.best_value`` / ``.baseline_value`` /
    ``.n_trials`` -- accepts a
    :class:`~pead.sub_sampling_ml.serving.tune.TuneResult`, or any object /
    namespace exposing the same attributes (missing attributes degrade
    gracefully to "n/a" rather than raising). ``tune_result=None`` means the
    search was never run for this bundle (e.g. it still uses the legacy
    hardcoded defaults) -- reported as such rather than guessed at.
    """
    if tune_result is None:
        return "Hyperparameter search not run (using default params)."

    best_params = getattr(tune_result, "best_params", None)
    best_value = getattr(tune_result, "best_value", None)
    baseline_value = getattr(tune_result, "baseline_value", None)
    n_trials = getattr(tune_result, "n_trials", None)

    lines = ["Hyperparameter search (Optuna, mean pinball loss):"]
    if n_trials is not None:
        lines.append(f"  trials run: {n_trials}")
    if best_params:
        lines.append("  winning params:")
        for k, v in best_params.items():
            lines.append(f"    {k}: {_fmt(v)}")
    else:
        lines.append("  winning params: n/a")

    have_scores = (isinstance(best_value, (int, float))
                  and isinstance(baseline_value, (int, float))
                  and np.isfinite(best_value) and np.isfinite(baseline_value)
                  and baseline_value != 0)
    if have_scores:
        improvement = baseline_value - best_value
        pct = improvement / baseline_value * 100.0
        lines.append(f"  tuned mean pinball:   {best_value:.4f}")
        lines.append(f"  default mean pinball: {baseline_value:.4f}")
        lines.append(f"  improvement: {improvement:.4f} ({pct:+.1f}%)")
    else:
        lines.append("  tuned vs default mean pinball: n/a")

    return "\n".join(lines)


def write_model_card(results: pd.DataFrame, summary: dict, *,
                     out_path: str | Path, title: str = "Drift model card",
                     tune_result=None) -> Path:
    """One-page PDF: headline metrics, calibration scatter, quantile-interval
    coverage, ``prob_up`` reliability, and a hyperparameter-search summary.

    ``tune_result``, when given, is duck-typed (see
    :func:`_hyperparameter_summary_text`) and drives the hyperparameter
    panel; ``None`` renders an explicit "not run" note there instead of
    leaving the section blank.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    calib = _calibration_table(results)
    rel = _reliability_table(results)

    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        fig.suptitle(title, fontsize=14, y=0.985)

        # (0,0) headline metrics as text, one caption line per metric.
        ax = axes[0, 0]; ax.axis("off")
        ax.text(0.02, 0.98, _metric_lines(summary), va="top", ha="left",
               family="monospace", fontsize=7.5, transform=ax.transAxes)
        ax.set_title("Headline OOS metrics", fontsize=10)

        # (0,1) calibration: raw per-event scatter (not decile-bucketed means),
        # with an optional binned-mean trend line overlaid.
        ax = axes[0, 1]
        y = pd.to_numeric(results["drift_raw"], errors="coerce").to_numpy()
        p50 = pd.to_numeric(results.get("pred_q50"), errors="coerce").to_numpy()
        m = np.isfinite(p50) & np.isfinite(y)
        if m.any():
            ax.scatter(p50[m], y[m], s=6, alpha=0.25, linewidths=0, color="steelblue",
                      label="events")
            lo = float(min(p50[m].min(), y[m].min()))
            hi = float(max(p50[m].max(), y[m].max()))
            ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1, label="y = x")
            if not calib.empty:
                ax.plot(calib["pred_mean"], calib["realized_mean"], "o-",
                       color="darkorange", ms=4, lw=1.2, label="binned mean")
            ax.legend(fontsize=6, loc="best", frameon=False)
        else:
            ax.text(0.5, 0.5, "no finite (p50, realized) pairs", ha="center", va="center",
                   transform=ax.transAxes)
        ax.set_xlabel("Predicted p50", fontsize=8)
        ax.set_ylabel("Realized drift", fontsize=8)
        ax.set_title("Calibration: p50 vs realized (per event)", fontsize=10)
        _add_caption(ax, "Each dot is one event: x = predicted median drift, "
                         "y = what actually happened. Dots hugging the dashed "
                         "y=x line mean well-calibrated predictions.")

        # (0,2) prediction interval coverage: quantile-by-quantile empirical
        ax = axes[0, 2]
        qs, emp = [], []
        for q in DEFAULT_QUANTILES:
            col = f"pred_{_quantile_name(q)}"
            if col not in results.columns:
                continue
            pq = pd.to_numeric(results[col], errors="coerce").to_numpy()
            mq = np.isfinite(pq) & np.isfinite(y)
            qs.append(q)
            emp.append(float((y[mq] <= pq[mq]).mean()) if mq.any() else float("nan"))
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.plot(qs, emp, "o-", color="steelblue")
        ax.set_xlabel("Nominal quantile", fontsize=8)
        ax.set_ylabel("Empirical P(y <= pred_q)", fontsize=8)
        ax.set_title("Quantile calibration", fontsize=10)
        _add_caption(ax, "For a well-calibrated model the dots sit on the "
                         "dashed line -- e.g. realized drift should fall "
                         "below the predicted q90 about 90% of the time.")

        # (1,0) prob_up reliability -- or a centered explanatory note when
        # there's no classifier / no usable data, instead of a blank axis.
        ax = axes[1, 0]
        if not rel.empty:
            ax.plot(rel["pred_prob"], rel["empirical_prob"], "o-", color="steelblue")
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        else:
            note = ("classifier not trained -- no prob_up"
                    if "prob_up" not in results.columns else
                    "prob_up present but no finite\n(pred, outcome) pairs to bin")
            ax.text(0.5, 0.5, note, ha="center", va="center", transform=ax.transAxes,
                   fontsize=8)
        ax.set_xlabel("Predicted P(up)", fontsize=8)
        ax.set_ylabel("Empirical P(up)", fontsize=8)
        ax.set_title("prob_up reliability", fontsize=10)
        _add_caption(ax, "Binned P(drift>0) predictions vs how often drift "
                         "was actually positive in each bin; dots on the "
                         "dashed line mean well-calibrated probabilities.")

        # (1,1) hyperparameter-search summary (doc S4).
        ax = axes[1, 1]; ax.axis("off")
        ax.text(0.02, 0.98, _hyperparameter_summary_text(tune_result), va="top",
               ha="left", family="monospace", fontsize=7.5, transform=ax.transAxes)
        ax.set_title("Hyperparameter search", fontsize=10)

        # (1,2) methodology footnote -- keeps the last cell from being dead space.
        ax = axes[1, 2]; ax.axis("off")
        ax.text(0.02, 0.98, _wrap(
            "Each test quarter above is scored using a model fit only on "
            "strictly earlier quarters, with a purge + embargo gap so no "
            "training label window overlaps the test quarter (doc S8). "
            "Nothing here has seen the events it is scored on.", width=42),
               va="top", ha="left", fontsize=7.5, style="italic", transform=ax.transAxes)
        ax.set_title("Methodology", fontsize=10)

        fig.subplots_adjust(left=0.055, right=0.98, top=0.90, bottom=0.06,
                            hspace=0.85, wspace=0.32)
        pdf.savefig(fig)
        plt.close(fig)

    return out_path


def run_backtest(cfg: DriftMLConfig, *, universe: str = "SP500",
                 out_dir: str | Path,
                 params: Optional[dict] = None,
                 tune_result=None,
                 test_quarters: Optional[list[str]] = None) -> dict:
    """End-to-end: walk-forward + summarise + write PDF; return summary dict.

    ``params`` and ``test_quarters`` are threaded into
    :func:`run_walk_forward`; ``tune_result`` is threaded into
    :func:`write_model_card`'s hyperparameter-search summary panel.
    """
    results = run_walk_forward(cfg, universe=universe, out_dir=out_dir,
                               params=params, test_quarters=test_quarters)
    summary = summarize(results)
    write_model_card(results, summary, out_path=Path(out_dir) / "model_card.pdf",
                     title=f"Drift model card ({universe}, h={cfg.primary_horizon})",
                     tune_result=tune_result)
    _log(f"Backtest complete. Summary: {summary}")
    return summary
