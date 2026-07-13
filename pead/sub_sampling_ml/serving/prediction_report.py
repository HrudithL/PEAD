"""Per-prediction PDF report: how good the model is, and how this one call did.

Emitted by ``run_predict_drift.py --report`` alongside the JSON record. Two
pages, reusing the same calibration/reliability machinery as the training-time
``model_card.pdf`` (:mod:`pead.sub_sampling_ml.serving.backtest`) so the
numbers agree with what training already measured:

* Page 1 -- model-level performance: headline OOS metrics, calibration,
  quantile calibration, ``prob_up`` reliability, and IC-by-quarter (how the
  walk-forward fit improves/degrades as more history accrues). Sourced from
  ``backtest_results.csv`` next to the loaded bundle, if that bundle was
  trained with ``--with-backtest``; otherwise this page says so plainly
  instead of fabricating numbers.
* Page 2 -- this specific ``(ticker, anndate)`` call: the predicted quantile
  fan against the realized CAR[+1, +horizon], if the event is old enough for
  that outcome to already exist on the price panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DriftMLConfig
from ..model import QUARTER_COL, regression_metrics
from .artifact import DriftModel
from .backtest import _calibration_table, _fmt, _reliability_table, summarize
from .featurize_one import realized_drift_one


def _load_backtest_results(model_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if model_dir is None:
        return None
    path = Path(model_dir) / "backtest_results.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, parse_dates=["anndats"])
    except Exception:
        return None


def _ic_by_quarter(results: pd.DataFrame) -> pd.DataFrame:
    """Spearman IC of pred_q50 vs realized drift, per test quarter.

    This is the walk-forward "does the model get better with more training
    history" view: each quarter in ``backtest_results.csv`` was scored by a
    booster refit on every *earlier* quarter only, so a rising trend here is
    genuine improvement, not a fit-on-the-past artifact.
    """
    if results is None or "pred_q50" not in results.columns:
        return pd.DataFrame(columns=["quarter", "ic", "n"])
    rows = []
    for q, grp in results.groupby(QUARTER_COL):
        m = regression_metrics(grp["drift_raw"].to_numpy(), grp["pred_q50"].to_numpy())
        rows.append({"quarter": q, "ic": m.get("spearman_ic"), "n": len(grp)})
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def _model_performance_page(fig, results: Optional[pd.DataFrame],
                            metadata_lines: list[str]) -> None:
    axes = fig.subplots(2, 2)

    ax = axes[0, 0]; ax.axis("off")
    if results is None:
        ax.text(0.02, 0.98,
                "No walk-forward backtest is bundled with this model.\n"
                "Retrain with --with-backtest to get calibration,\n"
                "coverage, and IC-by-quarter diagnostics here.\n\n"
                + "\n".join(metadata_lines),
                va="top", family="monospace", fontsize=9)
        ax.set_title("Headline OOS metrics")
        for a in (axes[0, 1], axes[1, 0], axes[1, 1]):
            a.axis("off")
            a.text(0.5, 0.5, "n/a -- no backtest_results.csv", ha="center", va="center")
        return

    summary = summarize(results)
    lines = [f"{k}: {_fmt(v)}" for k, v in summary.items()] + [""] + metadata_lines
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("Headline OOS metrics")

    calib = _calibration_table(results)
    ax = axes[0, 1]
    if not calib.empty:
        ax.plot(calib["pred_mean"], calib["realized_mean"], "o-")
        lo = float(min(calib["pred_mean"].min(), calib["realized_mean"].min()))
        hi = float(max(calib["pred_mean"].max(), calib["realized_mean"].max()))
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
    ax.set_xlabel("Predicted p50 (bin mean)")
    ax.set_ylabel("Realized drift (bin mean)")
    ax.set_title("Calibration (p50 vs realized)")

    ax = axes[1, 0]
    ic_q = _ic_by_quarter(results)
    if not ic_q.empty:
        x = np.arange(len(ic_q))
        ax.plot(x, ic_q["ic"], "o-")
        ax.axhline(0.0, color="grey", lw=1, ls="--")
        step = max(1, len(ic_q) // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(q) for q in ic_q["quarter"][::step]],
                           rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Walk-forward test quarter")
    ax.set_ylabel("Spearman IC (pred_q50 vs realized)")
    ax.set_title("Training improvement: OOS IC by quarter")

    rel = _reliability_table(results)
    ax = axes[1, 1]
    if not rel.empty:
        ax.plot(rel["pred_prob"], rel["empirical_prob"], "o-")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(up)")
    ax.set_ylabel("Empirical P(up)")
    ax.set_title("prob_up reliability")


def _event_page(fig, record: dict, actual: Optional[float]) -> None:
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})

    ax = axes[0]; ax.axis("off")
    lines = [
        f"ticker:          {record['ticker']}",
        f"anndate:         {record['anndate']}",
        f"horizon:         +{record['horizon']}d",
        f"model_version:   {record['model_version']}",
        f"coverage:        {record['coverage']:.0%} "
        f"({record['n_features_present']}/{record['n_features_total']})",
        "",
        f"expected_drift:  {_fmt(record['expected_drift'])}",
        f"interval_80:     [{_fmt(record['interval_80'][0])}, "
        f"{_fmt(record['interval_80'][1])}]",
    ]
    if actual is None:
        lines += ["", "actual_drift:    not yet observable",
                 "                 (needs +{}d of price history past".format(record["horizon"]),
                 "                  the announcement; event hasn't finished)"]
    else:
        err = actual - record["expected_drift"]
        qs = record["quantiles"]
        qvals = sorted(qs.items(), key=lambda kv: kv[1])
        rank = sum(1 for _, v in qvals if v <= actual) / max(1, len(qvals))
        lines += [
            "",
            f"actual_drift:    {_fmt(actual)}",
            f"error (a - e):   {_fmt(err)}",
            f"actual vs. predicted quantiles: ~{rank:.0%}-ile",
        ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("This prediction")

    ax = axes[1]
    qs = record["quantiles"]
    order = ["q10", "q25", "q50", "q75", "q90"]
    xs = [q for q in order if q in qs]
    ys = [qs[q] for q in xs]
    if xs:
        ax.plot(range(len(xs)), ys, "o-", color="tab:blue", label="predicted quantiles")
    if "q50" in qs:
        ax.axhline(qs["q50"], color="tab:blue", lw=1, ls=":", alpha=0.6)
    if actual is not None:
        ax.axhline(actual, color="tab:red", lw=2, label="realized drift")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_ylabel("CAR[+1, +horizon]")
    ax.set_title("Expected drift (predicted quantile fan) vs. actual")
    ax.legend(loc="best", fontsize=8)


def write_prediction_report(record: dict, cfg: DriftMLConfig, model: DriftModel, *,
                            out_path: str | Path,
                            model_dir: Optional[Path] = None,
                            compute_actual: bool = True) -> Path:
    """Write the two-page PDF described in the module docstring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    results = _load_backtest_results(model_dir)

    meta = model.metadata
    metadata_lines = [
        f"trained cutoff:  {meta.cutoff_date}",
        f"universe:        {meta.universe}",
        f"n_events:        {meta.n_events}",
        f"horizon:         {meta.horizon}d",
        f"date range:      {meta.start_year}-{meta.end_year}",
        f"git_sha:         {meta.git_sha}",
    ]

    actual = None
    if compute_actual:
        try:
            actual = realized_drift_one(record["ticker"], record["anndate"], cfg,
                                        horizon=record["horizon"])
        except Exception as exc:
            print(f"[drift-serve] realized_drift_one failed for "
                 f"{record['ticker']} {record['anndate']}: {exc}")
            actual = None

    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(f"Drift model performance -- {record['model_version']}",
                    fontsize=14, y=0.98)
        _model_performance_page(fig, results, metadata_lines)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 5.5))
        fig.suptitle(f"Prediction detail -- {record['ticker']} {record['anndate']}",
                    fontsize=14, y=0.98)
        _event_page(fig, record, actual)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

    return out_path
