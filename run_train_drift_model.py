"""CLI: tune, honestly backtest, and ship the frozen drift-serving model bundle.

One command runs all three jobs (doc S3.1 / S3.4 Jobs 1-3 / S5):

  1. Tune LightGBM hyperparameters (Optuna) on the tuning pool -- the earlier
     share of distinct event-quarters (``1 - test_frac``).
  2. Honestly backtest those params on the held-out test quarters, emitting
     ``backtest_results.csv`` + ``model_card.pdf`` (PDF #1) -- runs by default,
     since it is the deliverable that proves out-of-sample accuracy.
  3. Refit on ALL data (tuning pool + test quarters) at the winning
     hyperparameters and persist the frozen bundle.

Examples
--------
  python run_train_drift_model.py                      # full universe, tune + backtest + ship
  python run_train_drift_model.py --universe SP500 --cutoff 2024-12-31
  python run_train_drift_model.py --best-params models/2024-12-31_abc123/best_params.json
  python run_train_drift_model.py --no-tune --no-backtest   # legacy fast path
"""

from __future__ import annotations

import argparse
import json
import shutil
import types
from datetime import date
from pathlib import Path
from typing import Optional

from pead.sub_sampling_ml.config import build_parser, config_from_args
from pead.sub_sampling_ml.dataset import feature_columns
from pead.sub_sampling_ml.features import CATEGORICAL_FEATURES
from pead.sub_sampling_ml.model import QUARTER_COL
from pead.sub_sampling_ml.serving import backtest as backtest_mod
from pead.sub_sampling_ml.serving import train_final as train_mod
from pead.sub_sampling_ml.serving import tune as tune_mod


def build_cli_parser() -> argparse.ArgumentParser:
    """Return the ``run_train_drift_model`` argument parser (doc S3.1 / S3.4)."""
    p = build_parser(
        prog="run_train_drift_model",
        description="Tune, backtest, and fit the frozen drift-serving model bundle.")
    p.add_argument("--universe", default="ALL",
                   help="Ticker group (SP500|R1000|R2000|ALL) restricting training.")
    p.add_argument("--cutoff", default=date.today().isoformat(),
                   help="Only train on events with anndats <= cutoff (YYYY-MM-DD).")
    p.add_argument("--out", default="models",
                   help="Root directory for versioned model bundles.")
    p.add_argument("--no-tune", action="store_true",
                   help="Skip the Optuna hyperparameter search and use the legacy "
                        "hardcoded LightGBM defaults (ignored when --best-params "
                        "is given).")
    p.add_argument("--n-trials", type=int, default=400,
                   help="Optuna trial budget for the hyperparameter search.")
    p.add_argument("--timeout-hours", type=float, default=8.0,
                   help="Wall-clock budget, in hours, for the hyperparameter search.")
    p.add_argument("--best-params", default=None,
                   help="Path to a best_params.json from a previous search; reuse "
                        "it and skip the search entirely.")
    p.add_argument("--test-frac", type=float, default=0.25,
                   help="Fraction of distinct event-quarters held out as the "
                        "honest test set (doc S3.4 Job 2).")
    p.add_argument("--study-path", default=None,
                   help="SQLite path backing a resumable Optuna study.")
    p.add_argument("--no-backtest", action="store_true",
                   help="Skip the walk-forward backtest / model_card.pdf (PDF #1); "
                        "it runs by default.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_cli_parser().parse_args(argv)
    cfg = config_from_args(args)

    # Mirror train_final's own restriction so tuning/backtest see exactly the
    # training population the shipped bundle will (minus nothing -- Job 3
    # refits on everything, tuning pool AND held-out test quarters, below).
    df = train_mod._restrict_cutoff(
        train_mod._restrict_universe(train_mod._load_or_build(cfg), args.universe),
        args.cutoff, cfg.primary_horizon)
    if df.empty:
        raise SystemExit("Training set is empty -- widen universe or cutoff.")

    feature_cols = feature_columns(df)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    tuning_quarters, test_quarters = tune_mod.split_tuning_test(
        df[QUARTER_COL], args.test_frac)

    tag_dir = Path(args.out) / f"{args.cutoff}_pending"
    # Clear leftovers from an interrupted earlier run so stale model_card.pdf /
    # backtest_results.csv / best_params.json can't be moved into this run's
    # final bundle.
    if tag_dir.exists():
        shutil.rmtree(tag_dir, ignore_errors=True)
    tag_dir.mkdir(parents=True, exist_ok=True)

    best_params = None
    tune_result = None
    if args.best_params:
        payload = json.loads(Path(args.best_params).read_text())
        best_params = payload["best_params"]
        # Carry the prior search's scorecard through to PDF #1's hyperparameter
        # panel even though we never re-ran the search this time.
        tune_result = types.SimpleNamespace(
            best_params=best_params,
            best_value=payload.get("best_value"),
            baseline_value=payload.get("baseline_value"),
            n_trials=payload.get("n_trials"),
        )
    elif not args.no_tune:
        tuning_df = df[df[QUARTER_COL].astype(str).isin(
            {str(q) for q in tuning_quarters})]
        tune_result = tune_mod.tune_hyperparameters(
            tuning_df, feature_cols, cat_cols, cfg,
            n_trials=args.n_trials, timeout=int(args.timeout_hours * 3600),
            study_path=args.study_path)
        best_params = tune_result.best_params
        tune_mod.save_best_params(tune_result, tag_dir / "best_params.json")
    # else: --no-tune with no --best-params -> best_params/tune_result stay
    # None, i.e. the legacy hardcoded-defaults path.

    oos = None
    if not args.no_backtest:
        oos = backtest_mod.run_backtest(
            cfg, universe=args.universe, out_dir=tag_dir,
            params=best_params, tune_result=tune_result, test_quarters=test_quarters)

    model = train_mod.train_final(
        cfg, cutoff_date=args.cutoff, universe=args.universe,
        out_root=args.out, oos_metrics=oos, params=best_params)

    final_dir = train_mod._version_dir(args.out, model.metadata)
    for fname in ("backtest_results.csv", "model_card.pdf", "best_params.json"):
        src = tag_dir / fname
        if src.is_file():
            shutil.move(str(src), str(final_dir / fname))
    try:
        tag_dir.rmdir()
    except OSError:
        pass

    msg = f"[drift-serve] Done. Bundle -> {final_dir}"
    best_value = getattr(tune_result, "best_value", None) if tune_result else None
    baseline_value = getattr(tune_result, "baseline_value", None) if tune_result else None
    if isinstance(best_value, (int, float)) and isinstance(baseline_value, (int, float)):
        msg += (f" | tuned pinball {best_value:.4f} vs "
                f"baseline {baseline_value:.4f}")
    print(msg)


if __name__ == "__main__":
    main()
