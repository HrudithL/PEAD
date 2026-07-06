"""CLI: fit and persist the frozen drift-serving model bundle.

Examples
--------
  python run_train_drift_model.py                        # SP500, cutoff=today
  python run_train_drift_model.py --universe SP500 --cutoff 2024-12-31
  python run_train_drift_model.py --out models --with-backtest
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pead.sub_sampling_ml.config import build_parser, config_from_args
from pead.sub_sampling_ml.serving import backtest as backtest_mod
from pead.sub_sampling_ml.serving import train_final


def main() -> None:
    p = build_parser(
        prog="run_train_drift_model",
        description="Fit and persist the frozen drift-serving model bundle.")
    p.add_argument("--universe", default="SP500",
                   help="Ticker group (SP500|R1000|R2000|ALL) restricting training.")
    p.add_argument("--cutoff", default=date.today().isoformat(),
                   help="Only train on events with anndats <= cutoff (YYYY-MM-DD).")
    p.add_argument("--out", default="models",
                   help="Root directory for versioned model bundles.")
    p.add_argument("--with-backtest", action="store_true",
                   help="Run walk-forward backtest into the same bundle dir.")
    args = p.parse_args()
    cfg = config_from_args(args)

    oos = None
    tag_dir = Path(args.out) / f"{args.cutoff}_pending"
    if args.with_backtest:
        oos = backtest_mod.run_backtest(cfg, universe=args.universe,
                                        out_dir=tag_dir)

    model = train_final.train_final(
        cfg, cutoff_date=args.cutoff, universe=args.universe,
        out_root=args.out, oos_metrics=oos)

    if args.with_backtest:
        import shutil
        final_dir = train_final._version_dir(args.out, model.metadata)
        for fname in ("backtest_results.csv", "model_card.pdf"):
            src = tag_dir / fname
            if src.is_file():
                shutil.move(str(src), str(final_dir / fname))
        try:
            tag_dir.rmdir()
        except OSError:
            pass

    print("[drift-serve] Done.")


if __name__ == "__main__":
    main()
