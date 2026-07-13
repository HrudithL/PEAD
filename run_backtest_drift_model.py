"""CLI: honest walk-forward backtest of the frozen drift-serving pipeline (§8).

Emits ``backtest_results.csv`` + ``model_card.pdf`` under the chosen ``--out``.
"""

from __future__ import annotations

from pathlib import Path

from pead.sub_sampling_ml.config import build_parser, config_from_args
from pead.sub_sampling_ml.serving import backtest as backtest_mod


def main() -> None:
    p = build_parser(
        prog="run_backtest_drift_model",
        description="Honest walk-forward backtest of the drift-serving pipeline.")
    p.add_argument("--universe", default="SP500")
    p.add_argument("--out", default="models/backtest_latest")
    args = p.parse_args()
    cfg = config_from_args(args)

    out_dir = Path(args.out)
    summary = backtest_mod.run_backtest(cfg, universe=args.universe,
                                        out_dir=out_dir)
    print("[drift-serve] Backtest summary:")
    for k, v in summary.items():
        print(f"    {k:>22}: {v}")
    print(f"[drift-serve] Artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
