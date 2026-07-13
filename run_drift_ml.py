"""Command-line entry point for the sub-sample drift-attribution study.

Builds a per-event feature matrix that is point-in-time as of each earnings
announcement, labels every event with its realized post-announcement drift
(market-adjusted CAR[+1, +H]), and fits interpretable (Fama-MacBeth) and
non-linear (LightGBM + SHAP) models under leakage-aware purged walk-forward CV
to map *which features drive the drift, in which direction, and how consistently*.

Examples
--------
  python run_drift_ml.py                                   # full universe, defaults
  python run_drift_ml.py --start-year 2018 --horizons 60,20,5
  python run_drift_ml.py --tickers SP500 --no-classifier
  python run_drift_ml.py --no-wrds                         # repo-only features
  python run_drift_ml.py --refresh-wrds                    # re-pull WRDS, refresh cache

See docs/subsample_drift_ml.md for the full design.
"""

from __future__ import annotations

from pead.sub_sampling_ml.config import parse_args
from pead.sub_sampling_ml.pipeline import run


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
