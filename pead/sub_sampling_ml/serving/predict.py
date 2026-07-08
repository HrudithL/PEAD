"""Score events against a frozen :class:`DriftModel`.

Two entry points:

* :func:`predict_one` -- single ``(ticker, anndate)`` with optional surprise
  overrides. Returns a compact record + the raw feature row for inspection.
* :func:`predict_events` -- CSV/DataFrame of events; scores rows in a loop
  reusing the same model, one-per-event featurization. Adequate for the S&P
  500-scale universe; a fused batch featurizer is future work.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import DriftMLConfig
from .artifact import DriftModel
from .featurize_one import EventInputs, featurize_one
from .train_final import latest_model_dir


COVERAGE_WARN_THRESHOLD = 0.70   # §0.7


def load_model(models_root: str = "models",
               version: Optional[str] = None) -> DriftModel:
    """Load a bundle from ``models/<version>/`` (default: latest trained)."""
    if version is None:
        latest = latest_model_dir(models_root)
        if latest is None:
            raise SystemExit(
                f"No trained model under {models_root}/. Run "
                "run_train_drift_model.py first.")
        return DriftModel.load(latest)
    return DriftModel.load(Path(models_root) / version)


def _to_record(prediction: pd.DataFrame, ticker: str, anndate,
               used_cache: bool) -> dict:
    """Flatten the single-row prediction to a plain dict for CLI/JSON output."""
    row = prediction.iloc[0].to_dict()
    record = {
        "ticker": ticker.upper(),
        "anndate": pd.Timestamp(anndate).strftime("%Y-%m-%d"),
        "horizon": int(row.pop("horizon")),
        "expected_drift": float(row.get("expected_drift", float("nan"))),
        "interval_80": [
            float(row.get("interval_low", float("nan"))),
            float(row.get("interval_high", float("nan"))),
        ],
        "prob_up": float(row["prob_up"]) if "prob_up" in row else None,
        "drift_z_pred": float(row["drift_z_pred"]) if "drift_z_pred" in row else None,
        "coverage": float(row.get("coverage", float("nan"))),
        "n_features_present": int(row.get("n_features_present", 0)),
        "n_features_total": int(row.get("n_features_total", 0)),
        "model_version": str(row.get("model_version", "")),
        "used_ibes_cache": bool(used_cache),
        "quantiles": {k.replace("drift_raw_pred_", ""): float(v)
                      for k, v in row.items()
                      if k.startswith("drift_raw_pred_q")},
    }
    return record


def predict_one(user: EventInputs, cfg: DriftMLConfig, *,
                model: Optional[DriftModel] = None,
                models_root: str = "models",
                prefer_cache: bool = True) -> dict:
    """Score one event; returns the flat output record (see :func:`_to_record`)."""
    mdl = model or load_model(models_root)
    fr = featurize_one(user, cfg, model=mdl, prefer_cache=prefer_cache)
    pred = mdl.predict(fr.features)
    record = _to_record(pred, user.ticker, user.anndate, fr.used_cache)
    if record["coverage"] < COVERAGE_WARN_THRESHOLD:
        warnings.warn(
            f"[drift-serve] LOW FEATURE COVERAGE "
            f"({record['n_features_present']}/{record['n_features_total']} "
            f"= {record['coverage']:.0%}) for {record['ticker']} "
            f"{record['anndate']}; prediction may be unreliable.",
            stacklevel=2,
        )
    return record


def predict_events(events: pd.DataFrame, cfg: DriftMLConfig, *,
                   model: Optional[DriftModel] = None,
                   models_root: str = "models",
                   prefer_cache: bool = True) -> pd.DataFrame:
    """Batch scoring. ``events`` needs columns ``ticker``, ``anndate``; any of
    ``anntime, fpedats, actual, meanest, stdev, numest, numup, numdown, cname``
    are used as overrides when present.
    """
    mdl = model or load_model(models_root)
    records: list[dict] = []
    for _, r in events.iterrows():
        user = EventInputs(
            ticker=str(r["ticker"]),
            anndate=str(r["anndate"]),
            anntime=_opt(r, "anntime"),
            fpedats=_opt(r, "fpedats"),
            statpers=_opt(r, "statpers"),
            cname=_opt(r, "cname"),
            actual=_opt_num(r, "actual"),
            meanest=_opt_num(r, "meanest"),
            medest=_opt_num(r, "medest"),
            stdev=_opt_num(r, "stdev"),
            numest=_opt_num(r, "numest"),
            numup=_opt_num(r, "numup"),
            numdown=_opt_num(r, "numdown"),
        )
        try:
            record = predict_one(user, cfg, model=mdl, prefer_cache=prefer_cache)
        except Exception as exc:
            record = {
                "ticker": user.ticker.upper(),
                "anndate": user.anndate,
                "error": str(exc),
            }
        records.append(record)
    # Flatten the nested ``quantiles`` dict + ``interval_80`` list.
    out = pd.json_normalize(records)
    for i, side in enumerate(("interval_low", "interval_high")):
        col = "interval_80"
        if col in out.columns:
            out[side] = out[col].apply(
                lambda x: x[i] if isinstance(x, list) and len(x) > i else None)
    if "interval_80" in out.columns:
        out = out.drop(columns="interval_80")
    return out


def _opt(row: pd.Series, name: str) -> Optional[str]:
    if name not in row.index:
        return None
    val = row[name]
    if pd.isna(val):
        return None
    return str(val)


def _opt_num(row: pd.Series, name: str):
    if name not in row.index:
        return None
    val = row[name]
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
