"""``DriftModel`` bundle: self-contained inference artifact on disk.

Serialization (locked, §0.9): each LightGBM booster as native text via
``Booster.model_to_string()``; a JSON sidecar carries everything else -- the
feature schema and column order, categorical levels, standardization stats,
training metadata (cutoff, git SHA, universe, OOS metrics), and the
per-industry history needed to recompute :func:`industry_drift_base` for a new
event. No pickle: the bundle is human-diffable and version-robust.

Directory layout under ``PEAD/models/<version>/`` (all files small)::

    manifest.json
    booster_q10.txt   booster_q25.txt   booster_q50.txt
    booster_q75.txt   booster_q90.txt
    booster_z.txt
    booster_class.txt
    industry_history.csv
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)

# Files inside a bundle directory.
_MANIFEST = "manifest.json"
_INDUSTRY_HISTORY = "industry_history.csv"


def _quantile_name(q: float) -> str:
    """``0.10 -> 'q10'``. Uses `round(q*100)` to avoid ``0.1`` float noise."""
    return f"q{int(round(q * 100))}"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],  # PEAD/
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


@dataclass
class TrainingMetadata:
    """What the model was trained on / with. Stamped into ``manifest.json``."""

    cutoff_date: str            # inclusive: all events with anndats <= cutoff
    universe: str               # e.g. "sp500"
    horizon: int                # trading days, primary horizon
    quantiles: tuple[float, ...]
    n_events: int
    start_year: int
    end_year: int
    embargo_months: int
    trained_at_utc: str
    git_sha: str
    lightgbm_version: str
    pandas_version: str
    numpy_version: str
    oos_metrics: dict = field(default_factory=dict)  # from the backtest, optional


@dataclass
class FeatureSchema:
    """Everything needed to rebuild the exact inference feature matrix.

    ``category_levels`` maps a categorical column to the ordered list of levels
    seen at training time; unknown levels at inference become NaN (LightGBM
    native missing). ``mean``/``std`` are per-numeric-feature standardization
    stats fit on the final train window (all events <= cutoff).
    """

    feature_cols: list[str]
    cat_cols: list[str]
    numeric_cols: list[str]
    category_levels: dict[str, list]
    mean: dict[str, float]
    std: dict[str, float]


@dataclass
class DriftModel:
    """Frozen predictor bundle. Instantiate via :meth:`load` or :meth:`build`."""

    schema: FeatureSchema
    metadata: TrainingMetadata
    quantile_boosters: dict[str, object]   # {"q10": Booster, ...}
    z_booster: Optional[object] = None     # LightGBM regressor on drift_z
    class_booster: Optional[object] = None # LightGBM classifier on drift_class
    industry_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------ save
    def save(self, out_dir: str | os.PathLike) -> Path:
        """Persist the bundle to ``out_dir/`` (created if missing)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, booster in self.quantile_boosters.items():
            (out / f"booster_{name}.txt").write_text(booster.model_to_string())
        if self.z_booster is not None:
            (out / "booster_z.txt").write_text(self.z_booster.model_to_string())
        if self.class_booster is not None:
            (out / "booster_class.txt").write_text(
                self.class_booster.model_to_string())

        if not self.industry_history.empty:
            self.industry_history.to_csv(out / _INDUSTRY_HISTORY, index=False)

        manifest = {
            "schema": asdict(self.schema),
            "metadata": asdict(self.metadata),
            "quantiles_present": sorted(self.quantile_boosters.keys()),
            "has_z_booster": self.z_booster is not None,
            "has_class_booster": self.class_booster is not None,
            "industry_history_rows": int(len(self.industry_history)),
        }
        (out / _MANIFEST).write_text(json.dumps(manifest, indent=2, default=str))
        return out

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, in_dir: str | os.PathLike) -> "DriftModel":
        """Rehydrate a bundle previously written by :meth:`save`."""
        import lightgbm as lgb

        inp = Path(in_dir)
        manifest = json.loads((inp / _MANIFEST).read_text())

        schema = FeatureSchema(**manifest["schema"])
        metadata = TrainingMetadata(**manifest["metadata"])

        quantile_boosters: dict[str, object] = {}
        for name in manifest["quantiles_present"]:
            path = inp / f"booster_{name}.txt"
            quantile_boosters[name] = lgb.Booster(model_str=path.read_text())

        z_booster = None
        if manifest.get("has_z_booster"):
            z_booster = lgb.Booster(model_str=(inp / "booster_z.txt").read_text())
        class_booster = None
        if manifest.get("has_class_booster"):
            class_booster = lgb.Booster(
                model_str=(inp / "booster_class.txt").read_text())

        industry_history = pd.DataFrame()
        hist_path = inp / _INDUSTRY_HISTORY
        if hist_path.is_file():
            industry_history = pd.read_csv(hist_path,
                                           parse_dates=["close_date"])
        return cls(schema=schema, metadata=metadata,
                   quantile_boosters=quantile_boosters,
                   z_booster=z_booster, class_booster=class_booster,
                   industry_history=industry_history)

    # -------------------------------------------------------- feature prep
    def prepare_X(self, feats: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Align an inference feature frame to the training schema.

        - Adds any missing columns as NaN, drops extras, reorders.
        - Applies persisted mean/std z-scoring to numeric columns.
        - Encodes categoricals with the training-time level order (unknown
          levels -> NaN so LightGBM treats them as missing).
        Returns ``(X, coverage)`` where coverage is the fraction of
        ``feature_cols`` that had a non-null value in ``feats``.
        """
        X = pd.DataFrame(index=feats.index)
        # coverage over original (pre-alignment) values
        present = 0
        for col in self.schema.feature_cols:
            if col in feats.columns and feats[col].notna().any():
                present += 1
        coverage = present / max(1, len(self.schema.feature_cols))

        for col in self.schema.numeric_cols:
            s = pd.to_numeric(feats.get(col, pd.Series(np.nan, index=feats.index)),
                              errors="coerce")
            mu = self.schema.mean.get(col, 0.0)
            sd = self.schema.std.get(col, 1.0) or 1.0
            X[col] = (s - mu) / sd

        for col in self.schema.cat_cols:
            levels = self.schema.category_levels.get(col, [])
            raw = feats.get(col, pd.Series(pd.NA, index=feats.index))
            X[col] = pd.Categorical(raw, categories=levels)

        # Keep exact training column order.
        X = X[self.schema.feature_cols]
        return X, coverage

    # ------------------------------------------------------------ predict
    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Score prepared features and return the per-event output schema."""
        X, coverage = self.prepare_X(feats)

        out = pd.DataFrame(index=feats.index)
        # Quantile predictions.
        for name, booster in sorted(self.quantile_boosters.items()):
            out[f"drift_raw_pred_{name}"] = booster.predict(X)

        if "q50" in self.quantile_boosters:
            out["expected_drift"] = out["drift_raw_pred_q50"]
        if "q10" in self.quantile_boosters and "q90" in self.quantile_boosters:
            out["interval_low"] = out["drift_raw_pred_q10"]
            out["interval_high"] = out["drift_raw_pred_q90"]

        # Enforce monotonicity of the quantile family: per-row sort so
        # crossings from independently trained boosters cannot flip an
        # interval inside-out.
        qcols = [c for c in out.columns if c.startswith("drift_raw_pred_q")]
        if qcols:
            sorted_vals = np.sort(out[qcols].to_numpy(), axis=1)
            out[qcols] = sorted_vals

        if self.z_booster is not None:
            out["drift_z_pred"] = self.z_booster.predict(X)
        if self.class_booster is not None:
            out["prob_up"] = self.class_booster.predict(X)

        out["coverage"] = coverage
        out["n_features_present"] = int(round(
            coverage * len(self.schema.feature_cols)))
        out["n_features_total"] = len(self.schema.feature_cols)
        out["model_version"] = self._version_tag()
        out["horizon"] = self.metadata.horizon
        return out

    # ------------------------------------------------------------ metadata
    def _version_tag(self) -> str:
        try:
            ts = pd.Timestamp(self.metadata.cutoff_date).strftime("%Yq%q")
        except Exception:
            ts = "asof"
        return f"{ts}_{self.metadata.git_sha}"


def make_metadata(*, cutoff_date: str, universe: str, horizon: int,
                  quantiles: tuple[float, ...], n_events: int,
                  start_year: int, end_year: int, embargo_months: int,
                  oos_metrics: Optional[dict] = None) -> TrainingMetadata:
    """Build a :class:`TrainingMetadata` with git SHA + library versions filled in."""
    import lightgbm as lgb

    return TrainingMetadata(
        cutoff_date=str(pd.Timestamp(cutoff_date).date()),
        universe=universe,
        horizon=int(horizon),
        quantiles=tuple(float(q) for q in quantiles),
        n_events=int(n_events),
        start_year=int(start_year),
        end_year=int(end_year),
        embargo_months=int(embargo_months),
        trained_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        lightgbm_version=lgb.__version__,
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        oos_metrics=oos_metrics or {},
    )
