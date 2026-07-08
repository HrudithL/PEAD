"""Serving layer: turn the drift-attribution research pipeline into a frozen,
versioned predictor answering *"given this ticker+announcement, what drift
should we expect?"*

See ``docs/drift_ml_modeling.md`` for the design; §0 lists the locked decisions.

Modules
-------
* :mod:`artifact`         -- :class:`DriftModel` bundle (save/load, schema).
* :mod:`train_final`      -- fit all frozen models on data <= cutoff.
* :mod:`featurize_one`    -- PIT feature row for a single ``(ticker, anndate)``.
* :mod:`wrds_incremental` -- per-ticker append to the shared WRDS CSV cache.
* :mod:`predict`          -- score single event or batch; output schema.
* :mod:`backtest`         -- walk-forward calibration + model card.
"""

from __future__ import annotations

from .artifact import DriftModel, DEFAULT_QUANTILES

__all__ = ["DriftModel", "DEFAULT_QUANTILES"]
