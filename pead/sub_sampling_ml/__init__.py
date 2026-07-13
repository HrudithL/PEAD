"""Sub-sample drift attribution: which features drive post-earnings drift.

The base PEAD study (``pead.equities``) sorts earnings events into surprise
deciles and measures the average drift per decile. This subpackage asks the
next question: *across all drift occurrences, which firm/event characteristics
cause a large upward drift, and which cause a downward (or muted) drift?*

It builds a per-event feature matrix that is strictly point-in-time as of the
announcement, labels each event with its realized post-announcement drift
(market-adjusted ``CAR[+1, +H]``), and fits two complementary models -- an
interpretable Fama-MacBeth cross-sectional regression and a non-linear
gradient-boosted tree with SHAP attribution -- under leakage-aware
purged/embargoed walk-forward cross-validation.

Module map (see ``docs/subsample_drift_ml.md`` for the full design):

* ``config``      -- :class:`DriftMLConfig`, CLI parsing, path/credential wiring.
* ``labels``      -- CAR encodings (raw / z / decile / class) from the AR matrix.
* ``features``    -- point-in-time feature engineering per family.
* ``wrds_extract``-- CRSP/Compustat/IBES pulls via the ``wrds`` package + linking.
* ``dataset``     -- as-of joins assembling ``event_features.parquet``.
* ``model``       -- Fama-MacBeth + LightGBM under purged walk-forward CV.
* ``attribution`` -- SHAP, permutation, and cross-fold consistency aggregation.
* ``report``      -- the PDF deliverable.
"""

from __future__ import annotations

from .config import DriftMLConfig

__all__ = ["DriftMLConfig"]
