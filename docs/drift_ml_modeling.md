# Drift-ML Modeling: From Attribution Study to a Deployable Predictor

Status: **DRAFT / for discussion.** This spec proposes how to turn the existing
sub-sample drift-attribution research pipeline (`pead/sub_sampling_ml`, see
`docs/subsample_drift_ml.md`) into a **trained model that predicts the expected
post-earnings drift for a single new event**, given its point-in-time (PIT)
features at announcement.

Nothing here is built yet. Sections marked **[DECISION]** are design forks; the
choices below are now **locked** (see §0). Each section retains its rationale.

---

## 0. Resolved decisions (locked)

1. **Target (§4): raw-drift quantile models.** Train LightGBM quantile
   regressors on `drift_raw` at p10/p25/p50/p75/p90. Serve **p50 as expected
   drift** and **[p10,p90]** as the range. Keep the existing `drift_z` model for
   ranking/research, and expose **p(up)** from the `drift_class` classifier.
2. **Cadence (§5): one frozen fit** on all history ≤ cutoff, cutoff stamped in
   metadata, plus an `--as-of` flag that rebuilds the model as of a past date
   (same purged/embargoed splits) for honest backtesting.
3. **Inference (§6): single-event `(ticker, anndate)` is the primary caller
   path**; batch mode is a wrapper over the same core. The caller always
   supplies a specific announcement — the model never has to discover an
   announcement date. Surprise inputs (`actual`, `meanest`, `stdev`, `numest`)
   are auto-pulled from the IBES cache when available; the CLI prompts
   *"cache found — use cached values or supply your own?"*, and manual override
   flags are always accepted.
4. **Universe/horizon (§9): S&P 500, primary 60-day horizon only** for v1.
5. **Artifact location (§9): `PEAD/models/`** (gitignored, local).
6. Serving surface: **CLI + importable Python function** (no REST for v1).
7. Missing-feature policy (§7): score with NaNs (LightGBM-native), attach a
   coverage flag, **warn hard below 70% feature coverage**.
8. **Live WRDS pulls** at inference append incrementally to the shared
   `Data Source/wrds_cache/*.csv` (link tables, `crsp_daily`, `compustat_fundq`,
   `compustat_company`). No separate serving cache.
9. **Artifact format:** LightGBM native text (`Booster.model_to_string()`) for
   each booster + a JSON sidecar for schema, category levels, standardization
   stats, industry-drift history, and training metadata. No pickle.
10. **Backtest + model card ship in v1** alongside the serving CLI.

---

## 1. Goal

Given a *current* earnings event — a ticker, an announcement date/time, and the
surprise — produce:

> "Based on this firm's PIT features, the model expects a 60-day drift of
> **CAR[+1,+60] ≈ +X%**, with a probability of positive drift of **p**, and a
> plausible range of **[lo, hi]**."

This is a **point prediction + uncertainty + direction probability** for one
event, served from a **frozen, versioned model artifact** — not a per-fold
research model that is discarded after scoring.

### Explicit non-goals (for v1)
- Not a trading/backtest engine (P&L, turnover, costs). The decile-spread in the
  report is the only economic proxy we keep.
- Not a real-time/intraday system. Daily granularity, batch or on-demand.
- Not a retraining scheduler/MLOps platform. Retrain is a manual CLI step.

---

## 2. The gap (what exists vs. what's missing)

| Capability | Today | Needed |
|---|---|---|
| PIT feature build | ✅ `dataset.build_event_features` (batch over a date range) | Reuse; add a **single-event / incremental** entry point |
| Leakage-safe labels & CV | ✅ purged, embargoed walk-forward | Reuse for **honest backtest of the deployed model** |
| Model fit | ⚠️ `fit_lightgbm_cv` trains **one model per fold, then discards it** | A **final fit** on all data ≤ cutoff, **persisted** |
| Standardization | ⚠️ computed inline per fold, discarded | **Persist** train-window mean/std with the model |
| Categorical mapping | ⚠️ `astype("category")` ad hoc per run | **Persist** category levels so inference matches training |
| Target | ⚠️ `drift_z` (z-scored within calendar quarter) | Add an **interpretable raw-drift target** for serving (see §4) |
| Inference | ❌ none | `predict_drift(...)` API + `predict_drift.py` CLI |
| Artifact / versioning | ❌ none | A `DriftModel` bundle on disk with metadata + git SHA |

The feature/label engine is already PIT-correct and tested, so this is mostly a
**"freeze + serve"** layer, not a rewrite.

---

## 3. Proposed architecture

New module group inside the existing package, so it reuses config, dataset,
features, and WRDS extraction verbatim:

```
pead/sub_sampling_ml/
  serving/
    artifact.py      # DriftModel bundle: save/load, schema, metadata
    train_final.py   # fit the deployable model(s) on all data <= cutoff
    predict.py       # score new event(s) -> expected drift + p(up) + interval
    featurize_one.py # PIT feature build for a live/ad-hoc event (wraps dataset)
run_train_drift_model.py   # CLI: build the frozen artifact
run_predict_drift.py       # CLI: score a ticker+date (or a CSV of events)
models/                    # artifacts (gitignored), versioned by timestamp+sha
```

### 3.1 The `DriftModel` artifact (single source of truth at inference)
A self-contained bundle so inference never needs the training data:
- the fitted LightGBM booster(s) — see §4 for which targets,
- `feature_cols` (exact order), `cat_cols`, and **persisted category levels**,
- **standardization stats** (per-feature mean/std) from the final train window,
- target metadata: which target, any inverse transform to raw drift,
- training metadata: `start_year`/`end_year`, cutoff date, row count, horizon(s),
  library versions, **git commit SHA**, OOS metrics from the most recent fold,
- the FF12 industry drift history needed to recompute `industry_drift_base` for
  new events (a small per-industry table of closed-event drifts).

**[DECISION] Serialization format.** Recommend `joblib` of a dataclass holding
LightGBM `Booster`s (via `booster.model_to_string()`) + a JSON sidecar of
metadata for human/diff-friendliness. Alternative: pure-LightGBM text + JSON
(no pickle at all). Pickle/joblib of sklearn wrappers is simplest but version-
fragile. *Recommendation: LightGBM native text + JSON sidecar.*

---

## 4. [DECISION] Target representation — the central choice

The research model targets `drift_z` (cross-sectional z-score **within a
calendar quarter**). For a *single live event* there is no quarter cross-section
to standardize against, so a z-prediction is not directly interpretable as a
percentage.

Options:

- **(A) Train a second regressor on `drift_raw`** (the actual CAR[+1,+H], in
  return units). Output is directly "+3.2%". Keep the `drift_z` model for
  ranking/research. *Most interpretable for your stated goal.*
- **(B) Predict `drift_z`, then invert** using a persisted estimate of the
  cross-sectional mean/std (e.g. trailing-quarter average). Adds an assumption
  about the current cross-section that we can't fully observe for one event.
- **(C) Quantile regression on `drift_raw`** (LightGBM `objective="quantile"` at
  e.g. 0.1/0.5/0.25/0.75/0.9). Gives the **median point estimate + a prediction
  interval** in one family — directly answers "what should the drift look like."

**Recommendation: (A) + (C) together** — train raw-drift quantile models
(p10/p25/p50/p75/p90). Report **p50 as the expected drift** and **[p10,p90]** as
the range. Keep the existing `drift_z` model untouched for the attribution report.
Also expose **p(up)** from the existing `drift_class` classifier.

Trade-off to acknowledge up front: OOS skill is **modest** (h60 Spearman IC
≈ 0.04, decile spread ≈ +0.17, classifier ROC-AUC ≈ 0.55 on S&P 500). The
predictor is a **weak, diversified signal**, not a precise per-name forecast.
The interval (p10–p90) will be wide; that is honest and intended.

---

## 5. [DECISION] Training cadence & data cutoff

The deployed model must only see data **before** the event it scores.

- **(A) Single frozen fit on all history** up to a cutoff; retrain manually
  (e.g. quarterly) by re-running the train CLI. Simplest, reproducible.
- **(B) Rolling/expanding auto-retrain** at score time (train on everything
  strictly before the new event's quarter). Most "live-honest" but slower and
  non-deterministic per call.

**Recommendation: (A)** with the cutoff stamped into the artifact metadata, plus
a `--as-of` flag so a backtest can rebuild the model as of any past date using
the **same purged-split discipline**. This keeps live inference fast and
reproducible while preserving an honest backtest path.

---

## 6. [DECISION] Inference input mode

- **(A) Batch scoring of a provided event table** (CSV/parquet of
  ticker+anndate+[surprise]). We PIT-featurize each row and score. Lowest new
  plumbing — reuses `build_event_features` over the supplied events.
- **(B) Live single-event** `predict_drift(ticker, anndate)`: pulls the latest
  CRSP/Compustat/IBES as-of that date, builds the one-row feature vector, scores.
  More convenient, but needs recent price history + a WRDS round-trip at call
  time (cached).

**Recommendation: build (A) first** (it's a thin wrapper over existing batch
code and immediately useful), then **(B)** as `featurize_one.py` on top. Both
share the same `DriftModel.predict(X)` core.

`industry_drift_base` for a new event is recomputed from the artifact's stored
industry history (only prior **closed** events), preserving causality.

---

## 7. [DECISION] Uncertainty & output schema

Proposed prediction record per event:

```json
{
  "ticker": "AAPL", "anndate": "2025-01-30", "horizon": 60,
  "expected_drift": 0.031,        // p50, raw CAR[+1,+60]
  "interval_80": [-0.06, 0.13],   // [p10, p90]
  "prob_up": 0.57,                // from drift_class classifier
  "drift_z_pred": 0.22,           // research model, for ranking
  "model_version": "2025q1_<sha>",
  "n_features_present": 39, "n_features_total": 42
}
```

`n_features_present` matters: a live event may be missing WRDS fundamentals
(timing of the latest filing). We should **flag low-coverage predictions** rather
than silently imputing.

**[DECISION] Missing-feature policy at inference.** LightGBM handles NaN
natively, so we *can* score with gaps. Recommend: score anyway, but attach a
coverage flag and refuse (or warn hard) below a threshold (e.g. <70% present).

---

## 8. Honest evaluation of the *deployed* model

Distinct from the attribution report's OOS metrics:
- **Walk-forward backtest of the frozen pipeline**: for each test quarter,
  train-as-of that quarter (purged/embargoed), predict raw drift, and report
  calibration (predicted vs realized deciles), interval coverage (does ~80% of
  realized drift fall in [p10,p90]?), and `prob_up` reliability curve.
- Persist a one-page **model card** (metrics, coverage, calibration, caveats)
  next to the artifact.

---

## 9. File-structure / deployment questions (need your input)

1. **Artifact location.** `PEAD/models/` (gitignored, local) — or a shared drive
   under `Data Source/` like the WRDS cache, so multiple machines reuse it?
2. **Serving surface.** CLI only for now? Or also a thin Python function others
   import? (A REST endpoint is out of scope unless you want it.)
3. **Horizons.** Serve only the primary 60-day, or all of 60/20/5?
4. **Universe at train time.** Freeze on S&P 500, or full IBES universe (broader,
   slower, but more events → likely better generalization)?
5. **Retrain trigger.** Manual quarterly is assumed; want a documented checklist
   or leave it ad hoc?

---

## 10. Proposed build order (once decisions are settled)

1. `artifact.py` — `DriftModel` bundle (save/load, schema, metadata) + tests.
2. `train_final.py` + `run_train_drift_model.py` — fit & persist raw-drift
   quantile models, z model, classifier; stamp metadata + git SHA.
3. `predict.py` + `run_predict_drift.py` — batch scoring (mode A) + output schema.
4. `featurize_one.py` — live single-event featurization (mode B).
5. Backtest/calibration harness + model card (§8).
6. CI: a fast synthetic train→save→load→predict round-trip test.

---

## 11. Open decisions summary (please weigh in)

- §4 Target: raw-drift **quantile** models (recommended) vs z-only vs invert-z.
- §5 Cadence: single frozen fit + `--as-of` backtest (recommended) vs auto-retrain.
- §6 Input: batch table first (recommended) then live single-event.
- §7 Missing-feature policy + coverage threshold.
- §9 Artifact location, serving surface, horizons, train universe, retrain policy.
