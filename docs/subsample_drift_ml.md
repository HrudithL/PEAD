# Sub-sample analysis — what drives post-earnings drift (feature-attribution ML)

## 1. Objective

The base study sorts earnings events into surprise deciles and measures the
average drift per decile. This sub-sample study asks the next question:

> **Across all drift occurrences, which firm/event characteristics cause a large
> upward drift, and which cause a downward (or muted) drift?**

We are not trying to label specific firms as "permanently top decile." We are
building a **map of features → drift**: a model that, given what is knowable
*before* the drift begins, explains and predicts how strongly a stock drifts
after its earnings announcement, and tells us *which characteristics matter most
and most consistently*.

"Consistency" here is a property of a **feature**, not a firm: a feature is
consistent if it ranks as a top driver of drift repeatedly — across time folds,
across sectors, and across surprise regimes — not just once.

## 2. Unit of analysis & sample

- **One sample = one quarterly earnings event** `(firm, announcement)`.
- Each event has: a label (the realized drift) and a point-in-time feature
  vector (everything known at/just-before the announcement).
- **Universe:** full available universe (all tickers present in IBES + price
  data), 2015–2024 by default. No `--tickers` restriction.
- This reuses the existing event construction (`pead/equities/data_loader.py`,
  `surprise.py`, `event_study.py`); we only add a per-event label column and a
  feature matrix.

## 3. Target / label — "the drift"

Per the design decisions:

- **Drift = market-adjusted `CAR[+1, +60]`** — cumulative abnormal return from
  trading day **+1 through +60**, *excluding the day-0 announcement jump*. This
  isolates the post-announcement *drift* from the immediate reaction ("the work
  done before the drift").
- **Signed**, not absolute. Large positive = "good"/strong upward drift; large
  negative = "bad"/downward drift.
- Abnormal = stock return − SPY return (existing `benchmark="spy"` convention).
- Computed directly from the existing AR matrix: with `window_pre=5,
  window_post=60`, the label is `sum(AR[offset] for offset in 1..60)` per event
  (`event_study.compute_ar_matrix` already produces this matrix).

Two label encodings are produced from the same raw CAR:

| Encoding | Definition | Used for |
|---|---|---|
| `drift_raw` | market-adjusted CAR[+1,+60] | regression target |
| `drift_z` | cross-sectional z-score of `drift_raw` within each calendar quarter | regression target, regime-neutral |
| `drift_decile` | per-quarter decile (1–10) of `drift_raw` | descriptive / ranking views |
| `drift_class` | top decile = 1, bottom decile = 0 (middle dropped) | classifier target |

Quarter-relative encodings (`drift_z`, `drift_decile`) remove market-wide
regime shifts so the model learns the *cross-sectional* drivers rather than
calendar effects. The decile view is how we connect back to the original
"deciles of drift" framing — but ranking is now on **realized drift**, not
surprise.

**Robustness on the horizon:** the pipeline will also emit CAR[+1,+20] and
CAR[+1,+5] labels so we can confirm a driver is not an artifact of one horizon.

## 4. Point-in-time discipline (no look-ahead)

Every feature must be **knowable strictly before trading day +1**, otherwise the
"prediction" is leakage. Rules:

1. **Fundamentals** (Compustat): use only the most recent fiscal period whose
   public report date precedes the announcement. Quarterly facts use `RDQ`
   (report date) `< anndats`; annual facts are lagged with a reporting gap.
2. **Analyst data:** use the last consensus snapshot `STATPERS < ANNDATS_ACT`
   (already enforced in `load_events`).
3. **Price/return features:** computed over windows ending on or before day 0.
4. **The announcement-day reaction itself** (day-0 and day +1 open jump) ends
   *before* the [+1,+60] drift window, so the **announcement abnormal return
   (EAR)** is a legitimate, and likely powerful, point-in-time feature.

Leakage is also controlled at the **validation** level (Section 8) because each
label spans 60 future trading days and would otherwise overlap neighbouring
events' feature windows.

## 5. Feature catalog

Grouped by family. "Source" → `IBES`/`px` already in repo; `CRSP`,
`Compustat`, `OptionMetrics`, `ref` (constituent lists / SIC) require a WRDS
extract (Section 6). All features are point-in-time as of the announcement.

### 5.1 Earnings / analyst signal (the trigger)
| Feature | Definition | Source |
|---|---|---|
| `sue_std` | (ACTUAL − MEANEST) / STDEV of estimates | IBES |
| `sue_price` | (ACTUAL − MEANEST) / pre-price | IBES + px |
| `surprise_raw` | ACTUAL − MEANEST | IBES |
| `analyst_disp` | STDEV / |MEANEST| (forecast disagreement) | IBES |
| `n_analysts` | NUMEST (coverage) | IBES |
| `revision_net` | (NUMUP − NUMDOWN) / NUMEST in run-up | IBES |
| `forecast_staleness` | days between STATPERS and anndats | IBES |
| `beat_flag` | sign(surprise); beat / meet / miss | IBES |
| `ear` | **announcement abnormal return**, day-0 (and +1 open) | px |

### 5.2 Price / return & risk (pre-event)
| Feature | Definition | Source |
|---|---|---|
| `mom_12_1` | cumulative return [-252, -21] | CRSP/px |
| `mom_1m` | cumulative return [-21, -1] | CRSP/px |
| `reversal_1w` | return [-5, -1] | CRSP/px |
| `rvol_60` | realized daily-return vol, 60d | CRSP/px |
| `beta_252` | market-model beta vs SPY, 252d | CRSP/px |
| `ivol` | idiosyncratic vol (resid of market model) | CRSP/px |
| `high52_prox` | price / 52-week high | CRSP/px |
| `pre_run` | abnormal return [-5, -1] into the print | px |

### 5.3 Liquidity / size
| Feature | Definition | Source |
|---|---|---|
| `mktcap` | shares outstanding × price (log) | CRSP/Compustat |
| `dollar_vol` | mean daily $ volume, 60d (log) | CRSP |
| `turnover` | volume / shares outstanding | CRSP |
| `amihud` | mean(|ret| / $vol), illiquidity | CRSP |
| `price_level` | log close price | px |
| `idx_member` | S&P500 / Russell1000 / Russell2000 membership | ref |

### 5.4 Sector / industry (required)
| Feature | Definition | Source |
|---|---|---|
| `gics_sector` | GICS sector (categorical) | Compustat |
| `ff12` / `ff48` | Fama-French industry from SIC | SIC map |
| `industry_mom` | recent peer-industry return | CRSP + map |
| `industry_drift_base` | trailing avg PEAD of the industry (PIT, expanding) | derived |

### 5.5 Fundamentals / firm quality (WRDS — Compustat)
| Feature | Definition | Source |
|---|---|---|
| `book_to_market` | book equity / market cap | Compustat + CRSP |
| `roa` | net income / assets | Compustat |
| `gross_margin` | (rev − cogs) / rev | Compustat |
| `leverage` | total debt / assets | Compustat |
| `asset_growth` | yoy Δ total assets | Compustat |
| `sales_growth` | yoy Δ revenue | Compustat |
| `accruals` | balance-sheet accruals / assets | Compustat |
| `n_employees` | EMP (count, log) | Compustat |
| `earn_vol` | std of past 8q earnings | Compustat (fundq) |
| `rd_intensity` | R&D / sales | Compustat |

### 5.6 Options surface — DEFERRED to v2 (not in initial build)
The options-surface family is **out of scope for the first build** and listed
only so the design leaves room for it. Do not extract or model these until the
equities-feature model is validated.

| Feature | Definition | Source |
|---|---|---|
| `atm_iv_pre` | pre-event ATM implied vol | OptionMetrics |
| `iv_skew` | put−call IV skew | OptionMetrics |
| `opt_volume` | pre-event option volume / OI | OptionMetrics |
| `implied_move` | straddle-implied move vs realized | OptionMetrics |

### 5.7 Context / calendar
| Feature | Definition | Source |
|---|---|---|
| `fiscal_q` | reporting quarter (seasonality) | IBES |
| `report_time` | BMO vs AMC (ANNTIMS) | IBES |
| `vix_level` | market-implied vol regime at the print | OptionMetrics/ext |
| `mkt_ret_pre` | SPY return [-21,-1] | px |

## 6. Data sources & WRDS extraction plan

Existing repo data (IBES summary, master_stock, constituent lists) covers
families 5.1, parts of 5.2/5.3, and 5.4 partially. Families 5.3 (full), 5.4
(GICS), and 5.5 require WRDS. Plan:

**Access mode (decided):** pull via the **`wrds` Python package** (live
connection) as the primary path. If a needed field is not reachable through the
package, fall back to the **WRDS web/database directly** for the privileged
extract and drop the result into the external data dir the resolver knows about.
Either way, raw WRDS pulls stay outside git; only the derived parquet is cached.

**WRDS tables**
- `crsp.dsf` — daily stock file: returns, prices, volume, shares outstanding →
  size, turnover, Amihud, momentum, vol, beta.
- `comp.fundq` / `comp.funda` — quarterly/annual fundamentals (assets, book
  equity, employees `emp`, revenue, net income, debt, R&D), with `rdq` for PIT.
- `comp.company` — `gsector` (GICS), `sic`.
- `ibes.statsumu_epsus` / `ibes.det_epsus` — richer estimates/revisions if we
  want detail beyond the current summary file.

**Identifier linking** (the main data-engineering task — IBES is keyed by
ticker, the rest by PERMNO/GVKEY):
1. IBES ticker → CRSP PERMNO via `wrdsapps.ibcrsphist` (IBES–CRSP link).
2. PERMNO → Compustat GVKEY via `crsp.ccmxpf_lnkhist` (CCM link, date-valid).
3. Carry SIC/GICS from `comp.company`.

Extraction lands in `data/derived/` as compact parquet (same data-plane
convention as the rest of the repo — raw WRDS pulls stay outside git). One
firm-quarter fundamentals panel + one daily CRSP panel, both PIT-joined onto the
event table by `(permno, date<=anndats)` as-of merges.

## 7. Modeling approach

Two complementary models on the same event-feature-label table:

**A. Interpretable baseline — Fama-MacBeth cross-sectional regressions.**
For each calendar quarter, regress `drift_z` on the standardized features; average
the coefficients across quarters and report Newey-West t-stats. This gives a
classical, leakage-free statement of "a 1-σ increase in feature X moves drift by
β, and the effect is significant across quarters" — and the per-quarter
coefficient series *is* the consistency evidence for linear effects.

**B. Non-linear model + attribution — gradient-boosted trees (LightGBM).**
- Regressor on `drift_z` (primary) and classifier on `drift_class`
  (top vs bottom drift decile).
- Native categorical handling for `gics_sector`; high-cardinality categoricals
  target-encoded *inside* CV folds only.
- **SHAP** values for global importance, sign, and dependence (non-linear /
  interaction effects), plus permutation importance as a cross-check.

Descriptive layer (per decision "Both"): top-decile vs bottom-decile feature
distributions with effect sizes and tests — a plain-language table of "strong
upward-drift firms look like ___; downward-drift firms look like ___."

## 8. Validation — leakage-aware

Because each label spans 60 future trading days, naive k-fold leaks. We use:

- **Purged, embargoed walk-forward CV** (López de Prado): train on past
  quarters, test on the next, **purge** any training event whose [+1,+60] window
  overlaps the test period, and **embargo** ~3 months after each test block.
- Features cross-sectionally standardized **within the training window only**.
- All time-dependent encodings (target encoding, `industry_drift_base`) fit on
  train, applied to test.

**Metrics (out-of-sample):**
| Metric | Question it answers |
|---|---|
| Spearman rank IC (pred vs realized drift) | does the model rank drift correctly? |
| OOS R² on `drift_z` | how much drift variance is explained? |
| Top–bottom decile spread of model predictions | economic payoff of acting on it |
| Classifier AUC / PR-AUC | can it separate strong-up vs down drift? |

## 9. Feature-consistency / robustness protocol

This operationalises the user's notion of "consistency" (a feature repeatedly
shown most important):

1. Record SHAP/permutation importance **per walk-forward fold** → a feature is
   "consistent" if it stays in the top-k across most folds (report rank
   stability, e.g. mean rank + dispersion).
2. Re-fit **within each GICS sector** and **within size terciles** and
   **within surprise-sign groups** → confirm a driver is broad, not a one-sector
   artifact.
3. Re-run on the alternate horizons (CAR[+1,+20], [+1,+5]).
4. Compare A (Fama-MacBeth signs/t-stats) vs B (SHAP signs) — agreement is
   strong evidence; disagreement flags non-linearity worth a dependence plot.

The headline deliverable is a ranked, sign-annotated, stability-scored table:
**feature → direction of effect on drift → strength → consistency.**

## 10. Outputs / artifacts

- `data/derived/event_features.parquet` — event × feature matrix + labels.
- `outputs/drift_ml_report.pdf` — **PDF report** (matplotlib, matching the repo
  convention): fit metrics, SHAP summary & dependence plots, Fama-MacBeth
  coefficient table, top/bottom descriptive comparison, per-fold consistency
  table.
- `outputs/feature_importance.csv` — the ranked driver map.

## 11. Code structure (decided)

New subpackage `pead/sub_sampling_ml/`, mirroring the existing subpackage style;
reuses `pead/equities` for events and abnormal returns, `pead/io` for data
location. (The package directory uses underscores — `sub_sampling_ml` — because
Python module names cannot contain hyphens and must be importable.)

```
pead/
  sub_sampling_ml/
    __init__.py
    labels.py        # CAR[+1,+60] per event from the AR matrix; encodings
    features.py      # PIT feature engineering per family
    wrds_extract.py  # CRSP/Compustat/IBES pulls (wrds pkg) + identifier linking
    dataset.py       # as-of joins → event_features.parquet
    model.py         # Fama-MacBeth + LightGBM + purged-walk-forward CV
    attribution.py   # SHAP, permutation, consistency aggregation
    report.py        # PDF report
run_drift_ml.py      # CLI entry point
```

## 12. Resolved decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | Package / entry point | `pead/sub_sampling_ml/` + `run_drift_ml.py`. |
| 2 | WRDS access | `wrds` Python package first; privileged WRDS database directly for any field the package can't reach. |
| 3 | Options features (5.6) | **Deferred to v2.** Not extracted or modeled in the initial build. |
| 4 | Report format | **PDF** (matplotlib), matching the repo convention. |

All four blocking questions are resolved; the spec is build-ready. Build begins
only when explicitly instructed.
