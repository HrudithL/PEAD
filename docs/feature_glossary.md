# Drift-ML Glossary — features, labels, metrics, and plots

A plain-language reference for the sub-sample drift-attribution study
(`pead/sub_sampling_ml`, design in `docs/subsample_drift_ml.md`). It explains
**what every feature means**, **what the model is predicting**, **what each
metric in the PDF report means**, and **how to read each plot**.

The same descriptions are baked into the PDF report and the CSV exports
(`pead/sub_sampling_ml/descriptions.py` is the single source of truth), so this
file and the report never drift apart.

---

## 1. What the study predicts — the label ("the drift")

Every earnings event is labelled with its **realized post-announcement drift**:

| Encoding | Definition | Used for |
|---|---|---|
| `drift_raw` | **Market-adjusted CAR[+1,+H]** — cumulative abnormal return (stock − SPY) from trading day **+1 through +H**, *excluding the day-0 announcement jump*. "The drift" in return units. | regression target (interpretable) |
| `drift_z` | `drift_raw` **z-scored within each calendar quarter** — removes market-wide regime shifts so the model learns *cross-sectional* drivers. | **primary** regression target |
| `drift_decile` | Per-quarter **decile (1–10)** of `drift_raw`. | descriptive / ranking |
| `drift_class` | **1** = top-decile drift, **0** = bottom-decile drift, middle dropped. | classifier target |

Horizons: **H = 60** trading days is primary; **H = 20** and **H = 5** are
robustness checks. The day-0 jump is deliberately excluded so the label captures
the *drift after* the announcement, not the immediate reaction.

---

## 2. Feature catalog — what each input means

All features are **point-in-time**: knowable strictly *before* trading day +1,
so nothing leaks from the future into the prediction.

### 2.1 Earnings / analyst signal (the trigger)

| Feature | Plain meaning | Intuition (higher = …) |
|---|---|---|
| `sue_std` | Standardized unexpected earnings: (actual − mean estimate) / dispersion of estimates. | Bigger, more decisive beat vs. how much analysts disagreed. |
| `sue_price` | Surprise scaled by pre-announcement price: (actual − estimate) / price. | Larger beat in cents per dollar of price. |
| `surprise_raw` | Raw EPS surprise: actual EPS − consensus mean. | Positive = beat; negative = miss. |
| `analyst_disp` | Forecast dispersion: std of estimates / \|mean estimate\|. | Analysts disagreed more (more pre-print uncertainty). |
| `n_analysts` | Number of analysts in the consensus. | More closely followed; drift often larger for less-covered names. |
| `revision_net` | Net upward revisions in the run-up: (up − down) / total. | Analysts were turning more optimistic before the print. |
| `forecast_staleness` | Days between the last consensus snapshot and the announcement. | Consensus was older / less refreshed. |
| `beat_flag` | Sign of the surprise: +1 beat / 0 meet / −1 miss (categorical). | Direction of the surprise, ignoring size. |
| `ear` | **Announcement abnormal return** (day 0, stock − SPY). | Market reacted more positively on the day. Ends before the drift window, so it is a legitimate predictor. |

### 2.2 Price / return & risk (pre-event)

| Feature | Plain meaning | Intuition (higher = …) |
|---|---|---|
| `mom_12_1` | 12-1 momentum: cumulative return over offsets [−252, −21]. | Longer-run winner. |
| `mom_1m` | 1-month return into the print, offsets [−21, −1]. | Recent uptrend. |
| `reversal_1w` | 1-week return, offsets [−5, −1]. | Very-short-term move (often mean-reverts). |
| `rvol_60` | Realized daily-return volatility over the prior 60 days. | More volatile pre-event. |
| `beta_252` | Market-model beta vs SPY over 252 days. | Moves more with the market. |
| `ivol` | Idiosyncratic vol: std of the market-model residual. | More firm-specific noise. |
| `high52_prox` | Price / trailing 52-week high. | Near its yearly high. |
| `pre_run` | Abnormal run-up over offsets [−5, −1] (stock − SPY). | Ran up abnormally just before reporting. |

### 2.3 Liquidity / size

| Feature | Plain meaning | Intuition (higher = …) |
|---|---|---|
| `price_level` | Log closing share price at announcement. | Higher-priced share. |
| `mktcap` | Log market cap (price × shares). | Larger company; drift classically stronger in small caps. |
| `dollar_vol` | Log average daily dollar volume (60d). | More actively traded / liquid. |
| `turnover` | Avg daily volume / shares outstanding (60d). | Shares change hands faster. |
| `amihud` | Amihud illiquidity: avg \|return\| / dollar volume. | More illiquid (price moves on little volume). |
| `idx_member` | Index-membership flag (categorical; not populated in this build). | — |

### 2.4 Sector / industry

| Feature | Plain meaning | Intuition |
|---|---|---|
| `gics_sector` | GICS sector (categorical). | Broad sector. |
| `ff12` | Fama-French 12-industry bucket from SIC (categorical). | Coarse industry. |
| `ff48` | Fama-French 48-industry bucket from SIC (categorical). | Finer industry. |
| `industry_mom` | Equal-weighted trailing 1-month return of FF12 peers (own firm removed). | The firm's industry has been rallying. |
| `industry_drift_base` | Trailing avg realized drift of *prior* same-industry events whose window already closed (strictly causal). | This industry has tended to drift up after earnings recently. |

### 2.5 Fundamentals / firm quality (Compustat, point-in-time)

| Feature | Plain meaning | Intuition (higher = …) |
|---|---|---|
| `book_to_market` | Book equity / market cap. | More "value"; lower = more "growth". |
| `roa` | Net income / total assets. | More profitable per dollar of assets. |
| `gross_margin` | (Revenue − COGS) / revenue. | More pricing power / better unit economics. |
| `leverage` | Total debt / total assets. | More financial leverage. |
| `asset_growth` | Year-over-year change in total assets. | Faster balance-sheet expansion (often a negative for future returns). |
| `sales_growth` | Year-over-year change in revenue. | Faster top-line growth. |
| `accruals` | Balance-sheet accruals (Δ non-cash working capital net of depreciation) / assets. | Earnings rely more on accruals than cash (classically negative). |
| `n_employees` | Log employee count. | Size / labor-intensity proxy. |
| `earn_vol` | Std of the last 8 quarters of net income. | Less predictable earnings. |
| `rd_intensity` | R&D expense / sales. | More research-intensive (growth/tech). |

### 2.6 Calendar / context

| Feature | Plain meaning | Intuition |
|---|---|---|
| `fiscal_q` | Calendar quarter of the announcement (1–4, categorical). | Quarter-of-year seasonality. |
| `report_time` | Before market open (BMO) vs after close (AMC). | Release timing vs the trading day. |
| `mkt_ret_pre` | SPY return over offsets [−21, −1]. | Market backdrop into the print. |
| `vix_level` | Market-implied vol regime (not populated in this build). | Market fear gauge. |

> **Note on empty columns.** `idx_member` and `vix_level` are emitted for schema
> stability but are not populated in the current build, so they carry no signal.

---

## 3. Metric glossary — the report's vocabulary

| Metric | What it means |
|---|---|
| **Spearman rank IC** | Rank correlation between predicted and realized drift, out of sample. 0 = no skill; ~0.03–0.05 is a weak-but-real cross-sectional signal for this task. |
| **OOS R²** | Out-of-sample R² on `drift_z`. Often near zero / slightly negative because per-name drift is mostly noise — this does **not** mean the ranking is useless (check IC and decile spread). |
| **Decile spread** | Realized drift of the model's top-decile predictions minus its bottom-decile predictions. Positive = the ranking separated winners from losers. |
| **Classifier ROC-AUC** | Probability the classifier ranks a random strong-up event above a random down event. 0.50 = coin flip; just above 0.50 = weak edge. |
| **PR-AUC** | Precision-recall area for the strong-up vs down classifier; complements ROC-AUC under class imbalance. |
| **mean \|SHAP\|** | Average absolute SHAP value — overall importance / strength of a feature. |
| **SHAP direction (sign)** | Average signed SHAP — whether higher feature values push predicted drift up (+) or down (−). |
| **Cohen's d** | Standardized mean difference of a feature between top- and bottom-decile drift events. \|d\|≈0.2 small, 0.5 medium, 0.8 large. |
| **Fama-MacBeth coef** | Average per-quarter cross-sectional coefficient — linear effect of a +1σ change in the feature on `drift_z`. |
| **t-stat (Newey-West)** | Significance of the Fama-MacBeth coefficient, autocorrelation-corrected. \|t\| > 2 is the usual bar. |
| **top_k_hit_rate** | Fraction of walk-forward folds where the feature is in the top-10 most important — a stability measure. |
| **sign_agreement** | True when the linear (Fama-MacBeth) and non-linear (SHAP) views agree on direction — strong corroborating evidence. |
| **consistency_score** | 0–1 blend of top-k stability (50%) + low rank dispersion (30%) + linear significance (20%). High = a driver that shows up again and again. |

---

## 4. How to read each plot

**Figure 1 — Out-of-sample skill.** Rank IC and R² (left) and the top-minus-
bottom decile spread (right), per horizon. Look for a positive IC and a positive
decile spread — that is the model ranking drift correctly. R² near zero is
expected. Bars repeat for +60/+20/+5 days to confirm a result is not a
one-horizon artifact.

**Figure 2 — Which features drive the drift (SHAP).** Features ranked by mean
|SHAP| (importance). Color = direction: **green pushes drift up, red pushes it
down**. Longer bar = matters more. This is the non-linear analogue of the
Fama-MacBeth coefficients.

**Figure 3 — Consistency.** How stably each feature ranks as a driver across all
walk-forward folds, blended with linear significance (0–1). A feature can be
important once (high SHAP) yet inconsistent (low score); this chart separates the
robust drivers from the lucky ones.

**Figure 4 — Strong vs weak drift firms (Cohen's d).** For each feature, the
standardized difference between the strongest- and weakest-drift deciles. Green
to the right = strong-up-drift firms have a *higher* value; red to the left = the
opposite. A plain descriptive contrast ("winners look like ___"), not a model
output.

---

## 5. Outputs produced by a run

| File | Contents |
|---|---|
| `outputs/drift_ml_report_<timestamp>.pdf` | The full interpretable report (glossary, plots with explanations, driver map, Fama-MacBeth table). |
| `outputs/feature_importance.csv` | The ranked driver map: feature → importance, direction, stability, coef, t-stat, sign agreement, consistency score. |
| `outputs/event_features.csv` | The full **event × feature × label matrix** (one row per earnings event) — the same data as the cached parquet, exported for inspection. |
| `outputs/feature_summary.csv` | Per-feature coverage and summary stats (non-null count, % present, mean, std, min/median/max). |
| `data/derived/event_features.parquet` | The cached binary copy of the matrix (gitignored), reused across runs unless `--no-cache`. |
