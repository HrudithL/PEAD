"""Plain-language glossary: features, labels, metrics, and plots.

Single source of truth for every human-readable description used by the PDF
report (:mod:`report`) and mirrored in ``docs/feature_glossary.md``. Keeping the
text here (rather than scattered through the plotting code) means the report,
the CSV exports, and the standalone documentation never drift apart.

Each feature entry carries:

* ``family``  -- the catalog family it belongs to (Section 5 of the design).
* ``label``   -- a short, friendly axis label for charts (e.g. ``12-1 momentum``).
* ``desc``    -- one-sentence plain-English meaning.
* ``hint``    -- the economic intuition / what a high value tends to mean.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Feature glossary
# --------------------------------------------------------------------------- #
# family, friendly label, plain meaning, economic hint
FEATURES: dict[str, dict[str, str]] = {
    # 5.1 Earnings / analyst signal -- the trigger
    "sue_std": {
        "family": "Earnings / analyst",
        "label": "SUE (std-scaled)",
        "desc": "Standardized unexpected earnings: (actual EPS - mean estimate) divided by the dispersion of analyst estimates.",
        "hint": "Higher = a bigger, more decisive earnings beat relative to how much analysts disagreed.",
    },
    "sue_price": {
        "family": "Earnings / analyst",
        "label": "SUE (price-scaled)",
        "desc": "Earnings surprise scaled by the pre-announcement share price (actual - estimate) / price.",
        "hint": "Higher = a larger beat measured in cents per dollar of price.",
    },
    "surprise_raw": {
        "family": "Earnings / analyst",
        "label": "EPS surprise",
        "desc": "Raw earnings surprise in EPS dollars: actual EPS minus the consensus mean estimate.",
        "hint": "Positive = beat consensus; negative = missed.",
    },
    "analyst_disp": {
        "family": "Earnings / analyst",
        "label": "Analyst disagreement",
        "desc": "Forecast dispersion: standard deviation of estimates divided by the absolute mean estimate.",
        "hint": "Higher = analysts disagreed more going into the print (more uncertainty).",
    },
    "n_analysts": {
        "family": "Earnings / analyst",
        "label": "Analyst coverage",
        "desc": "Number of analysts in the consensus (IBES NUMEST).",
        "hint": "Higher = more closely followed; drift is often larger for less-covered names.",
    },
    "revision_net": {
        "family": "Earnings / analyst",
        "label": "Net estimate revisions",
        "desc": "Net upward revisions in the run-up: (analysts revising up - revising down) / total.",
        "hint": "Higher = analysts were getting more optimistic before the announcement.",
    },
    "forecast_staleness": {
        "family": "Earnings / analyst",
        "label": "Forecast staleness (days)",
        "desc": "Days between the last consensus snapshot and the announcement date.",
        "hint": "Higher = the consensus was older / less refreshed at announcement.",
    },
    "beat_flag": {
        "family": "Earnings / analyst",
        "label": "Beat / meet / miss",
        "desc": "Sign of the surprise: +1 beat, 0 meet, -1 miss (categorical).",
        "hint": "Direction of the surprise, ignoring magnitude.",
    },
    "ear": {
        "family": "Earnings / analyst",
        "label": "Announcement reaction (EAR)",
        "desc": "Earnings announcement abnormal return on day 0 (stock minus SPY), the immediate market reaction.",
        "hint": "Higher = the market reacted more positively on the day; ends before the drift window so it is a legitimate predictor.",
    },
    # 5.2 Price / return & risk (pre-event)
    "mom_12_1": {
        "family": "Price / risk",
        "label": "12-1 momentum",
        "desc": "Cumulative return from ~12 months ago to ~1 month ago (offsets [-252, -21]).",
        "hint": "Higher = the stock has been a longer-run winner.",
    },
    "mom_1m": {
        "family": "Price / risk",
        "label": "1-month return",
        "desc": "Cumulative return over the month before the print (offsets [-21, -1]).",
        "hint": "Recent trend heading into the announcement.",
    },
    "reversal_1w": {
        "family": "Price / risk",
        "label": "1-week reversal",
        "desc": "Return over the five trading days before the print (offsets [-5, -1]).",
        "hint": "Captures very-short-term moves that often mean-revert.",
    },
    "rvol_60": {
        "family": "Price / risk",
        "label": "Realized vol (60d)",
        "desc": "Standard deviation of daily returns over the prior 60 trading days.",
        "hint": "Higher = a more volatile stock pre-event.",
    },
    "beta_252": {
        "family": "Price / risk",
        "label": "Market beta (252d)",
        "desc": "Market-model beta versus SPY estimated over the prior 252 trading days.",
        "hint": "Higher = moves more with the overall market.",
    },
    "ivol": {
        "family": "Price / risk",
        "label": "Idiosyncratic vol",
        "desc": "Standard deviation of the market-model residual (stock-specific volatility).",
        "hint": "Higher = more firm-specific noise unexplained by the market.",
    },
    "high52_prox": {
        "family": "Price / risk",
        "label": "Proximity to 52w high",
        "desc": "Current price divided by the trailing 52-week high.",
        "hint": "Close to 1 = trading near its yearly high.",
    },
    "pre_run": {
        "family": "Price / risk",
        "label": "Pre-announcement run-up",
        "desc": "Abnormal return over the five days into the print (offsets [-5, -1], stock minus SPY).",
        "hint": "Higher = the stock ran up abnormally just before reporting.",
    },
    # 5.3 Liquidity / size
    "price_level": {
        "family": "Liquidity / size",
        "label": "Log price",
        "desc": "Natural log of the closing share price at announcement.",
        "hint": "A rough proxy for share-price level (low-priced stocks behave differently).",
    },
    "mktcap": {
        "family": "Liquidity / size",
        "label": "Log market cap",
        "desc": "Natural log of market capitalization (price x shares outstanding) from CRSP.",
        "hint": "Higher = a larger company; drift is classically stronger in smaller caps.",
    },
    "dollar_vol": {
        "family": "Liquidity / size",
        "label": "Log dollar volume",
        "desc": "Log of average daily dollar trading volume over the prior 60 days.",
        "hint": "Higher = more actively traded / more liquid.",
    },
    "turnover": {
        "family": "Liquidity / size",
        "label": "Turnover",
        "desc": "Average daily volume divided by shares outstanding over the prior 60 days.",
        "hint": "Higher = shares change hands faster (more speculative interest).",
    },
    "amihud": {
        "family": "Liquidity / size",
        "label": "Amihud illiquidity",
        "desc": "Average of |return| / dollar volume -- price impact per dollar traded.",
        "hint": "Higher = more illiquid (prices move a lot on little volume).",
    },
    "idx_member": {
        "family": "Liquidity / size",
        "label": "Index membership",
        "desc": "S&P 500 / Russell membership flag (categorical; not populated in this build).",
        "hint": "Index membership status.",
    },
    # 5.4 Sector / industry
    "gics_sector": {
        "family": "Sector / industry",
        "label": "GICS sector",
        "desc": "GICS sector code from Compustat (categorical).",
        "hint": "Broad sector the firm belongs to.",
    },
    "ff12": {
        "family": "Sector / industry",
        "label": "Fama-French 12 industry",
        "desc": "Fama-French 12-industry bucket mapped from the firm's SIC code (categorical).",
        "hint": "Coarse industry grouping.",
    },
    "ff48": {
        "family": "Sector / industry",
        "label": "Fama-French 48 industry",
        "desc": "Fama-French 48-industry bucket mapped from the firm's SIC code (categorical).",
        "hint": "Finer industry grouping.",
    },
    "industry_mom": {
        "family": "Sector / industry",
        "label": "Industry momentum",
        "desc": "Equal-weighted trailing one-month return of the firm's FF12 peers (own firm removed).",
        "hint": "Higher = the firm's industry has been rallying.",
    },
    "industry_drift_base": {
        "family": "Sector / industry",
        "label": "Industry baseline drift",
        "desc": "Trailing average realized drift of prior same-industry events whose window already closed (strictly causal).",
        "hint": "Higher = this industry has tended to drift up after earnings recently.",
    },
    # 5.5 Fundamentals / firm quality
    "book_to_market": {
        "family": "Fundamentals",
        "label": "Book-to-market",
        "desc": "Book equity divided by market cap (value vs growth).",
        "hint": "Higher = more 'value'; lower = more 'growth'.",
    },
    "roa": {
        "family": "Fundamentals",
        "label": "Return on assets",
        "desc": "Net income divided by total assets (profitability).",
        "hint": "Higher = more profitable per dollar of assets.",
    },
    "gross_margin": {
        "family": "Fundamentals",
        "label": "Gross margin",
        "desc": "(Revenue - cost of goods sold) / revenue.",
        "hint": "Higher = more pricing power / better unit economics.",
    },
    "leverage": {
        "family": "Fundamentals",
        "label": "Leverage",
        "desc": "Total debt divided by total assets.",
        "hint": "Higher = more financial leverage.",
    },
    "asset_growth": {
        "family": "Fundamentals",
        "label": "Asset growth (yoy)",
        "desc": "Year-over-year change in total assets.",
        "hint": "Higher = faster balance-sheet expansion (often a negative for future returns).",
    },
    "sales_growth": {
        "family": "Fundamentals",
        "label": "Sales growth (yoy)",
        "desc": "Year-over-year change in revenue.",
        "hint": "Higher = faster top-line growth.",
    },
    "accruals": {
        "family": "Fundamentals",
        "label": "Accruals",
        "desc": "Balance-sheet accruals (change in non-cash working capital net of depreciation) scaled by assets.",
        "hint": "Higher = earnings rely more on accruals than cash (classically a negative signal).",
    },
    "n_employees": {
        "family": "Fundamentals",
        "label": "Log employees",
        "desc": "Natural log of the employee count.",
        "hint": "A size / labor-intensity proxy.",
    },
    "earn_vol": {
        "family": "Fundamentals",
        "label": "Earnings volatility",
        "desc": "Standard deviation of the last eight quarters of net income.",
        "hint": "Higher = less predictable earnings.",
    },
    "rd_intensity": {
        "family": "Fundamentals",
        "label": "R&D intensity",
        "desc": "R&D expense divided by sales.",
        "hint": "Higher = more research-intensive (often growth/tech firms).",
    },
    # 5.7 Context / calendar
    "fiscal_q": {
        "family": "Calendar / context",
        "label": "Fiscal quarter",
        "desc": "Calendar quarter of the announcement (1-4; seasonality, categorical).",
        "hint": "Captures quarter-of-year seasonal effects.",
    },
    "report_time": {
        "family": "Calendar / context",
        "label": "Report timing (BMO/AMC)",
        "desc": "Whether the firm reported before market open (BMO) or after market close (AMC).",
        "hint": "Timing of the release relative to the trading day.",
    },
    "mkt_ret_pre": {
        "family": "Calendar / context",
        "label": "Market return (pre)",
        "desc": "SPY return over the month before the print (offsets [-21, -1]).",
        "hint": "The market backdrop heading into the announcement.",
    },
    "vix_level": {
        "family": "Calendar / context",
        "label": "VIX level",
        "desc": "Market-implied volatility regime at the print (not populated in this build).",
        "hint": "Overall market fear gauge.",
    },
}

# --------------------------------------------------------------------------- #
# Label / target glossary
# --------------------------------------------------------------------------- #
LABELS: dict[str, str] = {
    "drift_raw": "Market-adjusted CAR[+1,+H]: cumulative abnormal return (stock minus SPY) from trading day +1 through +H, excluding the day-0 announcement jump. This is 'the drift' in return units.",
    "drift_z": "The primary regression target: drift_raw z-scored within each calendar quarter, which removes market-wide regime shifts so the model learns cross-sectional drivers rather than calendar effects.",
    "drift_decile": "Per-quarter decile (1-10) of drift_raw, used for the descriptive top-vs-bottom comparison and ranking views.",
    "drift_class": "Classifier target: 1 for top-decile drift, 0 for bottom-decile drift, with the middle deciles dropped.",
}

# --------------------------------------------------------------------------- #
# Metric glossary -- the vocabulary the report uses
# --------------------------------------------------------------------------- #
METRICS: dict[str, str] = {
    "Spearman rank IC": "Rank correlation between predicted and realized drift out of sample. 0 = no skill; ~0.03-0.05 is a weak-but-real cross-sectional signal typical for this kind of task.",
    "OOS R\u00b2": "Out-of-sample R-squared on drift_z. Often near zero or slightly negative because per-name drift is mostly noise; a value near 0 does not mean the ranking is useless (see IC and decile spread).",
    "Decile spread": "Average realized drift of the model's top-decile predictions minus its bottom-decile predictions. A positive spread means acting on the model's ranking would have separated winners from losers.",
    "Classifier ROC-AUC": "Probability the classifier ranks a random strong-up-drift event above a random down-drift event. 0.50 = coin flip; values just above 0.50 indicate a weak edge.",
    "PR-AUC": "Area under the precision-recall curve for the strong-up vs down classifier; complements ROC-AUC when classes are imbalanced.",
    "mean |SHAP|": "Average absolute SHAP value -- how much, on average, a feature moves the model's drift prediction. This is the feature's overall importance (strength).",
    "SHAP direction (sign)": "Average signed SHAP value -- whether higher values of the feature tend to push the predicted drift up (positive) or down (negative).",
    "Cohen's d": "Standardized mean difference of a feature between top-decile and bottom-decile drift events. |d|~0.2 is small, ~0.5 medium, ~0.8 large; the sign says which group has the higher value.",
    "Fama-MacBeth coef": "Average per-quarter cross-sectional regression coefficient -- the linear effect of a 1-standard-deviation increase in the feature on drift_z, averaged across quarters.",
    "t-stat (Newey-West)": "Statistical significance of the Fama-MacBeth coefficient, corrected for autocorrelation across quarters. |t| > 2 is the usual significance bar.",
    "top_k_hit_rate": "Fraction of walk-forward folds in which the feature lands in the top-10 most important features -- a stability measure.",
    "sign_agreement": "True when the linear (Fama-MacBeth) and non-linear (SHAP) views agree on the direction of the effect; agreement is strong corroborating evidence.",
    "consistency_score": "A 0-1 blend of top-k stability (50%), low rank dispersion across folds (30%), and linear significance (20%). High = the feature shows up as a driver again and again, not just once.",
}

# --------------------------------------------------------------------------- #
# Plot guide -- what each figure shows and how to read it
# --------------------------------------------------------------------------- #
PLOTS: dict[str, dict[str, str]] = {
    "oos": {
        "shows": "Out-of-sample skill of the LightGBM drift model at each horizon: rank IC and R\u00b2 (left), and the top-minus-bottom decile spread of its predictions (right).",
        "read": "Look for a positive IC and a positive decile spread -- that is the model ranking drift correctly. R\u00b2 near zero is expected and not a red flag here. Bars are repeated for the +60/+20/+5 day horizons to confirm a result is not a one-horizon artifact.",
    },
    "shap": {
        "shows": "The strongest drivers of drift in the non-linear model, ranked by mean |SHAP| (importance). Bar color encodes direction: green features tend to push drift up, red features tend to push it down.",
        "read": "Longer bar = the feature matters more for the prediction. The color/sign tells you which way: e.g. a green 'idiosyncratic vol' bar means higher idiosyncratic vol is associated with higher subsequent drift. This is the non-linear analogue of the Fama-MacBeth coefficient table.",
    },
    "cons": {
        "shows": "How consistently each top feature acts as a driver across all walk-forward folds, blended with its linear significance (the consistency score, 0-1).",
        "read": "A high bar means the feature is repeatedly important across time, sectors, and folds -- the kind of robust driver to trust. A feature can be important once (high SHAP) yet inconsistent (low score); this chart separates the two.",
    },
    "desc": {
        "shows": "For each feature, the standardized difference (Cohen's d) between the strongest-drift decile and the weakest-drift decile of events.",
        "read": "A green bar to the right means strong-up-drift firms have a higher value of that feature than down-drift firms; red to the left means the opposite. This is a plain descriptive contrast ('winners look like ___'), not a model output.",
    },
}


def feature_label(name: str) -> str:
    """Friendly chart label for a feature column (falls back to the raw name)."""
    return FEATURES.get(name, {}).get("label", name)


def feature_desc(name: str) -> str:
    """One-line plain meaning for a feature column (empty string if unknown)."""
    return FEATURES.get(name, {}).get("desc", "")


def feature_family(name: str) -> str:
    return FEATURES.get(name, {}).get("family", "Other")
