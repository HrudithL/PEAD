# Data dictionary

What PEAD reads, and the columns it depends on. Raw data is never committed; see
`.env.example` for how paths are configured.

## OptionMetrics (parquet, year-partitioned)

Root: `OPTIONMETRICS_DIR` (e.g. `D:/OptionMetrics/parquet`). Year-partitioned
tables are named `f"{stem}{year}.parquet"`. Schema constants live in
`pead/io/om_schema.py`.

### `secnmd.parquet` — security name dimension (small, unpartitioned)

Maps tickers to OptionMetrics security ids. Used to attach `secid` to events.

| column | notes |
|---|---|
| `secid` | OptionMetrics security id (join key) |
| `ticker` | exchange ticker (a ticker may map to multiple secids over time) |
| `cusip`, `issuer`, `class`, `issue`, `sic`, `effect_date` | descriptive |

### `opprcd{year}.parquet` — option prices + greeks (core fact table)

One row per option contract per trading day. The largest table; the extract step
reads only the columns below and only rows inside each event window.

| column | notes |
|---|---|
| `secid` | join key to events / secnmd |
| `date` | quote date |
| `exdate` | expiration date |
| `cp_flag` | `C` call / `P` put |
| `strike_price` | **in 1/1000 dollars** — divide by 1000 (`strike` column does this) |
| `best_bid`, `best_offer` | quotes |
| `volume`, `open_interest` | activity |
| `impl_volatility` | Black-Scholes implied vol |
| `delta`, `gamma`, `vega`, `theta` | greeks |

### `secprd{year}.parquet` — underlying security daily prices

| column | notes |
|---|---|
| `secid`, `date` | keys |
| `open`, `high`, `low`, `close`, `volume`, `return` | daily OHLCV + return |

### `zerocd.parquet` — zero-coupon yield curve

| column | notes |
|---|---|
| `date`, `days`, `rate` | risk-free rates for Black-Scholes / discounting |

### `distrd.parquet` — distributions / dividends

Used for dividend adjustments (reserved for future use).

## Derived event panel (`data/derived/event_panel.parquet`)

Output of `pead/options/extract.py`; input to the compute engine. Schema in
`pead/options/panel.py`.

| column | source |
|---|---|
| `secid`, `date`, `exdate`, `cp_flag` | opprcd |
| `strike_price`, `strike` | opprcd (`strike` = strike_price / 1000) |
| `best_bid`, `best_offer`, `volume`, `open_interest` | opprcd |
| `impl_volatility`, `delta`, `gamma`, `vega`, `theta` | opprcd |
| `ann_date` | event (earnings announcement date) |
| `rel_day` | `date − ann_date` in calendar days |

## Per-event results (`data/derived/event_results.parquet`)

Output of the compute engine (native or pandas). One row per `(secid, ann_date)`.

| column | meaning |
|---|---|
| `atm_iv_pre` | mean ATM-call implied vol, `rel_day < 0` |
| `atm_iv_post` | mean ATM-call implied vol, `rel_day > 0` |
| `iv_drift` | `atm_iv_post − atm_iv_pre` |
| `n_pre`, `n_post` | ATM observation counts |
| `total_volume` | summed option volume in the window |

ATM = a call with `|abs(delta) − 0.5| ≤ 0.1`.

## IBES summary (equities + event construction)

CSV under `BEI_DATA_DIR`. Columns used by the options `events.py`:

| column | meaning |
|---|---|
| `TICKER` | firm ticker |
| `ANNDATS_ACT` | actual announcement date (→ `ann_date`) |
| `ACTUAL` | reported EPS |
| `MEANEST`, `STDEV`, `NUMEST` | consensus mean, dispersion, # estimates |

Surprise (SUE): `sue_std = (ACTUAL − MEANEST) / STDEV`.
