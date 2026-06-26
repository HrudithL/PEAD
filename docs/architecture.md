# Architecture

PEAD measures post-earnings announcement drift in two markets that share one
design principle: **the repository holds code only; data lives outside it and is
located at runtime.**

## Data plane vs. code

| | Where it lives | How code reaches it | In git? |
|---|---|---|---|
| Equities CSVs (`master_stock`, `IBES_Summary_*`) | external `Data Source/` | `BEI_DATA_DIR` → `pead.io.resolver` | no |
| OptionMetrics (~1 TB parquet) | external drive (`D:/OptionMetrics/parquet`) | `OPTIONMETRICS_DIR` → `pead.io.resolver` | no |
| Constituent lists (S&P/Russell) | `data/reference/` | committed | yes (small) |
| Derived event panels | `data/derived/` | generated cache | no (gitignored) |
| Run artifacts (PDF/CSV/PNG) | `outputs/` | generated | no (gitignored) |

The resolver tries an environment variable first, then a list of known candidate
paths, so the same code runs on the Windows workstation, the Linux GPU box, and
CI without edits.

## Equities track (`pead/equities/`)

```
IBES + master_stock ──► surprise ──► event study ──► report (PDF)
```

Classic PEAD: compute earnings surprise, align abnormal stock returns in an event
window, bucket by surprise, measure the long/short drift spread.

## Options track (`pead/options/`)

The options dataset is ~1 TB, so the pipeline is staged to read it **once** and
do all iteration on a compact reduction.

```
                          OPTIONMETRICS_DIR (≈1 TB parquet, year-partitioned)
                                        │
  (1) events.py     IBES → (ticker, secid, ann_date) + surprise
                                        │
  (2) extract.py    DuckDB: per-event date-window join over opprcd{year}
                    • partition pruning  → only the year files in range
                    • projection pushdown → only needed columns
                    • predicate pushdown  → only rows inside [ann-pre, ann+post]
                                        │
                                        ▼
                    data/derived/event_panel.parquet   (a few GB, or less)
                                        │  Arrow (zero-copy)
                                        ▼
  (3) engine        native/ C++/CUDA  (GPU)   ── or ──  pead/options/engine.py (pandas)
                    per-event ATM implied-vol drift (post − pre)
                                        │
                                        ▼
                    data/derived/event_results.parquet
                                        │
  (4) analysis.py   bucket iv_drift by earnings surprise; long/short spread
  (5) report.py     CSV + bar plot
```

### Why DuckDB for stage 2

DuckDB is the **data-access / reduction** layer, not the compute engine. It turns
"1 TB on a slow drive" into "the few GB you actually need" using pushdown, then
hands the result to the compute stage as Arrow. It never runs the heavy math.

### Why a separate native engine for stage 3

The per-event math (millions of option-days) is the compute-bound part. It runs
in C++/CUDA on the reduced panel, reading Arrow buffers directly (zero-copy). The
Python fallback produces an identical schema, so the pipeline always runs; the
native binary is a drop-in speedup once built.

### The Arrow hand-off

`extract.py` writes parquet; the engine reads it through Arrow C++. Arrow's
columnar layout is GPU-friendly, so columns can be pushed to CUDA (or cuDF) with
no serialization. DuckDB and the GPU are teammates: one feeds, the other crunches.

## Adding another study

A new market/study is a new subpackage under `pead/` that depends on `pead.io`
for data location and (optionally) the `native/` engine for compute. No existing
track needs to change.
