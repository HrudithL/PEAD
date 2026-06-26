# PEAD — Post-Earnings Announcement Drift

Research code measuring **post-earnings announcement drift (PEAD)** in two markets:

- **Equities** — abnormal stock drift bucketed by earnings surprise (working, see `pead/equities/`).
- **Options** — the same drift studied through the options surface (IV, greeks, volume),
  driven off OptionMetrics (see `pead/options/`).

## Architecture (data plane → reduce → compute → report)

```
External data (never in repo)                In-repo code
-----------------------------                ------------
Data Source/  (master_stock, IBES)  ─┐
D:/OptionMetrics/parquet (~1 TB,     ─┤──►  pead/io      locate data (env vars)
  year-partitioned: opprcd, secprd,  │      pead/options/extract.py  DuckDB reduce
  secnmd, zerocd, ...)               │          │  (partition prune + projection)
                                     │          ▼
                                     │      data/derived/*.parquet  (compact panel)
                                     │          │  Arrow (zero-copy)
                                     │          ▼
                                     │      native/   C++/CUDA engine (GPU + CPU fallback)
                                     │          │
                                     │          ▼
                                     └──►  pead/*/report.py  PDF report
```

The repository contains **code only**. Raw equities CSVs and the ~1 TB OptionMetrics
dataset stay on disk and are located at runtime via `BEI_DATA_DIR` and
`OPTIONMETRICS_DIR` (see `.env.example`).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # edit paths for your machine
python run_pead.py --help   # equities study
```

## Layout

| Path | Purpose |
|------|---------|
| `pead/equities/` | Stock-market PEAD (data load, surprise, event study, report) |
| `pead/options/`  | Options-market PEAD (DuckDB extract, panel, engine, report) |
| `pead/io/`       | Shared data-access layer (resolver, manifest, schema) |
| `native/`        | C++/CUDA compute engine consuming the derived Arrow panel |
| `data/reference/`| Small committed constituent lists |
| `data/derived/`  | Extracted event panels (gitignored cache) |
| `docs/`          | Architecture + data dictionary |

> Detailed docs land in `docs/` (added in a later PR).
