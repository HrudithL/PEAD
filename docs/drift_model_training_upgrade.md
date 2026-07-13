# Drift model training upgrade — full universe, tuned hyperparameters, Colab execution

> **⚠️ READ FIRST — build / development workflow.** How this gets implemented is
> as binding as what gets implemented. The rules below apply to every task in
> this doc.

## 0. Build / development workflow (READ FIRST)

### 0.1 Parallelize with sub-agents

Independent tasks are built in parallel by sub-agents rather than serially in the main thread.

- **Each parallel sub-agent runs in its own isolated git worktree** (`isolation: "worktree"`). This is not optional: a git repo has one working-tree checkout, so two agents on two branches in the *same* tree would clobber each other. Worktree isolation gives each agent its own checkout + branch, and the harness auto-cleans a worktree that ends up unchanged.
- The main thread stays orchestration-only: decompose into tasks, spawn agents, review their pushed branches, open PRs. It does not itself write the feature code when agents are handling it.
- **Task waves** (dependency-ordered — only same-wave tasks run in parallel):
  - **Wave 1 (independent, parallel):**
    - `wrds_extract.py` incremental checkpointing + tests (section 7.2).
    - Optuna tuning module (`tune.py`) + tests + `requirements.txt` (`optuna`) (sections 3.2 / 3.4).
  - **Wave 2 (depend on the tuning module from wave 1):**
    - `train_final.py` — accept tuned params, early stopping (section 3.2 / 3.4 Job 3).
    - `backtest.py` + PDF #1 (`model_card.pdf`) rewrite (section 4) — Job-2 test-set evaluation + report fixes.
    - `run_train_drift_model.py` CLI wiring (tune step, holdout split, reuse-best-params flag).

### 0.2 Branch hierarchy + small commits

- Every task gets its **own small feature branch**, based on the current serving-work head (**`feat/drift-serving` / `subsample-drift`**, *not* `main` — the serving/training code isn't on `main`).
- **Small, focused commits** on each branch (one logical change each), not one giant squash. Conventional-commit style, matching the repo's history.
- Feature branches **roll up into a single integration branch** (e.g. `feat/drift-training-upgrade`). That integration branch is the one intended to eventually reach `main`.
- Shape: `feat/<task>` (small) → `feat/drift-training-upgrade` (integration) → `main` (final).

### 0.3 Permission constraints (hard rules for Claude)

- **CAN**: push commits to any feature or integration branch.
- **CANNOT**: push to `main`.
- **CANNOT**: merge branches — no `git merge`, no `gh pr merge`, at any level (feature→integration or integration→main).
- Net: **Claude prepares branches and opens PRs; the user performs every merge** and the final promotion to `main`. Claude's role ends at "branch pushed, PR opened for review."

## Status

Planning document. No training code has been changed yet. This describes what
will change, why, and what's already verified vs. still assumed. Implementation
starts after this is confirmed.

## 1. End goal

One command produces a **fully-tuned model bundle trained on the entire
available universe**, plus **one PDF** reporting how good that model is,
measured honestly on a held-out **test set** the hyperparameter search never
sees (section 3.4). That PDF and bundle are the complete deliverable of the
training step. Scoring a single announcement against the finished bundle (and
*that* PDF) is a separate, later effort — not part of this round.

This directly matches the project's own original design intent
(`docs/subsample_drift_ml.md`, section 2): *"Universe: full available universe
(all tickers present in IBES + price data)... No `--tickers` restriction."*
The 3-ticker demo run earlier in this session was a shortcut for speed, not
the intended scope — this upgrade returns to (and then tunes) that original
scope.

## 2. Current state (verified)

| Question | Verified answer |
|---|---|
| How many tickers are already available locally, no new data pull? | 8,222 tickers have quarterly IBES actual-EPS announcements (`IBES_Summary_2015_2024.csv`, 456K announcement-rows); 4,703 tickers have price history (`master_stock.csv`). Usable universe = their intersection, thousands of tickers. |
| Is there any hyperparameter search today? | No. `train_final.py` and `model.py::fit_lightgbm_cv` both hardcode identical fixed params (`learning_rate=0.03, num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=20`) and a fixed round count (400, or 300 for the classifier). Nobody has ever tuned these. |
| How does WRDS get pulled today? | `wrds_extract.py` already batches: e.g. `extract_crsp_daily` chunks permnos 1,000 at a time into `WHERE permno IN (...)` SQL, not one query per ticker. Scaling to the full universe is a few large queries, not thousands of small ones. Results already cache to CSV under `WRDS_CACHE_DIR` (defaults inside `Data Source/`) — this is the exact "query WRDS, save to CSV in Data Source" behavior you asked for; it just needs to run un-scoped instead of `--tickers AAPL,MSFT,JPM`. |
| Does LightGBM here support GPU out of the box? | Partially unverified. `device_type="gpu"` ran without error in a local smoke test, but that's most likely this machine's integrated-GPU OpenCL driver — not evidence it'll work identically on a Colab NVIDIA GPU, which usually needs a CUDA-specific LightGBM build (`device_type="cuda"`), not the default `pip install lightgbm` wheel. Treating this as **to-confirm on first Colab run**, not assumed working. |
| Is the search itself compute-bound in a way GPU helps? | No, measured, not just theoretical. A real LightGBM fit (42 features, early stopping) on this machine: 5K rows -> 0.41s, 20K -> 0.33s, 50K -> 0.72s, 100K -> 1.41s. Sub-1.5-second CPU fits at full scale don't need acceleration; GPU is not the reason to use Colab here. |
| Real full-universe scale | Measured directly against the actual files: 4,209 tickers appear in *both* `IBES_Summary_2015_2024.csv` (actual-EPS announcements) and `master_stock.csv` (prices) -- the real usable universe. That intersection yields **104,656** candidate quarterly events before the calendar-location filter trims events too close to the edges of price history; realistic final count is likely 80,000-100,000. |
| Does the Colab kernel see the local `Data Source/` folder directly? | **No -- confirmed empirically.** Ran `scratch/colab_filesystem_check.ipynb` against a live Colab kernel via the VS Code extension: the kernel is a separate remote machine and cannot see `C:\Users\hrudi\...\Data Source`. GPU compute is transparent (that part *is* really Colab's T4); file access is not. Something has to ship the data to the remote side -- see 3.3. |
| Raw data file sizes | Measured: `IBES_Summary_2015_2024.csv` = 482 MB, `master_stock.csv` = 303 MB (~785 MB combined). Transport decided as Google Drive, not GitHub (section 3.3) — the earlier GitHub/LFS analysis is moot; sizes retained here only as facts. |

## 3. What changes

### 3.1 Universe: 3 tickers -> full intersection

- Drop `--tickers AAPL,MSFT,JPM` from the training command entirely — run un-scoped, matching the original design's default.
- `use_wrds=True` (not `--no-wrds`) so `wrds_extract.build_wrds_panels()` pulls real Compustat/CRSP fundamentals for the whole universe, via the existing batched-query path — no new pulling code needed, just don't restrict it.
- If any *additional* WRDS fields turn out to be missing once we're at full scale (e.g. a fundamentals column not currently extracted), add it to the existing `extract_*` functions in `wrds_extract.py` following the same batched-query pattern, and it lands in the same CSV cache under `Data Source/wrds_cache/`. No new pulling mechanism — extend the existing one.
- Expected scale: tens of thousands of quarterly events, 2015-2024. This is still small-to-medium tabular data, not "big data" — no distributed compute needed, just more wall-clock time than the 110-event demo.

### 3.2 Hyperparameter search: none -> wide search for best result

- New search module (likely `pead/sub_sampling_ml/serving/tune.py` or added to `model.py`) using **Optuna** (add to `requirements.txt`) for a wide randomized/TPE search rather than a small manual grid, per your "large search space, best possible result" call.
- **Search space** (LightGBM params): `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, `max_depth` — wide ranges, not narrow defaults-tweaking.
- **One shared param set across all 5 quantile boosters** (they differ only in `alpha`). The search optimizes a single set, evaluated by mean pinball across the five — ~5x cheaper than tuning each quantile independently, and the standard approach. The winning set is then **reused for the z-regressor and the classifier** too (those two aren't separately tuned; the pinball objective only applies to the quantile heads, and reusing keeps the whole bundle consistent and the search tractable).
- **Objective = mean pinball (quantile) loss** — matches exactly what the quantile boosters are trained to minimize. Spearman IC is still computed and shown in PDF #1 as a diagnostic, but it is *not* the search target.
- **Where the search runs**: entirely inside the **tuning pool**, using purged walk-forward CV (`purged_walk_forward_splits`) to score each trial. The leakage-safe walk-forward is *not* up for debate — overlapping 60-day label windows leak under a plain split. The test set is never touched during the search. Full split spec in **3.4**.
- **Early stopping** replaces the fixed 400/300-round count. Three levels of held-out data are in play, kept distinct on purpose: **(A)** the 25% *test set* (never touched during tuning at all — 3.4); **(B)** each tuning walk-forward fold's *scored quarter* (the next quarter after the fold's training window, used to compute the trial's pinball); **(C)** the *early-stop validation quarter*. For (C): within each walk-forward fold, hold out the **most recent 1 quarter of that fold's own training window** (time-ordered, not random), fit on the rest, and stop when that quarter's pinball stops improving. Neither the scored quarter (B) nor the test set (A) is ever used to decide when to stop.
- Once the search picks a winner, that single best hyperparameter set is used for the honest test-set evaluation (PDF #1) *and* the final deployed fit — so the reported accuracy reflects the params actually shipped (see 3.4 for how those two differ).

### 3.3 Execution: local -> Colab (via the VSCode Colab extension) — Google Drive

Confirmed empirically (section 2): the Colab kernel cannot see the local `Data Source/` folder. **Decided: Google Drive**, not GitHub/LFS — simpler, and the account has room.

**Data (done):** Google Drive for Desktop is installed and mounts at `G:\My Drive` locally. Both raw CSVs have been copied into a dedicated folder:

| Local (Drive for Desktop) | In Colab (after `drive.mount`) |
|---|---|
| `G:\My Drive\PEAD_data\IBES_Summary_2015_2024.csv` | `/content/drive/MyDrive/PEAD_data/IBES_Summary_2015_2024.csv` |
| `G:\My Drive\PEAD_data\master_stock.csv` | `/content/drive/MyDrive/PEAD_data/master_stock.csv` |

The pipeline needs **no code change** to use these: `pead/io/resolver.py` already resolves the data directory from the `BEI_DATA_DIR` env var (env wins over all built-in candidates). The Colab bootstrap sets `BEI_DATA_DIR=/content/drive/MyDrive/PEAD_data` and everything downstream (`DEFAULT_IBES`, `DEFAULT_STOCK`, etc.) points at the mounted files automatically.

⚠️ Note on Drive account space: `G:\` reports ~4.4 GB free (Drive account quota). The two files (~750 MB) fit, but the account is fairly full — watch this once WRDS-cache CSVs and model bundles also start landing on Drive.

**Code:** the PEAD repo is **public** on GitHub (`https://github.com/HrudithL/PEAD.git`), so the remote kernel gets the code with a plain `git clone` — no token, no LFS, no Drive copy of the code needed. The bootstrap clones it fresh each session.

**Credentials:** WRDS username/password go into Colab's built-in **Secrets** (`google.colab.userdata`), injected into the environment at runtime — never committed, never written to a plaintext `.env`.

**Resumability:**
- **Derived data** (`event_features.parquet`) — after the one-time build, written to `/content/drive/MyDrive/PEAD_data/derived/` on Drive. Any later session reloads it (via the existing `use_cache=True` path) instead of rebuilding from the 750 MB raw CSVs or re-pulling WRDS.
- **WRDS cache** — set `WRDS_CACHE_DIR` to a Drive path too, so the (slow, one-time) CRSP/Compustat pull persists across sessions instead of being re-fetched every time.
- **Hyperparameter-search progress** — Optuna study in a SQLite file kept on Colab's **local `/content` disk** during the run (`sqlite:////content/optuna_study.db`, `load_if_exists=True`), **snapshotted to Drive every N trials** (via an Optuna callback) and on completion. On a fresh/resumed session, copy the Drive snapshot back to `/content` first, then continue. Rationale: SQLite over a Drive FUSE mount is unreliable (network-filesystem file locking is exactly what SQLite warns against), and the 8h cap will likely truncate the 400-trial search on a slower Colab CPU — so resume-across-sessions genuinely matters and must use reliable local storage, not a live file on the FUSE mount. Worst case on a crash is losing the last < N trials since the snapshot.
- **Final artifacts** (model bundle + `backtest_results.csv` + PDF #1) written to a Drive path so they sync back down to this machine automatically — no manual download.

**GPU:** request a GPU runtime, but optional/best-effort per the caveat above — if `device_type="cuda"` doesn't work cleanly, fall back to CPU (measured fits are sub-1.5s; the search parallelizes across CPU cores regardless).

### 3.4 Train / tune / test split (single holdout)

The data is split by time into two pieces up front, and three distinct jobs run against them. This is the spec the code implements.

```
2015 ───────────────────────── ~2022Q2 │ ~2022Q3 ──────── 2024
          TUNING POOL (~75%)            │   TEST SET (~25%, held out)
```

- **Test set = the most recent ~25% of distinct event-quarters**, rounded to the nearest whole quarter, computed from the data at load time (not hardcoded — adapts if the date range changes). With the current 2015-2024 span (~40 event-quarters) that's the last ~10 quarters (~2022Q3 onward). The tuning pool is the earlier ~75%.

**Job 1 — select hyperparameters (tuning pool only).** Optuna runs its 400-trial / 8h search *inside* the tuning pool, scoring each trial by mean pinball across purged walk-forward folds (3.2). These fold scores are a *selection instrument* only — they are cherry-picked (best of 400) and are **never reported as model accuracy**. The test set is untouched here.

**Job 2 — report honest accuracy (test set only) → PDF #1.** Take the single winning param set and walk it forward *through the test-set quarters*: for each test quarter, fit on all prior data (with the same purge/embargo gap before the quarter as everywhere else, so 60-day label windows don't leak) using the fixed params, and predict that quarter. Because the params were chosen without ever seeing the test set, these numbers are honest. PDF #1 states them explicitly as "with these hyperparameters, on data never used to choose them, the model scored X." This reuses the existing `backtest.py` walk-forward machinery, restricted to the test-set quarters with fixed (pre-tuned) params.

**Job 3 — ship the deployed bundle (all data).** Refit the winning params on *all* data (tuning pool + test set) to produce the bundle used later for live single-event scoring. Standard practice: the *procedure* (params + method) is validated on the holdout in Job 2; the shipped model is then retrained on everything so deployment uses the most data. The Job-2 number is the honest estimate of how this shipped model behaves.

**Why single holdout, not nested CV** (decided — cost/value): nested CV would rerun the entire search once per outer test fold (~5x), turning the ~8-16h single search into ~40-75h across 5-9 Colab sessions, to gain honest metrics on ~5 test years instead of 1. The most-recent holdout is already the most deployment-relevant test, and the marginal bias-reduction from nesting is small once the holdout is clean. Not worth 5x the compute and operational risk at this scale.

## 4. PDF #1 (training report) — what's new here

Beyond the fixes already agreed (scatter instead of decile-bucketing, captions, fixing the blank `prob_up` panel, explanatory text for every metric): this report now also needs a short section reporting **which hyperparameters won the search and by how much** (e.g. best trial's score vs. the old hardcoded defaults, so the tuning is visibly justified, not just "trust me, we searched").

## 5. Concrete file changes (implementation phase, not yet started)

- `requirements.txt` — add `optuna`.
- `pead/sub_sampling_ml/wrds_extract.py` — incremental checkpointing per section 7.2 (`extract_crsp_daily`, `extract_compustat_fundq`, `extract_company`: skip cached IDs, append per chunk). **Decided, not yet built.**
- New: hyperparameter search module (name TBD, likely `pead/sub_sampling_ml/serving/tune.py`).
- `pead/sub_sampling_ml/serving/train_final.py` — accept tuned params instead of the hardcoded `_BASE_PARAMS`, replace fixed boosting rounds with early stopping.
- `pead/sub_sampling_ml/serving/backtest.py` — use the same tuned params per fold; extend `write_model_card` (this is PDF #1) with the fixes already agreed (scatter, captions, blank-panel fix, hyperparameter-search summary section).
- `pead/sub_sampling_ml/serving/prediction_report.py` — **strip its page 1** (the model-wide performance page). That content now belongs solely to PDF #1 (`model_card.pdf`); PDF #2 becomes the single-event page only, so the two reports don't overlap. (Deferred to the predict-side round, but recorded here so it isn't lost.)
- `run_train_drift_model.py` — drop the 3-ticker demo scoping, wire in the search step (with a flag to skip search and reuse previously-found best params, so re-running doesn't always re-search from scratch).

**Already written this session (Colab plumbing):**
- `scratch/colab_filesystem_check.ipynb` — the empirical remote-kernel test (already run; confirmed section-2 finding).
- `scratch/colab_train_bootstrap.ipynb` — mounts Drive, clones the public repo (branch `subsample-drift`), sets `BEI_DATA_DIR`/`WRDS_CACHE_DIR`, loads WRDS creds from Colab Secrets, runs training. Its train cell is currently the pre-upgrade command and gets updated when the tuned path lands.

## 6. Decided

1. **Tuning objective**: mean pinball/quantile loss (matches what the quantile boosters are actually trained on; Spearman IC still reported in the PDF as a diagnostic, not the search target).
2. **Compute**: Colab Pro, via the VS Code Colab extension, T4 High-RAM runtime. Kernel runs remotely with no access to the local `Data Source/` folder (empirical test, section 2).
3. **Data transport**: Google Drive (installed, files copied — section 3.3). `BEI_DATA_DIR` env var points the pipeline at the mounted path, zero code change. Code comes from the public GitHub repo via `git clone`.
4. **Resumability**: Optuna SQLite study on Colab-local `/content`, snapshotted to Drive every N trials (survives session death; avoids the FUSE-locking problem of a live SQLite file on Drive) + derived-parquet + WRDS-cache reuse on Drive (avoids re-pulling raw data / WRDS every session) — section 3.3.

## 7. Decided (implementation inputs)

1. **Trial budget**: Optuna `n_trials=400`, `timeout=8h`, whichever hits first (the cap is a safety net; the intent is to let it run to completion).
2. **Stage 1 checkpointing**: the WRDS build must checkpoint **intermittently** and **never re-pull IDs already cached on Drive**. Concretely, in `wrds_extract.py`:
   - `extract_crsp_daily` — today pulls all permno chunks then writes the cache once at the end (crash = total loss). Change to: read cached permnos, pull only the missing ones, and **append each chunk to the cache as it completes**, so a crash keeps finished chunks.
   - `extract_compustat_fundq` — today a single giant IN-list query. Change to chunked-by-gvkey with the same skip-cached + append-per-chunk behavior.
   - `extract_company` — same treatment for consistency (small, low-risk).
   - Net effect: re-running after a crash (or on a fresh session that mounts the same Drive cache) resumes from where it stopped and pulls only what's genuinely missing.

## 8. Methodology — decided

1. **Early-stopping validation split** — within each walk-forward fold, hold out the **most recent 1 quarter of that fold's own training window** (time-ordered, not random) as the early-stopping validation set. The test/holdout quarter is never used to decide when to stop. (See 3.2.)

2. **Optimism / how accuracy is reported** — **single holdout**, not nested CV. The most-recent ~25% of event-quarters is a held-out **test set** the hyperparameter search never sees; PDF #1's accuracy is measured there, with the winning params stated explicitly. The walk-forward folds used *during* tuning are a selection instrument, not the accuracy report. Nested CV rejected on cost/value (~5x compute for marginal gain — see 3.4).

3. **Search granularity** — **one shared hyperparameter set** across all 5 quantile boosters (evaluated by mean pinball across them), reused for the z-regressor and classifier. Not five independent searches. (See 3.2.)

4. **Test-set size** — the most recent **~25% of distinct event-quarters**, rounded to the nearest whole quarter, computed from the data at runtime. ~10 of ~40 quarters given the 2015-2024 span. (See 3.4.)
