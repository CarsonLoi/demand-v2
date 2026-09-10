# Data exploration for the long-history pipeline — design

Status: approved by user 2026-09-10, pending spec review.

## Problem

The long-history pipeline (`long_history/`) trains on 2015-2026 data through a
174-feature engine, but there is no exploratory analysis of either the raw
series or the engineered matrix. Specific questions the user wants answered:

- What do the descriptive statistics of `demand` / `floortables` /
  `demand_per_table` look like, overall and cut by year, weekday, and regime?
- **Is the monthly seasonality shape consistent across years?** ("aligned")
- **Is each holiday's demand impact consistent across years?** ("aligned") —
  especially the pre-CNY collapse that keeps producing mid-January
  under-forecasts.
- What does one row of the feature matrix actually contain after feature
  engineering?
- Which features are correlated with each other (redundancy) and with the
  target?
- What is the feature importance, as **SHAP values** — including *why* a
  specific forecast (e.g. mid-Jan 2026) came out where it did?

## Goal

A standalone exploration deliverable under `research/explore_longhistory/`
that profiles both data layers and produces:

1. Descriptive statistics (overall / by year / by weekday / by regime) as CSVs.
2. Monthly-seasonality-by-year analysis with an explicit **alignment verdict**
   (year-to-year spread + shape correlation), COVID years excluded from the
   comparison.
3. Holiday-impact-by-year analysis with an explicit **alignment verdict**
   (coefficient of variation of the multiplier across clean years), per
   holiday and — for wide-window holidays — per day-offset.
4. A separate COVID section (timeline + closure days + the "2023 is recovery
   not normal" check), kept out of the year-over-year charts.
5. A readable sample of the feature matrix after feature engineering (tall +
   wide + data dictionary).
6. Feature correlation analysis (group heatmap, high-correlation pairs,
   target correlation).
7. SHAP feature importance via LightGBM's native `pred_contrib` — global
   ranking plus per-row local explanations for hand-picked days.
8. A written `EXPLORATION.md` report and a re-runnable `explore.ipynb`
   notebook.

## Non-goals

- Not wired into `run_pipeline.py`. Standalone, run on demand.
- Changes nothing in `long_history/` or `v2/`. The engine is imported
  read-only; `research/explore_longhistory/` is entirely new.
- Not a modelling change. This is analysis only — no new features, no tuning,
  no claim that the model should change. Any redundancy or contamination the
  analysis surfaces is *reported*, not acted on here.
- No new pip dependency. SHAP is computed with `lightgbm`'s built-in
  `pred_contrib` (exact TreeSHAP for tree models), not the `shap` library.

## Design

### Approach

Layered: a pure-function library, an orchestrator script, and a notebook that
imports the same library, so the notebook and script cannot drift and each
computation has one definition.

### Folder layout

```
research/explore_longhistory/
├── eda_lib.py          pure functions, no side effects; imports
│                       long_history/{config,engine} read-only
├── run_explore.py      orchestrator: builds the matrix (cached), calls
│                       eda_lib, writes every CSV/PNG + the EXPLORATION.md
│                       skeleton
├── explore.ipynb       notebook: every analysis as a runnable cell,
│                       charts inline, imports eda_lib
├── EXPLORATION.md      the written report — findings, alignment verdicts,
│                       redundancy list, the SHAP story, chart references
└── output/
    ├── *.csv
    └── *.png
```

Rationale for one self-contained folder rather than the
`research/<name>.py` + `research/<name>_output/` split used elsewhere this
session: this deliverable is five artifact types (library, script, notebook,
report, output), and a folder keeps them together.

### Data

- Raw: `long_history/data/raw/rawdata_long.csv` via
  `F.load_long_demand()` — schema `date,demand,floortables`, 2015-01-01 …
  2026-05-28, one row per day, no gaps (enforced by `data/validate.py`).
- `demand_per_table = demand / floortables.clip(lower=1)`.
- Matrix: `F.build_base(demand, history_start=None, quiet=True)` then
  `F.apply_exclusions(base, demand, [])` — exactly the shipped `off` variant,
  174 features, cached via `config.CACHE_MATRIX`.

### Definitions used throughout

- **Clean years**: 2015-2019 and 2023-2026. COVID years 2020, 2021, 2022 are
  excluded from every year-over-year comparison (per `config.EXCLUDE_FROM_TRAINING`
  the excluded span is 2020-01-23 … 2022-12-31; for the yearly cuts this
  design excludes whole calendar years 2020/2021/2022).
- **Normal day**: `demand > 0` and `floortables > 0` and not inside any
  holiday window (`S.holiday_date_set()` with the extended anchors).
- **Pre-holiday baseline** for holiday *H* in year *Y*: mean demand
  (and mean demand-per-table) over the **normal days in the 28 calendar days
  immediately before the start of H's window that year**. If fewer than 5
  normal days are found, widen the lookback to 56, then 84 days. Lead-in
  only — never uses days inside or after the window. Returns NaN if still
  under 5 normal days (reported, not silently skipped).
- **Holiday window**: `S.HOLIDAY_WINDOWS[name]` offsets around
  `S.HOLIDAY_ANCHORS[name]` for the matching year, with the extended
  2016-2030 anchor set (`F.extended_holidays()` context).
- **Alignment metric**: coefficient of variation (std / mean) of a quantity
  across clean years. Also, for the 12-vector monthly shape, the Pearson
  correlation between each pair of clean years.

### Part A — raw series

**A1. Descriptive statistics**
Functions: `describe_series`, `describe_by_year`, `describe_by_dow`,
`describe_by_regime`, `closure_days`.
For `demand`, `floortables`, `demand_per_table`: count, span, gap count
(expect 0), min, p10, median, mean, p90, max, std, CV, skew, kurtosis.
Cut by: calendar year; day-of-week (Mon-Sun); the four `config.REGIMES`.
Closure/zero list: every date with `floortables == 0` or `demand == 0`, with
the demand and floortables values.
Outputs: `stats_overall.csv`, `stats_by_year.csv`, `stats_by_dow.csv`,
`stats_by_regime.csv`, `closure_days.csv`.

**A2. Monthly seasonality by year — alignment check**
Function: `monthly_index_by_year(demand, clean_years)` →
`monthly_index = (month mean demand_per_table) / (that year's annual mean
demand_per_table)`, a (year × 12) table.
`monthly_alignment(index_table)` → per-month std across clean years; per-month
min/max; a year × year Pearson correlation matrix of the 12-vectors.
Verdict printed: months whose across-year std exceeds 0.10 (index points)
flagged as "seasonally unstable"; mean off-diagonal shape correlation
reported as the headline alignment number.
Outputs: `monthly_index_by_year.csv`, `monthly_alignment.csv`,
`monthly_seasonality.png` (one line per clean year, x = month 1-12,
y = index), `monthly_shape_corr.png` (heatmap).

**A3. Holiday impact by year — alignment check**
Functions: `pre_holiday_baseline`, `holiday_multiplier_by_year`,
`holiday_offset_shape_by_year`, `holiday_alignment`.
- `holiday_multiplier_by_year`: for each of the 9 holidays × each anchor year,
  `mult_raw = mean(demand in window) / mean(demand in pre-holiday baseline)`
  and `mult_per_table` likewise on demand-per-table.
- `holiday_offset_shape_by_year`: for the wide-window holidays
  (`CNY` −7…+10, `GoldenWeek` 0…6, `Labour` 0…4, `Christmas` −3…+3,
  `Easter` −2…+1, `DragonBoat` −2…+1), the multiplier at each day-offset,
  per year.
- `holiday_alignment`: CV of `mult_per_table` across clean years, per holiday.
  Low CV (< 0.10) → "consistent, safe to pool years"; high CV → "each
  occurrence differs — the model needs occurrence-specific signal".
- CNY sub-analysis: the minimum multiplier over offsets −7…−1 by year (the
  pre-CNY trough depth), tabulated and charted — this is the documented
  mid-January failure mode.
Outputs: `holiday_multiplier_by_year.csv`, `holiday_offset_shape_by_year.csv`,
`holiday_alignment.csv`, `cny_trough_by_year.csv`,
`holiday_impact.png` (per-holiday multiplier by year, bar),
`holiday_offset_shape.png` (per-offset line, one line per year, one panel per
wide-window holiday), `cny_trough_by_year.png`.

**A4. COVID section (separate)**
Function: `covid_timeline(demand)` → monthly mean demand-per-table
2019-01 … 2024-12, plus the closure spans, plus the 2023 monthly per-table
trajectory with the config claim ("11.7 → 19.7 across 2023, never reaching the
2024+ norm") checked against the actual numbers.
Outputs: `covid_timeline.csv`, `covid_timeline.png`, and a `covid_notes.md`
fragment folded into `EXPLORATION.md`.

### Part B — engineered matrix

**B1. Sample data after feature engineering**
Functions: `sample_tall`, `sample_wide`, `feature_dictionary`.
- tall: 500 rows sampled evenly across the date range (fixed seed), all
  columns → `sample_matrix_tall.csv`.
- wide: hand-picked target dates at horizon 7 — a normal mid-week day, a
  normal Saturday, `CNY anchor − 3`, `CNY anchor + 1`, a Golden Week day, and
  the mid-January 2026 dates whose forecast ran low (target dates around
  2026-01-24 … 2026-01-31) — transposed so each feature is a row, each
  chosen day a column → `sample_matrix_wide.csv`.
- dictionary: feature name, group (from the documented feature grouping),
  dtype, % non-null over all rows, % non-null over trainable rows, min, mean,
  max → `feature_dictionary.csv`.

**B2. Feature correlation**
Functions: `corr_matrix` (Pearson + Spearman on trainable rows, numeric
features), `high_corr_pairs`, `target_corr`, `group_corr`.
- `high_corr_pairs`: every feature pair with `|r| > 0.9`, ordered by `|r|`,
  labelled with both features' groups — redundancy candidates.
- `target_corr`: each feature's Pearson and Spearman correlation with `y`
  (trainable rows), ranked by `|Spearman|`.
- `group_corr`: mean absolute cross-correlation between feature groups, as a
  small readable heatmap (the full 174×174 grid is not rendered).
Outputs: `corr_high_pairs.csv`, `corr_target.csv`, `corr_group.csv`,
`corr_heatmap_grouped.png`, `corr_target.png` (top ± 30 bar).

**B3. SHAP via LightGBM native `pred_contrib`**
Functions: `train_horizon_model`, `shap_global`, `shap_local`.
- `train_horizon_model(mat, feats, horizon, as_of)`: trains one
  `LGBMRegressor(**config.LGBM_PARAMS)` on `mat` rows with
  `target_date <= as_of`, `horizon == h`, `dropna(subset=["y"])` **only**
  (LightGBM handles NaN features natively — dropping on the full feature list
  empties the training set), with `S.make_sample_weights(...,
  half_life_days=config.DEFAULT_HALF_LIFE)` and
  `S.holiday_mask_from_matrix`. `as_of` = the data's max date: the goal is to
  explain what the deployed model learned from the full history, not to
  benchmark held-out accuracy. Stated plainly in the report.
- `shap_global`: for horizons 1, 4, 7, 14, 21, 28 (CLI `--horizons` can widen
  to all 28, ~15 min), run `model.predict(X, pred_contrib=True)` on a sample
  of up to 4000 trainable rows per horizon; the last column is the bias term.
  Aggregate `mean(|SHAP|)` per feature across horizons, ranked →
  `shap_importance.csv`. Beeswarm-style summary for the top 25:
  per feature, a horizontal strip plot of SHAP value coloured by the
  (rank-normalised) feature value → `shap_summary.png`.
- `shap_local`: for each hand-picked day from B1, at horizon 7, the full
  per-feature SHAP vector for that single row, sorted by contribution →
  `shap_local_<date>.csv`, and a waterfall PNG
  (`shap_waterfall_<date>.png`) showing base value → the ± contributions of
  the top ~15 features → final prediction. For the mid-Jan-2026 days this
  makes the `yoy_ratio` / `lag_365` contribution visible directly.

### Part C — report + notebook

- `EXPLORATION.md`: sections mirroring A1-A4, B1-B3; each states the numeric
  finding, the verdict where there is one (monthly alignment, per-holiday CV
  table, redundancy list, SHAP top features and the Jan-2026 local story),
  and references the relevant PNG(s) by relative path. Written by hand from
  the computed CSVs; `run_explore.py` emits a skeleton with the tables
  pre-filled and `TODO(prose)` markers where narrative is needed.
- `explore.ipynb`: one section per analysis, each a cell (or short cell group)
  that calls `eda_lib` and renders the chart inline, with a markdown cell
  above it restating the question. Re-runnable end to end; reads the same
  cached matrix.

### CLI (`run_explore.py`)

```
uv run python research/explore_longhistory/run_explore.py
    --horizons 1,4,7,14,21,28     # SHAP horizons; "all" = 1..28
    --shap-sample 4000            # rows per horizon for pred_contrib
    --skip-shap                   # Parts A + B1 + B2 only (seconds)
    --baseline-window 28          # pre-holiday lead-in start; widens 28→56→84
```

Prints a progress log and a final summary (the alignment verdicts and the
SHAP top-10) to stdout; writes everything under `output/`.

## Testing

`research/explore_longhistory/` gets a small `test_eda_lib.py` (pytest,
consistent with the repo's `research` test style) covering the pure functions
where a wrong answer would be silent:

- **Pre-holiday baseline is lead-in only**: construct a demand frame with a
  known holiday window and known surrounding values; assert the baseline
  equals the mean of the *preceding* normal days and never includes a
  window-or-later day; assert the widen-to-56/84 fallback triggers and that
  NaN is returned (not a partial mean) when under 5 normal days remain.
- **Monthly index normalisation**: a synthetic flat series → every monthly
  index is 1.0; a series with one month doubled → that month's index is the
  correct ratio.
- **Alignment CV**: identical values across years → CV 0; known spread → known
  CV.
- **Clean-year filter**: assert 2020, 2021, 2022 never appear in any
  year-over-year output.
- **SHAP additivity**: for a trained horizon model and a sample of rows,
  `pred_contrib` rows sum (feature contributions + bias) to the model's
  `predict` output within 1e-6 — the defining property of exact SHAP; a
  failure means the contributions are being read wrong.
- **`train_horizon_model` dropna**: assert it drops on `y` only and that the
  training row count matches `mat[(horizon==h)].y.notna().sum()`.

Plus a `--self-check` fast path in `run_explore.py` that runs the raw-series
parts on the real data and asserts: no gaps, closure days match the known
list (2018-09-16 Mangkhut; 2020-02-05…02-19; 2022-07-11…07-22), and every
holiday-year cell either produced a baseline or is explicitly listed as
"insufficient normal days".

## Honesty / leakage notes

- Part A is descriptive only. The single place a lookahead could sneak in is
  the pre-holiday baseline; the test above pins it to lead-in days.
- Part B2 correlations are computed on trainable rows of the full matrix — a
  descriptive property of the data, no model, no leakage concern.
- Part B3 SHAP models are trained the same way the pipeline trains
  (params, weights, half-life, `dropna` on `y` only), so the explanation
  reflects a realistic model rather than a toy. SHAP explains the fitted
  function on in-sample rows *by design* — it is an attribution of the
  model's own output, not a generalisation claim. The report states this.
- `as_of` = data max is deliberate and disclosed: this explains the
  full-history model, not a walk-forward one.
- Nothing here changes the model. Redundancy pairs and any contamination the
  SHAP local plots reveal are findings for a *separate* decision, recorded in
  `EXPLORATION.md`, not acted on in this work.

## Risks / caveats to carry into the report

- With only ~7 clean years (fewer for late-year or moving holidays), a
  per-holiday CV is itself a small-sample number — report the n alongside it
  and treat a single divergent year as a flag, not a conclusion (consistent
  with the project's "a result from one event is not a result" standard).
- `demand_per_table` still carries the post-2022 market-structure shift even
  after capacity adjustment; 2015-2019 vs 2023-2026 comparisons must note
  this rather than treat the two blocks as interchangeable.
- The mid-Jan-2026 "wide" sample dates are chosen from memory of the
  under-forecast; `run_explore.py` should also print, for those rows, the
  actual vs a quick reference forecast so the sample is anchored to a real
  error, not an assumed one.

## Open questions

None outstanding — form (standalone in `research/`), scope (both layers),
SHAP method (native `pred_contrib`), normalisation (both per-table and
lead-in ratio-to-baseline), COVID handling (excluded from YoY), and
deliverables (report + notebook + CSVs + PNGs) were all resolved during
brainstorming.
