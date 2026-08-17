# Plan — extending history to 2016, and a pipeline completeness audit

Two separate pieces of work, written up before any code is changed.

**Part 1** — how to add pre-2024 history (and handle COVID) without breaking
the model.
**Part 2** — an honest audit of what the current pipeline does and does not
cover.

---

# Part 1 — Extending history back to 2016

## 1.0 Three findings that shape everything below

These were measured against the current code, not assumed.

### Finding A — old data currently counts for nothing

`make_sample_weights` applies exponential recency decay with a **240-day
half-life**. For a run dated 2026-05-28:

| record | days back | training weight |
|---|---|---|
| May 2026 | 27 | 0.925 |
| CNY 2025 | 484 | 0.247 |
| CNY 2024 | 838 | 0.089 |
| CNY 2019 | 2,669 | **0.00045** |
| mid-2016 | 3,648 | **0.0000266** |

A 2016 row would carry roughly **1/35,000th** the weight of a recent row.

> **Appending 2016–2023 data on its own will change the forecast by
> approximately nothing.** The weighting change is not an optional extra — it
> *is* the work. Anyone who loads ten years of history, sees no improvement,
> and concludes "more data doesn't help" will have drawn the wrong conclusion
> from a config value.

### Finding B — deleting COVID rows does not skip the period, it punches a hole

Lag features are **calendar lookups**, not positional ones:

```python
date_to_demand.get(T - pd.Timedelta(days=L))
```

Measured, by deleting a four-month block and re-reading the lags:

| feature | source date | before delete | after delete |
|---|---|---|---|
| `lag_7` | outside block | 7,392 | 7,392 |
| `lag_60` | **inside block** | 6,179 | **NaN** |
| `lag_91` | **inside block** | 8,293 | **NaN** |
| `lag_182` | outside block | 7,252 | 7,252 |

Deleting rows does not compress the timeline. Every lag, rolling window and
EWMA that reaches across the deleted span returns NaN — for **years** of
subsequent dates, since `lag_728` reaches back two years. This is the same
mechanism the README already warns about ("a single missing day silently
corrupts every rolling window and lag feature"), just at a 1,400-day scale.

**So: do not delete the COVID rows.** The correct lever is separating *"which
rows the model learns from"* (training targets) from *"which demand values
feed the lag features"* (the input series). These are independent, and the code
already supports the distinction — training does `.dropna(subset=["y"])`, so a
row whose `y` is NaN is excluded from learning while its demand value remains
available to `date_to_demand` for other dates' lags.

### Finding C — `chinese_calendar` covers 2015–2026 and fails in 2027

| year | `is_workday()` |
|---|---|
| 2015–2026 | OK |
| **2027** | **NotImplementedError** |

Good news for this project: the mainland holiday-block features will work
across the whole 2016+ extension.

Bad news, unrelated but urgent: **the model cannot build features for 2027
dates at all.** A 28-day forecast run in December 2026 will cross into 2027
and fail. This is a hard deadline independent of this plan — see Part 2, gap 6.

---

## 1.1 First: a data question that must be settled before anything else

The request says *"headcount data back to 2016"*. The model's target is
**`demand` (patron hours)**, and it also requires **`floortables`**. Before
planning further:

1. Is the historical series actually patron hours on the **same definition**
   as 2024+ — same measurement method, same venues, same opening hours?
2. Does `floortables` exist for 2016–2023, or would it need backfilling?
3. If "headcount" means something different (e.g. dealer headcount, or
   headcount as a *proxy* for demand), that is a **different target variable**
   and cannot simply be concatenated.

A definition change mid-series is worse than no data: it teaches the model a
step change that is an artefact of measurement, not of demand. This needs a
concrete answer before any code is written.

---

## 1.2 The regime problem is bigger than COVID

The request frames 2020–2023 as the exception. It is more complicated:

| period | regime | usable? |
|---|---|---|
| 2016–2019 | pre-COVID normal | level almost certainly different; shape may transfer |
| 2020-01 – 2022-12 | COVID + border closures | **not representative of anything** |
| 2023 | recovery, non-stationary | transitional; level moving throughout |
| 2024–2026 | current regime | the model's actual operating regime |

Two structural breaks matter beyond the pandemic itself:

- **The junket crackdown (2022)** permanently changed the Macau VIP market.
  "Recovery" did not return demand to the 2019 baseline — it returned to a
  *different market structure*.
- **Any property-level changes** over ten years (renovations, new floors,
  smoking policy, table mix) will show up as level shifts the model has no
  feature to explain.

The practical consequence: **pre-2020 data is plausibly informative about
holiday and weekday SHAPE, and probably misleading about LEVEL.** The plan
below is built around that distinction.

---

## 1.3 Recommended approach — staged, each stage independently validated

The goal is more examples of *rare repeating events* (there are currently only
1–2 usable Chinese New Years — the root cause of the CNY anchor's existence).
2016–2019 would add four more CNY occurrences, four more Golden Weeks, and so
on.

### Stage 0 — data preparation (no model changes)

1. Assemble `rawdata.csv` covering 2016-01-01 → present, **fully
   calendar-contiguous, no gaps**, with COVID rows *present* (see Finding B).
2. Add an explicit `regime` column rather than inferring it from dates in code:
   `pre_covid` / `covid` / `recovery` / `current`, with the boundary dates
   recorded in a config file and justified in writing.
3. Validate: no gaps, no duplicates, no zero/negative demand, no
   definition-change discontinuities. (Part 2, gap 1 proposes making this a
   first-class script rather than a README one-liner.)
4. Plot the full ten-year series and **look at it** before modelling anything.

### Stage 1 — holiday configuration back to 2016

`ANCHOR_YEARS` is currently `range(2024, 2030)`. Fixed-date and computed
holidays extend trivially:

- **Fixed** (New Year, Labour, Golden Week, Christmas): change to
  `range(2016, 2030)` — generated, cannot go stale.
- **Easter**: already computed by `_easter(y)` for any year — no work.

**Moving holidays must be hand-entered** — there are currently 3 anchors each
and 8 more years are needed. Required anchor dates:

| year | CNY | Mid-Autumn | Dragon Boat | Ching Ming |
|---|---|---|---|---|
| 2016 | 02-08 | 09-15 | 06-09 | 04-04 |
| 2017 | 01-28 | 10-04 | 05-30 | 04-04 |
| 2018 | 02-16 | 09-24 | 06-18 | 04-05 |
| 2019 | 02-05 | 09-13 | 06-07 | 04-05 |
| 2020 | 01-25 | 10-01 | 06-25 | 04-04 |
| 2021 | 02-12 | 09-21 | 06-14 | 04-04 |
| 2022 | 02-01 | 09-10 | 06-03 | 04-05 |
| 2023 | 01-22 | 09-29 | 06-22 | 04-05 |

Two cautions:

- **Every one of these must be verified against an authoritative source**
  before use — `chinese_calendar.get_holiday_detail()` covers 2015+ and can
  confirm CNY. A wrong anchor is worse than a missing one: it teaches the model
  a holiday shape on the wrong days.
- **Ching Ming is a solar term** and moves between 4 and 5 April (occasionally
  6). It is currently hardcoded to 04-04 for all years, which is already
  slightly wrong for 2018/2019/2022/2023. Fix while extending.

Also extend `research/audit_holiday_features.py` (which already checks
anchor coverage and moving-holiday drift) to run across the full range and
confirm every year resolves.

### Stage 2 — exclude COVID from training targets, keep it in the feature series

Add a config list of excluded date ranges, and set `y = NaN` for those rows in
`build_matrix`. Effects:

- Those dates are **not learned from** (training already drops NaN targets).
- Their demand values **remain available** to `date_to_demand`, so the calendar
  stays contiguous and no lag returns NaN because of the exclusion.

Then a second, separate decision: **should COVID demand values feed the lag
features of post-COVID dates?** `lag_728` reaches back two years, so a
2024-01-01 forecast currently would read 2022-01-03 — deep COVID. Options,
in increasing order of intervention:

- **(a)** Leave them — the model sees the real history, distorted lags and all.
- **(b)** Cap the usable lookback so no feature crosses a regime boundary.
- **(c)** Substitute a regime-adjusted value for COVID dates in the feature
  series only.

Note that (c) is essentially what I tried for typhoon closure days earlier in
this project — and **it measurably made things worse** (pooled MAPE 6.28% →
6.75%). That is a caution, not a prohibition: the closure case was two days,
this is three years, and the mechanism may differ. But it must be measured, not
assumed.

**Recommendation: start with (a).** It is the null hypothesis, requires no new
code, and if the regime flag and re-weighting do their job the model can learn
to discount that period itself.

### Stage 3 — the actual lever: re-weighting

This is where the value is (Finding A). Options to test, cheapest first:

1. **Lengthen the half-life.** 240 → 480 / 720 / 1095 days. At 720 days, CNY
   2019 goes from weight 0.00045 to ~0.076 — from irrelevant to comparable
   with CNY 2024 today.
2. **Regime-aware weighting.** Down-weight or zero the COVID block explicitly
   rather than relying on decay; optionally up-weight pre-COVID *holiday* rows
   specifically, since shape is what they contribute.
3. **Decouple holiday weighting from recency.** The current
   `HOLIDAY_UPWEIGHT = 3.0` multiplies a decayed weight, so old holidays are
   still nearly invisible. A floor on holiday-row weight regardless of age
   would directly target the "only 1–2 CNY examples" problem.

**Expect a trade-off.** A longer half-life means more history *and* slower
adaptation to genuine recent change. There will be an optimum; it must be
found on held-out data, not chosen.

### Stage 4 — level normalisation (only if Stage 3 is insufficient)

If old data helps holiday shape but drags the level, the next step is to make
the target scale-free — modelling demand *relative to* a slow local baseline
so that a 2017 CNY and a 2026 CNY are comparable in shape even at different
absolute levels.

Flagged honestly: **a ratio-to-baseline target was already tested and rejected**
(`research/ratio_probe.py`, documented in TUTORIAL §8). That test was on
2024+ data where no regime break exists, so the conclusion may not transfer —
but it is prior evidence against, and this stage should not be attempted until
Stages 1–3 are exhausted.

---

## 1.4 Validation protocol — non-negotiable

Given that **five** promising changes in this project have already failed their
held-out test (horizon pooling, two `yoy_ratio` fixes, closure-day cleaning,
two anchor extensions), this must be measured the same way:

1. **Baseline first.** Record current production MAPE on the six standard
   windows *before* touching anything.
2. **One change at a time.** Data extension, then holiday config, then
   weighting — never bundled. A bundled improvement cannot be attributed, and
   a bundled regression cannot be diagnosed.
3. **The gate is per-window, not average.** No window may be worse than
   production. An improved mean hiding one bad window is a fail.
4. **Report CNY and non-CNY separately.** The whole point is more holiday
   examples; if CNY improves and ordinary days degrade, that is a real
   trade-off requiring a decision, not a win.
5. **Confirm on the 2026 YTD rolling backtest** (21 weekly origins) — the
   closest thing to the operational number.

### Honest prior

More data addressing a data-sparsity problem is the **most plausible remaining
lever** — the TUTORIAL's own figures (15 months → ~8%, 22 → ~5%, 28 → ~3%)
show history length dominating. But the regime break is severe, and it is
entirely possible that:

- pre-2020 data helps *only* holiday shape, not level; or
- it helps nothing, because the 2022 market-structure change makes even the
  shape non-transferable.

Both are real outcomes. The plan is designed so either can be detected cheaply
and recorded, rather than discovered after a large refactor.

---

# Part 2 — Is the current script a complete pipeline?

**Short answer: it is a complete and unusually well-validated *forecasting
script*. It is not a complete production *pipeline*.** The modelling core is
strong; the operational scaffolding around it is largely absent.

## What is genuinely solid

| Area | State |
|---|---|
| Feature engineering | Horizon-safe by construction; I tried to break the leakage gating and could not |
| Leakage discipline | Training filtered to `<= run_date`; verified empirically with an injected 99,999 sentinel |
| Training / prediction | Retrains from scratch each run — no stale artefacts, nothing to drift |
| Run archiving | Every run saved by date with predictions, chart, metadata |
| Documentation | Well above typical — README, TUTORIAL, MODEL_REVIEW, REPRODUCTION |
| Negative-results record | Rejected approaches documented with numbers, so they are not re-tried |

## Gaps, ordered by risk

### 1. No enforced data validation — highest risk
`load_demand()` is a bare `read_csv`. It does not check for gaps, duplicates,
non-positive values or type coercion. The single most damaging documented
failure mode ("a single missing day silently corrupts every lag") is guarded
only by a one-liner in the README that a human must remember to run.
**Fix:** a `validate_data.py` that hard-fails on gaps/dupes/negatives, called
automatically at the top of `forecast.py`.

### 2. No tests at all
No `test_*.py`, no `conftest.py`, no pytest config, no CI. `v2/_shared.py` was
modified repeatedly during this project with **no automated safety net** — the
only protection was manual backtesting after each change.
**Fix:** a small suite covering the properties that actually matter — leakage
gating (no feature reads past `origin-1`), anchor no-op on non-CNY days,
holiday-window membership, matrix shape/column stability.

### 3. Prediction intervals are wrong — actively misleading
P10–P90 is built from **in-sample** residuals; measured coverage is **46–54%**
against a nominal 80%. Anyone setting staffing to a service level from that
band is being misled about their risk.
**Fix:** out-of-fold residuals. This is a correctness fix, not an accuracy one.

### 4. No accuracy monitoring
Nothing tracks whether live accuracy is degrading. `evaluate.py` and
`research/backtest.py` exist but are manual.
**Fix:** log realised error per run as actuals arrive; alert on drift.

### 5. Derived-data cache can silently serve stale results
`data/derived/base_preds_*.csv` are keyed only by window name, not by code
version or input hash. This already caused a real incident — the deck published
numbers from a 17 June cache against a 30 July feature set, and they no longer
reproduced (see REPRODUCTION.md).
**Fix:** hash the inputs + relevant config into the cache key, or delete on any
`_shared.py` change.

### 6. `chinese_calendar` expires at the end of 2026 — hard deadline
Mainland block features raise `NotImplementedError` for 2027 dates. A 28-day
run in December 2026 crosses into 2027 and **will fail**, not degrade.
**Fix:** bump the package when 2027 support ships; add a startup check that
warns well ahead of the boundary.

### 7. Manual data ingestion
Actuals are appended to a CSV by hand. No schema enforcement, no provenance, no
audit trail of what changed when.

### 8. Downstream stages incomplete
`hourly/` exists but is **not operational** (requires
`data/raw/hourly_demand.csv`, which does not exist). Deck stages 3–5 (table
count → headcount → shift assignment) are designed, not built. The deck is
honest about this, labelling them "DESIGNED".

### 9. No orchestration
No scheduler, no CI, no Makefile, no container. Every run is a human typing a
command.

### 10. Known-stale config
`pyproject.toml` still describes the project as a *"standalone simple
demand-forecasting playground (v1, v2, v3, hybrid)"* and pulls in
**NeuralProphet → torch + pytorch-lightning** — a very large dependency for a
model that was benchmarked and rejected, and whose scripts have now been
deleted. Removing it would substantially shrink install time and surface area.
Also: no lint/test dependency group despite a `[tool.ruff]` section.

## Suggested priority

| Priority | Item | Why |
|---|---|---|
| **P0** | Data validation (1) | Prevents the worst silent failure |
| **P0** | 2027 calendar deadline (6) | Hard failure, fixed date |
| **P1** | Tests (2) | Everything else is riskier without it |
| **P1** | Interval calibration (3) | Currently misleading a real decision |
| **P2** | Cache keying (5) | Already caused one incident |
| **P2** | Accuracy monitoring (4) | Cannot manage what is not measured |
| **P3** | Dependency cleanup (10) | Cheap, low risk |
| **P3** | Ingestion / orchestration (7, 9) | Quality-of-life |

**Note on sequencing:** items 1 and 2 should land *before* the Part 1 history
extension, not after. That work will touch `_shared.py` substantially, and
doing so without data validation or a test suite is how silent regressions get
introduced.
