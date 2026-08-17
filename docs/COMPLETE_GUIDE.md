# The Demand Forecast Model — a complete walkthrough

Written for someone who has never seen this project. No prior context assumed.
Read top to bottom; each section builds on the last.

---

## 1. What this system does

Every day it answers one question:

> **How many patron-hours will we see on each of the next 28 days?**

"Patron-hours" is the demand measure — roughly, how much customer presence the
floor has to serve. The forecast drives staffing: how many tables to open, how
many dealers to roster, and which shifts they work. Those decisions are made
days or weeks ahead and are expensive to get wrong in both directions — too few
tables and patrons wait or leave; too many and you pay dealers to stand idle.

For each of the 28 days it outputs three numbers:

| output | meaning | how to use it |
|---|---|---|
| **P50** | the median, "most likely" forecast | plan against this |
| **P10** | low case | a pessimistic bound |
| **P90** | high case | an optimistic bound |

⚠️ **Important caveat, stated up front:** P10–P90 is supposed to contain the
actual 80% of the time. Measured, it contains it only **46–54%** of the time.
The band is roughly half as wide as it claims. Treat it as a *relative*
confidence signal (wider = less certain), never as a literal 80% guarantee.
See §10.

**Typical accuracy: about 4.3% MAPE** across a full year, meaning the forecast
is on average within 4.3% of actual demand. Ordinary days run ~3.8%; Chinese
New Year runs ~7.3%.

---

## 2. The mental model — how a forecast is made

The entire system is **one script reading one data file**.

```
data/raw/rawdata.csv  ──►  forecast.py  ──►  forecasts/run_<date>/predictions.csv
```

There is **no separate "training" step**. Every run retrains the models from
scratch on all data available at that moment, makes its predictions, and throws
the models away. Nothing is stored between runs, so nothing can silently go
stale.

### The one rule that shapes everything: no peeking

A forecast made 14 days ahead **must not** use data from the last 13 days —
that data does not exist yet at forecast time. Violating this is called
*leakage*, and it produces a model that looks brilliant in testing and fails in
production.

This rule has a big architectural consequence:

> **Each forecast distance gets its own model.** 28 separate models — one for
> "tomorrow", one for "2 days out", … one for "28 days out" — because each is
> allowed to see a different amount of recent history.

| forecasting… | may use | may NOT use |
|---|---|---|
| 1 day ahead | data up to yesterday | today (not closed yet) |
| 7 days ahead | data up to 8 days ago | the last 7 days |
| 28 days ahead | data up to 29 days ago | the last 28 days |

This is also why accuracy degrades with distance: the 28-day model is simply
working with less information.

---

## 3. The data

### Required — `data/raw/rawdata.csv`

```csv
date,demand,floortables
2024-01-01,6099,299
2024-01-02,5178,299
```

| column | meaning |
|---|---|
| `date` | ISO date, **one row per calendar day, no gaps** |
| `demand` | patron hours — the thing being predicted |
| `floortables` | how many tables were open that day (capacity) |

**Gaps are the single most dangerous data problem.** Lag features are
*calendar* lookups (`this date minus 365 days`), not positional ones. Delete a
row and you don't shorten the timeline — you create a hole that turns every
lag, rolling average and moving average reaching across it into "missing", for
up to two years afterwards. One absent day can quietly degrade months of
forecasts.

A useful convention already present in the data: **`floortables = 0` means the
casino was closed.** Those are supply-zero days, not demand observations, and
are excluded from training automatically.

### Optional — `data/raw/typhoons.csv`

Historical typhoon signals, scraped from the Hong Kong Observatory. **Not a
model input** (see §8) — kept for audit and for deciding manual overrides.

### Optional and currently unused — `data/raw/reservations.csv`

Hotel booking pace. The code is fully written and leakage-guarded but the file
does not exist, so it is switched off. **This is the biggest untapped
opportunity in the system** — see §11.

---

## 4. Features — what the model actually sees

About **170 features** per row. They fall into four families.

### 4a. Calendar (~12% of the model's decision-making)

`dow` (day of week), `month`, `quarter`, `day_of_year`, `week_of_month`,
weekend/Friday/Saturday/Sunday flags, month-start/end flags, plus **cyclical
encodings** (`dow_sin`, `dow_cos`, `doy_sin`, `doy_cos`).

The cyclical ones exist because 31 December and 1 January are one day apart in
reality but 364 apart as plain numbers. Encoding the date as a position on a
circle fixes that. Note the smooth `doy_sin`/`doy_cos` do most of the seasonal
work — raw `month` barely registers.

### 4b. History / momentum (~15%)

- **Lags** — demand exactly N days ago: `lag_2` … `lag_728`
- **Rolling statistics** — mean/std/max/min over 3, 7, 14, 28 days
- **Moving averages** — exponentially weighted, spans 3/7/14/28
- **Same-weekday means** — average of the last 4 and 8 same-weekdays

Every one is *horizon-gated*: for the 14-day model, none of them may read
anything newer than 15 days before the target.

Measured contribution by lag distance: short (`lag_2`–`lag_28`) 6.5%, medium
(`lag_35`–`lag_182`) 5.9%, long (`lag_365`, `lag_728`) 2.3%. **`lag_728` is
nearly dead weight at 0.13%.**

### 4c. Holidays (~47% — the single biggest block)

Nearly half the model is holidays. Nine are modelled: Chinese New Year, Golden
Week, Labour Day, Mid-Autumn, Dragon Boat, Ching Ming, Christmas, New Year and
Easter.

Three mechanisms:

1. **Windows, not days.** CNY runs −7 to +10 days around the anchor date;
   demand is disrupted for weeks, not one day.
2. **Per-offset flags.** Each day-offset inside a window gets its own feature,
   so the model learns that day +1 behaves differently from day +4.
3. **Fixed vs moving split** — the key distinction, explained in §7.

Also included: distance to the next holiday, distance from the last one, and
the Mainland-China working-day calendar (which drives visitor flows).

### 4d. Interactions

Combinations like holiday × recent-demand-level, or lag-per-table.

---

## 5. The model

**LightGBM** — gradient-boosted decision trees. Chosen after testing
alternatives: classical time series (ARIMA/ETS) cannot absorb 170 features,
and neural networks need far more than a decade of daily data.

Gradient boosting builds many small, deliberately weak trees in sequence, each
correcting the previous ones' mistakes. It suits this problem because it finds
interactions itself, handles missing values natively (important — long-horizon
models have lags deliberately withheld), and trains in minutes on a laptop.

### Three modes

| command | time | what it does |
|---|---|---|
| `forecast.py` | ~3 min | single LightGBM per horizon — the daily driver |
| `forecast.py --full` | ~15–20 min | 6 different models averaged |
| `forecast.py --full --blend selection` | ~30–40 min | 6 models + per-weekday model selection + fixed-holiday anchor |

The six models in the ensemble are LightGBM-L2, LightGBM-quantile, XGBoost,
CatBoost, bagged LightGBM (3 seeds), and a two-stage holiday specialist. They
disagree in different ways, so averaging cancels individual quirks.

### Sample weighting

Not all training rows count equally:

- **Recency decay** — 240-day half-life: a row 240 days old counts half as
  much as today's, 480 days old a quarter, and so on.
- **Holiday upweight** — holiday-window rows count 3× (rare but high-stakes).

The half-life matters more than it looks. At 240 days a row from 2016 carries
**1/35,000th** the weight of a recent one — effectively invisible. Loading a
decade of history without changing this changes nothing.

---

## 6. Chinese New Year — the special case

CNY is the hardest part of the year and gets its own treatment **in every mode,
automatically**.

**The problem:** demand more than halves before CNY — down to **0.46–0.53×**
normal on the eve — then rebounds above normal for a week. And CNY *moves*: it
fell on 10 Feb in 2024, 29 Jan in 2025, 17 Feb in 2026. Up to a three-week
swing. So the model has at most one or two comparable prior examples, which is
far too few to learn a −50% collapse.

**The fix — a post-model correction.** After the normal forecast is produced,
CNY-window days are blended toward a separately-computed estimate:

```
final = 0.20 × model + 0.80 × profile

profile = (last year's demand at the SAME POSITION in CNY,
           divided by last year's normal trade for that weekday)
          × (this year's normal trade for that weekday)
```

Matching by *position within the holiday* (day −3 to day −3) rather than by
calendar date is what makes it work despite CNY moving.

**Measured, with the 0.80 weight calibrated on a prior year and applied blind:**

| | before | after |
|---|---|---|
| CNY, short lead | 11.9% | **6.9%** |
| CNY, long lead | 16.1% | **7.4%** |
| CNY, full year rolling | 14.2% | **7.3%** |
| non-CNY days | unchanged | **unchanged** |

It touches only CNY-window days — mathematically impossible for it to harm the
other ~95% of the forecast.

**Deliberately NOT applied to Mid-Autumn, Dragon Boat or Easter.** Tested: it
made all three *worse* in 7 of 12 cases. Those holidays move demand only
0.85–1.15× over 3–4 days where the model is already accurate; nudging them
toward a thin sample adds noise.

### Why CNY still runs ~7% — and why that is close to the limit

Measured across seven usable years, the CNY shape varies year to year with a
standard deviation of **0.076** against a mean lift of ~0.88 — about 8.6%
variation. **Even a predictor with the perfect average profile would miss by
~7% on a typical CNY day.** Current performance is essentially at that floor.

Two alternatives were tested and are worse: matching on CNY's *weekday*
(13.0%) and averaging all prior years (9.4%), versus 7.1% for simply using
last year.

**Getting materially below ~7% requires new information, not better use of
demand history** — hotel booking pace being the obvious candidate (§11).

---

## 7. Fixed vs moving holidays — why they're handled differently

This is the most important technical distinction in the system.

| | **Fixed-date** | **Moving** |
|---|---|---|
| which | New Year, Labour, Golden Week, Christmas, Ching Ming | CNY, Mid-Autumn, Dragon Boat, Easter |
| behaviour | same calendar date every year | shifts up to 3 weeks |
| year-ago match | **364 days back** (52 weeks: same weekday, ~same date) | **position within the holiday** |
| why | weekday dominates casino demand, and 364 preserves it | a 364-day lag lands on an ordinary day |

**The subtlety worth understanding:** the same matching rule can be right in
one role and wrong in another. 364 days preserves weekday but shifts holiday
*phase* by one day. A tree can learn around that; a directly-applied profile
cannot. So for a 7-day holiday with a strong day-by-day arc (Golden Week),
using 364-day matching as an anchor is destructive (5.2% → 13.1%) even though
it works fine as a model feature.

Fixed holidays also get their own anchor, but only in `--blend selection` mode.
It nudges them toward `last year's same date × recent growth`, since base
models systematically under-predict holiday peaks.

---

## 8. Typhoons — deliberately NOT in the model

Nine typhoon days exist in the record. Their impact ranges from **+10% to
−87%**:

| typhoon | signal | hours | impact |
|---|---|---|---|
| TORAJI 2024 | T8 | 10.3 | **+10.1%** |
| RAGASA 2025 | T8 | 9.7 | **−81.3%** |

Near-identical storm exposure, opposite outcomes. Neither signal level nor
duration explains it. What explains it: **RAGASA suspended casino operations
and TORAJI did not.** That is a business decision, not a weather variable, and
nothing in the weather feed predicts it.

With two closure events on record, a learned coefficient would be memorising a
single event. So `USE_WEATHER = False`, and typhoons are handled by a person:

```bash
uv run python typhoon_override.py --dates 2026-08-12 --show      # inspect
uv run python typhoon_override.py --dates 2026-08-12 --closure   # ops suspended
uv run python typhoon_override.py --dates 2026-08-12 --signal-t8 # ops continuing
```

This writes `predictions_adjusted.csv` alongside the original — the model's
honest output is never overwritten.

---

## 9. Running it

### One-time setup
```bash
cd "D:\Applications\Work\Models\Demand Forecast v2"
uv sync
```

### Routine forecast
```bash
# 1. append yesterday's actuals to data/raw/rawdata.csv
# 2. check for gaps
uv run python -c "import pandas as pd; d=pd.read_csv('data/raw/rawdata.csv', parse_dates=['date']); print('gaps:', list(pd.date_range(d.date.min(), d.date.max()).difference(d.date)))"
# 3. forecast
uv run python forecast.py
```

Confirm these two console lines before trusting anything:
```
Historical data:  2024-01-01 .. 2026-06-16   <- should end at your latest row
Forecasting:      2026-06-17 .. 2026-07-14   <- the next 28 days
```

### Scheduling cycle
```bash
# Day 1 — freeze the plan for a window
uv run python forecast.py --full --blend selection --initial-forecast 2026-06-17 2026-07-14
# Each following day — append actuals, then re-forecast (no flag)
uv run python forecast.py --full --blend selection
```

### Outputs (`forecasts/run_<YYYYMMDD>/`)

| file | contents |
|---|---|
| `predictions.csv` | `date, p10, p50, p90` — the forecast |
| `forecast.png` | 60 days of actuals + the 28-day band |
| `metadata.json` | audit trail: model, data range, config |
| `../comparison.png` | last ~6 runs overlaid — **the key sanity check** |
| `../tracking_comparison.png` | actual vs frozen plan vs latest |

**Always glance at `comparison.png`.** If a new forecast diverges sharply from
recent runs with no obvious reason, investigate before acting on it.

### Measuring accuracy
```bash
uv run python research/backtest.py --fast                       # last 28 days
uv run python evaluate.py --start 2026-01-01 --end 2026-05-28 --n 0   # any range
```

---

## 10. Known limitations — read before relying on this

1. **Prediction intervals are too narrow.** 46–54% coverage against a nominal
   80%. Cause: residuals are measured on the training data, which the model
   fits far more tightly than unseen data. Fix is known (out-of-fold
   residuals), not yet implemented.
2. **CNY runs ~7%** and is close to the information floor (§6).
3. **Accuracy varies a lot by window** — 2.3% to 9.5% across held-out tests on
   the same model. A single month's number is not representative.
4. **`chinese_calendar` expires after 2026.** A 28-day run in December 2026
   will cross into 2027 and **fail outright**. Hard deadline.
5. **No automated data validation** in the production path — the gap check is
   a manual command.
6. **No test suite.** Changes to the feature code are verified only by manual
   backtesting.
7. **Downstream stages not built.** The hourly split exists but lacks its input
   file; table-count, headcount and shift assignment are designed only.

---

## 11. Where the next real gains are

**In priority order, based on measurement rather than intuition:**

1. **Hotel booking data (`reservations.csv`).** Code written, leakage-guarded,
   unused for want of a file. It is the only *forward-looking* signal
   available, and forward-looking is exactly what CNY needs. Highest expected
   value in the system.
2. **Fix the prediction intervals.** Correctness, not accuracy — but people
   may be sizing risk off a band that is half as wide as advertised.
3. **The 2027 calendar deadline.** Cheap, dated, and a hard failure.
4. **More history + a longer half-life.** Tested on 2015–2026 data: a 3-year
   half-life with full history gives ~−0.30pp pooled, improving most windows
   but regressing two. Modest on average — but it also removes a class of
   *catastrophic single-day* errors the short-window model is structurally
   exposed to. Measured: 2026-02-02 was missed by **27.7%** by the production
   model and by **3.2%** by the long-history model, with no other change. That
   failure mode matters more operationally than a 0.3pp average. See §12.

### What has already been tested and rejected

Recorded so nobody re-treads them. Each was measured on held-out data:

| approach | result |
|---|---|
| Horizon pooling (share training across nearby horizons) | worse on 6 of 6 windows |
| Typhoon as a learned feature | unlearnable; driver is a closure decision |
| Cleaning closure days out of the feature series | worse (6.28% → 6.75%) |
| Feature pruning to 40 or 60 features | worse than the full set |
| Anchoring Mid-Autumn / Dragon Boat / Easter | worse in 7 of 12 cases |
| Anchoring fixed-date holidays with the CNY method | worse in 8 of 16 cases |
| Guarding `yoy_ratio` against holiday contamination | net wash, two attempts |
| Narrow demand-driven version of that guard | helped on 1 event, failed on 6 (see §12) |
| Replacing LightGBM with a GAM | worse at all 9 horizons tested |
| GAM for holidays only (hybrid) | GAM not better on holidays either |
| Matching CNY on weekday instead of last year | 13.0% vs 7.1% |
| Averaging all prior CNYs instead of last year | 9.4% vs 7.1% |
| Excluding the 2023 recovery year | worse (4.35% → 4.54%) |

**The pattern:** feature-level tinkering does not work on this problem.
Architectural changes and post-model corrections do — and even those mostly
fail. Any proposed improvement should be assumed neutral until measured on
held-out windows, per-window rather than on the average.

---

## 12. The long-history pipeline (`long_history/`)

### What it is, in one sentence

A **second, complete copy of the whole system** that trains on 11 years of data
(2015–2026) instead of 2 years, kept in its own folder so you can experiment
freely without any risk of breaking the live forecast.

### Why it exists

The production model only trains on data from 2024 onward. That is deliberate —
recent data reflects today's business. But it creates a blind spot: **the model
has only ever seen two Chinese New Years.** When something unusual happens, it
has almost no precedent to reason from.

Here is the clearest example we found.

> **What went wrong on 2 February 2026**
>
> One of the model's features is "how does demand right now compare to the same
> day last year?" On 2026-02-02, "the same day last year" pointed at
> 2025-01-27 — which happened to be two days before Chinese New Year 2025, when
> demand had collapsed to about half of normal.
>
> So the model was handed a comparison like *"today is 1.97× last year"* — a
> number far outside anything in its training data. It had no idea what to do
> with it, and predicted **4,964** when the real answer was **6,868**. That is a
> **27.7% miss**.
>
> The long-history model, given the exact same day and the exact same feature,
> was off by **3.2%**. Not because it was told about the problem — but because
> it had seen roughly ten Chinese New Years drift around the calendar and had
> simply learned what that pattern looks like.

That is the real argument for long history: not a slightly better average, but
**far fewer disasters**.

### The one thing that makes long history actually work

Adding old data alone does nothing. The model weights recent days more heavily
using a setting called **half-life** — how many days until a day counts half as
much.

| half-life | how much a day from 2016 counts, in 2026 |
|---|---|
| 240 days (production) | about 1/35,000th of today |
| 1095 days (long-history default) | roughly 1/900th |

At the production setting of 240, a 2016 row is mathematically invisible. So
**the half-life change *is* the experiment** — not a follow-up to it. That is
why `config.py` defaults to `DEFAULT_HALF_LIFE = 1095`.

### It cannot break production

Everything in `long_history/` uses its own vendored copy of the feature engine
(`core.py`), not the production one. Nothing in the folder imports from `v2/`
or writes anywhere outside `long_history/output/`. Delete the folder and the
live forecast is unaffected.

Three deliberate corrections live in that fork:

1. Holiday dates extended back to 2016 (production only ships 2024+).
2. Moving-holiday dates verified against the `chinese_calendar` library.
3. **A real bug fixed:** production hardcodes Ching Ming to 4 April every year.
   It is a solar term and actually falls on 5 April in 2018, 2019, 2022 and
   2023.

### Running it — the normal case

One command does everything:

```bash
uv run python long_history/run_pipeline.py
```

That runs three steps in order and **stops at the first failure**:

| step | what it does | why it matters |
|---|---|---|
| **validate** | checks the data file for gaps, duplicates, impossible values | a single missing day silently corrupts every lag feature — this is a hard gate |
| **evaluate** | measures how accurate the model is, and produces charts | tells you what to expect before you trust a number |
| **forecast** | produces the real 28-day forecast | the actual deliverable |

### Running individual pieces

```bash
# Just check the data is usable
uv run python long_history/run_pipeline.py --steps validate

# How accurate were we in 2026, forecasting 7 days ahead?
uv run python long_history/evaluate_long.py \
    --start 2026-01-01 --end 2026-05-28 --n 6 --mode rolling_window --by-lead

# A 28-day forecast from a specific date
uv run python long_history/forecast_long.py --run-date 2026-04-30

# Which settings are best? (slow — only when the data or business changes)
uv run python long_history/run_pipeline.py --steps experiment --quick
```

### Understanding `--n` (this trips people up)

`--n` sets **how far ahead the forecast was made**, which is the single biggest
driver of accuracy. A "MAPE" number means nothing without it.

| setting | meaning | plain English |
|---|---|---|
| `--n 0` | data up to `t-1` predicts `t` | "tomorrow's forecast" |
| `--n 6` | data up to `t-1` predicts `t .. t+6` | "the week-ahead forecast" |
| `--n 27` | data up to `t-1` predicts `t .. t+27` | "the full 28-day plan" |

Worked example — evaluating May 2026 at `--n 6`:

```
origin 2026-05-01  ->  predicts May 2, 3, 4, 5, 6, 7, 8
origin 2026-05-08  ->  predicts May 9, 10, 11, 12, 13, 14, 15
...and so on, tiling the month
```

Each forecast is trained **only** on data up to its own origin. May 2's
forecast never sees May 3.

And the two modes:

- **`fixed_lead`** — every point measured at the same distance. Answers *"how
  good are we 7 days out?"*
- **`rolling_window`** — mimics re-forecasting every 7 days. Answers *"what
  would our actual operating forecast have looked like?"*

### What you get back

A real run over May 2026 at a 7-day lead:

```
=== Accuracy ===
  MAPE        3.18%   (average % miss)
  WAPE        3.15%   (total error ÷ total demand)
  RMSE          271    (patron hours)
  Bias          -30    (+ = over-forecast)
  Coverage   78.6%   (P10-P90, nominal 80%)
  Days          28

=== By segment ===
  ordinary     n=  23   MAPE   3.21%
  Labour       n=   5   MAPE   3.04%
```

Plus charts in `long_history/output/evaluation/`: actual vs forecast with
holiday windows shaded and a % error panel, and an accuracy-vs-lead-time curve.

**Read `Bias` alongside `MAPE`.** MAPE says how far off you were; bias says
which direction. A bias of −30 means the forecast ran slightly low overall —
which for staffing is a different problem from running high.

### Speed

Building the feature matrix takes about 4 minutes on 11 years of data, so it is
**cached to disk** (`config.CACHE_MATRIX = True`). The first run pays the cost;
later runs load it in about 0.1 seconds.

The cache key includes a hash of `core.py` itself, so **editing the feature
engine invalidates the cache automatically**. A stale matrix can never be
silently served. Each cached matrix is roughly 90 MB, under
`long_history/output/matrix_cache/`.

### The files

| file | role |
|---|---|
| `run_pipeline.py` | **start here** — runs the steps in order |
| `config.py` | every setting; edit this, not the scripts |
| `validate_data.py` | the data gate |
| `core.py` | vendored feature engine (the fork of `v2/_shared.py`) |
| `features.py` | builds the matrix, applies exclusions, caches |
| `holidays_extended.py` | holiday dates back to 2016, verified |
| `forecast_long.py` | produces a 28-day forecast |
| `evaluate_long.py` | measures accuracy over any range and lead |
| `experiment.py` | compares settings (history length × half-life) |
| `charts.py` | charts from an experiment run |

### How COVID is handled

2020–2022 demand is not a real signal — borders were closed and the casino was
shut for weeks. Two separate things are done about it, and the difference
matters:

1. **The model does not learn from those days.** Their answer (`y`) is blanked
   out, so they never become training examples.
2. **Their demand values stay in the series anyway.** This sounds wrong but is
   essential: if you *deleted* the rows, every "what happened 365 days ago"
   feature reaching across the gap would break. Blanking the answer while
   keeping the value keeps the calendar continuous.

On top of that, any year-ago feature whose **source** lands in COVID gets
blanked too — otherwise a 2023 row would compare itself against a collapsed
2022 baseline.

Closure days are found from the data, not hardcoded: `floortables == 0` means
zero tables were open. That correctly catches typhoon Mangkhut (2018-09-16) and
both COVID closures. These are *supply* zeros, not demand observations.

### Honest status

- The long-history setup gives roughly **−0.3pp** pooled MAPE versus
  production, improving most held-out windows but regressing two. It does not
  pass a strict no-regression bar.
- Its stronger argument is the **catastrophic-error reduction** shown above
  (27.7% → 3.2% on the worst measured day).
- Two ideas tested here and **rejected**: a narrow demand-driven contamination
  mask (helped on the 2026 event, then failed when validated across six), and
  swapping LightGBM for a GAM (worse at all nine horizons tested). Both are
  recorded in §11 so nobody re-treads them.

---

## 13. File map

**Core**
| file | role |
|---|---|
| `forecast.py` | the entry point — trains, predicts, saves, charts |
| `v2/_shared.py` | all feature engineering + the CNY anchor + every config flag |
| `blend.py` | model selection + fixed-holiday anchor (`--blend selection` only) |

**Operations**
| file | role |
|---|---|
| `run_schedule.py` | drives a day-by-day scheduling cycle |
| `typhoon_override.py` | manual typhoon/closure adjustment |
| `scrape_typhoon.py` | refreshes the typhoon audit file from HKO |

**Evaluation**
| file | role |
|---|---|
| `evaluate.py` | accuracy over any date range / lead time / cadence |
| `research/backtest.py` | quick MAPE check on the most recent 28 days |
| `research/holiday_anchor.py` | re-derives the fixed-holiday anchor calibration |
| `research/*.py` (others) | the record of rejected approaches |

**Long-history pipeline** — `long_history/` — a fully self-contained second
copy of the system trained on 2015–2026. Start with
`long_history/run_pipeline.py`. See §12 and the folder's own README.

**Documentation**
| file | role |
|---|---|
| `README.md` | operations guide |
| `TUTORIAL.md` | step-by-step how-to |
| `docs/MODEL_REVIEW.md` | full technical review, incl. leakage audit |
| `docs/REPRODUCTION.md` | how the published benchmark numbers were derived |
| `docs/COMPLETE_GUIDE.md` | this document |

---

## 14. The five things to remember

1. **One script, one data file, retrains every run.** Nothing persists.
2. **No peeking** — 28 separate models exist because each forecast distance may
   legally see a different amount of history.
3. **Half the model is holidays**, and CNY gets a dedicated correction because
   it is unlearnable from one or two examples.
4. **Typhoons are a human decision, not a model feature** — the driver is
   whether operations close.
5. **Measure before believing.** Most plausible improvements tested here made
   things worse. The bar is per-window, not on the average — and a result from
   a single event is not a result. The narrow mask (§12) looked convincing on
   one Chinese New Year and collapsed when tested against six.
