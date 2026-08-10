# The Long-History Demand Forecast — complete guide

Everything this system does, why it does it, and what it cannot do.
Written for someone who has never seen it before. No statistics background
needed — every term is explained where it first appears.

For "just tell me the commands", read [TUTORIAL.md](TUTORIAL.md) instead.

---

## Contents

1. [What this system does](#1-what-this-system-does)
2. [The mental model](#2-the-mental-model)
3. [The one rule that shapes everything: no peeking](#3-the-one-rule-that-shapes-everything-no-peeking)
4. [The data](#4-the-data)
5. [COVID, closures, and other bad days](#5-covid-closures-and-other-bad-days)
6. [What the model looks at](#6-what-the-model-looks-at)
7. [How the model decides](#7-how-the-model-decides)
8. [Half-life — the setting that makes long history work](#8-half-life--the-setting-that-makes-long-history-work)
9. [Chinese New Year — the hard part](#9-chinese-new-year--the-hard-part)
10. [Fixed vs moving holidays](#10-fixed-vs-moving-holidays)
11. [Typhoons](#11-typhoons)
12. [How accurate is it](#12-how-accurate-is-it)
13. [Folder layout](#13-folder-layout)
14. [Configuration reference](#14-configuration-reference)
15. [Known limitations](#15-known-limitations)
16. [Tested and rejected](#16-tested-and-rejected)
17. [Troubleshooting](#17-troubleshooting)
18. [Hourly demand splitting](#18-hourly-demand-splitting)

---

## 1. What this system does

It predicts **patron hours** — roughly, how many people are on the floor
multiplied by how long they stay — for the **next 28 days**, one number per
day. That figure drives staff scheduling: too low and the floor is
understaffed, too high and you pay for labour you did not need.

For each of the 28 days you get three numbers:

| output | meaning | how to use it |
|---|---|---|
| **P50** | the median, most-likely figure | plan against this |
| **P10** | low case | a pessimistic bound |
| **P90** | high case | an optimistic bound |

> ⚠️ **Caveat stated up front.** P10–P90 is supposed to contain the real
> answer 80% of the time. Measured, it manages **46–54%**. Treat the band as a
> *relative* confidence signal (wider = less certain), never as a literal
> guarantee. See §15.

**Typical accuracy**, measured on held-out days it had never seen:

| how far ahead | average miss | days within 5% of actual |
|---|---|---|
| 1 day | 3.78% | 79% |
| 7 days | 4.07% | 71% |
| 14 days | 4.96% | 63% |
| 28 days | 4.84% | 62% |

A "4% miss" means: if the real answer was 7,000 patron hours, a typical
forecast lands about 280 either side.

---

## 2. The mental model

The entire system is **one command reading one data file**.

```
long_history/data/raw/rawdata_long.csv
        │
        ▼
  run_pipeline.py   →   validate → evaluate → forecast
        │
        ▼
long_history/output/run_<date>/predictions.csv
```

### Every run starts from scratch

There is **no separate training step** and **no saved model**. Each time you
ask for a forecast, the models are rebuilt from all data available at that
moment, they make their predictions, and they are thrown away. Nothing
persists between runs — so nothing can quietly go stale without anyone
noticing.

The only thing cached is the *feature table* (an intermediate calculation),
and that cache invalidates itself automatically whenever the feature code
changes. See §14.

---

## 3. The one rule that shapes everything: no peeking

**A forecast may only use information that genuinely existed at the time it
was made.**

Suppose today is 1 March and you want to predict 15 March — 14 days ahead:

```
… up to 28 Feb        1 – 14 March           15 March
 data you MAY use   FORBIDDEN (not yet      the day you
                     known on 1 March)       predict
```

On 1 March, demand for 5 March has not happened yet. A model allowed to see it
while learning would score brilliantly in testing and fail completely in real
use. This mistake is called **leakage**, and it is the single most common way
forecasting projects fail.

**How it is enforced:** every backward-looking number carries a rule — it may
not read anything newer than `(days ahead + 1)`. This was verified
empirically, not just by reading the code: a sentinel value was planted in
future data and confirmed never to reach the predictions.

### Why this means 28 separate models

A forecast for tomorrow may use yesterday's numbers. A forecast for four weeks
out may not. They are genuinely different problems, so **each forecast
distance gets its own model** — 28 of them, retrained every run.

| model | predicts | may use |
|---|---|---|
| 1 | tomorrow | data up to yesterday |
| 7 | one week out | nothing from the last 7 days |
| 28 | four weeks out | nothing from the last 28 days |

This is also why accuracy degrades with distance — there is simply less to go
on. **Never quote an accuracy figure without saying how far ahead it was
measured.**

---

## 4. The data

### Required — `data/raw/rawdata_long.csv`

```csv
date,demand,floortables
2015-01-01,4463,240
2015-01-02,4102,240
```

| column | meaning |
|---|---|
| `date` | one row per calendar day, **no gaps** |
| `demand` | patron hours — the thing being predicted |
| `floortables` | tables open on the floor that day |

Current file: **2015-01-01 to 2026-05-28, 4,166 days.**

> **The single most damaging failure mode is a missing day.** Every
> "what happened N days ago" calculation counts backwards through the
> calendar, so one absent row silently corrupts a large number of inputs —
> with no error message. `data/validate.py` is a hard gate that refuses to let
> anything else run if it finds a gap. Never skip it, and never "fix" a gap by
> deleting surrounding rows.

### Optional — `data/raw/typhoons.csv`

Historical T8+ typhoon signal days, refreshed from the Hong Kong Observatory
by `data/scrape_typhoon.py`. Used for record-keeping and the manual override
tool, **not** as a model input — see §11 for why.

---

## 5. COVID, closures, and other bad days

Eleven years of history includes periods where demand was not a real signal.
Three separate mechanisms handle this, and the differences matter.

### Average demand by year

| period | years | what it is |
|---|---|---|
| Normal trade | 2015–2019 | usable history (~4,000–5,900) |
| **COVID** | 2020–2022 | borders shut, casino closed for weeks (~890–2,100) |
| Recovery | 2023 | real but still climbing all year (~4,900) |
| Current | 2024– | today's business (~6,500–7,000) |

### 1. Do not learn from COVID days

The *answer* for each COVID day is blanked out, so those days never become
training examples. The model is never taught that demand of 889 is normal.

### 2. But keep the rows

This sounds wrong but is essential. **Delete the rows and every "what happened
N days ago" calculation reaching across the gap breaks.** Blanking the answer
while keeping the row keeps the calendar continuous.

### 3. Block the year-ago lookups too

A day in 2023 comparing itself against "the same day last year" would be
comparing against collapsed 2022 trade. Any year-ago input whose *source* date
lands inside COVID is blanked as well.

### Closures are found in the data, not typed in

`floortables == 0` means zero tables were open — the floor was shut. That rule
alone correctly finds typhoon Mangkhut (2018-09-16) and both COVID closures,
**26 days in total**. These are *supply* zeros, not demand observations, and
`demand == 0` would also break the accuracy maths (it divides by actuals).

---

## 6. What the model looks at

About **170 pieces of information** per day, in four families. The percentages
are *measured* contribution to the model's decisions, not guesses.

### Holidays — 47%

Nearly half the model. Nine holidays are modelled: Chinese New Year, Golden
Week, Labour Day, Mid-Autumn, Dragon Boat, Ching Ming, Christmas, New Year and
Easter. Three mechanisms:

- **Windows, not days.** CNY runs −7 to +10 days around the anchor date;
  demand is disrupted for weeks, not one day.
- **Per-offset flags.** Each day-offset inside a window gets its own feature,
  so the model learns that day +1 behaves differently from day +4.
- **Fixed vs moving split** — see §10.

Also included: distance to the next holiday, distance from the last one, and
the Mainland-China working-day calendar (which drives visitor flows).

### Recent history — 15%

- **Lags** — demand exactly N days ago (`lag_2` … `lag_728`)
- **Rolling statistics** — mean/std/max/min over 3, 7, 14, 28 days
- **Moving averages** — exponentially weighted, spans 3/7/14/28
- **Same-weekday means** — average of the last 4 and 8 same-weekdays

Every one is horizon-gated (§3).

### Calendar — 12%

Day of week, month, quarter, day of year, week of month, weekend/Friday/
Saturday/Sunday flags, month start/end. Plus **cyclical encodings**: 31
December and 1 January are one day apart in reality but 364 apart as plain
numbers, so the date is also encoded as a position on a circle.

Two features here are **not in the short-history production model**, because
they are noise on 2 years but learnable on 11:

- `week_of_year` — a useful middle granularity (52 buckets vs 12 months vs 365 days)
- `regime` — lets the model condition on pre-covid / covid / recovery / current
  instead of averaging across structurally different periods

### Everything else — 26%

Tables open on the floor, and interactions (holiday × recent level,
lag-per-table, and so on).

---

## 7. How the model decides

The technique is **LightGBM** — gradient-boosted decision trees. The name is
scarier than the idea.

### A decision tree is a stack of yes/no questions

Here is the **actual first question** the trained model asks:

> *Was the busiest day in the last two weeks below 6,708 patron hours?*
> - **YES** → it has been a quiet fortnight → expect a quieter day
> - **NO** → it has been a busy fortnight → expect a busier day

…then another question, and another — day of week? holiday? how many tables
are open? — up to five deep.

Nobody wrote that question. The model found it by trying every possible
question across eleven years of days and keeping whichever one best separated
busy days from quiet ones.

### One tree is weak — 500 together are strong

This is called **boosting**: build a crude rule, see where it is wrong, build a
second rule that fixes those mistakes, then a third that fixes what is still
wrong — 500 times.

A real forecast, watched tree by tree (predicting 2026-05-28, true answer
**6,993**):

| trees used | guess | off by |
|---|---|---|
| 1 | 6,025 | 13.9% |
| 20 | 6,257 | 10.5% |
| 100 | 6,704 | 4.1% |
| 500 | 6,791 | 2.9% |

Most of the gain arrives early. **No single tree is clever; added together
they are.** (Reproduce with `research/model_demo.py`.)

### The whole system in one line

```
170 facts  →  500 trees  →  × 28 models  →  + CNY correction
about the     each correcting   one per         applied
day           the last          forecast        afterwards
                                distance
```

### Why this model and not another

| alternative | why not |
|---|---|
| Classic time-series (ARIMA) | cannot absorb 170 different inputs |
| Neural networks | need far more than a decade of daily data |
| GAM (smoother statistical model) | tested here — worse at all 9 forecast distances |

---

## 8. Half-life — the setting that makes long history work

Older days are deliberately given less say. **Half-life** is how many days
until a day counts half as much.

| half-life | a day from 2016 counts, in 2026 |
|---|---|
| 240 days (short-history production default) | about **1/35,000th** of today |
| **1095 days (this model's default)** | roughly **1/900th** |

**This is the trap.** At a 240-day half-life a 2016 row is mathematically
invisible, so loading eleven years of history and changing nothing else does
*literally nothing*. The half-life is not a follow-up to adding history — **it
IS the change.**

Set by `DEFAULT_HALF_LIFE = 1095` in `config.py`.

Holiday-window rows also get an extra **3× boost** — they are rare but
high-stakes.

---

## 9. Chinese New Year — the hard part

CNY is by far the largest error source. Demand does not dip, it **collapses**:
on the eve of CNY 2025 it fell to **3,331** against a typical 7,300 — under
half of normal — then rebounded within days.

The model struggles for two reasons at once:

1. It has few prior CNYs to learn from (the short-history model has **two**;
   this one has about **ten**).
2. Its "how busy have we been lately" inputs are *themselves* dragged down by
   the same collapse, exactly when it needs them most.

### The fix: look it up instead of learning it

The collapse is dramatic but remarkably **consistent year to year**, which
makes it something you can look up:

1. **Find the matching day** — not the same date, the same *position*. "Three
   days before CNY" this year matches "three days before CNY" last year.
2. **Work out the shape** — last year that day ran at 0.47× normal trade.
3. **Apply to today's level** — multiply this year's normal trade by that
   ratio, corrected for day of week.
4. **Blend it in** — final answer = 20% model + 80% this lookup.

**Measured:** CNY error roughly halved — 11.9% → 6.9% at short range, 16.1% →
7.4% at long range. Calibrated on 2025 and applied blind to 2026.

**CNY only.** The same trick was tried on Easter, Dragon Boat and Mid-Autumn
and made all three *worse*. Their swings are mild (0.85–1.15×) over ranges the
model already handles, so a lookup built from one or two prior years is mostly
noise.

Controlled by `ANCHOR_HOLIDAYS` and `MOVING_ANCHOR_ALPHA` in `engine/core.py`.

---

## 10. Fixed vs moving holidays

This distinction causes more subtle bugs than anything else in the system.

| | fixed | moving |
|---|---|---|
| which | New Year, Labour, Golden Week, Christmas, Ching Ming | CNY, Mid-Autumn, Dragon Boat, Easter |
| behaviour | same date every year | follow the lunar calendar, shift up to 3 weeks |
| last year's version | count back 364 days (= 52 weeks, same weekday) | must match by *position within the holiday* |

### The bug this causes

CNY 2025 fell on **29 January**; CNY 2026 falls on **17 February** — 19 days
later. So for a date in late January 2026, "the same date last year" landed
deep inside the 2025 collapse — comparing an ordinary Tuesday against a day
running at half normal trade.

The short-history model was handed a number far outside anything it had seen
and **missed 2 February 2026 by 27.7%** (predicting 4,964 when the answer was
6,868). The long-history model, same day and same inputs, was off by **3.2%** —
not because anything was patched, but because it had watched about ten Chinese
New Years drift around the calendar.

### A real calendar bug fixed here

Ching Ming is treated as 4 April every year. It is **not** a fixed date — it
is a solar term, and it actually falls on **5 April in 2018, 2019, 2022 and
2023**. This package corrects that, and extends every holiday date back to
2016, each verified at run time against the `chinese_calendar` library rather
than typed in by hand. The pipeline refuses to run if verification fails.

---

## 11. Typhoons

**Typhoons are deliberately not a model input.** The storm is not what closes
the casino — a management decision is.

| storm | operations | demand |
|---|---|---|
| RAGASA (Sept 2025) | suspended | collapsed to 879, an 87% drop |
| TORAJI (Nov 2024) | stayed open | roughly normal (+10%) |

Similar storms, opposite outcomes. No amount of weather data separates them,
because the deciding factor is human.

### What RAGASA actually cost

On 23–24 September 2025 demand fell to 1,295 and 879 against a normal ~7,000.
The forecast said about 5,000 — a **468% miss** on the worst day, the single
worst forecast in the whole eleven years.

Two consequences worth understanding:

- **Those two days alone move the 2025 annual figure from 4.94% to 7.71%.**
  Two days out of 365.
- **The following week was damaged too.** Trade recovered but the forecast
  stayed low, because the model's "how busy have we been lately" inputs were
  poisoned by the two collapsed days. One shock hurts the next week as well —
  the same mechanism that makes CNY hard.

Note that `floortables` stayed at ~285 through RAGASA, so the automatic
closure detection (§5) did **not** catch it.

### What to do instead

Use `ops/typhoon_override.py` to apply a manual adjustment to a saved
forecast when management decides to suspend operations. It writes
`predictions_adjusted.csv` beside the original and records the override in
`metadata.json`; the original `predictions.csv` is never modified.

---

## 12. How accurate is it

All figures below are **held-out**: for each forecast the model trained only
on data available at that moment.

### By forecast distance (2026 year-to-date, 148 days)

| lead | average miss | within 5% | bias |
|---|---|---|---|
| 1 day | 3.78% | 79.1% | +0.30% |
| 7 days | 4.07% | 70.9% | +0.17% |
| 14 days | 4.96% | 62.8% | −0.14% |
| 28 days | 4.84% | 61.5% | +0.74% |

`bias` is direction: positive means the forecast ran high. Read it alongside
the average miss — for staffing, running consistently low is a different
problem from running high.

### Versus the short-history model

Identical protocol — same days, same 7-day lead, same test, only the training
history differs:

| | long history (11 yr) | short memory (2 yr) |
|---|---|---|
| average miss | **4.07%** | 5.08% |
| days missed by >15% | **4** | 8 |
| worst single day | **19.0%** | 60.4% |

**The average understates it.** On a typical day the two are close; the gap
opens at Chinese New Year. On 16 February the actual was 3,518 — long history
said 3,550 (0.9% off), short memory said 5,414 (53.9% off).

The real argument for long history is **fewer disasters**, not a better
average.

### A full year (2025, 7-day lead)

7.71% across all 365 days — but 4.94% excluding the RAGASA storm week. See
§11.

---

## 13. Folder layout

Everything lives under `long_history/`. **Nothing outside this folder is read
or written at runtime.**

```
long_history/
├── run_pipeline.py         ← START HERE. Runs the steps in order.
├── config.py                  Every setting. Edit this, not the scripts.
├── README.md
│
├── engine/                    The feature engine
│   ├── core.py                  feature building, CNY anchor, all flags
│   ├── features.py              matrix assembly, exclusions, caching
│   └── holidays.py              holiday dates 2016-2030, self-verifying
│
├── data/                      Data and its gatekeeping
│   ├── raw/
│   │   ├── rawdata_long.csv     THE input file
│   │   └── typhoons.csv         T8+ signal days (record-keeping)
│   ├── validate.py              hard gate: gaps, duplicates, bad values
│   └── scrape_typhoon.py        refresh typhoons.csv from HKO
│
├── training/                  Model fitting and measurement
│   ├── forecast.py              produce the 28-day forecast
│   ├── evaluate.py              measure accuracy over any range/lead
│   └── experiment.py            compare settings (history × half-life)
│
├── ops/
│   └── typhoon_override.py      manual closure adjustment
│
├── viz/
│   ├── charts.py                charts from an experiment run
│   └── deck_data.py             actual-vs-forecast series for slides
│
├── research/                  One-off investigations, kept for the record
│   ├── alpha_test.py            how hard should the CNY anchor pull?
│   └── model_demo.py            boosting convergence, real numbers
│
├── docs/
│   ├── COMPLETE_GUIDE.md        this document
│   └── TUTORIAL.md              step-by-step how-to
│
└── output/                    Everything the pipeline produces
    ├── run_<YYYYMMDD>/          predictions.csv, forecast.png, metadata.json
    ├── evaluation/              accuracy charts and CSVs
    ├── matrix_cache/            cached feature tables (regenerable)
    └── eval_cache/              cached backtest results (regenerable)
```

### Independence

`engine/core.py` is a **fork** of the production feature engine, copied so
this package is self-contained. Nothing here imports from `v2/` or the repo
root. You can delete the rest of the repository and this folder still runs.

---

## 14. Configuration reference

Everything is in `config.py`. The settings that actually matter:

| setting | default | what it does |
|---|---|---|
| `DATA_FILE` | `data/raw/rawdata_long.csv` | the input file, relative to `long_history/` |
| `DEFAULT_HALF_LIFE` | `1095` | recency weighting (§8). **The key knob.** |
| `EXCLUDE_FROM_TRAINING` | COVID range | date ranges whose answers are blanked |
| `EXCLUDE_CLOSURES` | `True` | drop `floortables == 0` days from training |
| `GUARD_LAGS_CROSSING_EXCLUDED` | `True` | block year-ago lookups into excluded periods |
| `ADD_WEEK_OF_YEAR` | `True` | week-of-year feature (§6) |
| `ADD_REGIME_FEATURE` | `True` | regime feature (§6) |
| `LGBM_PARAMS` | 500 trees, depth 5 | the model itself |
| `CACHE_MATRIX` | `True` | cache the feature table to disk |
| `OUT_DIR` | `output` | where results go, relative to `long_history/` |

### About the caches

Building the feature table takes about **4 minutes** on eleven years, and
every script needs the same one — so it is cached. Later runs load it in about
0.1 seconds.

Both caches (feature table and evaluation results) include a **hash of the
feature engine** in their key, so editing `engine/core.py` or
`engine/features.py` invalidates them automatically. A stale result can never
be silently served. Each cached matrix is roughly 90 MB under
`output/matrix_cache/`; delete the folder any time to reclaim space.

---

## 15. Known limitations

Read these before relying on the output.

1. **The high/low range is too narrow.** P10–P90 should contain the actual 80%
   of the time; measured it manages 46–54%. The band comes from in-sample
   residuals, which understate real uncertainty. Treat it as a relative
   signal. *This is a correctness issue, not an accuracy one — but people may
   be sizing risk off it.*

2. **It cannot see one-off events.** A concert, a competitor closing, a policy
   change. If it is not in the calendar and not in the past, the model does
   not know.

3. **Typhoons are not predicted.** Deliberately — see §11.

4. **The holiday calendar ends in 2026.** The underlying `chinese_calendar`
   library has no 2027 dates and raises an error for them. A 28-day forecast
   made in December 2026 will reach past that boundary. **A known, dated
   deadline.**

5. **It does not pass a strict no-regression bar versus the short-history
   model.** About 0.3 points better on average, improving most held-out
   windows but making two worse. Its stronger claim is the tail (§12).

6. **Eleven years takes minutes, not seconds.** The first run of the day pays
   ~4 minutes to build the feature table.

---

## 16. Tested and rejected

Recorded so nobody re-treads them. Each was measured on held-out data.

| approach | result |
|---|---|
| Blocking the contaminated year-ago comparison (narrow mask) | helped on 1 CNY event, **failed across 6** — long history had already solved it |
| Replacing LightGBM with a GAM | worse at all 9 forecast distances |
| GAM for holidays only (hybrid) | GAM not better on holidays either; CNY a dead tie |
| Horizon pooling (sharing training across nearby distances) | worse on 6 of 6 windows |
| Typhoon as a learned feature | unlearnable — the driver is a closure decision |
| Cleaning closure days out of the feature series | worse (6.28% → 6.75%) |
| Feature pruning to 40 or 60 features | worse than the full set |
| Anchoring Mid-Autumn / Dragon Boat / Easter | worse in 7 of 12 cases |
| Matching CNY on weekday instead of last year | 13.0% vs 7.1% |
| Averaging all prior CNYs instead of last year | 9.4% vs 7.1% |
| Excluding the 2023 recovery year | worse (4.35% → 4.54%) |

**The pattern:** feature-level tinkering does not work on this problem.
Assume any idea is neutral until measured on held-out windows, per-window
rather than on the average.

**And the most transferable lesson:** *a result from a single event is not a
result.* The narrow mask looked convincing on one Chinese New Year and
collapsed when tested against six.

---

## 17. Troubleshooting

| symptom | fix |
|---|---|
| `FATAL: ... does not exist` | put the CSV at `data/raw/rawdata_long.csv`, or edit `DATA_FILE` |
| `MISSING DAY(S) in the date range` | fill the gaps. **Do not** delete surrounding rows |
| `ABORTED: fix the data errors above first` | validation failed — read the message above it |
| Holiday anchors failed to verify | a holiday date disagrees with `chinese_calendar`; check `engine/holidays.py` |
| `NotImplementedError` mentioning 2027 | the calendar deadline in §15 |
| Forecast looks stale after editing features | it should not — the cache self-invalidates. If in doubt, delete `output/matrix_cache/` |
| Everything is slow | first run of the day builds the feature table (~4 min); later runs are seconds |
| Want to reclaim disk | `rm -rf output/matrix_cache output/eval_cache` — both regenerate |

---

## 18. Hourly demand splitting

Splits each day's forecast into 24 hourly values, using the actual
historical hourly distribution — no trained model does the splitting
itself.

### Why an empirical distribution, not a model

For each forecast day, the split picks between two *measured* shapes: the
average hourly pattern from the last 90 similar (non-holiday) days, or —
for holidays whose shape genuinely differs — a recency-weighted average of
past occurrences of that exact holiday-day. The choice is data-driven,
using Total Variation Distance to check whether a holiday actually looks
different from an ordinary day of that type.

A small comparison model exists (`long_history/hourly/comparison_model.py`)
but is used **only inside the validation script**, never to produce an
actual forecast — its whole purpose is giving the "the empirical approach
is good enough" claim a real number to be measured against, rather than
leaving it asserted.

### Current status: pipeline complete, validation pending real data

`long_history/data/raw/hourly_demand.csv` does not exist yet. Every script
in `long_history/hourly/` runs cleanly and has been smoke-tested against a
10-day format-reference sample — but that is a **mechanism check**, not
evidence of accuracy. Real validation requires real hourly history (18-24+
months, multiple occurrences of each major holiday) and runs with one
command once that data lands: `uv run python long_history/hourly/validate.py`.

See [long_history/hourly/README.md](../hourly/README.md) and
[long_history/hourly/TUTORIAL.md](../hourly/TUTORIAL.md) for full detail.

---

## The five things to remember

1. **Eleven years only helps if old days actually count.** The half-life
   setting *is* the change; without it the extra history does nothing.
2. **No peeking.** There are 28 separate models because each forecast distance
   may legally see a different amount of history.
3. **The win is fewer disasters, not a better average** — 0.3 points typically,
   but 27.7% → 3.2% on the worst measured day.
4. **Half the model is holidays**, and Chinese New Year gets its own
   correction because it cannot be learned from one or two examples.
5. **Measure before believing.** Most sensible-sounding improvements tested
   here made things worse, and a result from a single event is not a result.
