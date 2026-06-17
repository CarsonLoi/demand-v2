# Demand Forecast — Detailed Tutorial

A step-by-step guide to running, understanding, and maintaining the daily casino
demand forecast. Written for an operator who did not build the model.

---

## 0. Mental model (read this first)

The system answers one question: **how many patron-hours will we see on each of
the next 28 days?** It outputs three numbers per day:

- **P50** — the median ("most likely") forecast. Use this for planning.
- **P10 / P90** — a low/high band. Actual demand should land between them most of
  the time. Use the width as a "how confident am I" signal.

Everything runs from **one script (`forecast.py`)** reading **one data file
(`data/raw/rawdata.csv`)**. The model retrains from scratch every run — there is
no separate "training" step to manage.

There are three accuracy tiers, trading speed for precision:

| Command | Time | What it does | When to use |
|---|---|---|---|
| `uv run python forecast.py` | ~3 min | Single LightGBM-L2 model | Daily driver |
| `uv run python forecast.py --full` | ~15-20 min | 6-model equal blend | Weekly / committed |
| `uv run python forecast.py --full --blend selection` | ~30-40 min | 6-model + smart selection + holiday anchor | Monthly planning, holiday months |

The third tier is the most accurate (see Section 6) and is the recommended one
for any forecast you'll stake a budget or staffing plan on.

---

## 1. One-time setup

```powershell
cd "D:\Applications\Work\Models\Demand Forecast v2"
uv sync
```

This installs Python 3.11 and all dependencies into `.venv\`. Takes ~5 minutes
the first time, <10 seconds thereafter.

Smoke test:

```powershell
uv run python forecast.py
```

If you see `=== Done — open forecasts\run_<date>\forecast.png ===` and a PNG
appears under `forecasts\`, you're operational.

---

## 2. The daily / weekly workflow

### Step 1 — Update the demand data

Open `data\raw\rawdata.csv`, append yesterday's actuals at the bottom:

```csv
date,demand,floortables
...
2026-06-15,6993,289
2026-06-16,7104,289      <- new row
```

Rules: one row per day, **no gaps**, ISO dates, integer demand & floortables.
Save as CSV (UTF-8).

Verify after editing:

```powershell
uv run python -c "import pandas as pd; d=pd.read_csv('data/raw/rawdata.csv', parse_dates=['date']); gaps=pd.date_range(d.date.min(), d.date.max()).difference(d.date); print('rows:', len(d), 'last:', d.date.max().date(), 'gaps:', list(gaps))"
```

You want `gaps: []`. A single missing day silently corrupts the lag features.

### Step 2 — (Optional) Refresh typhoon data

Only needed after a typhoon season, or if a T8+ signal occurred recently:

```powershell
uv run python scrape_typhoon.py
```

This rebuilds `data\raw\typhoons.csv` from the Hong Kong Observatory. The model
uses one weather feature, `is_typhoon_t8plus`. (See README "Weather data".)

### Step 3 — Run the forecast

For routine use:

```powershell
uv run python forecast.py
```

For a committed monthly plan (recommended for accuracy):

```powershell
uv run python forecast.py --full --blend selection
```

Watch the first two console lines to confirm the data loaded correctly:

```
  Historical data:    2024-01-01 .. 2026-06-16 (898 days)   <- should end at your latest row
  Forecasting:        2026-06-17 .. 2026-07-14              <- next 28 days
```

### Step 4 — Read the output

Everything lands in `forecasts\run_<YYYYMMDD>\`:

| File | What it is |
|---|---|
| `predictions.csv` | `date, p10, p50, p90` for the next 28 days — the forecast |
| `forecast.png` | Chart: 60 days of actuals + 28-day forecast band |
| `metadata.json` | Run audit: model used, data range, P50 summary |
| `..\comparison.png` | Overlay of the last ~6 runs (the key sanity check) |

**Always glance at `comparison.png`.** If the new forecast diverges wildly from
recent runs without an obvious reason (new data, holiday, table change),
investigate before trusting it.

---

## 3. What each forecast mode actually does

### Default (`forecast.py`)

Trains one LightGBM-L2 model per horizon (28 models, one for "1 day ahead", one
for "2 days ahead", ... "28 days ahead"). Fast, solid, slightly less accurate
than the blends.

### Equal hybrid (`--full`)

Trains **6 different models** per horizon and averages them:

1. LightGBM-L2 (squared-error)
2. LightGBM-quantile (median objective)
3. XGBoost
4. CatBoost
5. LightGBM bagged (3 seeds)
6. Two-stage holiday specialist (baseline + holiday-residual model)

Averaging diverse models cancels their individual quirks. This is the original
"hybrid".

### Selection + anchor (`--full --blend selection`) — the recommended mode

Same 6 models, but two honest improvements layered on top (both validated by
extensive backtesting in `research/`):

**A. Per-(day-of-week, horizon) selection.** Instead of averaging all 6 models
equally, it learns *which 2 models are best for each weekday-and-lead-time
combination* and averages only those. A model that's great at "Saturday, 1 day
ahead" may be poor at "Tuesday, 14 days ahead" — selection exploits that.

**B. Holiday anchor.** Base models systematically under-predict fixed-date
holiday spikes (Labour Day, Golden Week, New Year, Christmas, Ching Ming). The
anchor nudges those days toward *last year's same date × recent growth*, which
is a far better holiday estimate. Lunar holidays (Chinese New Year, Mid-Autumn)
are left alone because last-year's-date doesn't line up for them.

**Both are calibrated honestly** — on a recent *validation* slice of history with
known answers, never on the future being predicted. That's why this mode takes
~2× longer: it does a calibration round, then a production round.

---

## 4. How "honest calibration" works (and why it matters)

The danger in forecasting is **leakage** — accidentally letting the model peek at
the answers it's supposed to predict. A model tuned against the test period looks
brilliant in a demo and fails in production.

The selection mode avoids this with a two-round design:

```
ROUND 1 (calibration):
   Train the 6 models on data BEFORE the last 180 days.
   Predict those last 180 days (we know the real answers).
   - Learn which 2 models win each (weekday, horizon).
   - Learn how hard to anchor holidays (the "alpha").

ROUND 2 (production):
   Re-train the 6 models on ALL available data.
   Predict the real future 28 days.
   Apply the recipe learned in Round 1.
```

The recipe is learned from data the production models hadn't seen, so it reflects
*generalization*, not memorization. Nothing from the forecast period ever
informs the recipe.

---

## 4b. Day-by-day scheduling cycle (with initial-plan tracking)

When you build a 28-day staffing schedule, you freeze a **forecast on day 1** and
use it for planning. Over the following days you re-forecast as actuals arrive and
watch how reality and the updated view diverge from that initial plan. The
`--initial-forecast` flag and the tracking chart support exactly this.

### Day 1 of the cycle — freeze the initial plan for a date range

```powershell
uv run python forecast.py --full --blend selection --initial-forecast 2026-06-17 2026-07-14
```

`--initial-forecast FROM TO` takes **two ISO dates** marking the scheduling window
you are planning for, and flags this run as the **frozen baseline** for it. It is
saved like any run under `forecasts\run_<YYYYMMDD>\`, with the range recorded in
its `metadata.json` (`"initial_forecast_range": ["2026-06-17", "2026-07-14"]`).
The tracking chart focuses on exactly this window.

The range is normally the next 28 days (`run_date+1 .. run_date+28`), but you can
set any sub-window you actually schedule against.

### Every following day — update and track

```powershell
# 1. append yesterday's actual demand to data\raw\rawdata.csv
# 2. re-forecast (regular update, NOT a new baseline — omit --initial-forecast):
uv run python forecast.py --full --blend selection
```

A regular update omits `--initial-forecast`. Each run refreshes two tracking
artifacts:

| File | Contents |
|---|---|
| `forecasts\tracking_comparison.png` | Line chart: **actual demand** (black) vs **initial plan** (orange dashed, with P10–P90 band) vs **latest forecast** (blue) |
| `forecasts\tracking_comparison.csv` | The same three series joined by date: `date, initial_p50, latest_p50, actual` |

The chart answers the operational question: *"is the day we planned for still
looking like the plan, or has it drifted — and does the latest forecast say we
should adjust staffing?"* The vertical dotted line marks the latest actual, so
everything left of it is realized and everything right is still forecast.

### Starting a new cycle

When you begin the next scheduling cycle, run with `--initial-forecast FROM TO`
again using the new window's dates. The tracking chart always uses the **most
recent** run that has an initial-forecast range, so the new plan automatically
becomes the reference and the chart re-focuses on the new window.

### Notes

- `--initial-forecast FROM TO` = "these figures are the initial scheduling plan
  for this date range". A run without the flag is a routine update.
- Both `--initial-forecast` and `--initial_forecast` spellings are accepted.
- FROM must be on or before TO (validated).
- The tracking chart needs at least one run with an initial-forecast range; until
  then it is skipped with a message.

---

## 5. Maintenance schedule

| Task | Frequency | Command |
|---|---|---|
| Append demand actuals | Daily / per batch | edit `data\raw\rawdata.csv` |
| Run forecast | Daily or weekly | `forecast.py [--full --blend selection]` |
| Refresh typhoons | After typhoon events / quarterly | `scrape_typhoon.py` |
| Extend holiday anchors | Before forecasting into a new year | edit `v2\_shared.py` (see README) |
| Re-validate accuracy | Quarterly | `research\backtest.py` |

---

## 6. Accuracy: what to expect, honestly

Forecast accuracy (MAPE = mean absolute % error) depends heavily on **how much
history you have** and **what kind of days are in the window**. From backtests:

| Window type | Typical MAPE |
|---|---|
| Normal month, full history | ~2.5 - 3.0% |
| Holiday-heavy month | ~3 - 5% |
| Thin history (< 2 years) | 5 - 8% |

On the validated May 2026 window (Labour Day month, ~2.4 years of data):

| Mode | MAPE |
|---|---|
| Default LGBM-L2 | 3.47% |
| `--full` equal hybrid | 3.30% |
| `--full --blend selection` | **2.52%** |

**Important honest caveats:**

1. **2.5% is achievable on good windows, not guaranteed on every window.** Months
   with thin data or overlapping lunar holidays can run 5%+. There is no setting
   that fixes this — the information isn't in the data yet.

2. **Accuracy improves automatically as data accumulates.** The single biggest
   driver of MAPE is history length (15 months → ~8%, 22 → ~5%, 28 → ~3%). Just
   running daily and letting `rawdata.csv` grow will pull typical-window MAPE
   toward 2.5% over the coming year, with no code changes.

3. **The P10-P90 band is currently optimistic** (covers ~55-60% of actuals vs the
   80% it implies). Treat the band as a *relative* confidence signal (wider =
   less certain), not a literal 80% guarantee. Honest interval calibration is a
   known future improvement.

---

## 7. Validating accuracy yourself

To measure MAPE on the most recent 28 days (held out from training):

```powershell
# Fast: LGBM-L2 only
uv run python research\backtest.py --fast

# Full: both LGBM-L2 and the 6-model hybrid
uv run python research\backtest.py

# Test a specific historical window
uv run python research\backtest.py --holdout-end 2026-04-30
```

To reproduce the selection + anchor result and per-day breakdown:

```powershell
uv run python research\holiday_anchor.py --holdout 2026-05
```

---

## 8. The research/ folder

Everything in `research\` is the experimentation that produced the production
choices. You don't need it to run forecasts, but it documents *why* the model is
built the way it is and lets you re-derive every number:

| Script | What it answers |
|---|---|
| `backtest.py` | Baseline MAPE per model on a holdout |
| `compare_strategies.py` | Equal blend vs per-(DOW,horizon) selection |
| `holiday_anchor.py` | The holiday-anchor calibration and gain |
| `optimize_mape.py` | Log-transform, NNLS, DOW-scaling experiments (all rejected) |
| `ratio_probe.py` | Ratio-to-baseline target experiment (rejected) |
| `model_zoo_probe.py` | Non-GBM models — Ridge/ElasticNet/ExtraTrees/RandomForest |

Key findings from this work (so nobody re-treads dead ends):

- **Log-transform, ratio-target, NNLS weights, and per-DOW scaling all made
  things worse or didn't transfer across windows.** Don't reach for them.
- **ExtraTrees was the single best model on one window but unreliable on
  another** — kept out of production for that reason.
- **Selection + holiday anchor are the only two techniques that improved results
  honestly without overfitting a single window.** They're in production.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Historical data` ends earlier than expected | Stale CSV — append latest rows |
| `gaps: [...]` non-empty | Missing days in `rawdata.csv` — fill them |
| Holiday under-predicted in forecast | Holiday anchor missing for that year — check `HOLIDAY_ANCHORS` in `v2\_shared.py` covers the forecast year, and that it's a fixed-date holiday |
| Forecast looks too flat vs recent surge | Default mode reacts slowly; try `--full --blend selection` |
| Run is slow (>40 min) | Normal for selection mode (2 training rounds); use default for daily |
| `typhoons.csv not found` | Run `scrape_typhoon.py`, or ignore (feature defaults to 0) |

---

## 10. Quick reference card

```powershell
cd "D:\Applications\Work\Models\Demand Forecast v2"

# Setup (once)
uv sync

# Daily forecast (fast)
uv run python forecast.py

# Committed / monthly forecast (most accurate)
uv run python forecast.py --full --blend selection

# Day-by-day scheduling cycle
uv run python forecast.py --full --blend selection --initial-forecast 2026-06-17 2026-07-14  # day 1: freeze plan for window
uv run python forecast.py --full --blend selection                                           # daily updates

# Specific run date
uv run python forecast.py --run-date 2026-06-16 --full --blend selection

# Refresh typhoon data
uv run python scrape_typhoon.py

# Validate accuracy
uv run python research\backtest.py --fast

# Outputs land in:
#   forecasts\run_<YYYYMMDD>\predictions.csv  (the forecast)
#   forecasts\run_<YYYYMMDD>\forecast.png     (chart)
#   forecasts\comparison.png                  (run-over-run sanity check)
#   forecasts\tracking_comparison.png         (actual vs initial plan vs latest)
#   forecasts\tracking_comparison.csv         (the three series, joined by date)
```
