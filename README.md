# Demand Forecast — Operations Guide

Production system for producing a **28-day daily demand forecast** (patron hours)
for casino floor planning, with prediction intervals and run-over-run tracking.

The entire production pipeline lives in a few files and one CSV. Everything else
in this repository is research/archive.

> **New here? Read [TUTORIAL.md](TUTORIAL.md)** for a full step-by-step guide to
> running, understanding, and maintaining the forecast.

---

## What you actually need

```
forecast.py              <-- the production script (run this)
blend.py                 <-- selection + holiday-anchor blending (used by --blend selection)
v2/_shared.py            <-- feature engineering (imported by forecast.py)
scrape_typhoon.py        <-- builds data/raw/typhoons.csv (weather feature)
data/raw/rawdata.csv     <-- your daily demand history
pyproject.toml + uv.lock <-- Python environment lockfile
```

### Three forecast modes

```powershell
uv run python forecast.py                          # LGBM-L2, ~3 min (daily driver)
uv run python forecast.py --full                   # 6-model equal blend, ~15-20 min
uv run python forecast.py --full --blend selection # 6-model + selection + holiday anchor,
                                                    # ~30-40 min, most accurate (recommended
                                                    # for committed / holiday-month forecasts)
```

The `selection` mode adds two backtest-validated, leakage-free improvements:
per-(day-of-week, horizon) model selection and fixed-date holiday anchoring.
On the validated May 2026 window it improved MAPE from 3.30% (equal) to 2.52%.
See [TUTORIAL.md](TUTORIAL.md) Section 3-4 and the `research/` folder for details.

That is the complete production system. Everything below in the "Archive" section
exists only for benchmark reproducibility and can be ignored during normal
operation.

---

## One-time setup (~5 minutes)

You need `uv` installed (https://docs.astral.sh/uv/getting-started/installation/).

```powershell
cd "D:\Applications\Work\Models\Demand Forecast v2"
uv sync
```

This installs Python 3.11 (if missing), creates `.venv\`, and pins all
dependencies (LightGBM, XGBoost, CatBoost, pandas, matplotlib, chinese_calendar).
Re-running `uv sync` later is a no-op when nothing has changed (<10 sec).

Verify it works with a quick smoke test (~3 minutes):

```powershell
uv run python forecast.py
```

If you see `=== Done -- open forecasts\run_<date>\forecast.png ===` and a
non-empty PNG appears under `forecasts\`, the system is operational.

---

## The recurring workflow

You do this every time you want a fresh forecast (typically weekly or after a
batch of new actuals arrives):

1. **Update `data/raw/rawdata.csv`** with new daily actuals
2. (Only when new holidays/events are in the forecast window) **Update
   `v2/_shared.py`** with new anchor dates
3. **Run `forecast.py`** to retrain and produce the forecast
4. **Review** the run folder and comparison chart

Each step is detailed below.

---

## Step 1 -- Update `data/raw/rawdata.csv`

### Format

```csv
date,demand,floortables
2024-01-01,6099,299
2024-01-02,5178,299
...
2026-05-28,6993,289
```

Three columns, in any order:

| Column | Type | Notes |
|---|---|---|
| `date` | ISO `YYYY-MM-DD` | One row per day. **No gaps allowed.** |
| `demand` | non-negative integer | patron hours for that day |
| `floortables` | positive integer | table capacity for that day |

### How to append

Open the CSV in Excel or a text editor, append your new daily rows in date
order at the bottom, save as **CSV (UTF-8)** -- not `.xlsx`. Don't reorder or
rewrite the existing rows. Trailing spaces in numeric values are harmless;
trailing spaces on the `date` column will break parsing.

### Sanity check after appending

```powershell
uv run python -c "import pandas as pd; d=pd.read_csv('data/raw/rawdata.csv', parse_dates=['date']); print('rows:', len(d)); print('first:', d.date.min().date()); print('last:', d.date.max().date()); print('expected days:', (d.date.max()-d.date.min()).days+1); gaps=pd.date_range(d.date.min(), d.date.max()).difference(d.date); print('gaps:', list(gaps))"
```

If `rows == expected days` and `gaps: []`, you are good. If there are gaps,
fix them before forecasting -- missing days silently corrupt every lag and
rolling-window feature.

---

## Step 2 -- Add new holidays or events

Holiday anchor dates are hardcoded in `v2/_shared.py` (lines 37-51). The
system has no calendar service; you must extend the list manually as new
holidays enter the next 12 months of your forecast window.

**You only need to do this when:**
- Your forecast window will cross into a new calendar year, OR
- A new recurring event needs to be modeled

### Open `v2/_shared.py` and find these two dicts (around line 37):

```python
HOLIDAY_ANCHORS = {
    "CNY":        [pd.Timestamp(d) for d in ["2024-02-10", "2025-01-29", "2026-02-17"]],
    "GoldenWeek": [pd.Timestamp(d) for d in ["2024-10-01", "2025-10-01", "2026-10-01"]],
    "Labour":     [pd.Timestamp(d) for d in ["2024-05-01", "2025-05-01", "2026-05-01"]],
    "MidAutumn":  [pd.Timestamp(d) for d in ["2024-09-17", "2025-10-06", "2026-09-25"]],
    "NewYear":    [pd.Timestamp(d) for d in ["2024-01-01", "2025-01-01", "2026-01-01"]],
    "Christmas":  [pd.Timestamp(d) for d in ["2024-12-25", "2025-12-25"]],
    "ChingMing":  [pd.Timestamp(d) for d in ["2024-04-04", "2025-04-04", "2026-04-04"]],
    "Easter":     [pd.Timestamp(d) for d in ["2024-03-31", "2025-04-20", "2026-04-05"]],
}
HOLIDAY_WINDOWS = {
    "CNY":        (-7, 10), "GoldenWeek": (0, 6), "Labour": (0, 4),
    "MidAutumn":  (-1, 1),  "NewYear":    (0, 0), "Christmas": (-3, 3),
    "ChingMing":  (0, 0),   "Easter":     (-2, 1),
}
```

### How to extend an existing holiday into a new year

Example: adding CNY 2027 (which falls on 2027-02-06):

```python
"CNY": [pd.Timestamp(d) for d in ["2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06"]],
```

That single edit registers the holiday. The window definition (`-7, +10` for
CNY) is reused automatically.

### How to add a brand-new event

Example: adding a new annual concert series that boosts demand for 3 days:

1. Add a new key to `HOLIDAY_ANCHORS` with all the historical and future anchor dates:

   ```python
   HOLIDAY_ANCHORS = {
       ...
       "ConcertSeries": [pd.Timestamp(d) for d in ["2024-08-15", "2025-08-20", "2026-08-18", "2027-08-17"]],
   }
   ```

2. Add the corresponding window to `HOLIDAY_WINDOWS`:

   ```python
   HOLIDAY_WINDOWS = {
       ...
       "ConcertSeries": (0, 2),   # day-of and 2 days after
   }
   ```

   Window format: `(days_before_anchor, days_after_anchor)`. `(0, 0)` = single day only.

3. **Important**: include at least 2 historical anchor dates (events that have
   already happened and are in `rawdata.csv`) so the model has data to learn
   from. A future-only anchor is treated as an unknown feature and has no effect.

That is the entire extension mechanism -- "holidays" and "events" are
mechanically identical to the model; both are anchor-dates with windows.

### Mainland China holidays are auto-handled

Block features for PRC official holidays (CNY, Labour, Golden Week, etc.) are
sourced live from the `chinese_calendar` Python package -- you don't maintain
them yourself. But the package needs to support the year you're forecasting:

```powershell
uv run python -c "import chinese_calendar; import datetime; print(chinese_calendar.is_workday(datetime.date(2027,1,1)))"
```

If that throws `NotImplementedError`, bump the package:

```powershell
uv add chinese-calendar@latest
```

### Step 3 of holiday edit -- verify the new anchor took effect

```powershell
uv run python -c "import sys; sys.path.insert(0, 'v2'); from _shared import HOLIDAY_ANCHORS, HOLIDAY_WINDOWS; print('CNY anchors:', HOLIDAY_ANCHORS['CNY']); print('Windows:', HOLIDAY_WINDOWS)"
```

You should see your newly-added date in the output.

---

## Step 3 -- Run the forecast

There is **no separate "train" step**. `forecast.py` retrains from scratch on
every run using all historical data up to the run date.

### Default mode (LightGBM-L2 single model, ~3 minutes)

```powershell
uv run python forecast.py
```

This is the fastest path and the recommended daily driver. Single-model MAPE
is around 1.94% on backtests.

### Full hybrid mode (6-model blend, ~15-20 minutes)

```powershell
uv run python forecast.py --full
```

This trains 5 base GBMs (LightGBM-L2, LightGBM quantile, XGBoost, CatBoost,
bagged LightGBM) plus a two-stage holiday specialist (baseline + holiday
residual). Blended-MAPE benchmark is **1.29%**. Use this for monthly planning,
forecasts you'll commit to in a budget, or any time the extra accuracy is
worth 15-20 minutes.

### Specifying a custom run date

By default, `forecast.py` uses **the last date in `rawdata.csv`** as the run
date, and forecasts the next 28 days. Override if you want to forecast from
a specific date (e.g. for historical backtest comparisons):

```powershell
uv run python forecast.py --run-date 2026-06-08
uv run python forecast.py --run-date 2026-06-08 --full
```

If your CSV extends past the run date (i.e. you have actuals after the run
date), the script will print a warning and exclude those rows from training
to prevent data leakage.

### What you should see in the console

```
=== Forecast run ===
  Historical data:    2024-01-01 .. 2026-06-08 (890 days)
  Run date:           2026-06-08
  Forecasting:        2026-06-09 .. 2026-07-06
  Building feature matrix...
  Training LightGBM-L2 (~3 min)...

=== Saving outputs ===
  -> forecasts\run_20260608\predictions.csv
  -> forecasts\run_20260608\metadata.json
  -> forecasts\run_20260608\forecast.png

=== Building comparison chart ===
  -> forecasts\comparison.png

=== Done -- open forecasts\run_20260608\forecast.png ===

P50 statistics:
  min:  5234
  max:  7891
  mean: 6543
  sum:  183204
```

**Verify two things before trusting the output:**

1. `Historical data` ends at the day you expected (your latest actuals).
2. `Forecasting` starts at the day after that.

If either is wrong, you forgot to update the CSV or you passed a wrong
`--run-date`.

---

## Step 4 -- Review the outputs

Every run creates a new folder under `forecasts\` named for the run date:

```
forecasts\
  run_20260601\
    predictions.csv      # date, p10, p50, p90 for the next 28 days
    forecast.png         # chart of trailing 60d actuals + 28d forecast band
    metadata.json        # run config, train range, P50 summary stats
  run_20260608\
    predictions.csv
    forecast.png
    metadata.json
  comparison.png         # overlay of the most recent 6 runs vs trailing actuals
```

### `predictions.csv`

The forecast itself. Three quantile columns:

| Column | Meaning |
|---|---|
| `p50` | The median forecast (the "point" prediction) |
| `p10` | 10th percentile -- low-case; demand likely above this 90% of the time |
| `p90` | 90th percentile -- high-case; demand likely below this 90% of the time |

The gap `p90 - p10` is your uncertainty band. Wider band = the model is less
confident about that day.

### `metadata.json`

Run audit trail. Useful when you need to know *what* produced a forecast:

```json
{
  "run_date": "2026-06-08",
  "model": "lightgbm_l2_v2",
  "horizon_days": 28,
  "historical_data_range": ["2024-01-01", "2026-06-08"],
  "forecast_range": ["2026-06-09", "2026-07-06"],
  "p50_min": 5234.0,
  "p50_max": 7891.0,
  "p50_mean": 6543.2,
  "p50_sum": 183204.5,
  "generated_at": "2026-06-09T14:23:11"
}
```

### `forecast.png`

Trailing 60 days of actuals (black dots) + the 28-day forecast band (blue
line = P50, shaded blue = P10-P90 interval). The vertical red dashed line
marks the run date. Quick visual sanity check that the forecast level is
plausible relative to recent actuals.

### `comparison.png`

The most important sanity check. Overlay of up to the last 6 runs. If your
new forecast diverges sharply from the previous runs without an obvious data
reason (new actuals, holiday, floortables change), something is wrong --
investigate before committing to the new numbers.

### Re-running on the same date

`forecasts\run_<YYYYMMDD>\` is overwritten silently if you re-run with the
same run date. If you want to preserve a prior run for audit, rename its
folder before re-running:

```powershell
Rename-Item forecasts\run_20260608 forecasts\run_20260608_v1
```

---

## Common gotchas

Ranked by how often they cause problems:

1. **Gaps in `rawdata.csv`** -- even a single missing day silently corrupts
   every rolling window and lag feature. Run the sanity check at the end of
   Step 1 after every data update.

2. **Forgot to extend holiday anchors into the new year** -- if your forecast
   window crosses Jan 1, the model will treat all 2027 holidays as ordinary
   days and under-predict them. Update `v2/_shared.py` per Step 2.

3. **CSV ends earlier than you thought** -- the console line
   `Historical data: ... .. <date>` is your truth source. If it doesn't match
   yesterday, you have a stale CSV.

4. **`floortables` step change in the future** -- the script pads future days
   with the last observed value of `floortables`. If you know tables are
   changing (renovation, new floor), the forecast won't anticipate that.
   Either update the historical column to reflect the actual schedule, or
   accept the limitation.

5. **`hybrid` mode is slow** -- 15-20 minutes is normal on a laptop. If it
   takes meaningfully longer, check Task Manager -- another process may be
   contending for CPU.

6. **CSV encoding** -- save as UTF-8 without BOM. PowerShell's default
   `Out-File` writes UTF-16 LE; if you ever script the append, pass
   `-Encoding utf8`.

---

## Run records

Every forecast is automatically recorded under `forecasts\run_<YYYYMMDD>\`
with its CSV, chart, and metadata. To find the latest run:

```powershell
Get-ChildItem forecasts\run_* | Sort-Object Name -Descending | Select-Object -First 5
```

To list all runs with their P50 sum (for quick trend review):

```powershell
uv run python -c "import json, pathlib; runs=sorted(pathlib.Path('forecasts').glob('run_*/metadata.json')); [print(r.parent.name, '->', json.loads(r.read_text())['p50_sum']) for r in runs]"
```

---

## Reference -- what each file does

### Production files

- **`forecast.py`** -- entry point. Trains models, produces the 28-day
  forecast, writes outputs, builds comparison chart.
- **`v2/_shared.py`** -- feature engineering. Calendar features, holiday
  windows, lag features, rolling statistics, EWMA, Mainland China holiday
  blocks, sample weighting. Imported by `forecast.py`.

### Configuration knobs inside `v2/_shared.py`

- `HOLDOUT_DAYS = 28` -- forecast horizon length. Changing this affects both
  training (per-horizon models) and the forecast length.
- `HOLIDAY_ANCHORS` / `HOLIDAY_WINDOWS` -- holidays/events to model (Step 2).
- `LAG_DAYS`, `ANCHOR_LAGS`, `ROLLING_WINDOWS`, `EWMA_SPANS` -- which
  historical lookbacks to feed the model. Defaults are well-tuned; only
  change if you know what you're doing.
- `HOLIDAY_UPWEIGHT = 3.0` -- how much extra weight holiday rows get during
  training. Higher = the model cares more about holidays at the expense of
  ordinary days.
- `USE_RESERVATIONS = False` -- set to `True` if you have a
  `data/raw/reservations.csv` file with on-the-books demand snapshots.
  Format documented in the file.
- `USE_WEATHER = True` -- exposes the single `is_typhoon_t8plus` feature
  built from `data/raw/typhoons.csv`. Set False to remove from the model.
  See "Weather data" section below.

## Weather data

The model uses a **single binary weather feature** -- `is_typhoon_t8plus`,
1 when HKO Tropical Cyclone Warning Signal 8 (or 9 / 10) was active on
that date, 0 otherwise. T8+ is the only weather variable that materially
moves Macau casino demand (it triggers border crossing restrictions and
ferry suspensions). Temperature, humidity, wind, precipitation, and lower
signals (T3) added noise without improving backtested MAPE and were
removed.

### Source file -- `data/raw/typhoons.csv`

```csv
date,typhoon_name,highest_signal
2023-09-01,SAOLA,10
2023-09-02,SAOLA,10
2024-09-05,YAGI,8
2024-09-06,YAGI,8
2025-09-23,RAGASA,8
2025-09-24,RAGASA,10
```

One row per affected calendar day; if multiple signals (e.g. T8 -> T9 -> T10)
were active during the same day, the highest is kept. Dates not in this file
default to `is_typhoon_t8plus = 0`.

### Building / refreshing the file -- `scrape_typhoon.py`

Sourced from HKO's Tropical Cyclone Warning Signals Database
(https://www.hko.gov.hk/en/wxinfo/climat/warndb/warndb1.shtml). The
underlying data feed at https://www.hko.gov.hk/dps/wxinfo/climat/warndb/tc.dat
contains records back to 1946.

```powershell
# Default: signal >= 8, since 2023
uv run python scrape_typhoon.py

# Wider historical window
uv run python scrape_typhoon.py --since 2018
```

Re-run quarterly (or after each Macau typhoon season) to pick up new events.
HKO confirms records "during the first working day after the expiry or
cancellation," so wait a few days after a signal is lowered before re-running.

### Disabling

Set `USE_WEATHER = False` in `v2/_shared.py` to remove the feature entirely.
If `typhoons.csv` is missing, the feature defaults to 0 for every row and
the model proceeds.

---

### Archive

The following exist for historical reference (benchmark reproducibility,
documentation of rejected approaches). Production does not use them:

- `v2/simple_*.py` -- standalone scripts that train individual base models
  for benchmarking (LightGBM, XGBoost, CatBoost, NeuralProphet, etc.).
- `v2/plot_feature_importance.py` -- diagnostic plot script.
- `v2/output/` -- pickled models and benchmark charts from the v2 baseline
  exercise.
- `v3/` -- the entire folder. Contains an alternative feature set (`_shared.py`
  with trend features and cross-holiday transfer) that was tested and rejected
  for overfitting, plus the original `simple_lightgbm_2stage.py` script whose
  logic has been inlined into `forecast.py` as the `_train_2stage` function.

You can delete the archive without breaking production. The only cost is
losing the ability to re-derive the 1.29% MAPE benchmark from scratch.

---

## Quick reference card

```powershell
# Setup (one time)
cd "D:\Applications\Work\Models\Demand Forecast v2"
uv sync

# Daily / weekly: update data, then forecast
# 1. append rows to data\raw\rawdata.csv
# 2. retrain + forecast (fast path)
uv run python forecast.py
# 3. open forecasts\run_<today>\forecast.png and forecasts\comparison.png

# Monthly planning / committed forecast (slow path)
uv run python forecast.py --full

# Specific historical run date
uv run python forecast.py --run-date 2026-06-08 --full

# After adding a holiday: verify the edit took
uv run python -c "import sys; sys.path.insert(0,'v2'); from _shared import HOLIDAY_ANCHORS; print(HOLIDAY_ANCHORS)"
```
