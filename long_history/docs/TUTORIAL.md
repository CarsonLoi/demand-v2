# Tutorial — using the long-history forecast

Step-by-step instructions for every routine task. Copy-paste ready.

For *why* it works this way, read [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md).

All commands are run from the **repository root** (the folder containing
`long_history/`).

---

## 0. One-time setup

```bash
uv sync
```

Check the data is where it should be:

```bash
uv run python long_history/data/validate.py
```

You should see `PASSED` — possibly with a warning about 26 full-closure days,
which is expected and correct (COVID shutdowns and typhoon Mangkhut).

---

## 1. The daily routine

This is what you will do most days.

### Step 1 — add yesterday's actual demand

Append one row to `long_history/data/raw/rawdata_long.csv`:

```csv
2026-05-29,7104,294
```

Columns are `date,demand,floortables`. **One row per day, no gaps ever.**

### Step 2 — run the pipeline

```bash
uv run python long_history/run_pipeline.py
```

That runs three steps in order and **stops at the first failure**:

| step | what it does |
|---|---|
| validate | checks the data file for gaps, duplicates, impossible values |
| evaluate | measures recent accuracy and draws the charts |
| forecast | produces the 28-day forecast |

### Step 3 — read the output

```
long_history/output/run_20260529/
├── predictions.csv     date, p10, p50, p90
├── forecast.png        chart: recent actuals + the 28-day band
└── metadata.json       every setting used, for auditability
```

**Plan against `p50`.** See the caveat on `p10`/`p90` in the guide (§15).

---

## 2. Producing a forecast only

Skip validation and evaluation when you just want numbers:

```bash
uv run python long_history/training/forecast.py
```

From a past date (a backtest-style rerun — the leakage guard still applies, so
it trains only on data up to that date):

```bash
uv run python long_history/training/forecast.py --run-date 2026-04-30
```

Other options:

```bash
# a different recency weighting
uv run python long_history/training/forecast.py --half-life 720

# without the Chinese New Year correction, to see what it is worth
uv run python long_history/training/forecast.py --no-anchor
```

---

## 3. Measuring accuracy

### "How accurate were we last month, one week ahead?"

```bash
uv run python long_history/training/evaluate.py \
    --start 2026-05-01 --end 2026-05-28 --n 6 --mode rolling_window
```

`--n` is the window depth and **lead = n + 1**:

| flag | meaning |
|---|---|
| `--n 0` | tomorrow's forecast (lead 1) |
| `--n 6` | the week-ahead forecast (lead 7) |
| `--n 13` | two weeks ahead (lead 14) |
| `--n 27` | the full 28-day plan |

### Two modes

| mode | answers |
|---|---|
| `fixed_lead` | *"how accurate are we exactly N days out?"* — every point at the same distance |
| `rolling_window` | *"what would our real operating forecast have looked like?"* — mimics re-forecasting every N days |

### Add the accuracy-vs-distance curve

```bash
uv run python long_history/training/evaluate.py \
    --start 2026-01-01 --end 2026-05-28 --n 6 --mode rolling_window --by-lead
```

### What you get

```
=== Accuracy ===
  MAPE        3.18%   (average % miss)
  WAPE        3.15%   (total error ÷ total demand)
  RMSE          271    (patron hours)
  Bias          -30    (+ = over-forecast)
  Coverage   78.6%   (P10-P90, nominal 80% — see caveat)
  Days          28

=== By segment ===
  ordinary     n=  23   MAPE   3.21%
  Labour       n=   5   MAPE   3.04%
```

Plus charts in `long_history/output/evaluation/`.

> **Read `Bias` alongside `MAPE`.** MAPE says how far off you were; bias says
> which direction. For staffing, running consistently low is a different
> problem from running high.

---

## 4. When a typhoon closes the floor

The model cannot predict this — see guide §11. Handle it manually.

### Refresh the typhoon record

```bash
uv run python long_history/data/scrape_typhoon.py
```

### Adjust a saved forecast

```bash
uv run python long_history/ops/typhoon_override.py
```

It lists your saved runs and walks you through applying an adjustment. It
writes `predictions_adjusted.csv` beside the original and records the override
in `metadata.json`. **The original `predictions.csv` is never modified.**

Afterwards, add the real actual to `rawdata_long.csv` as normal. If the floor
was fully shut, record `floortables` as `0` so the closure detection picks it
up automatically.

---

## 5. Changing settings

Everything lives in `long_history/config.py`. **Edit that file, not the
scripts.**

The setting most worth understanding is the recency half-life:

```python
DEFAULT_HALF_LIFE = 1095   # days until an old day counts half as much
```

At 240 (the short-history default), a day from 2016 counts about 1/35,000th of
today — so eleven years of history would do nothing at all. See guide §8.

### Testing whether a change actually helps

Never adopt a setting change on a hunch. Compare it:

```bash
uv run python long_history/training/experiment.py --quick   # sampled, faster
uv run python long_history/training/experiment.py           # all 28 distances
```

This compares history length × half-life over the same held-out windows and
applies a **per-window gate**: a variant is only adopted if *no* window gets
worse. An average improvement hiding one bad window is a fail.

Then draw the charts:

```bash
uv run python long_history/viz/charts.py
```

---

## 6. Running individual pipeline steps

```bash
# data gate only
uv run python long_history/run_pipeline.py --steps validate

# validate then forecast, skipping evaluation
uv run python long_history/run_pipeline.py --steps validate forecast

# everything including the settings comparison (slow)
uv run python long_history/run_pipeline.py \
    --steps validate experiment evaluate forecast --quick

# evaluate a specific range
uv run python long_history/run_pipeline.py --steps evaluate \
    --eval-start 2026-01-01 --eval-end 2026-05-28 --eval-n 6
```

---

## 7. Regenerating the slide data

If you maintain the presentation deck:

```bash
uv run python long_history/viz/deck_data.py       # actual-vs-forecast series
uv run python long_history/research/model_demo.py # boosting example numbers
node docs/build_longhistory_deck.js               # build the .pptx
```

The first command takes roughly an hour — it runs five full backtests at
different forecast distances.

---

## 8. Housekeeping

### Reclaim disk space

```bash
rm -rf long_history/output/matrix_cache long_history/output/eval_cache
```

Both regenerate automatically. The matrix cache is ~90 MB per variant.

### Do I ever need to clear the cache after changing the code?

**No.** Both caches include a hash of the feature engine, so editing
`engine/core.py` or `engine/features.py` invalidates them automatically. A
stale result cannot be silently served.

---

## Common problems

| symptom | what to do |
|---|---|
| `MISSING DAY(S) in the date range` | fill the gaps in the CSV. **Never** delete surrounding rows — every backward-looking calculation counts through the calendar |
| `ABORTED: fix the data errors above first` | validation failed; the specific problem is printed above |
| `end=... is past the last actual` | you asked to evaluate dates you have no actuals for |
| Pipeline stops at `validate` | that is the gate doing its job — fix the data, do not bypass it |
| First run takes 4 minutes | building the feature table. Later runs are seconds |
| `NotImplementedError` about 2027 | the holiday-calendar deadline; see guide §15 |

---

## Quick reference

```bash
# the daily command
uv run python long_history/run_pipeline.py

# forecast only
uv run python long_history/training/forecast.py

# accuracy, one week ahead, this year
uv run python long_history/training/evaluate.py \
    --start 2026-01-01 --end 2026-05-28 --n 6 --mode rolling_window --by-lead

# check the data
uv run python long_history/data/validate.py

# typhoon adjustment
uv run python long_history/ops/typhoon_override.py

# does this setting change actually help?
uv run python long_history/training/experiment.py --quick
```
