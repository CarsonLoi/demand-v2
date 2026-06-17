# Hourly forecast module

Splits the 28-day daily forecast produced by `forecast.py` into hourly
demand. For each forecast day, decides whether to use:

1. **A locally-current DOW pattern** — built from the prior 90 non-holiday
   days relative to the run date (weekday / Friday / Saturday / Sunday).
2. **A holiday-specific pattern** — a recency-weighted average of past
   occurrences of that exact holiday-day (e.g. CNY day 1 across all years).

The choice is data-driven: an analyzer compares each historical holiday-day
against its own contemporary DOW baselines (prior 90 days of THAT
occurrence) using Total Variation Distance (TVD). When the holiday looks
indistinguishable from a normal DOW, the system inherits the DOW pattern;
otherwise it uses the holiday-specific pattern.

## Files

| File | Purpose |
|---|---|
| `utils.py` | Shared functions: data loading, TVD, local DOW baselines |
| `analyze_hourly_patterns.py` | One-time / quarterly: runs the TVD analysis, emits decisions and plots |
| `split_hourly.py` | Daily: applies decisions + current local DOW baseline to `predictions.csv` |
| `sample_hourly_data.csv` | Example of the expected `data/raw/hourly_demand.csv` format |

## Inputs

### `data/raw/hourly_demand.csv`

Long format, one row per (date, hour):

```csv
date,hour,demand
2024-01-01,0,234
2024-01-01,1,189
2024-01-01,2,142
...
2024-01-01,23,412
2024-01-02,0,256
...
```

| Column | Type | Notes |
|---|---|---|
| `date` | ISO `YYYY-MM-DD` | Every day must have all 24 hours |
| `hour` | integer 0..23 | Local-time hour bucket |
| `demand` | non-negative number | Patron hours observed in that hour |

**Data-depth requirement**: at least 18 months of hourly history, ideally
24+ months, with **multiple historical occurrences of each major holiday**
(CNY, GoldenWeek, Labour, MidAutumn). The analyzer requires `>=2`
occurrences per `(holiday, day_offset)` cell to make a decision.

A small example file is included at `hourly/sample_hourly_data.csv` for
format reference only — it is not large enough to run the analyzer.

## Workflow

### One-time / quarterly: build the decisions

```powershell
cd "D:\Applications\Work\Models\Demand Forecast v2"
uv run python hourly\analyze_hourly_patterns.py
```

Produces:

| Output | Contents |
|---|---|
| `data\derived\hourly_pattern_decisions.csv` | Per `(holiday, day_offset)`: TVD per occurrence, modal best-DOW match, final decision (`USE_SATURDAY`, `USE_HOLIDAY_PROFILE`, etc.) |
| `data\derived\hourly_holiday_profiles.csv` | Recency-weighted hourly shares for cells decided to use holiday-specific patterns |
| `data\derived\hourly_pattern_plots\*.png` | Side-by-side hourly charts for any cell that was borderline or holiday-specific. **Review these manually.** |

Decision rules:

| Condition | Decision |
|---|---|
| `tvd_mean < 0.05` AND `tvd_max < 0.08` AND consistent modal DOW | `USE_<dow>` -- close match, inherit DOW |
| `tvd_mean < 0.10` AND consistent modal DOW | `USE_<dow>` -- borderline, flagged for review |
| `tvd_mean >= 0.10` | `USE_HOLIDAY_PROFILE` -- shape is materially different |
| Inconsistent modal DOW (occurrences disagree) | `USE_HOLIDAY_PROFILE` -- pattern is drifting, use holiday's own shape |

Re-run quarterly as new holiday samples accumulate. Decisions can change
as more years stabilize the holiday-specific pattern.

### Every forecast: split daily into hourly

```powershell
# After forecast.py has produced forecasts\run_<date>\predictions.csv:
uv run python hourly\split_hourly.py --run-date 2026-06-09
```

Produces `forecasts\run_20260609\predictions_hourly.csv`:

```csv
date,hour,p10,p50,p90,strategy
2026-06-10,0,12.4,15.8,19.2,dow_weekday
2026-06-10,1,9.1,11.6,14.1,dow_weekday
...
2026-07-01,0,18.7,23.5,28.4,CNY_d+1_holiday_profile
...
```

The `strategy` column shows which pattern was applied for each day --
useful when something looks wrong.

The script also recomputes the prior-90d DOW baselines fresh on each run,
so the applied DOW patterns track current behavior over time. The
holiday-specific decision and profile are reused as-is from the analyzer
run.

## Configuration knobs (in `utils.py`)

| Constant | Default | Effect |
|---|---|---|
| `LOCAL_WINDOW_DAYS` | 90 | How many prior days form the local DOW baseline |
| `TVD_CLOSE` | 0.05 | Below this TVD = "close match, use DOW" |
| `TVD_BORDERLINE` | 0.10 | Below this = "borderline, use DOW but flag for review" |
| `MIN_DOW_SAMPLES` | 5 | Minimum non-holiday days per DOW bucket in the window |
| `MIN_HOLIDAY_OCCURRENCES` | 2 | Minimum historical samples per `(holiday, day_offset)` |

Tune `TVD_CLOSE` / `TVD_BORDERLINE` based on spot-checking the comparison
plots. If the algorithm is too eager to use DOW patterns, lower them.
If it is too eager to use holiday-specific patterns (which are noisier),
raise them.

## Holiday set

The analyzer evaluates these holidays (sourced from `v2/_shared.py` so
the daily and hourly pipelines agree on which dates are holidays):

```
CNY, GoldenWeek, Labour, MidAutumn, Christmas, Easter
```

The single-day low-impact holidays (NewYear, ChingMing) are not analyzed
-- they fall through to the default DOW path during splitting.

## Expected accuracy

The daily model averages ~1.3% MAPE (with `--full`). After splitting:

| Day type | Typical hourly MAPE |
|---|---|
| Normal weekday | 3-5% |
| Friday / Saturday / Sunday | 4-6% |
| Holiday with DOW decision | 5-8% |
| Holiday with `USE_HOLIDAY_PROFILE` | 6-10% (more variance, fewer samples) |

Hourly MAPE is naturally larger than daily because you're dividing a
similar absolute error by smaller hourly denominators. Anything below 8%
on holidays is operationally useful for staffing decisions.

## Operational checklist

Before trusting an hourly forecast for staffing:

1. Has `analyze_hourly_patterns.py` been run in the last quarter?
2. Did the splitter print `[WARN] insufficient prior-90d samples`? If so,
   your hourly history may have gaps -- some forecast days will fall back
   to NaN strategies.
3. Spot-check the `strategy` column in `predictions_hourly.csv` -- do the
   strategies for upcoming holiday days match what the decisions file says?
4. If you see `dow_<bucket>` for a date you expected to be flagged as a
   holiday, check that the holiday anchor exists in `v2/_shared.py`'s
   `HOLIDAY_ANCHORS` and covers that year.
