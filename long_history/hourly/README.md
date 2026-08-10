# Hourly demand splitting

Splits the 28-day daily forecast produced by `training/forecast.py` into
hourly demand, using the actual historical hourly distribution — not a
trained model. For each forecast day, picks between:

1. **A locally-current DOW pattern** — built from the prior 90 non-holiday
   days relative to the run date (weekday / Friday / Saturday / Sunday).
2. **A holiday-specific pattern** — a recency-weighted average of past
   occurrences of that exact holiday-day (e.g. CNY day +1 across all years).

The choice is data-driven: `analyze.py` compares each historical
holiday-day against its own contemporary DOW baselines using Total
Variation Distance (TVD). When the holiday looks indistinguishable from a
normal DOW, the system inherits the DOW pattern; otherwise it uses the
holiday-specific pattern. Full detail: [TUTORIAL.md](TUTORIAL.md).

## Status: pipeline complete, validation pending real data

⚠️ **`long_history/data/raw/hourly_demand.csv` does not exist yet.** Every
script here runs and has been smoke-tested against
`sample_hourly_data.csv` (10 days, format reference only — the README
for that file is explicit that it is NOT enough data to run the
analyzer meaningfully). Nothing here has been validated against real
hourly history. See `validate.py`'s module docstring and
[COMPLETE_GUIDE.md §18](../docs/COMPLETE_GUIDE.md#18-hourly-demand-splitting)
for what "validated" will mean once real data lands.

## Files

| File | Purpose |
|---|---|
| `patterns.py` | Shared functions: data loading, TVD, local DOW baselines, holiday membership |
| `analyze.py` | Quarterly: runs the TVD analysis, emits decisions and plots |
| `split.py` | Daily: applies decisions + current local DOW baseline to a saved forecast |
| `comparison_model.py` | A lightweight regressor, used ONLY inside `validate.py` — never produces an actual forecast |
| `validate.py` | Leakage-free backtest: empirical method vs the comparison model, by day-type |
| `checks.py` | Assert-based sanity checks (conservation, leakage, share-vector correctness) |
| `sample_hourly_data.csv` | 10-day format reference for `data/raw/hourly_demand.csv` — NOT sufficient for real validation |

## Independence

Holiday dates come from `long_history/engine/core.py` — the extended,
corrected 2016-2030 set — not `v2/_shared.py`. Nothing here imports from
outside `long_history/`.

## Quick start

See [TUTORIAL.md](TUTORIAL.md).
