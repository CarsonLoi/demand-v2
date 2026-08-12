# Tutorial — hourly demand splitting

Step-by-step commands for the hourly module. For *why* it works this way,
read [README.md](README.md) and
[../docs/COMPLETE_GUIDE.md §18](../docs/COMPLETE_GUIDE.md#18-hourly-demand-splitting).

All commands run from the **repository root**.

---

## 0. Before you start

⚠️ **You need `long_history/data/raw/hourly_demand.csv`** — real hourly
history, at least 18 months, ideally 24+, covering multiple occurrences of
each major holiday (CNY, Golden Week, Labour, Mid-Autumn). Format:

```csv
date,hour,demand
2024-01-01,0,234
2024-01-01,1,189
...
2024-01-01,23,412
```

Every date needs all 24 hours present. `long_history/hourly/sample_hourly_data.csv`
shows the format but is only 10 days — not enough to run anything
meaningfully. Until the real file exists, the commands below will either
fail with a clear "file not found," or (for `analyze.py` /
`validate.py` against the sample) run as a **mechanism check only** — see
the warnings each script prints.

---

## 1. One-time / quarterly: build the decisions

```bash
uv run python long_history/hourly/analyze.py
```

Produces, under `long_history/output/hourly/`:

| Output | Contents |
|---|---|
| `decisions.csv` | Per `(holiday, day_offset)`: TVD per occurrence, modal best-DOW match, final decision |
| `holiday_profiles.csv` | Recency-weighted hourly shares for cells decided to use holiday-specific patterns |
| `pattern_plots/*.png` | Side-by-side hourly charts for any borderline or holiday-specific cell — **review these manually** |

Re-run quarterly as new holiday samples accumulate.

## 2. Every forecast: split daily into hourly

```bash
# After training/forecast.py has produced output/run_<date>/predictions.csv:
uv run python long_history/hourly/split.py --run-date 2026-06-09
```

Produces `long_history/output/run_20260609/predictions_hourly.csv`:

```csv
date,hour,p10,p50,p90,strategy
2026-06-10,0,12.4,15.8,19.2,dow_weekday
...
2026-07-01,0,18.7,23.5,28.4,CNY_d+1_holiday_profile
```

The `strategy` column shows which pattern was applied — useful when
something looks wrong.

## 3. Validating the approach

```bash
uv run python long_history/hourly/validate.py --holdout-days 90
```

Runs an honest, walk-forward backtest: for each day in the holdout window,
using only data strictly before it, computes what the empirical method
would have predicted and what a lightweight comparison model would have
predicted, and applies both to that day's *actual* total. Covers **both**
holiday and ordinary days, reported in two tiers:

- **Tier 1 — by bucket** (operational): `weekday` (Mon–Thu, pooled — this
  is the actual prediction the split uses), `friday`, `saturday`, `sunday`,
  `holiday_inherited_dow`, `holiday_specific_profile`.
- **Tier 2 — by individual weekday** (diagnostic, non-holiday days only):
  Monday through Sunday reported separately, even though Monday–Thursday
  share one predicted baseline. This checks whether pooling those four
  days is actually justified — if Monday's error is consistently worse
  than Wednesday's despite using the same prediction, that's evidence the
  `weekday` bucket in `patterns.DOW_BUCKETS` should be split further. The
  script prints the Mon–Thu spread and flags it directly if it looks too
  wide.

Produces, under `long_history/output/hourly/`:
- `validation_report.csv` — one row per (date, method), with both the
  bucket (`day_type`) and the individual `weekday` recorded
- `validation_summary.csv` — Tier 1: MAPE by (bucket, method)
- `validation_summary_by_weekday.csv` — Tier 2: MAPE by (weekday, method)

**Read every segment, not just the average — for both tiers.** The
empirical method is only "good enough, no complicated model needed" if it
holds up across *every* bucket, and the weekday pooling is only
justified if Tier 2 doesn't show one weekday quietly worse than the rest.
A good pooled average hiding one bad segment is not a pass.

## 4. Running the sanity checks

```bash
uv run python long_history/hourly/checks.py
```

Confirms the mechanism itself is sound: shares sum to 1, TVD behaves
correctly, no split or baseline ever reads same-day or future data. Fast,
no real hourly data required — safe to run any time.

## 5. Wiring into the main pipeline

```bash
uv run python long_history/run_pipeline.py --steps hourly-analyze hourly-split
```

Not part of the default `run_pipeline.py` steps (`validate`, `evaluate`,
`forecast`) — opt in explicitly once real hourly data exists.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `ERROR: ... hourly_demand.csv not found` | you haven't added real hourly history yet — see §0 |
| `analyze.py` prints "No cells had enough data" | expected until you have ≥2 occurrences per `(holiday, day_offset)` — needs real, multi-year data |
| `split.py` warns "no share vector" for a date | that date's DOW bucket had fewer than 5 non-holiday days in the prior 90 — expected on a short history |
| `validate.py` shows empty holiday segments | your holdout window doesn't contain any holiday days — widen `--holdout-days` or pick a different window |
