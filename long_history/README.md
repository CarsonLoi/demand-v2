# Long-History Demand Forecast

Predicts casino patron hours for the next 28 days, trained on eleven years of
history (2015–2026).

**This folder is self-contained.** It has its own feature engine, its own data,
and its own outputs. Nothing outside `long_history/` is read or written at
runtime — you could delete the rest of the repository and this would still
run.

---

## Start here

```bash
uv run python long_history/run_pipeline.py
```

Runs three steps in order, stopping at the first failure:
**validate → evaluate → forecast**.

Results land in `long_history/output/run_<date>/`:

| file | contents |
|---|---|
| `predictions.csv` | `date, p10, p50, p90` — plan against `p50` |
| `forecast.png` | recent actuals plus the 28-day band |
| `metadata.json` | every setting used, for auditability |

---

## Documentation

| document | read it for |
|---|---|
| **[docs/TUTORIAL.md](docs/TUTORIAL.md)** | step-by-step commands for every routine task |
| **[docs/COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md)** | how it works and why — written for someone new to it |
| **[hourly/TUTORIAL.md](hourly/TUTORIAL.md)** | step-by-step commands for splitting a forecast into hourly demand |

---

## Layout

```
run_pipeline.py     ← start here
config.py              every setting; edit this, not the scripts

engine/                the feature engine (core.py, features.py, holidays.py)
data/                  raw/ CSVs, validate.py (hard gate), scrape_typhoon.py
training/              forecast.py, evaluate.py, experiment.py
ops/                   typhoon_override.py — manual closure adjustment
viz/                   charts.py, deck_data.py
hourly/                daily-to-hourly demand splitting (own README/TUTORIAL)
research/              one-off investigations, kept for the record
docs/                  the two documents above
output/                everything produced
```

---

## Accuracy

Held-out — for each forecast the model trained only on data available at that
moment.

| how far ahead | average miss | days within 5% |
|---|---|---|
| 1 day | 3.78% | 79% |
| 7 days | 4.07% | 71% |
| 14 days | 4.96% | 63% |
| 28 days | 4.84% | 62% |

Against the short-history (2-year) model on an identical 7-day-lead test:
**4.07% vs 5.08%**, and — more importantly — **4 days missed by over 15%
versus 8**. The real gain is fewer disasters, not a better average.

---

## Three things that will bite you

1. **A missing day in the CSV silently corrupts a lot of inputs.** Every
   "N days ago" calculation counts through the calendar. `data/validate.py`
   is a hard gate; never bypass it, and never fix a gap by deleting rows.

2. **The P10–P90 band is narrower than it claims** — measured coverage is
   46–54% against a nominal 80%. Treat it as a relative signal only.

3. **Typhoons are not predicted, on purpose.** RAGASA shut the floor and the
   forecast missed by 468%; TORAJI did not and demand was normal. The driver
   is a human decision, so use `ops/typhoon_override.py`.

See [docs/COMPLETE_GUIDE.md §15](docs/COMPLETE_GUIDE.md#15-known-limitations)
for the full list, including the **2027 holiday-calendar deadline**.
