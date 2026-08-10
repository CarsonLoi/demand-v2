# Presentation & evaluation assets

## Files

| File | What it is |
|---|---|
| `Demand_Forecast_Model.pptx` | The presentation (34 slides) |
| `build_deck.js` | Generator for the deck — edit and re-run to regenerate |
| `qa_images/` | Slide PNGs used for visual QA (regenerable, safe to delete) |

## Regenerating the deck

```powershell
cd "D:\Applications\Work\Models\Demand Forecast v2"
node docs\build_deck.js
```

The generator reads two things from the project at build time, so refreshed
analysis flows into the deck automatically:

- `data/derived/accuracy_by_lead.csv` — the MAPE-by-lead-time chart
- `forecasts/chart_actual_vs_forecast.png` — the actual-vs-forecast chart

If either is missing the deck still builds, with a placeholder in its place.

## Deck structure

| Slides | Section | Audience |
|---|---|---|
| 1–2 | Title + executive summary | Everyone |
| 3–6 | The business problem, cost of error, pipeline overview | Management |
| 7–12 | Data sources and feature engineering | Technical |
| 13–17 | Methodology: architecture, ensemble, blending, validation | Technical |
| 18–23 | Results, negative results, limitations | Everyone |
| 24–28 | Downstream pipeline: hourly, tables, headcount, shifts | Both |
| 29–32 | Operations, maintenance, roadmap | Management |
| 33–34 | Close + technical appendix | Everyone |

Speaker notes are attached to the key slides.

---

# evaluate.py — actual vs forecast analysis

Produces the accuracy charts used in the deck, and supports ad-hoc analysis.

## The `n` parameter

For a target date `t`, a forecast built from data up to `t-1` is a 1-day-ahead
forecast. Using data up to `t-1` to predict `t .. t+n` makes the final day of
that window an **(n+1)-day-ahead** forecast. So `n` controls how far ahead the
forecast was made:

| `n` | Data used | Evaluates |
|---|---|---|
| 0 | up to t−1 | 1-day-ahead accuracy |
| 6 | up to t−1 | 7-day-ahead accuracy |
| 10 | up to t−1 | 11-day-ahead accuracy |

## Two modes

**`fixed_lead`** (default) — every plotted point uses the same lead time.
Answers *"how accurate are we at N days out?"* Only one horizon is trained per
origin, so it is fast.

**`rolling_window`** — origins step by `n+1` days, each contributing a full
`t .. t+n` segment that tiles the range. Answers *"what did our operational
forecast actually look like?"* — mirrors re-forecasting every `n+1` days.

## Usage

```powershell
# 1-day-ahead accuracy across a month
uv run python evaluate.py --start 2026-05-01 --end 2026-05-28 --n 0

# 11-day-ahead accuracy over the same window
uv run python evaluate.py --start 2026-05-01 --end 2026-05-28 --n 10

# Operational view: re-forecast every 7 days
uv run python evaluate.py --start 2026-05-01 --end 2026-05-28 --n 6 --mode rolling_window

# Also produce the MAPE-vs-lead-time curve
uv run python evaluate.py --start 2026-05-01 --end 2026-05-28 --n 0 --by-lead
```

From Python:

```python
from evaluate import rolling_backtest, plot_actual_vs_forecast, summarize

df = rolling_backtest("2026-05-01", "2026-05-28", n=0)
metrics = plot_actual_vs_forecast(df, out_path="june.png")
print(summarize(df))
```

## Honesty guarantee

For each origin `O`, the model trains **only** on demand dates `<= O`. Reservation
snapshots taken after `O` are dropped. Nothing after the origin can influence its
forecast, so reported accuracy is what the model would genuinely have achieved.

## Speed

Results are cached under `data/derived/eval_cache/` keyed by the exact settings,
so re-plotting is instant. First run costs roughly one model fit per origin
(~3 s each) plus a one-off feature-matrix build (~70 s). Raise `--step` to
trade resolution for speed on long ranges.
