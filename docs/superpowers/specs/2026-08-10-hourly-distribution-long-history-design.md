# Hourly demand splitting for the long-history pipeline — design

Status: approved by user 2026-08-10, pending spec review.

## Problem

The long-history pipeline (`long_history/`) produces daily 28-day forecasts.
Staffing needs an hourly breakdown of each day's total. A prototype for this
already exists at the repo root (`hourly/`), but it:

- imports production's holiday config (`v2/_shared.py`) rather than
  `long_history`'s own (extended 2016-2030, with the Ching Ming date bug
  fixed)
- reads production's forecast output (`forecasts/run_<date>/predictions.csv`)
  rather than `long_history`'s (`long_history/output/run_<date>/predictions.csv`)
- has no validation script proving the approach is adequate
- has no real hourly data to validate against — only a 10-day format-reference
  sample (`hourly/sample_hourly_data.csv`), explicitly marked in its own
  README as too small to run the analyzer

The user's requirement, verbatim: split daily demand into hourly using the
actual historical hourly distribution, and produce validation proving that
approach is good enough — **no trained model needed** to do the splitting
itself.

## Goal

1. Port the hourly-splitting pipeline into `long_history/hourly/`, fully
   independent (own holiday config, own data path, no reference to `v2/` or
   the repo root).
2. Build a validation script that, using only prior data at each point (same
   leakage discipline as the rest of `long_history/`), measures the empirical
   distribution method's hourly accuracy against a lightweight comparison
   model — a real head-to-head, not an isolated number.
3. Run that validation now against the 10-day sample file as a **mechanism
   check only** — confirms the code runs and produces sane output. Explicitly
   NOT proof: too few days, zero holiday coverage, no statistical power.
   Real validation is deferred until `long_history/data/raw/hourly_demand.csv`
   exists with real history (18-24+ months, per the original README's stated
   requirement).
4. Document it: a concepts section in `long_history/docs/COMPLETE_GUIDE.md`,
   and a **dedicated new markdown** for the hourly tutorial (not merged into
   the existing `long_history/docs/TUTORIAL.md`).

## Non-goals

- Does not require or wait for real hourly data to exist before this work is
  done — the pipeline and validation script must be complete and correct now,
  ready to run the moment real data lands.
- Does not change anything in `v2/`, the repo-root `hourly/`, or any
  production file. Repo-root `hourly/` is left as-is (untouched reference).
- Does not attempt to derive an hourly proxy from other data (floortables,
  reservations) — ruled out during brainstorming.

## Design

### Method — unchanged from the prototype, this IS "the actual distribution"

For every forecast day, decide between two sources of hourly *shares*
(fractions of the day, summing to 1.0), using the existing decision cascade:

1. **Local DOW baseline** — mean hourly-share vector from the prior 90
   non-holiday days, bucketed into Mon-Thu / Friday / Saturday / Sunday.
   Computed fresh at forecast time so it tracks current behavior.
2. **Holiday-specific profile** — recency-weighted mean hourly-share vector
   across past occurrences of that exact `(holiday, day_offset)` cell.

A day is classified holiday-window via the long_history holiday config. Its
occurrences are compared to their own contemporary DOW baseline using Total
Variation Distance (TVD):

| condition | decision |
|---|---|
| `tvd_mean < 0.05` and `tvd_max < 0.08`, consistent modal DOW | inherit DOW pattern (close match) |
| `tvd_mean < 0.10`, consistent modal DOW | inherit DOW pattern, flagged for manual review |
| `tvd_mean >= 0.10` | use the holiday-specific profile — shape is materially different |
| modal DOW inconsistent across occurrences | use the holiday-specific profile — pattern is drifting |

No gradient boosting, no neural net — averages plus a distance threshold.
Logic ported as-is from the prototype; only the holiday-config and I/O paths
change.

### Comparison model — exists only to be measured against

One LightGBM regressor, predicting a single hour's *share* of the daily
total. Features: hour-of-day, day-of-week, month, a holiday flag plus
day-offset-within-holiday-window, and the prior-N-days average share for that
same hour (its own honest lag feature, horizon-gated the same way every other
lag in this project is — no reading same-day or future hours). Trained once
per validation run on all rows before the current backtest split, not
per-day-refit, to keep it a fair "the alternative," not a needlessly heavy
one. Never used to produce an actual forecast; exists only inside
`validate.py`. Its purpose is to give the "good enough" claim a real
reference point — the same measure-don't-assert standard used for the
GAM-vs-LightGBM comparison earlier in this project's daily-model work.

### Validation protocol — honest, leakage-free, same discipline as the rest of `long_history/`

For each historical day with real hourly data:

1. Using only data strictly before that day, compute what the empirical
   method would have predicted (DOW baseline or holiday profile, per the
   cascade above) and what the comparison model would have predicted.
2. Apply both to that day's **actual** daily total (not a forecast daily
   total) — this isolates hourly-split error from daily-forecast error,
   which is a separate, already-measured concern.
3. Compare both predictions to the real hourly demand: report hourly MAPE
   and TVD.
4. Split results by day-type: ordinary weekday / Friday-Saturday-Sunday /
   holiday-inherited-DOW / holiday-specific-profile — mirroring the accuracy
   table already documented in the prototype's README.

**Verdict rule**, consistent with every other claim made in this project: the
empirical method is only "good enough, no complicated model needed" if it is
competitive with or better than the comparison model across **every**
day-type segment, not just on a pooled average. If the comparison model wins
on some segment, that must be reported plainly, not averaged away.

### Folder layout

```
long_history/
└── hourly/
    ├── __init__.py
    ├── patterns.py          shared utilities: load hourly data, TVD,
    │                        local DOW baselines, holiday membership
    │                        (uses long_history/engine's holiday config)
    ├── analyze.py            quarterly: builds decisions + holiday profiles
    ├── split.py               daily: applies decisions to a saved forecast
    ├── comparison_model.py   the strawman model, used only by validate.py
    ├── validate.py            NEW: the leave-one-out honest backtest
    ├── README.md              module overview (adapted from the prototype)
    └── sample_hourly_data.csv  10-day format reference, copied in,
                               explicitly labeled non-validating
```

Data expectation: `long_history/data/raw/hourly_demand.csv` — same schema as
the prototype (`date,hour,demand`, every day all 24 hours). Does not exist
yet; user will supply it later.

### Independence

- Holiday dates come from `long_history/engine/core.py` (`HOLIDAY_ANCHORS`,
  `HOLIDAY_WINDOWS`) — the extended, corrected set — not `v2/_shared.py`.
- `split.py` reads `long_history/output/run_<date>/predictions.csv`.
- Nothing under `long_history/hourly/` imports from `v2/` or the repo root.
- All outputs land under `long_history/output/hourly/` (decisions, profiles,
  plots, validation results) or beside the forecast run
  (`predictions_hourly.csv`).

### Testing

- **Conservation check**: a day's split hourly values sum back to (within
  floating-point tolerance of) the daily total being split.
- **Leakage check**: for any split, assert no same-day or future hourly row
  ever participates in the source pattern — same sentinel-style discipline
  used elsewhere in this project.
- **Mechanism smoke test**: run `validate.py` against
  `hourly/sample_hourly_data.csv` (10 days, no holidays). Confirms the
  pipeline runs end-to-end and produces sane, bounded numbers. Report is
  explicitly labeled "mechanism check only — not evidence of accuracy."

### Documentation

- `long_history/docs/COMPLETE_GUIDE.md` — new section: what the hourly
  module does, why an empirical distribution and not a trained model, current
  status (pipeline complete, validation pending real data).
- **New standalone file**, `long_history/hourly/TUTORIAL.md` — step-by-step
  commands: building decisions, splitting a forecast, running validation.
  Not merged into the main `long_history/docs/TUTORIAL.md`.
- `long_history/README.md` — one-line pointer to the hourly module.

## Open questions

None outstanding — both forks (comparison model: yes; validation-script
timing: build now) were resolved during brainstorming.

## Risks / honest caveats to carry into the docs

- Until real hourly data exists, **no claim of "good enough" is actually
  proven** — only the mechanism is confirmed to run. This must be stated
  plainly in both docs, not softened.
- The holiday-specific decision path is entirely untested against real
  holiday data (the sample has zero holiday coverage) and cannot be until
  real data with multiple holiday occurrences exists.
