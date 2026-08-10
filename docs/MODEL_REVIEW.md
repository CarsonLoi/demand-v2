# Model review — August 2026

A full review of the demand-forecast model, covering (1) leakage and evaluation
honesty, (2) the moving-holiday features, and (3) whether typhoons deserve to be
a model feature.

Data reviewed: `data/raw/rawdata.csv`, 2024-01-01 .. 2026-05-28 (879 days).
Every number below is reproducible from the scripts named alongside it.

---

## 0. Executive summary

| Area | Verdict |
|---|---|
| Leakage in **production** forecasting | None found. The horizon-aware feature design is correct. |
| Leakage in **backtests** | One real instance: the typhoon feature (`USE_WEATHER=True` on committed `main`). It inflates any backtest covering a typhoon. |
| Moving-holiday features | Materially incomplete. CNY runs 16–22% MAPE against 3–4% on ordinary days, and the information needed to fix it is already in the data. |
| Typhoon as a learned feature | Should stay off. The driver is a casino **closure decision**, not a weather variable. |
| Prediction intervals | Over-confident by construction (in-sample residuals). |
| Blend calibration | Not leaky, but tuned on far less independent data than it appears. |

The single biggest accuracy opportunity is the moving-holiday profile, not the
model architecture.

---

## 1. Leakage audit

### 1.1 What is correct

The feature matrix is built from **all** history, but every backward-looking
feature is gated on `min_safe = horizon + 1`. For a row `(target_date T,
horizon h)` nothing earlier than `T-(h+1)` is ever read. Since a forecast from
origin `O` uses `h = T - O`, this resolves to `O-1` exactly. The design is
right, and I could not break it:

| Feature group | Gate | Verdict |
|---|---|---|
| `lag_L` | only set when `L >= h+1` | safe |
| `lag_anchor_L` | `eff = max(L, h+1)` | safe |
| `rolling_*`, `ewma_*` | offsets start at `h+1` | safe |
| `same_dow_mean_4w/8w` | only multiples of 7 that are `>= h+1` | safe |
| `yoy_ratio` | `T-(h+1)` and `T-(h+1)-365` | safe |
| `same_holiday_lastyear_lag` | `T-364` | safe |
| `holiday_yoy_lag` / `_lift` | previous year's anchor ±30–60d | safe |
| calendar / holiday flags / Mainland blocks | pure calendar facts | safe |
| `y` | training filtered to `target_date <= origin` | safe |
| `make_sample_weights` | max taken over the training subset only | safe |

Evaluation paths are also correctly walled off:

- `forecast.py` excludes rows after `run_date` and warns when the CSV extends
  past it.
- `evaluate.py` sets `origin = target - lead` and queries `horizon = lead`, so
  the gate above resolves to `origin-1`.
- `research/backtest.py` builds the matrix on full data but trains only on
  `train_dates_set`; safe for the same reason.
- `blend.py` calibrates on a validation window that ends before the production
  models' training data, and `yoy_growth` filters `date < before_date`.
- The reservations path filters snapshots by `update_date <= as_of`.

### 1.2 The one real leak — the typhoon feature in backtests

On committed `main`:

```python
USE_WEATHER = True
MASK_FUTURE_TYPHOON = False
```

`add_weather_features` joins `typhoons.csv` onto `target_date` with no cutoff.
In **production** this is harmless — the scraped file only contains typhoons
that have already happened, so future rows get 0.

In a **backtest** it is not harmless. `research/backtest.py` and `evaluate.py`
build the matrix over the whole series, so held-out days carry their *true*
typhoon flag. The model is told "a T8 occurred on this future day" — a fact not
available at forecast time. Any backtest spanning a typhoon is inflated.

This matters for one number already recorded in `v2/_shared.py`: the claim that
weather "saves −1.26pp on a typhoon window (Sep 2025)". That saving *is* the
leak. The measured cost on a clean window (+0.42pp) is the honest half of that
comparison.

The uncommitted working-tree change already sets `USE_WEATHER = False`, which
resolves it. The recommendation below is to keep it off and delete the
ambiguity rather than rely on a toggle.

### 1.3 Not leakage, but the numbers are softer than they look

**Prediction intervals are in-sample.** `forecast_lgbm_l2` computes
`residuals = y - m.predict(X)` on the *training* matrix, then takes their 10th
and 90th percentiles as P10/P90. A 500-tree LightGBM fits its training set far
more tightly than unseen data, so the band is too narrow. This is exactly the
documented symptom — P10–P90 covers ~55–60% of actuals rather than 80%. It is
an honest-reporting problem, not a leak, and it is fixable with out-of-fold
residuals.

**The holiday anchor is tuned on very little.** `blend.calibrate_anchor`
searches 3 growth windows × 4 post offsets × 21 alphas = **252 configurations**
and keeps the best. On a 180-day validation window it scores those against:

| `post` | distinct fixed-holiday dates | rows scored |
|---|---|---|
| 0 | 14 | 392 |
| 3 | 26 | 728 |

The MAPE is computed over rows, but each date contributes 28 near-identical
rows, so the effective sample is the **date** count — roughly 14 independent
points for 252 configurations. Expect the calibrated alpha to be noisy.

**Model selection is thin too.** `build_selection_table` picks the best 2 of 6
models for each of 7 × 28 = 196 `(dow, horizon)` cells from ~26 observations
per cell. A large share of cells will be selecting on noise.

Neither is dishonest — both are calibrated on data the production models never
saw. `research/holiday_anchor.py` is careful about this: it calibrates alpha on
validation holidays falling in Nov–Apr and applies it to a May holdout, so the
transfer is genuine.

But the headline that follows from it is thinner than it reads. README and
TUTORIAL present **"3.30% (equal) → 2.52% (selection)"** as the validated
production result. That is **one 28-day window** (May 2026), with an alpha
picked from 252 configurations on ~14 independent dates. It is evidence, not a
performance guarantee, and it should be quoted with the window attached rather
than as a general figure. Re-deriving it across several windows would cost a
few hours and would either firm it up or reveal it as window-specific.

### 1.4 Checked and cleared: `floortables`

Backtests read the target date's *true* `floortables`, while production pads
future days with the last observed value — a genuine train/serve skew. Measured
over 731 rolling 28-day windows:

- mean absolute error from padding: **1.89 tables (0.68%)**
- correlation of demand deviation with floortables deviation: **+0.031**

The variable barely moves demand once weekday and trend are removed, so the
skew is immaterial. No action needed.

---

## 2. Moving holidays

### 2.1 The problem, measured

Accuracy splits sharply by day type (origin-based 28-day forecasts, LGBM-L2):

| Window | overall MAPE | ex-CNY | CNY |
|---|---|---|---|
| Feb-2026, forecast from 2026-01-31 | 11.55% | 3.30% | **16.13%** |

CNY days are roughly **5× worse** than ordinary days. This is the dominant
error source in the whole model.

### 2.2 Why — the failure is concentrated in the pre-holiday collapse

Demand relative to surrounding normal trade, by offset from the CNY anchor:

| year | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|---|
| 2024 | 0.66 | 0.56 | 0.51 | **0.43** | 0.65 | 0.85 | 1.07 | 1.17 |
| 2025 | 0.77 | 0.64 | 0.51 | **0.47** | 0.75 | 0.88 | 1.02 | 1.09 |
| 2026 | 0.71 | 0.62 | 0.55 | **0.49** | 0.73 | 0.91 | 1.11 | 1.09 |

Demand **more than halves** on the eve of CNY, then rebounds above normal. The
shape is strikingly consistent across all three years.

The model misses it because the features that could carry the signal don't:

- `lag_364` (the legacy "same holiday last year" feature) lands on an ordinary
  day for a lunar holiday — CNY 2026 falls **384** days after CNY 2025, so a
  364-day lookback misses by 20 days.
- `rolling_mean_*` and `same_dow_mean_*` are themselves dragged down by the
  slump, so the "current level" is wrong exactly when it matters.
- `CNY_d{offset}` dummies exist, but with a 240-day sample-weight half-life,
  CNY 2024 carries a weight of ~0.12. The model is effectively learning an
  18-day profile from about 1.5 examples.

### 2.3 The information is already in the data

A pure arithmetic predictor — no machine learning — using
**last year's demand at the same position in the same holiday, weekday-corrected,
re-based to today's clean weekday level**:

CNY 2026 window (18 days), forecast from origin 2026-02-09:

| Predictor | MAPE |
|---|---|
| Holiday profile, weekday-corrected | **6.68%** |
| Holiday profile, no weekday correction | 7.05% |
| `lag_364` (the current legacy feature) | 20.69% |
| Clean weekday level, no holiday adjustment | 26.73% |
| *(the GBM, for reference)* | *11–22%* |

The worst individual days show the mechanism plainly:

| date | offset | actual | profile | `lag_364` |
|---|---|---|---|---|
| 2026-02-15 | −2 | 3,936 | 4,087 (3.8%) | 7,109 (**80.6%**) |
| 2026-02-16 | −1 | 3,518 | 3,310 (5.9%) | 6,926 (**96.9%**) |

So the model is not short of capacity — it is short of the right feature.

### 2.4 Weekday correction

Matching by holiday position lands on the wrong weekday almost always: for CNY
2026, **0 of 18** matched days share a weekday with their 2025 counterpart
(2026-02-17 is a Tuesday; 2025-01-29 was a Wednesday). With a Saturday/Monday
swing of 1.17×, that is a real error.

Dividing last year's matched demand by *last year's same-weekday* baseline
removes it. Effect on year-over-year profile stability (lower = more transferable):

| holiday | raw spread | weekday-corrected |
|---|---|---|
| CNY | 0.051 | 0.045 |
| DragonBoat | 0.133 | 0.109 |
| Easter | 0.100 | 0.096 |
| MidAutumn | 0.178 | 0.170 |

Real but modest. The large win is having a holiday-aligned profile at all; the
weekday correction is a refinement worth roughly 0.4pp on the CNY window.

**Honest caveat on MidAutumn:** its profile does *not* stabilise (0.170 spread).
2025's Mid-Autumn fell inside Golden Week, so the two observations measure
different things. With two confounded examples, a YoY feature for MidAutumn is
speculative — treat its forecasts with low confidence regardless of what is
implemented.

### 2.5 The holiday windows are cut too short

Probing wider than the configured windows shows a post-holiday dip that is
currently outside the modelled range for **every** moving holiday:

| holiday | configured | effect found outside |
|---|---|---|
| CNY | (−7, +10) | −8 at 0.91× |
| MidAutumn | (−1, +1) | +2 at 0.80×, +3 at 0.88×, +4 at 0.91× |
| DragonBoat | (−2, +1) | −3 at 0.93×, +2 at 0.87×, +3 at 0.90×, +4 at 0.92× |
| Easter | (−2, +1) | +2 at 0.87×, +3 at 0.91× |

A day running at 0.80× normal that the model treats as ordinary is a guaranteed
20% miss. Widening is a config change and is being validated on held-out error
before it is recommended, not adopted from this table alone.

---

## 3. Typhoons

### 3.1 The signal level does not predict the impact

All T8+ days inside the data, against a same-weekday baseline:

| date | typhoon | max signal | hours T8+ | impact |
|---|---|---|---|---|
| 2024-09-05 | YAGI | 8 | 5.7 | −11.8% |
| 2024-09-06 | YAGI | 8 | 12.7 | −9.1% |
| 2024-11-13 | TORAJI | 8 | 0.8 | **+10.1%** |
| 2024-11-14 | TORAJI | 8 | 10.3 | **+6.1%** |
| 2025-07-20 | WIPHA | 10 | 19.3 | −22.5% |
| 2025-09-07 | TAPAH | 8 | 2.7 | −13.3% |
| 2025-09-08 | TAPAH | 8 | 13.2 | −11.7% |
| 2025-09-23 | RAGASA | 8 | 9.7 | **−81.3%** |
| 2025-09-24 | RAGASA | 10 | 20.3 | **−87.5%** |

Nine days in 2.4 years, with impacts spanning **+10% to −87%**. Two T8 days with
comparable hours under signal (TORAJI 10.3h, RAGASA 9.7h) differ by 87
percentage points.

Duration doesn't rescue it either — `scrape_typhoon.py` currently discards the
`start_HHMM`/`end_HHMM`/`duration_HHMM` fields the HKO feed provides, and
recovering them (table above) still leaves the contradiction.

### 3.2 The actual driver

Confirmed with the business: **RAGASA stopped casino operations; TORAJI did
not.** The variable that matters is an operational closure decision, which is
not present in — nor derivable from — the HKO signal level.

That makes this unlearnable at the available sample size: 2 closure days, 7
non-closure typhoon days, and no observable feature separating them ahead of
time. A learned coefficient would be fitting the label of one event.

**Verdict: keep `USE_WEATHER = False`.** Handle typhoons as an operational
override when a closure is announced. This also removes the backtest leak in §1.2.

### 3.3 The aftermath is the bigger cost, and it is fixable

The closure days themselves are two days a forecaster can override by hand. The
undiagnosed cost is what those two values do to every downstream feature.

With 879 and 1,295 patron-hours sitting in the series against a ~7,000 norm:

| feature | worst distortion | days affected |
|---|---|---|
| `rolling_mean_3` | **−58.5%** | ~4 |
| `rolling_mean_7` | **−24.6%** | ~8 |
| `rolling_mean_14` | −12.0% | ~15 |
| `rolling_mean_28` | −6.1% | **~29** |

`lag_7/14/21/28` each carry the raw 879 straight into a specific future day.

Concretely, the model entered Golden Week 2025 (2025-10-01, actual 8,165) with a
`rolling_mean_28` of 6,464 instead of 6,884 — a 6% low baseline, for four weeks,
caused entirely by two days the casino was shut.

The obvious inference — a closure day is a supply-zero observation, not a
demand observation, so substitute a same-weekday estimate for feature
construction and leave the target alone — **is wrong**. It was implemented
(backward-looking imputation only, so it stays honest at every origin) and
measured:

| origin | raw MAPE | cleaned MAPE | Δ | raw bias | cleaned bias |
|---|---|---|---|---|---|
| 2025-09-25 | 10.09% | 11.13% | **+1.05** | −599 | −655 |
| 2025-09-30 | 5.56% | 6.02% | **+0.46** | −260 | −313 |
| 2025-10-10 | 5.78% | 6.25% | **+0.47** | −346 | −365 |
| 2025-10-20 | 3.71% | 3.60% | −0.11 | +136 | +105 |
| **pooled** | **6.28%** | **6.75%** | **+0.47** | −267 | −307 |

Cleaning made accuracy worse at three of four origins and made the
under-forecast bias *worse*, not better — the opposite of the predicted
symptom. The most likely explanation is that demand stays genuinely depressed
for some days after a typhoon, and the "contaminated" rolling baseline partly
predicts that recovery; erasing it removes real signal along with the artefact.

**Conclusion: leave closure days in the data.** The distortion is real but no
correction tested beats doing nothing. Recorded here so it is not re-attempted.

---

## 4. Measured results

### 4.1 Protocol

Six forecast origins, each producing a production-shaped 28-day forecast: for
origin `O` the target `O+h` is taken at horizon `h`, so every feature resolves
to data at or before `O-1`. Training uses only `target_date <= O`. The model is
held fixed (LightGBM-L2, production hyperparameters) so the comparison isolates
the features. Scoring is always against **true** actuals, never a cleaned
series.

Note the baseline is **not** "no holiday feature". It already includes the
position-aligned `holiday_yoy_lag` / `holiday_yoy_lift` from the current
working tree. What is being measured is the *incremental* value of weekday
correction plus a level term.

### 4.2 Feature variants

| Window | base | v4_moving | Δ |
|---|---|---|---|
| Oct-2025 GoldenWeek + post-RAGASA | 5.25% | 5.49% | +0.24 |
| Feb-2026 CNY long lead — **CNY days** | **16.13%** | **15.10%** | **−1.03** |
| Feb-2026 CNY long lead — ex-CNY | 3.30% | 3.27% | −0.03 |
| CNY-2026 short lead — CNY days | 11.88% | 11.84% | −0.04 |
| Mar/Apr-2026 normal | 3.74% | 3.83% | +0.09 |
| Apr/May-2026 Easter + Ching Ming | 4.05% | 3.74% | **−0.31** |

The weekday-corrected features buy about **1pp on long-lead CNY** and about
0.3pp on the Easter window — both moving-holiday windows, which is where the
mechanism predicts gains. Elsewhere they are a wash, drifting ±0.1–0.2pp, which
is within the noise introduced by adding five sparse columns to a 173-column
matrix under `colsample_bytree=0.8`.

Extending the alignment to **all** holidays rather than moving ones only
(`v4_all`) does not settle either way — it is better at long lead and clearly
worse at short lead:

| CNY days | base | v4_moving | v4_all |
|---|---|---|---|
| Feb-2026 long lead | 16.13% | 15.10% | 14.98% |
| CNY-2026 short lead | 11.88% | 11.84% | **13.40%** |

So the existing note in `v2/_shared.py` — that fixed-date holidays are better
served by the weekday-preserving 364-day lag — survives. Weekday correction
does not rescue them. `MOVING_HOLIDAYS` should stay as it is.

**This is well short of the 6.68% the arithmetic profile reaches (§2.3).** The
likely reason is structural: the holiday features are populated on roughly 4%
of rows, and for the Feb-2026 long-lead forecast the model has seen exactly
**one** prior CNY with them populated. Learning "multiply `lift` by a level
column" is a two-column interaction, which trees discover poorly from one
example. That motivates §4.3.

### 4.3 The anchor — where the gain actually is

If the profile carries information the tree cannot extract from a sparse
column, the fix is to apply it *outside* the model. `blend.py` already does
exactly this for fixed-date holidays (`anchor = lag_365 x growth`) but excludes
lunar ones, because `lag_365` lands ~20 days off for them. The position-aligned,
weekday-corrected profile removes that objection:

    blended = (1 - alpha) * model + alpha * profile
    profile = lift_dow(matched day last year) x clean_dow_level(now)

**Alpha was calibrated on the CNY 2025 window and applied unchanged to CNY
2026.** Nothing from 2026 informed it.

| alpha | CNY 2025 (calibration) | CNY 2026 (held out) |
|---|---|---|
| 0.00 (model alone) | 22.47% | 11.88% |
| 0.50 | 11.81% | 8.25% |
| 0.70 | 8.51% | 7.30% |
| **0.80 — selected on 2025** | **8.02%** | **6.87%** |
| 0.85 | 8.03% | 6.65% |
| 0.90 | 8.29% | 6.64% |
| 1.00 (profile alone) | 10.86% | 6.68% |

**CNY 2026 held out: 11.88% -> 6.87%, a -5.01pp improvement.**

Three things make this credible rather than a lucky fit:

1. Alpha was chosen on a different year and transferred. The best attainable on
   2026 was 6.64% at alpha=0.90; the blindly-transferred 0.80 gave 6.87%, so
   only 0.23pp was left on the table.
2. The curve is **flat between 0.75 and 0.95 on both years**. The result does
   not depend on hitting alpha precisely.
3. The improvement is five times what the feature-based approach achieved on
   the same problem (§4.2), which matches the mechanism: the profile is applied
   directly instead of having to be discovered by a tree from one example.

The anchor touches **only** moving-holiday window days. Ordinary days are
mathematically unchanged, so it cannot degrade the other ~95% of the forecast.

### 4.4 The anchor must be restricted to CNY

Applying alpha = 0.80 **unchanged** to every moving holiday, at two lead
profiles each, shows it is not a general technique:

| holiday | year | lead | model | profile | blended | Δ |
|---|---|---|---|---|---|---|
| CNY | 2025 | 1 | 22.47% | 10.86% | 8.02% | **−14.45** |
| CNY | 2025 | 10 | 23.93% | 10.86% | 8.65% | **−15.28** |
| CNY | 2026 | 1 | 11.88% | 6.68% | 6.87% | **−5.01** |
| CNY | 2026 | 10 | 16.13% | 6.63% | 7.43% | **−8.70** |
| Easter | 2025 | 1 | 8.47% | 10.27% | 9.91% | +1.44 |
| Easter | 2025 | 10 | 10.77% | 9.62% | 9.85% | −0.92 |
| Easter | 2026 | 1 | 3.48% | 14.01% | 11.27% | **+7.79** |
| Easter | 2026 | 10 | 5.03% | 15.76% | 13.24% | **+8.20** |
| DragonBoat | 2025 | 1 | 7.55% | 12.28% | 11.10% | +3.54 |
| DragonBoat | 2025 | 10 | 6.60% | 12.58% | 11.07% | +4.48 |
| MidAutumn | 2025 | 1 | 7.43% | 33.83% | 28.51% | **+21.08** |
| MidAutumn | 2025 | 10 | 7.54% | 33.73% | 28.49% | **+20.95** |

**The anchor hurt in 7 of 12 cases.** Pooling everything gives −5.47pp, but that
average is an artefact of CNY's size — quoting it would be misleading.

The split is mechanical, not arbitrary:

- **CNY** has a very large, very stable profile (demand more than halves before
  the anchor, §2.2) over an 18-day window. The model cannot learn it from one
  or two prior occurrences; the profile can state it directly.
- **Easter, DragonBoat, MidAutumn** move demand by only 0.85–1.15×, over windows
  of 3–4 days, and **the model is already accurate on them** — Easter 2026 is
  3.48%. There is nothing to repair, and a profile built from one or two prior
  occurrences contributes more noise than signal.
- **MidAutumn** is the extreme case, exactly as §2.4 predicted: its profile does
  not transfer between years (2025's fell inside Golden Week), so its profile
  MAPE is 33.8% and anchoring quadruples the error.

**Decision: anchor CNY only.** Restricted to CNY, with alpha calibrated on 2025
and applied blind to 2026:

| | model | anchored |
|---|---|---|
| CNY 2026, short lead | 11.88% | **6.87%** |
| CNY 2026, long lead | **16.13%** | **7.43%** |

The long-lead figure is the worst number anywhere in this review, and it more
than halves. Note the CNY 2025 rows above are the *calibration* set for alpha;
only the 2026 rows are held out.

### 4.5 Why fixed holidays need a different mechanism, not just a different lag

Fixed-date holidays are forecast almost as badly as CNY — the Dec-2025
Christmas + New Year window runs **8.63%** against ~4.2% on ordinary months —
so the obvious move is to give them the same anchor treatment with a
fixed-date matching rule. Measured, that does not work, and the reason is
worth recording.

For a fixed-date holiday the natural match is **T − 364** (exactly 52 weeks):
it preserves the DAY OF WEEK and lands within a day of the same calendar date.
But 365 − 364 = 1, so it also **shifts the holiday phase by one day**.

That one-day shift behaves completely differently in the two roles:

| role | effect of the 1-day phase shift |
|---|---|
| **model feature** | Harmless. Weekday dominates casino demand, and the tree can learn around a small phase error. |
| **anchor** | Fatal on multi-day windows. The profile is applied directly, so a 7-day arc gets compared against last year's arc offset by one day. |

Measured per holiday (model MAPE / profile MAPE; **bold** = better):

| holiday | year | window | lead 1 | lead 10 |
|---|---|---|---|---|
| Ching Ming | 2025 | 1 day | **0.96** / 1.17 | 3.18 / **1.22** |
| Ching Ming | 2026 | 1 day | 11.14 / **6.71** | 14.82 / **8.38** |
| Christmas | 2025 | 7 days | **5.93** / 6.57 | 9.55 / **6.54** |
| Golden Week | 2025 | 7 days | **5.18** / 13.07 | **5.22** / 8.03 |
| Labour | 2025 | 5 days | 10.85 / **8.59** | **7.59** / 8.50 |
| Labour | 2026 | 5 days | **5.27** / 7.65 | **3.64** / 7.32 |
| New Year | 2025 | 1 day | **3.72** / 7.60 | — |

The profile wins only 5 of 13 cases, with no rule that separates the wins from
the losses in advance. Golden Week is the clearest failure — 5.18% to 13.07% —
and it is exactly the case the phase shift predicts: a 7-day arc compared
against last year's arc offset by one day.

Adopting this per-holiday would mean fitting a separate alpha for each of five
holidays from one or two prior occurrences each — the same overfitting this
review criticises in `blend.calibrate_anchor` (§1.3). Not adopted.

There is also a clean lead-time pattern in the Christmas rows: the model
degrades sharply with lead (5.93% → 9.55%) while the profile barely moves
(6.57% → 6.54%), because the profile is built from last year's shape and does
not depend on recent actuals. The same shows in CNY, where the anchor's gain
was −5.01pp at short lead but −8.70pp at long lead. A lead-dependent alpha is
therefore likely better than a flat one, for CNY as well.

**Conclusion: fixed holidays get FEATURES, moving holidays get an ANCHOR.**
The `fix_hol_*` block plus horizon pooling already took the Dec window from
8.63% to 7.37% with no anchor involved.

### 4.6 What is deliberately not adopted

- **The sparse holiday features** (§4.2). A wash overall, and redundant once the
  anchor handles the same days directly.
- **Alignment for fixed-date holidays.** Mixed, and worse at short lead.
- **Widened holiday windows** (§2.5). The profile evidence is suggestive, but
  the validation run was stopped in favour of the anchor work. Unvalidated —
  do not adopt on the strength of the profile table alone.

---

## 5. Reproducing this review

| Finding | Script |
|---|---|
| Holiday profiles, typhoon impact | `research/audit_holiday_features.py`, and the diagnostics in §2–3 |
| Typhoon hours from HKO | re-scrape with times (see §3.1) |
| Feature A/B | `ab_run.py` |
