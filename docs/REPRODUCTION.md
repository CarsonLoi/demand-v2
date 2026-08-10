# Reproducing the deck's benchmark numbers

`docs/Demand_Forecast_Model.pptx` slide 22 presents an accuracy ladder, all on
the held-out window **1–28 May 2026, trained only on data before 1 May**. This
note records what those numbers are, how to reproduce them, and the one thing
that stops them reproducing from current code.

---

## The short version

**Every number in the deck is correctly computed and reproducible — at commit
`0da57cc` (17 June 2026), not at `HEAD`.**

| deck claim | at `0da57cc` | at `HEAD` (63f676b) |
|---|---|---|
| Same day last week — 7.36% | **7.36%** ✓ | 7.36% |
| Flat 28-day average — 5.12% | **5.12%** ✓ | 5.12% |
| Day-of-week average — 4.70% | 4.75% (≈) | 4.75% |
| LightGBM single — 3.47% | **3.47%** ✓ | **3.84%** |
| Six-model ensemble — 3.30% | **3.30%** ✓ | **3.88%** |

The two model rows were confirmed by a **full retrain** at `0da57cc`, not read
from cache: `research/backtest.py --holdout-end 2026-05-28` reports
`MAPE = 3.47%` (LightGBM-L2, 58s) and `MAPE = 3.30%` (6-model hybrid, 766s).
Note the ensemble *helps* at that commit (+0.16pp over the single model) but
*hurts* at `HEAD` (−0.05pp) — another consequence of the feature change below.
| + best-two selection — 3.02% | **3.02%** ✓ | — |
| + holiday anchor — 2.52% | **2.52%** ✓ | — |

The anchor calibration reproduces parameter-for-parameter as well:
`growth_window=28d, post=3, alpha=0.25`, 8 days anchored, `G_hold=1.023`.

---

## Why `HEAD` gives different numbers

`data/derived/base_preds_*_2026-05.csv` — the cached base-model predictions the
selection and anchor figures are computed from — were last regenerated in
commit `0da57cc` (**2026-06-17**).

`v2/_shared.py` changed twice afterwards:

| commit | date | change |
|---|---|---|
| `f266a4c` | 2026-06-20 | **add dragon boat** |
| `63f676b` | 2026-07-30 | reservations `as_of` guard, `MASK_FUTURE_TYPHOON` |

The deck was built **2026-08-02** from the June cache, so it reports the
pre-Dragon-Boat feature set. `data/raw/rawdata.csv` is unchanged since 4 June,
so the underlying data is identical — only the features moved.

### Adding Dragon Boat cost 0.37pp on this window

| | matrix columns | LightGBM-L2 MAPE |
|---|---|---|
| `0da57cc` (no Dragon Boat) | 167 | **3.47%** |
| `HEAD` (with Dragon Boat) | 172 | **3.84%** |

The five new columns are `is_DragonBoat` plus four per-offset dummies. But the
cost is not really about those five columns — adding a holiday adds **anchor
dates**, which change `days_to_next_holiday` and `days_from_last_holiday` for
dates right across the series. Those two are the **#2 and #3 most important
features by gain**. For the May 2026 window, Dragon Boat 2026-06-19 becomes the
next upcoming anchor, so the value of `days_to_next_holiday` changes for every
day in the window.

Dragon Boat was added without a backtest, and it regressed this window.

---

## How to reproduce

The current working tree has further uncommitted changes, so reproduce in an
isolated worktree rather than by editing files in place:

```bash
# June state -- reproduces every deck number
git worktree add --detach /tmp/orig617 0da57cc
cp data/raw/rawdata.csv /tmp/orig617/data/raw/     # rawdata.csv is untracked
cd /tmp/orig617
uv run python research/backtest.py --holdout-end 2026-05-28   # 3.47% / 3.30%
uv run python research/holiday_anchor.py --holdout 2026-05 --use-cache  # 3.02% / 2.52%
```

Drop `--use-cache` to regenerate the base predictions instead of reading them.

---

## Two things to know when quoting these numbers

### 1. The naive baseline had an information advantage

The deck's "same day last week" baseline uses `t-7` unconditionally, so for any
target after 7 May it reads actuals **from inside the test window** — data the
model was denied.

| definition | MAPE |
|---|---|
| `t-7`, reaching into the test window (deck's) | 7.36% |
| last full week before the cutoff, repeated (equal-information) | **6.03%** |

So the headline "**−66% vs naive**" is **−58%** on a like-for-like basis
(6.03% → 2.52%). Still a large, real gain — and note the error runs *against*
the model's favour everywhere else: the model figures are honest, only the
comparison baseline was given an edge.

### 2. The model figures are a production-shaped forecast; the cache is not

`base_preds_hold_2026-05.csv` holds **784 rows** — 28 dates × 28 horizons.
Pooled over all of them, LightGBM-L2 scores 3.27%. The deck's 3.47% is the
**natural-horizon** subset (28 rows: day *i* of the window at horizon *i*),
which is the shape a real 28-day forecast takes. The deck is right to use it;
just don't confuse the two when reading the cache directly.

---

## Recommended follow-ups

1. **Regenerate the cached predictions** whenever `v2/_shared.py` changes, or
   delete them so `--use-cache` cannot silently serve stale results.
2. **Backtest holiday additions before committing them.** Dragon Boat looked
   free and cost 0.37pp.
3. **Re-state the naive comparison** as −58%, or redefine the baseline to be
   information-equivalent.
4. Note that P10–P90 coverage measures **46–54%** against a nominal 80% in both
   states — the intervals are over-confident by construction (see
   `docs/MODEL_REVIEW.md` §1.3).
