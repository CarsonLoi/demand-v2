"""config.py — every knob for the long-history pipeline in one place.

Edit this file, not the scripts. This folder is FULLY SELF-CONTAINED: it uses
its own vendored feature engine (core.py), so nothing here reads from or
writes to production. Experiments cannot affect the live model.
"""
from __future__ import annotations
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────
# Point this at the long-history CSV. Same schema as production:
#     date,demand,floortables
# One row per calendar day, NO GAPS (validate_data.py enforces this).
DATA_FILE = "data/raw/rawdata_long.csv"      # relative to long_history/

# ── Regimes ───────────────────────────────────────────────────────────────
# Demand history is not one homogeneous series. Boundaries are EDITABLE and
# should be set from your own knowledge of the business, not left at these
# defaults. They are used for (a) optional exclusion from training and
# (b) an optional `regime` feature.
#
# Two structural breaks matter, not one:
#   COVID          border closures, near-zero demand
#   Junket crackdown (2022) permanently changed the Macau VIP market, so
#                  "recovery" did NOT return to the 2019 baseline -- it
#                  arrived at a different market structure.
REGIMES = [
    ("pre_covid", "2000-01-01", "2020-01-22"),
    ("covid",     "2020-01-23", "2022-12-31"),
    ("recovery",  "2023-01-01", "2023-12-31"),
    ("current",   "2024-01-01", "2100-01-01"),
]

# Date ranges excluded from TRAINING TARGETS (y set to NaN).
# Their demand values REMAIN in the series feeding lag features -- deleting
# rows would punch a hole that NaNs out every lag reaching across it. See
# README section 3.
#
# Boundaries confirmed against demand-per-table in the actual file:
#   2019-12  21.2   normal
#   2020-01  18.2   declining (Wuhan lockdown 23 Jan)
#   2020-02   3.8   collapse
#   2022-12   2.9   still collapsed
#   2023-01  11.7   recovery begins
#   2023-12  19.7   near-normal
#   2024-01  19.4   normal resumed
EXCLUDE_FROM_TRAINING = [
    ("2020-01-23", "2022-12-31"),   # COVID
    # Typhoon RAGASA: operations suspended ~33h. These are SUPPLY-zero days,
    # not demand observations -- but floortables stayed >0 (a partial-day
    # suspension), so EXCLUDE_CLOSURES below does NOT catch them and they
    # would otherwise be learned as real demand.
    #     2025-09-22   6,292   pre-storm shoulder
    #     2025-09-23   1,295   -82% vs surrounding week
    #     2025-09-24     879   -88%
    #     2025-09-25   6,184   recovery shoulder
    # Listing them here also triggers GUARD_LAGS_CROSSING_EXCLUDED, which NaNs
    # lag_365 / lag_anchor_365 / lag_728 / yoy_ratio for any 2026 target whose
    # yearly lookback lands on them -- without it the Sep-2026 forecast reads
    # 879 as "same day last year" and collapses Sep 23-24 for no reason.
    ("2025-09-23", "2025-09-25"),   # RAGASA suspension + recovery
]

# Exclusion variants to COMPARE. Each entry is (label, extra ranges on top of
# EXCLUDE_FROM_TRAINING). The 2023 recovery year is genuinely ambiguous: it is
# real demand, but non-stationary -- per-table demand climbs 11.7 -> 19.7 over
# the year and never reaches the 2024+ norm. Including it adds 365 days of
# data; excluding it removes a trend the model may misread as signal.
# Decide by measurement, not by argument.
EXCLUSION_SETS = [
    ("covid_only",      []),
]

# Full-closure days are detected from the DATA, not hardcoded: floortables == 0
# means zero tables open. Found in this file:
#   2018-09-16          typhoon Mangkhut (first Macau casino closure)
#   2020-02-05..02-19   COVID government-ordered closure
#   2022-07-11..07-22   COVID lockdown
# These are SUPPLY-zero, not demand observations. demand == 0 also breaks MAPE
# (it divides by actuals), so they must leave the training targets.
EXCLUDE_CLOSURES = True

# Mask yearly-lookback features (lag_365 / lag_728 / lag_anchor_365 /
# yoy_ratio) when their SOURCE date falls in an excluded regime or a closure.
# Without this, 2023-2024 training rows divide by a collapsed COVID baseline
# on the single largest feature block in the model (holiday + YoY features
# measured at ~47% of total gain).
GUARD_LAGS_CROSSING_EXCLUDED = True

# ── Experiment grid ───────────────────────────────────────────────────────
# History start dates to compare. None = use everything in the file.
HISTORY_STARTS = [
    ("recent_only", "2024-01-01"),   # reproduces current production
    ("full",        None),           # everything available
]

# Recency half-life in days. THIS IS THE KEY VARIABLE.
# At the production default of 240, a 2016 row carries ~1/35,000th the weight
# of a recent row -- adding history without changing this does nothing.
HALF_LIVES = [240, 1095]

# The half-life a real forecast/evaluation uses when you do not pass one.
# Stated explicitly rather than taken as HALF_LIVES[-1]: reordering the
# comparison list above must never silently change what a live forecast does.
DEFAULT_HALF_LIFE = 1095

# ── Evaluation ────────────────────────────────────────────────────────────
# Held-out windows are generated automatically from the END of your data, so
# this works whatever range you have. Each origin produces a production-shaped
# 28-day forecast; only days with actuals are scored.
N_EVAL_WINDOWS = 8          # how many held-out windows
EVAL_WINDOW_STRIDE = 28     # days between window origins
EVAL_HORIZONS = None        # None = all 1..28; or e.g. [1,3,7,14,21,28] for speed

# ── Features ──────────────────────────────────────────────────────────────
# `month`, `week_of_month`, `quarter`, `day_of_year` (+ sin/cos) already exist
# in production. ISO week-of-year does NOT -- it is added here. With only ~2
# years of history it would be noise; with 10 years it becomes learnable.
ADD_WEEK_OF_YEAR = True
ADD_REGIME_FEATURE = True

# ── Model ─────────────────────────────────────────────────────────────────
# Held fixed across all variants so differences are attributable to the data
# and weighting, not to the model.
LGBM_PARAMS = dict(
    objective="regression", n_estimators=500, learning_rate=0.02,
    num_leaves=15, max_depth=5, min_child_samples=5,
    reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, verbose=-1, random_state=123,
)

OUT_DIR = "output"                           # relative to long_history/

# Cache the built feature matrix to disk. Building it takes 5-8 minutes on 11
# years, and every script in this folder needs the same one, so re-running the
# pipeline without this is mostly waiting. The cache key covers the data file,
# the feature toggles above AND a hash of core.py, so editing the feature
# engine invalidates it automatically -- a stale matrix can never be served.
# Costs roughly 150-250 MB per variant under output/matrix_cache/.
CACHE_MATRIX = True


def regime_of(d: pd.Timestamp) -> str:
    d = pd.Timestamp(d)
    for name, lo, hi in REGIMES:
        if pd.Timestamp(lo) <= d <= pd.Timestamp(hi):
            return name
    return "unknown"
