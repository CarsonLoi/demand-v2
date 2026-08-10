"""Enhanced shared module for v2 — Tier 1 holiday improvements.

Additions vs v1 (../  _shared.py):
  1. Mainland-China holiday block features (via chinese_calendar)
     - mainland_is_workday: 1/0
     - mainland_block_length: consecutive non-working days in current block
     - mainland_block_day_index: position within block (1..N)
     - mainland_is_first_block_day, mainland_is_last_block_day
  2. Same-holiday-last-year lag (relative to block position)
  3. Holiday sample upweight (3x) — applied in model fit functions
  4. USE_RESERVATIONS toggle (default False). When True, loads aggregated
     reservation snapshots and adds per-horizon OTB features.
"""
from __future__ import annotations

from datetime import date as _date
from functools import lru_cache
from pathlib import Path

import chinese_calendar
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent  # project root
DATA_DEMAND = ROOT / "data" / "raw" / "rawdata.csv"
DATA_RES = ROOT / "data" / "raw" / "reservations.csv"
DATA_TYPHOONS = ROOT / "data" / "raw" / "typhoons.csv"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

HOLDOUT_DAYS = 28

# ----- Horizon pooling for the default single-model path.
# Widens horizon h's training set to h-POOL .. h+POOL and adds `horizon` as a
# feature so the models can still separate the leads. 0 = one model per
# horizon (the original, and current, behaviour).
#
# MEASURED AND REJECTED (2026-08). Keep at 0.
#
# An earlier sweep appeared to show pooling helping (TUNE 6.55% -> 6.06% at
# +/-2, 5.80% at +/-3). That result was an ARTEFACT of the matrix it was run
# on: it included the v5 candidate features (clean_dow_level, fix_hol_*,
# mov_hol_*), which are sparse (~4% populated) and were diluting a 180-column
# matrix trained on ~850 rows per model. Pooling helped only because it
# rescued THOSE features from dilution -- it was never a standalone win.
#
# Re-tested on the PRODUCTION feature set (173 cols, no v5), full 1..28
# horizons, 6 windows, anchored as shipped: pooling was WORSE on 6 of 6
# windows (+0.09 to +1.75pp, pooled 5.08% -> 5.49%). Non-CNY days 4.51% ->
# 4.93%; CNY days 7.15% -> 7.54%.
#
# Lesson worth keeping: a structural change validated alongside a feature set
# does not transfer to a different feature set. Runtime, for the record, was
# only 1.3x (not the ~5x expected) -- LightGBM overhead dominates at these
# row counts, so runtime was never the reason to reject it. Accuracy was.
HORIZON_POOL = 0

# ----- Toggle: include reservation features (default OFF per user request)
USE_RESERVATIONS = False
# ------------------------------------------------------------------------

# ----- Toggle: include weather features.
# When True, expects data/raw/weather.csv with at minimum:
#   date, temp_max, temp_min, temp_avg, precip_mm, typhoon_signal
# Missing future dates are filled with day-of-year climatology computed
# from the available history. Missing column -> NaN (tree models handle).
# MEASURED (2026-08): the typhoon feature cannot work on this dataset, and the
# earlier evidence for it was contaminated.
#
# Nine T8+ days exist in 2024-01-01..2026-05-28. Their impact against a
# same-weekday baseline spans +10.1% (TORAJI) to -87.5% (RAGASA) — and those
# two had near-identical exposure (10.3 vs 9.7 hours under T8). Neither signal
# level nor duration separates them. What separates them is that RAGASA
# SUSPENDED CASINO OPERATIONS and TORAJI did not: an operational decision, not
# a weather variable, and nothing in the HKO feed predicts it.
#
# The previously recorded "-1.26pp saving on a typhoon window (Sep 2025)" was
# the backtest leak described below, not a real gain.
#
# Typhoons are handled by typhoon_override.py instead. See README "Typhoons".
USE_WEATHER = False
# Masks typhoon signals on rows after the run date.
#
# DEFAULT IS NOW True. add_weather_features joins typhoons.csv onto target_date
# with no cutoff, so in any BACKTEST the held-out days carry their true signal
# and the model is told a typhoon will occur — it is reading the answer. HKO
# does give a day or two of warning, but the forecast window is 28 days, so
# treating a signal 28 days out as "known" is not defensible.
#
# Set False only for a live production run where typhoons.csv genuinely holds
# forecast (not hindsight) signals for the window ahead.
MASK_FUTURE_TYPHOON = True
# ------------------------------------------------------------------------

# ----- Toggle: lunar-safe holiday year-over-year features.
# Matches a date to the same POSITION within the previous occurrence of the
# same holiday, instead of a fixed 364-day lag (which misaligns lunar
# holidays such as CNY by up to three weeks). Leakage-free: anchors are known
# calendar facts and every value looked up is ~1 year old.
USE_HOLIDAY_YOY = True

# Fixed-364-day "same holiday last year" lag. 364 = exactly 52 weeks, so this
# preserves DAY OF WEEK and lands within 1 day of any FIXED-date holiday —
# genuinely useful there. It is only misleading for MOVING holidays, which the
# holiday-aligned feature above now covers. The two are complementary: keep
# BOTH on. (Turning this off cost 0.66pp on a Labour-Day window.)
USE_LEGACY_HOLIDAY_LAG = True
# ------------------------------------------------------------------------

# ----- Toggle: mask yoy_ratio/lag_365/lag_anchor_365/lag_728 to NaN when their
# reference date falls inside a MOVING holiday's window (guard_yearly_lags,
# defined below MOVING_HOLIDAYS). Fixes a real, confirmed artifact: forecasting
# from 2026-01-28, yoy_ratio compared an ordinary day (2026-01-27) against
# 2 days pre-CNY-2025 (2025-01-27, 51% of normal), producing a value 4.5 std
# devs outside training -> one horizon-model mispredicted 2026-02-02 by 27.7%
# (4,964 vs actual 6,868; 0.8% error once masked).
#
# DEFAULT IS FALSE. MIXED on the 6 validated windows: 2 improved (Mar/Apr
# -0.21pp, Labour -0.06pp) but 4 got WORSE, including two real CNY windows
# (long-lead CNY +1.05pp, short-lead CNY +0.49pp, Easter+ChingMing +0.45pp).
# It IS a net improvement on a 21-origin weekly-cadence 2026 YTD backtest
# (non-CNY 3.84%->3.55%, CNY without anchor 14.16%->13.44%), but that
# contradicts the 6-window full-horizon result closely enough that it is not
# safe to ship broadly. Likely cause: masking is all-or-nothing across the
# whole configured window, including low-effect EDGE days (e.g. Feb 8, 2025
# is CNY-2025 offset +10, ratio ~1.03 -- barely contaminated) where the
# feature was still informative. A narrower mask (core collapse days only) is
# the next thing to try before enabling this. See docs/MODEL_REVIEW.md.
GUARD_YEARLY_LAGS = False
# ------------------------------------------------------------------------

def _easter(year: int) -> pd.Timestamp:
    """Western (Gregorian) Easter — anonymous Gregorian algorithm.
    Computed, not hard-coded, so it can never go stale."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    month, day = divmod(h + L - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


# Years for which the auto-generated (fixed-date and computed) anchors run.
# Extend freely — these three groups can never be wrong.
ANCHOR_YEARS = range(2024, 2030)

# ── Holiday anchors ───────────────────────────────────────────────────────
#  FIXED-DATE and COMPUTED holidays are generated, so they never go stale.
#  MOVING (lunar / solar-term) holidays MUST be maintained by hand — verify
#  each new year against an authoritative source. `chinese_calendar` is
#  authoritative but only covers 2004-2026:
#      import chinese_calendar as cc, datetime as dt
#      cc.get_holiday_detail(dt.date(2026, 2, 17))   -> (True, 'Spring Festival')
#  check_holiday_coverage() warns at run time when an anchor is missing.
HOLIDAY_ANCHORS = {
    # --- MOVING: hand-maintained, verified against chinese_calendar to 2026 ---
    "CNY":        [pd.Timestamp(d) for d in ["2024-02-10", "2025-01-29", "2026-02-17"]],
    "MidAutumn":  [pd.Timestamp(d) for d in ["2024-09-17", "2025-10-06", "2026-09-25"]],
    "DragonBoat": [pd.Timestamp(d) for d in ["2024-06-10", "2025-05-31", "2026-06-19"]],
    "ChingMing":  [pd.Timestamp(d) for d in ["2024-04-04", "2025-04-04", "2026-04-04"]],
    # --- COMPUTED ---
    "Easter":     [_easter(y) for y in ANCHOR_YEARS],
    # --- FIXED DATE: generated ---
    "GoldenWeek": [pd.Timestamp(f"{y}-10-01") for y in ANCHOR_YEARS],
    "Labour":     [pd.Timestamp(f"{y}-05-01") for y in ANCHOR_YEARS],
    "NewYear":    [pd.Timestamp(f"{y}-01-01") for y in ANCHOR_YEARS],
    "Christmas":  [pd.Timestamp(f"{y}-12-25") for y in ANCHOR_YEARS],
}
HOLIDAY_WINDOWS = {
    "CNY":        (-7, 10), "GoldenWeek": (0, 6), "Labour": (0, 4),
    "MidAutumn":  (-1, 1),  "NewYear":    (0, 0), "Christmas": (-3, 3),
    "ChingMing":  (0, 0),   "Easter":     (-2, 1),
    "DragonBoat": (-2, 1),   # soft pre-1~2 days captured by the -2 lookback
}

LAG_DAYS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56, 91, 182, 365, 728]
ANCHOR_LAGS = [7, 14, 21, 28, 56, 91, 182, 365]
ROLLING_WINDOWS = [3, 7, 14, 28]
EWMA_SPANS = [3, 7, 14, 28]

# Holiday upweight factor for training sample_weight
HOLIDAY_UPWEIGHT = 3.0


# ============================================================================
# Data loading
# ============================================================================
def load_demand() -> pd.DataFrame:
    return pd.read_csv(DATA_DEMAND, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def split_train_test(demand, holdout_days=HOLDOUT_DAYS):
    return demand.iloc[:-holdout_days].copy(), demand.iloc[-holdout_days:].copy()


# ============================================================================
# Mainland-China holiday block features (NEW in v2)
# ============================================================================
@lru_cache(maxsize=None)
def _is_mainland_workday(d) -> bool | None:
    try:
        return chinese_calendar.is_workday(d if isinstance(d, _date) else d.date())
    except (NotImplementedError, KeyError, ValueError):
        return None


@lru_cache(maxsize=None)
def _compute_block_info(d: pd.Timestamp) -> tuple[int, int, int]:
    """Return (block_length, day_index_in_block, is_first, is_last) for date d.

    block_length = 0 if d is a workday
    day_index_in_block = position from 1 to block_length, or 0 if workday

    Memoised: the matrix has 28 rows per date, and add_same_holiday_last_year_lag
    re-queries the same dates again, so an uncached call does the calendar walk
    tens of thousands of times for ~900 distinct days. Both functions are pure
    calendar lookups, so caching is value-identical (verified over 943 dates).
    """
    wd = _is_mainland_workday(d)
    if wd is None or wd is True:
        return (0, 0, 0, 0)

    # Walk backward to find block start
    start = d
    while True:
        prev = start - pd.Timedelta(days=1)
        if _is_mainland_workday(prev) is False:
            start = prev
        else:
            break
    # Walk forward to find block end
    end = d
    while True:
        nxt = end + pd.Timedelta(days=1)
        if _is_mainland_workday(nxt) is False:
            end = nxt
        else:
            break
    length = (end - start).days + 1
    idx = (d - start).days + 1
    is_first = int(idx == 1)
    is_last = int(idx == length)
    return (length, idx, is_first, is_last)


def add_mainland_block(df: pd.DataFrame, date_col: str = "target_date") -> pd.DataFrame:
    """Add Mainland-China holiday block features."""
    dates = pd.to_datetime(df[date_col])
    info = [_compute_block_info(d) for d in dates]
    df["mainland_block_length"] = [x[0] for x in info]
    df["mainland_block_day_index"] = [x[1] for x in info]
    df["mainland_is_first_block_day"] = [x[2] for x in info]
    df["mainland_is_last_block_day"] = [x[3] for x in info]
    df["mainland_is_workday"] = [
        1 if _is_mainland_workday(d) is True else (0 if _is_mainland_workday(d) is False else -1)
        for d in dates
    ]
    # Position normalized (0..1 within block)
    df["mainland_block_pos_norm"] = np.where(
        df["mainland_block_length"] > 0,
        df["mainland_block_day_index"] / df["mainland_block_length"].clip(lower=1),
        0.0,
    )
    return df


# ============================================================================
# Calendar features
# ============================================================================
def add_calendar(df: pd.DataFrame, date_col: str = "target_date") -> pd.DataFrame:
    d = pd.to_datetime(df[date_col])
    df["dow"] = d.dt.weekday
    df["day"] = d.dt.day
    df["week_of_month"] = ((d.dt.day - 1) // 7) + 1
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["year"] = d.dt.year
    df["day_of_year"] = d.dt.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_sunday"] = (df["dow"] == 6).astype(int)
    df["is_month_start"] = (d.dt.day <= 3).astype(int)
    df["is_month_end"] = (d.dt.day >= 28).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def add_holiday_flags(df: pd.DataFrame, date_col: str = "target_date") -> pd.DataFrame:
    target = pd.to_datetime(df[date_col])
    for name, anchors in HOLIDAY_ANCHORS.items():
        ws, we = HOLIDAY_WINDOWS[name]
        flag = pd.Series(0, index=df.index, dtype="int8")
        for a in anchors:
            window = pd.date_range(a + pd.Timedelta(days=ws), a + pd.Timedelta(days=we))
            flag |= target.isin(window).astype("int8")
        df[f"is_{name}"] = flag

    for name in ("CNY", "GoldenWeek", "Labour", "DragonBoat"):
        ws, we = HOLIDAY_WINDOWS[name]
        for offset in range(ws, we + 1):
            col = f"{name}_d{offset:+d}".replace("+", "p").replace("-", "m")
            flag = pd.Series(0, index=df.index, dtype="int8")
            for a in HOLIDAY_ANCHORS[name]:
                target_date = a + pd.Timedelta(days=offset)
                flag |= (target == target_date).astype("int8")
            df[col] = flag

    all_anchors = sorted({a for v in HOLIDAY_ANCHORS.values() for a in v})
    arr = np.array([a.value for a in all_anchors])
    target_arr = target.astype("int64").to_numpy()
    next_idx = np.searchsorted(arr, target_arr, side="left")
    has_next = next_idx < len(arr)
    days_to = np.full(len(df), 366, dtype="int32")
    days_to[has_next] = ((arr[next_idx[has_next]] - target_arr[has_next])
                         // (24 * 3600 * 10**9)).astype("int32")
    df["days_to_next_holiday"] = np.clip(days_to, 0, 366)

    last_idx = np.searchsorted(arr, target_arr, side="right") - 1
    has_last = last_idx >= 0
    days_from = np.full(len(df), 366, dtype="int32")
    days_from[has_last] = ((target_arr[has_last] - arr[last_idx[has_last]])
                            // (24 * 3600 * 10**9)).astype("int32")
    df["days_from_last_holiday"] = np.clip(days_from, 0, 366)

    # Holiday × DOW interactions
    if "dow" in df.columns:
        for name in ("CNY", "GoldenWeek", "Labour", "MidAutumn"):
            for d in range(7):
                df[f"{name}_dow{d}"] = (df[f"is_{name}"] & (df["dow"] == d)).astype("int8")
        any_holiday = sum(df[f"is_{n}"] for n in HOLIDAY_ANCHORS.keys()) > 0
        for d in range(7):
            df[f"holiday_dow{d}"] = (any_holiday & (df["dow"] == d)).astype("int8")
    return df


# ============================================================================
# Horizon-aware lag features (with same-holiday-last-year)
# ============================================================================
def add_horizon_aware_lags(df, date_to_demand,
                            lags=LAG_DAYS, anchor_lags=ANCHOR_LAGS,
                            rolling_windows=ROLLING_WINDOWS, ewma_spans=EWMA_SPANS):
    target_dates = pd.to_datetime(df["target_date"]).to_numpy()
    horizons = df["horizon"].to_numpy().astype(int)

    out_arrays = {}
    for L in lags: out_arrays[f"lag_{L}"] = np.full(len(df), np.nan, dtype="float64")
    for L in anchor_lags: out_arrays[f"lag_anchor_{L}"] = np.full(len(df), np.nan, dtype="float64")
    for w in rolling_windows:
        for stat in ("mean", "std", "max", "min"):
            out_arrays[f"rolling_{stat}_{w}"] = np.full(len(df), np.nan, dtype="float64")
    for span in ewma_spans:
        out_arrays[f"ewma_{span}"] = np.full(len(df), np.nan, dtype="float64")
    out_arrays["same_dow_mean_4w"] = np.full(len(df), np.nan, dtype="float64")
    out_arrays["same_dow_mean_8w"] = np.full(len(df), np.nan, dtype="float64")
    out_arrays["yoy_ratio"] = np.full(len(df), np.nan, dtype="float64")

    for i in range(len(df)):
        T = pd.Timestamp(target_dates[i])
        h = horizons[i]
        min_safe = h + 1

        for L in lags:
            if L >= min_safe:
                out_arrays[f"lag_{L}"][i] = date_to_demand.get(T - pd.Timedelta(days=L), np.nan)
        for L in anchor_lags:
            eff = max(L, min_safe)
            out_arrays[f"lag_anchor_{L}"][i] = date_to_demand.get(T - pd.Timedelta(days=eff), np.nan)
        for w in rolling_windows:
            vals = [date_to_demand.get(T - pd.Timedelta(days=min_safe + k), np.nan) for k in range(w)]
            arr = np.array([v for v in vals if not (v is None or np.isnan(v))])
            if len(arr) > 0:
                out_arrays[f"rolling_mean_{w}"][i] = arr.mean()
                out_arrays[f"rolling_max_{w}"][i] = arr.max()
                out_arrays[f"rolling_min_{w}"][i] = arr.min()
                if len(arr) > 1:
                    out_arrays[f"rolling_std_{w}"][i] = arr.std()
        for span in ewma_spans:
            alpha = 2.0 / (span + 1)
            ew_sum, ew_w = 0.0, 0.0
            for k in range(span * 3):
                v = date_to_demand.get(T - pd.Timedelta(days=min_safe + k), np.nan)
                if not (v is None or np.isnan(v)):
                    w = (1 - alpha) ** k
                    ew_sum += v * w; ew_w += w
            if ew_w > 0:
                out_arrays[f"ewma_{span}"][i] = ew_sum / ew_w
        for n_weeks, key in [(4, "same_dow_mean_4w"), (8, "same_dow_mean_8w")]:
            same_dow_vals = []
            for k in range(1, n_weeks + 1):
                offset = 7 * k
                if offset >= min_safe:
                    v = date_to_demand.get(T - pd.Timedelta(days=offset), np.nan)
                    if not (v is None or np.isnan(v)):
                        same_dow_vals.append(v)
            if same_dow_vals:
                out_arrays[key][i] = float(np.mean(same_dow_vals))

        v_recent = date_to_demand.get(T - pd.Timedelta(days=min_safe), np.nan)
        v_yoy = date_to_demand.get(T - pd.Timedelta(days=min_safe + 365), np.nan)
        if (not (v_recent is None or np.isnan(v_recent))
                and not (v_yoy is None or np.isnan(v_yoy)) and v_yoy > 0):
            out_arrays["yoy_ratio"][i] = v_recent / v_yoy

    for k, v in out_arrays.items():
        df[k] = v
    return df


def moving_holiday_date_set() -> set:
    """Every date inside a MOVING holiday's window (CNY, Mid-Autumn, Dragon
    Boat, Easter only — fixed-date holidays are deliberately excluded, see
    guard_yearly_lags below)."""
    out = set()
    for name in MOVING_HOLIDAYS:
        ws, we = HOLIDAY_WINDOWS[name]
        for a in HOLIDAY_ANCHORS[name]:
            for k in range(ws, we + 1):
                out.add(a + pd.Timedelta(days=k))
    return out


def guard_yearly_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Mask yearly-lookback features (yoy_ratio, lag_365, lag_anchor_365,
    lag_728) to NaN when their reference date falls inside a MOVING holiday's
    window.

    Root cause: for a forecast run at origin O, min_safe = horizon+1, so
    yoy_ratio's two reference dates simplify to O-1 and O-1-365 — the SAME two
    dates for every horizon in that run, regardless of which day is being
    predicted. If either date falls inside a MOVING holiday's window (which
    shifts by up to ~3 weeks year to year), the ratio compares mismatched
    points in the holiday cycle. Measured case: forecasting from 2026-01-28,
    O-1=2026-01-27 (ordinary Tuesday) vs O-1-365=2025-01-27 (2 days before
    CNY-2025, demand at 51% of normal) gives yoy_ratio=1.97 — a value 4.5
    standard deviations outside anything in the training data. LightGBM does
    not extrapolate gracefully, so different horizon-models route this
    out-of-distribution input to different, sometimes badly wrong, leaves —
    even for a target date (here, 2 days later) that isn't itself anywhere
    near a holiday window.

    FIXED-date holidays are deliberately NOT guarded: a 365-day lag correctly
    preserves their calendar alignment (that is the whole reason the legacy
    364-day lag is kept for them, see MOVING_HOLIDAYS above), so masking there
    would discard genuine signal rather than remove contamination.

    MEASURED on a 2026 YTD rolling-origin backtest (21 weekly re-forecasts,
    Jan-May, horizons 1-7): non-CNY MAPE 3.84% -> 3.55%, CNY MAPE (no anchor)
    14.16% -> 13.44%, combined with the CNY anchor 4.26% -> 3.99%. The
    specific case above: predicted 4,964 (actual 6,868, 27.7% error) -> 6,814
    (0.8% error). Improved on every measured slice, no regressions found.
    """
    T = pd.to_datetime(df["target_date"])
    H = df["horizon"].astype(int)
    min_safe = H + 1
    moving_hol = moving_holiday_date_set()

    v_recent_date = T - pd.to_timedelta(min_safe, unit="D")
    v_yoy_date = v_recent_date - pd.Timedelta(days=365)
    bad_yoy = v_recent_date.isin(moving_hol) | v_yoy_date.isin(moving_hol)
    df.loc[bad_yoy, "yoy_ratio"] = np.nan

    t365 = T - pd.Timedelta(days=365)
    bad_365 = t365.isin(moving_hol)
    if "lag_365" in df.columns:
        df.loc[bad_365, "lag_365"] = np.nan
    if "lag_anchor_365" in df.columns:
        df.loc[bad_365, "lag_anchor_365"] = np.nan

    if "lag_728" in df.columns:
        t728 = T - pd.Timedelta(days=728)
        bad_728 = t728.isin(moving_hol)
        df.loc[bad_728, "lag_728"] = np.nan

    return df


def add_same_holiday_last_year_lag(
    df: pd.DataFrame, date_to_demand: dict, date_col: str = "target_date",
) -> pd.DataFrame:
    """For each row, demand from same RELATIVE position in last year's
    Mainland holiday block (if both this year's date AND last year's match
    are in non-working blocks)."""
    target = pd.to_datetime(df[date_col])
    out = np.full(len(df), np.nan, dtype="float64")
    for i, T in enumerate(target):
        # Only for holiday-window days (non-workdays in Mainland calendar)
        wd = _is_mainland_workday(T)
        if wd is True or wd is None:
            continue
        T_last = T - pd.Timedelta(days=364)
        if _is_mainland_workday(T_last) is False:
            v = date_to_demand.get(T_last, np.nan)
            if not (v is None or np.isnan(v)):
                out[i] = v
    df["same_holiday_lastyear_lag"] = out
    return df


def check_holiday_coverage(demand: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS,
                            verbose: bool = True) -> list:
    """Warn loudly if the forecast window runs past the last anchor of any
    holiday — those days would silently be treated as ordinary.

    Holiday anchors are hand-maintained, so this is the guard that stops a
    forecast into an un-anchored period from quietly losing its holiday signal.
    """
    last = pd.Timestamp(demand["date"].max())
    window_end = last + pd.Timedelta(days=holdout_days)
    gaps = []
    for name, anchors in HOLIDAY_ANCHORS.items():
        latest = max(anchors)
        ws, we = HOLIDAY_WINDOWS[name]
        # The holiday recurs roughly annually; if the newest anchor's window
        # ends before our forecast window does, the next occurrence is missing.
        if latest + pd.Timedelta(days=we) < window_end:
            gaps.append((name, latest))
    if gaps and verbose:
        print(f"  [holidays] WARNING — forecast window reaches {window_end.date()} "
              f"but these holidays have no anchor that far out:")
        for name, latest in gaps:
            print(f"             {name:12s} last anchor {latest.date()}")
        print("             Those dates will be treated as ORDINARY days. "
              "Add the next occurrence to HOLIDAY_ANCHORS in v2/_shared.py.")
    return gaps


# Holidays whose date MOVES year to year (lunar or computed). Only these need
# holiday-position alignment.
#
# For FIXED-date holidays the legacy 364-day lag is strictly better: 364 is
# exactly 52 weeks, so it lands within 1 day of the anchor AND preserves the
# DAY OF WEEK. Holiday-position alignment hits the exact date but the wrong
# weekday (1 May 2026 is a Friday; 1 May 2025 was a Thursday) — and for casino
# demand, weekday dominates. Applying it to fixed-date holidays measurably
# hurt (May window 2.52% -> 3.22%), so they are excluded here.
MOVING_HOLIDAYS = {"CNY", "MidAutumn", "DragonBoat", "Easter"}


def _holiday_alignment_map(which: set | None = None):
    """date -> (holiday_name, offset_from_anchor, previous_anchor_of_same_holiday).

    Defaults to MOVING_HOLIDAYS — see the note above for why fixed-date
    holidays are deliberately left to the weekday-preserving legacy lag.
    Pass `which` to narrow it further (the CNY anchor below does).
    """
    which = MOVING_HOLIDAYS if which is None else which
    out = {}
    for name, anchors in HOLIDAY_ANCHORS.items():
        if name not in which:
            continue
        ws, we = HOLIDAY_WINDOWS[name]
        srt = sorted(anchors)
        for j, a in enumerate(srt):
            if j == 0:
                continue                      # no prior occurrence to match to
            prev = srt[j - 1]
            for off in range(ws, we + 1):
                d = a + pd.Timedelta(days=off)
                out.setdefault(d, (name, off, prev))
    return out


def add_holiday_aligned_yoy(df, date_to_demand, date_col: str = "target_date"):
    """Lunar-safe year-over-year holiday features.

    A fixed 364-day lag misaligns lunar holidays badly — CNY 2026 falls 384
    days after CNY 2025, so `T - 364` lands on an ordinary day. Here we align by
    POSITION WITHIN THE HOLIDAY instead: a date k days from this year's anchor
    is matched to k days from the previous anchor of the same holiday.

    Holiday anchor dates are calendar facts known years in advance, and every
    looked-up value is ~1 year old, so this is leakage-free at any horizon.

    Adds:
      holiday_yoy_lag  — demand on the matched day last year
      holiday_yoy_lift — that demand ÷ the surrounding non-holiday baseline
                         last year (how much the holiday lifted demand)
    """
    align = _holiday_alignment_map()
    target = pd.to_datetime(df[date_col])

    base_cache = {}

    def baseline_around(anchor):
        """Mean demand 30–60 days either side of the anchor — normal trade,
        deliberately outside the holiday window itself."""
        if anchor in base_cache:
            return base_cache[anchor]
        vals = []
        for lo, hi in ((-60, -30), (30, 60)):
            for k in range(lo, hi + 1):
                v = date_to_demand.get(anchor + pd.Timedelta(days=k), np.nan)
                if v is not None and not np.isnan(v):
                    vals.append(v)
        base_cache[anchor] = float(np.mean(vals)) if vals else np.nan
        return base_cache[anchor]

    # Resolve once per unique date, then map onto rows.
    lag_by_date, lift_by_date = {}, {}
    for d in target.unique():
        d = pd.Timestamp(d)
        hit = align.get(d)
        if hit is None:
            continue
        _name, off, prev = hit
        v = date_to_demand.get(prev + pd.Timedelta(days=off), np.nan)
        if v is None or np.isnan(v):
            continue
        lag_by_date[d] = float(v)
        b = baseline_around(prev)
        if b and not np.isnan(b) and b > 0:
            lift_by_date[d] = float(v) / b

    df["holiday_yoy_lag"] = target.map(lag_by_date).astype("float64")
    df["holiday_yoy_lift"] = target.map(lift_by_date).astype("float64")
    return df


# ============================================================================
# CNY anchor — post-model correction for the pre-Chinese-New-Year collapse
#
# CNY is by far the largest error source in the model: on an origin-based
# 28-day forecast it runs 11.9% MAPE at short lead and 16.1% at long lead,
# against 3.3% on ordinary days. The cause is that demand MORE THAN HALVES on
# the eve of CNY (0.43x / 0.47x / 0.49x of normal in 2024 / 2025 / 2026) and
# the model has at most one or two prior occurrences to learn that from, while
# its "current level" features are themselves dragged down by the same slump.
#
# The shape, however, is extremely stable year to year. So instead of hoping a
# tree discovers it, the prediction is blended toward it directly:
#
#     blended = (1 - alpha) * model + alpha * profile
#     profile = lift_dow(matched day last year) * clean_dow_level(now)
#
# MEASURED, alpha calibrated on CNY 2025 and applied blind to CNY 2026:
#     CNY 2026 short lead   11.9%  ->  6.9%
#     CNY 2026 long lead    16.1%  ->  7.4%
# The alpha curve is flat between 0.75 and 0.95 on both years, so the result
# does not depend on hitting alpha precisely.
#
# CNY ONLY. Applying the same correction to Easter, Dragon Boat and Mid-Autumn
# made all three WORSE (Easter 2026 3.5% -> 11.3%; Mid-Autumn 2025 7.4% ->
# 28.5%). Those holidays move demand by only 0.85-1.15x over 3-4 day windows
# where the model is already accurate, so a profile built from one or two prior
# occurrences is mostly noise. Mid-Autumn is worst because its 2025 occurrence
# fell inside Golden Week, so its profile does not transfer at all.
# See docs/MODEL_REVIEW.md section 4.
# ============================================================================
ANCHOR_HOLIDAYS = {"CNY"}
MOVING_ANCHOR_ALPHA = 0.80
CLEAN_DOW_WEEKS = 6


def holiday_date_set() -> set:
    """Every date inside any configured holiday window.

    Baselines meant to represent NORMAL trade must exclude these — during the
    CNY run-up demand runs at ~0.45x, so an uncleaned same-weekday mean is
    dragged down exactly when the anchor needs it most.
    """
    out = set()
    for name, anchors in HOLIDAY_ANCHORS.items():
        ws, we = HOLIDAY_WINDOWS[name]
        for a in anchors:
            for k in range(ws, we + 1):
                out.add(a + pd.Timedelta(days=k))
    return out


def _dow_baseline_around(anchor, dow, date_to_demand, hol) -> float:
    """Mean demand on weekday `dow`, 30-60 days either side of `anchor`,
    excluding holiday-window days — last year's normal trade for that weekday.

    Matching by holiday POSITION lands on the wrong weekday almost always (0 of
    18 CNY-2026 days share a weekday with their 2025 counterpart), and the
    Sat/Mon swing is 1.17x. Dividing by a same-weekday baseline removes it.
    """
    vals = []
    for lo, hi in ((-60, -30), (30, 60)):
        for k in range(lo, hi + 1):
            t = anchor + pd.Timedelta(days=k)
            if t in hol or t.weekday() != dow:
                continue
            v = date_to_demand.get(t)
            if v is not None and not np.isnan(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _clean_dow_level(T, h, date_to_demand, hol, n_weeks=CLEAN_DOW_WEEKS) -> float:
    """Normal level for T's weekday, from data at or before T-(h+1), skipping
    holiday days. Horizon-gated, so it is safe at any lead."""
    vals, k = [], 1
    while len(vals) < n_weeks and k <= 26:
        off = 7 * k
        if off >= h + 1:
            s = T - pd.Timedelta(days=off)
            if s not in hol:
                v = date_to_demand.get(s)
                if v is not None and not np.isnan(v):
                    vals.append(v)
        k += 1
    return float(np.mean(vals)) if vals else np.nan


def moving_holiday_profile(demand: pd.DataFrame, targets, horizons) -> np.ndarray:
    """Profile estimate for each (target, horizon) inside an ANCHOR_HOLIDAYS
    window; NaN everywhere else.

    Leakage-free: holiday anchors are calendar facts known years ahead, the
    matched day's demand is ~1 year old, and the level term reads only data at
    or before target-(horizon+1).
    """
    dm = dict(zip(demand["date"], demand["demand"].astype(float)))
    hol = holiday_date_set()
    align = _holiday_alignment_map(ANCHOR_HOLIDAYS)
    base_cache = {}
    out = np.full(len(targets), np.nan, dtype="float64")

    for i, (T, h) in enumerate(zip(pd.to_datetime(pd.Series(targets)), horizons)):
        T = pd.Timestamp(T)
        hit = align.get(T)
        if hit is None:
            continue
        _name, off, prev = hit
        m = prev + pd.Timedelta(days=off)
        v = dm.get(m)
        if v is None or np.isnan(v):
            continue
        key = (prev, m.weekday())
        if key not in base_cache:
            base_cache[key] = _dow_baseline_around(prev, m.weekday(), dm, hol)
        b = base_cache[key]
        if not b or np.isnan(b) or b <= 0:
            continue
        lvl = _clean_dow_level(T, int(h), dm, hol)
        if not np.isnan(lvl):
            out[i] = (v / b) * lvl
    return out


def apply_moving_anchor(preds: pd.DataFrame, demand: pd.DataFrame,
                        run_date, alpha: float | None = None) -> pd.DataFrame:
    """Blend predictions toward the CNY profile on CNY-window days.

    Applied in EVERY forecast mode — the default single-model path has exactly
    the same blind spot as the hybrid. Non-holiday days are untouched, so this
    cannot degrade the other ~95% of the forecast.

    `run_date` is required rather than inferred from preds: the horizon must be
    the true lead time, and inferring it silently produces wrong horizons if the
    frame is ever filtered or reordered.
    """
    alpha = MOVING_ANCHOR_ALPHA if alpha is None else alpha
    if not alpha or preds.empty:
        return preds
    run_date = pd.Timestamp(run_date)
    horizons = [(pd.Timestamp(d) - run_date).days for d in preds["date"]]
    if min(horizons) < 1:
        raise ValueError("preds contains dates on or before run_date")

    prof = moving_holiday_profile(demand, preds["date"].tolist(), horizons)
    m = ~np.isnan(prof)
    if not m.any():
        return preds
    out = preds.copy()
    for c in ("p10", "p50", "p90"):
        if c in out.columns:
            out.loc[m, c] = (1 - alpha) * preds.loc[m, c].values + alpha * prof[m]
    print(f"  [cny-anchor] alpha={alpha:.2f} applied to {int(m.sum())} "
          f"CNY-window day(s)")
    return out


def add_floortables(df, date_to_floortables, date_col="target_date"):
    df["floortables"] = pd.to_datetime(df[date_col]).map(date_to_floortables)
    return df


def add_interaction_features(df, date_to_demand):
    if "lag_anchor_7" in df.columns and "floortables" in df.columns:
        df["lag_anchor_7_per_table"] = df["lag_anchor_7"] / df["floortables"].clip(lower=1)
        df["lag_anchor_14_per_table"] = df["lag_anchor_14"] / df["floortables"].clip(lower=1)
    if "rolling_mean_7" in df.columns and "rolling_mean_28" in df.columns:
        df["trend_7_vs_28"] = df["rolling_mean_7"] / df["rolling_mean_28"].clip(lower=1)
        df["trend_diff_7_28"] = df["rolling_mean_7"] - df["rolling_mean_28"]
    if "lag_anchor_7" in df.columns and "rolling_std_28" in df.columns:
        df["lag_anchor_7_zscore"] = (
            (df["lag_anchor_7"] - df["rolling_mean_28"])
            / df["rolling_std_28"].clip(lower=1)
        )
    if "same_dow_mean_4w" in df.columns and "rolling_mean_28" in df.columns:
        df["same_dow_vs_overall"] = df["same_dow_mean_4w"] / df["rolling_mean_28"].clip(lower=1)
    if "lag_365" in df.columns and "lag_anchor_7" in df.columns:
        df["yoy_anchor_diff"] = df["lag_anchor_7"] - df["lag_365"]
    if "rolling_mean_28" in df.columns:
        for hol in ("CNY", "GoldenWeek", "Labour", "MidAutumn"):
            col = f"is_{hol}"
            if col in df.columns:
                df[f"{hol}_x_recent"] = df[col] * df["rolling_mean_28"]
    return df


# ============================================================================
# Reservation features (only if USE_RESERVATIONS=True)
# ============================================================================
# Max lead time (days) kept from reservation snapshots. Data with lead 1..30 is
# fine; anything beyond this is dropped (and missing leads simply yield NaN).
RES_MAX_LEAD = 60


def _load_and_aggregate_reservations(as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load + aggregate reservations (handles both patron-level and pre-aggregated).

    as_of: if given, drop every snapshot with update_date > as_of. This mirrors
    the demand-side `train_dates_set = dates <= run_date` filter — it's what
    prevents a run's res_dow_zscore (which pools OTB across the whole matrix)
    from being normalized using pacing snapshots that didn't exist yet at
    forecast time. Pass the run_date here.

    Robust to: missing file, missing/blank rooms_otb, duplicate snapshots, and
    inhouse dates with no coverage. Returns an empty frame if unusable — the
    model then runs with NaN reservation features (tree models handle NaN)."""
    if not DATA_RES.exists():
        print(f"  [reservations] {DATA_RES} not found — features will be NaN")
        return pd.DataFrame()
    try:
        res = pd.read_csv(DATA_RES, parse_dates=["update_date", "inhouse_date"])
    except (ValueError, KeyError) as e:
        print(f"  [reservations] could not parse {DATA_RES} ({e}) — features will be NaN")
        return pd.DataFrame()

    if "patron_id" in res.columns:
        # Patron-level: count rows per (update_date, inhouse_date)
        snap = res.dropna(subset=["update_date", "inhouse_date"]).groupby(
            ["update_date", "inhouse_date"], as_index=False).size().rename(
            columns={"size": "rooms_otb"})
    elif "rooms_otb" in res.columns:
        snap = res[["update_date", "inhouse_date", "rooms_otb"]].copy()
        snap["rooms_otb"] = pd.to_numeric(snap["rooms_otb"], errors="coerce")
    else:
        print("  [reservations] need either 'rooms_otb' (Format A) or 'patron_id' "
              "(Format B) column — features will be NaN")
        return pd.DataFrame()

    # Drop rows missing any key/value; dedupe (keep the last snapshot for a pair)
    before = len(snap)
    snap = snap.dropna(subset=["update_date", "inhouse_date", "rooms_otb"])
    snap = snap.drop_duplicates(subset=["update_date", "inhouse_date"], keep="last")
    snap["lead_time"] = (snap["inhouse_date"] - snap["update_date"]).dt.days
    snap = snap[(snap["lead_time"] >= 1) & (snap["lead_time"] <= RES_MAX_LEAD)]

    if as_of is not None:
        before_asof = len(snap)
        snap = snap[snap["update_date"] <= as_of]
        cut = before_asof - len(snap)
        if cut > 0:
            print(f"  [reservations] filtered out {cut} snapshot(s) with "
                  f"update_date > {as_of.date()} (run-date leakage guard)")

    snap = snap.reset_index(drop=True)
    dropped = before - len(snap)
    if dropped > 0:
        print(f"  [reservations] dropped {dropped} invalid/duplicate/out-of-lead/future row(s)")
    if not snap.empty:
        print(f"  [reservations] lead range {int(snap.lead_time.min())}..{int(snap.lead_time.max())} days")
    return snap


def add_reservation_features(df: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """For each (target_date, horizon h), look up rooms_otb at lead h+1."""
    if snap is None or snap.empty:
        df["res_otb"] = np.nan
        df["res_pickup_7d"] = np.nan
        df["res_dow_zscore"] = np.nan
        return df

    snap_lookup = snap.set_index(["update_date", "inhouse_date"])["rooms_otb"].to_dict()
    target_dates = pd.to_datetime(df["target_date"]).to_numpy()
    horizons = df["horizon"].to_numpy().astype(int)
    otb = np.full(len(df), np.nan, dtype="float64")
    pickup_7d = np.full(len(df), np.nan, dtype="float64")
    for i, (T_np, h) in enumerate(zip(target_dates, horizons)):
        T = pd.Timestamp(T_np)
        snap_date = T - pd.Timedelta(days=int(h) + 1)
        snap_date_prev = T - pd.Timedelta(days=int(h) + 8)
        v_now = snap_lookup.get((snap_date, T), np.nan)
        v_prev = snap_lookup.get((snap_date_prev, T), np.nan)
        if v_now is not None and not (isinstance(v_now, float) and np.isnan(v_now)):
            otb[i] = float(v_now)
        if (v_now is not None and v_prev is not None
                and not np.isnan(float(v_now)) and not np.isnan(float(v_prev))):
            pickup_7d[i] = float(v_now) - float(v_prev)
    df["res_otb"] = otb
    df["res_pickup_7d"] = pickup_7d

    # Coverage diagnostic: how many rows actually found an OTB snapshot.
    n_cov = int(np.sum(~np.isnan(otb)))
    print(f"  [reservations] res_otb coverage: {n_cov:,}/{len(df):,} rows "
          f"({100*n_cov/max(len(df),1):.0f}%); missing snapshots -> NaN (handled)")

    # DOW z-score (across same-DOW history; in-distribution)
    dow = pd.to_datetime(df["target_date"]).dt.weekday.to_numpy()
    z = np.full(len(df), np.nan, dtype="float64")
    for d in range(7):
        mask = dow == d
        vals = otb[mask]
        if np.sum(~np.isnan(vals)) >= 3:
            mu = np.nanmean(vals); sd = np.nanstd(vals)
            if sd > 0:
                z[mask] = (vals - mu) / sd
    df["res_dow_zscore"] = z
    return df


# ============================================================================
# Weather features (only if USE_WEATHER=True)
#
# Single binary feature kept: is_typhoon_t8plus
#   1 if HKO Tropical Cyclone Warning Signal 8 (or 9 / 10) was active on the
#   target date, 0 otherwise.
#
# Source: data/raw/typhoons.csv (produced by scrape_typhoon.py from HKO).
# When that file is missing, the feature is 0 for every row -- model proceeds.
#
# All other weather features (temperature, humidity, wind, precipitation,
# rolling weather averages, T3 signal, climatology flag) were removed: in
# Macau, T8+ is the only weather signal that materially moves casino demand.
# Lesser weather variables added noise and feature-importance churn without
# moving backtested MAPE.
# ============================================================================


def _load_typhoons() -> pd.DataFrame:
    """Load data/raw/typhoons.csv (date, typhoon_name, highest_signal).
    Returns empty DataFrame if the file is missing -- is_typhoon_t8plus
    will then default to 0 for every row."""
    if not DATA_TYPHOONS.exists():
        print(f"  [weather] {DATA_TYPHOONS} not found "
              "— is_typhoon_t8plus will be 0 everywhere")
        return pd.DataFrame()
    df = pd.read_csv(DATA_TYPHOONS, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def add_weather_features(df: pd.DataFrame, typhoons: pd.DataFrame,
                          as_of: pd.Timestamp | None = None,
                          date_col: str = "target_date") -> pd.DataFrame:
    """Attach is_typhoon_t8plus to each row by joining on target_date.

    as_of: the run_date. Typhoon signals ARE available ahead of time via HKO
    weather forecasts, so future forecast-window signals are legitimate (kept
    by default). Only when MASK_FUTURE_TYPHOON is True are forecast rows
    (target_date > as_of) masked to 0 — for a strict hindsight-free backtest
    where typhoons.csv holds actuals rather than forecasts."""
    if typhoons.empty:
        df["typhoon_signal"] = 0
        df["is_typhoon_t8plus"] = 0
        df["is_typhoon_t9plus"] = 0
        return df

    # Severity matters enormously and is NOT captured by a binary flag:
    # a T8 typically costs ~10-15% of demand, while a T10 has taken it down
    # by ~88% (24 Sep 2025: 879 patron hours against a ~7,000 norm).
    # We therefore expose the ordinal signal level as well as the flag.
    sig_map = (typhoons.set_index(pd.to_datetime(typhoons["date"]))["highest_signal"]
               .to_dict())
    target = pd.to_datetime(df[date_col])
    level = target.map(sig_map).fillna(0).astype("float64")
    if MASK_FUTURE_TYPHOON and as_of is not None:
        masked = int((level.gt(0) & (target > as_of)).sum())
        level = level.where(target <= as_of, 0.0)
        if masked > 0:
            print(f"  [weather] MASK_FUTURE_TYPHOON: masked {masked} future "
                  f"typhoon row(s) (target_date > {pd.Timestamp(as_of).date()})")
    df["typhoon_signal"] = level.astype("int8")          # 0 / 8 / 9 / 10
    df["is_typhoon_t8plus"] = (level >= 8).astype("int8")
    df["is_typhoon_t9plus"] = (level >= 9).astype("int8")  # the severe ones
    return df


# ============================================================================
# Matrix builder
# ============================================================================
def build_matrix(demand: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS,
                  as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """as_of: the run_date. Reservation snapshots with update_date > as_of are
    excluded — mirrors the demand-side train_dates_set <= run_date filter, so
    a run never has access to booking-pacing data that didn't exist yet.
    If omitted, defaults to demand's max date (no-op for a single fresh run)."""
    if as_of is None:
        as_of = demand["date"].max()
    date_to_demand = dict(zip(demand["date"], demand["demand"].astype(float)))
    date_to_floor = dict(zip(demand["date"], demand["floortables"].astype(float)))

    rows = []
    for d in demand["date"]:
        for h in range(1, holdout_days + 1):
            rows.append({"target_date": d, "horizon": h})
    mat = pd.DataFrame(rows)

    mat = add_calendar(mat)
    mat = add_holiday_flags(mat)
    mat = add_mainland_block(mat)        # NEW
    mat = add_horizon_aware_lags(mat, date_to_demand)
    if GUARD_YEARLY_LAGS:
        mat = guard_yearly_lags(mat)
    if USE_LEGACY_HOLIDAY_LAG:
        # Superseded: uses a fixed 364-day offset, which lands 10-20 days away
        # from the matching day for every MOVING holiday (CNY, Mid-Autumn,
        # Dragon Boat, Easter). Kept only for backward comparison.
        mat = add_same_holiday_last_year_lag(mat, date_to_demand)
    if USE_HOLIDAY_YOY:
        mat = add_holiday_aligned_yoy(mat, date_to_demand)
    check_holiday_coverage(demand, holdout_days)
    mat = add_floortables(mat, date_to_floor)
    mat = add_interaction_features(mat, date_to_demand)

    if USE_RESERVATIONS:
        print(f"  [reservations] loading + aggregating (as_of={as_of.date()})...")
        snap = _load_and_aggregate_reservations(as_of=as_of)
        print(f"  [reservations] {len(snap):,} snapshot rows")
        mat = add_reservation_features(mat, snap)
    else:
        print("  [reservations] USE_RESERVATIONS=False — skipping")

    if USE_WEATHER:
        typhoons = _load_typhoons()
        if not typhoons.empty:
            t8plus = (typhoons["highest_signal"] >= 8).sum()
            print(f"  [weather] typhoons.csv: {len(typhoons)} affected day(s), "
                  f"{t8plus} at T8+")
        mat = add_weather_features(mat, typhoons, as_of=as_of)
    else:
        print("  [weather] USE_WEATHER=False — skipping")

    mat["y"] = mat["target_date"].map(date_to_demand)
    return mat


# ============================================================================
# Sample weighting (with holiday upweight)
# ============================================================================
def make_sample_weights(target_dates: pd.Series, is_holiday: pd.Series | None = None,
                         half_life_days: int = 240,
                         holiday_upweight: float = HOLIDAY_UPWEIGHT) -> np.ndarray:
    """Recency decay × holiday upweight."""
    target_dates = pd.to_datetime(target_dates)
    max_date = target_dates.max()
    days_back = (max_date - target_dates).dt.days.values
    w = 0.5 ** (days_back / half_life_days)
    if is_holiday is not None:
        # Upweight holiday-window samples
        w = w * np.where(is_holiday.values, holiday_upweight, 1.0)
    return w


def holiday_mask_from_matrix(sub: pd.DataFrame) -> pd.Series:
    """Identify holiday-window rows for sample weighting."""
    cols = [c for c in sub.columns if c.startswith("is_") and c not in
            ("is_weekend", "is_friday", "is_saturday", "is_sunday",
             "is_month_start", "is_month_end")]
    if not cols: return pd.Series(False, index=sub.index)
    return (sub[cols].sum(axis=1) > 0)


# ============================================================================
# Metrics
# ============================================================================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if len(y_true) == 0: return dict(wape=np.nan, mape=np.nan, rmse=np.nan, bias=np.nan)
    return dict(
        wape=float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()),
        mape=float(np.mean(np.abs((y_true - y_pred) / y_true))),
        rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        bias=float(np.mean(y_pred - y_true)),
    )


def print_metrics(name: str, m: dict, target_mape: float = 0.02):
    flag = " ✅2%" if m["mape"] < target_mape else (" ✅3%" if m["mape"] < 0.03 else "")
    print(f"  {name:22s} WAPE={m['wape']*100:5.2f}%  MAPE={m['mape']*100:5.2f}%  "
          f"RMSE={m['rmse']:7.0f}  Bias={m['bias']:+6.0f}{flag}")
