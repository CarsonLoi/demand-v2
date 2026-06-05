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
from pathlib import Path

import chinese_calendar
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent  # project root
DATA_DEMAND = ROOT / "data" / "raw" / "rawdata.csv"
DATA_RES = ROOT / "data" / "raw" / "reservations.csv"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

HOLDOUT_DAYS = 28

# ----- Toggle: include reservation features (default OFF per user request)
USE_RESERVATIONS = False
# ------------------------------------------------------------------------

# Holiday config (same as v1)
HOLIDAY_ANCHORS = {
    "CNY":        [pd.Timestamp(d) for d in ["2024-02-10", "2025-01-29", "2026-02-17"]],
    "GoldenWeek": [pd.Timestamp(d) for d in ["2024-10-01", "2025-10-01", "2026-10-01"]],
    "Labour":     [pd.Timestamp(d) for d in ["2024-05-01", "2025-05-01", "2026-05-01"]],
    "MidAutumn":  [pd.Timestamp(d) for d in ["2024-09-17", "2025-10-06", "2026-09-25"]],
    "NewYear":    [pd.Timestamp(d) for d in ["2024-01-01", "2025-01-01", "2026-01-01"]],
    "Christmas":  [pd.Timestamp(d) for d in ["2024-12-25", "2025-12-25"]],
    "ChingMing":  [pd.Timestamp(d) for d in ["2024-04-04", "2025-04-04", "2026-04-04"]],
    "Easter":     [pd.Timestamp(d) for d in ["2024-03-31", "2025-04-20", "2026-04-05"]],
}
HOLIDAY_WINDOWS = {
    "CNY":        (-7, 10), "GoldenWeek": (0, 6), "Labour": (0, 4),
    "MidAutumn":  (-1, 1),  "NewYear":    (0, 0), "Christmas": (-3, 3),
    "ChingMing":  (0, 0),   "Easter":     (-2, 1),
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
def _is_mainland_workday(d) -> bool | None:
    try:
        return chinese_calendar.is_workday(d if isinstance(d, _date) else d.date())
    except (NotImplementedError, KeyError, ValueError):
        return None


def _compute_block_info(d: pd.Timestamp) -> tuple[int, int, int]:
    """Return (block_length, day_index_in_block, is_first, is_last) for date d.

    block_length = 0 if d is a workday
    day_index_in_block = position from 1 to block_length, or 0 if workday
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

    for name in ("CNY", "GoldenWeek", "Labour"):
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
def _load_and_aggregate_reservations() -> pd.DataFrame:
    """Load + aggregate reservations (handles both patron-level and pre-aggregated)."""
    if not DATA_RES.exists():
        print(f"  [reservations] {DATA_RES} not found — skipping")
        return pd.DataFrame()
    res = pd.read_csv(DATA_RES, parse_dates=["update_date", "inhouse_date"])
    if "patron_id" in res.columns:
        snap = res.groupby(["update_date", "inhouse_date"], as_index=False).size().rename(
            columns={"size": "rooms_otb"}
        )
    else:
        snap = res[["update_date", "inhouse_date", "rooms_otb"]].copy()
    snap["lead_time"] = (snap["inhouse_date"] - snap["update_date"]).dt.days
    snap = snap[(snap["lead_time"] >= 1) & (snap["lead_time"] <= 60)].reset_index(drop=True)
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
        otb[i] = v_now
        if not (np.isnan(v_now) or np.isnan(v_prev)):
            pickup_7d[i] = v_now - v_prev
    df["res_otb"] = otb
    df["res_pickup_7d"] = pickup_7d

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
# Matrix builder
# ============================================================================
def build_matrix(demand: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> pd.DataFrame:
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
    mat = add_same_holiday_last_year_lag(mat, date_to_demand)  # NEW
    mat = add_floortables(mat, date_to_floor)
    mat = add_interaction_features(mat, date_to_demand)

    if USE_RESERVATIONS:
        print("  [reservations] loading + aggregating...")
        snap = _load_and_aggregate_reservations()
        print(f"  [reservations] {len(snap):,} snapshot rows")
        mat = add_reservation_features(mat, snap)
    else:
        print("  [reservations] USE_RESERVATIONS=False — skipping")

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
