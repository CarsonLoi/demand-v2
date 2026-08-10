"""comparison_model.py -- a lightweight regressor, used ONLY inside
validate.py to give the empirical-distribution method's "good enough" claim
a real reference point.

NEVER used to produce an actual forecast. One LightGBM regressor predicting
a single hour's SHARE of the daily total, from: hour-of-day, day-of-week,
month, a holiday flag + day-offset-within-holiday-window, and the
prior-N-days average share for that same hour (an honest lag feature -- it
only reads days strictly before the row's own date, same discipline as
every other lag in this project).

Deliberately small and untuned: its only job is being a fair "the
alternative you'd otherwise have to build," not a competitive product.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import patterns as P  # noqa: E402

FEATURE_COLS = ["hour", "dow", "month", "is_holiday", "holiday_offset", "lag_share_90d"]

LGBM_PARAMS = dict(
    objective="regression", n_estimators=200, learning_rate=0.05,
    num_leaves=15, max_depth=4, min_child_samples=10,
    n_jobs=-1, verbose=-1, random_state=123,
)


def build_hourly_feature_matrix(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    """df: long-format date,hour,demand (from patterns.load_hourly()).

    Returns one row per (date, hour) with FEATURE_COLS + date, hour, share.
    Days with fewer than 24 hours, or a zero/NaN daily total, are dropped --
    their share is undefined.
    """
    date_to_hours = P.date_to_hourly_array(df)
    rows = []
    for d, hours in date_to_hours.items():
        if np.isnan(hours).any():
            continue
        shares = P.share_vector(hours)
        if np.isnan(shares).any():
            continue

        # Honest lag feature: mean share for THIS hour over the prior 90
        # non-holiday days, same window discipline as compute_local_dow_baselines.
        prior_shares = {h: [] for h in range(24)}
        window_start = d - pd.Timedelta(days=P.LOCAL_WINDOW_DAYS)
        for d2, hours2 in date_to_hours.items():
            if d2 < window_start or d2 >= d:      # strictly before d
                continue
            if d2 in holiday_dates or np.isnan(hours2).any():
                continue
            s2 = P.share_vector(hours2)
            if np.isnan(s2).any():
                continue
            for h in range(24):
                prior_shares[h].append(s2[h])

        is_hol = d in holiday_dates
        hol_name, hol_offset = P.find_holiday_membership(d) if is_hol else (None, None)

        for h in range(24):
            lag = float(np.mean(prior_shares[h])) if prior_shares[h] else np.nan
            rows.append({
                "date": d, "hour": h, "dow": d.weekday(), "month": d.month,
                "is_holiday": int(is_hol),
                "holiday_offset": hol_offset if hol_offset is not None else 999,
                "lag_share_90d": lag,
                "share": float(shares[h]),
            })
    return pd.DataFrame(rows)


def train_comparison_model(feat: pd.DataFrame, train_mask: pd.Series) -> lgb.LGBMRegressor:
    """Fits one LightGBM regressor to predict `share` from FEATURE_COLS,
    using only rows where train_mask is True."""
    sub = feat[train_mask].dropna(subset=FEATURE_COLS + ["share"])
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(sub[FEATURE_COLS], sub["share"])
    return model


def predict_day_shares(model: lgb.LGBMRegressor, feat: pd.DataFrame,
                       date: pd.Timestamp) -> np.ndarray:
    """Predicts and NORMALIZES a 24-vector of shares (sums to 1.0) for `date`.

    The model predicts each hour independently, so raw outputs will not sum
    to exactly 1 -- normalizing here keeps the comparison fair: both methods'
    final step is "a 24-vector that sums to 1, applied to the daily total."
    """
    day_rows = feat[feat["date"] == date].sort_values("hour")
    if len(day_rows) != 24 or day_rows[FEATURE_COLS].isna().any().any():
        return np.full(24, np.nan)
    raw = model.predict(day_rows[FEATURE_COLS])
    raw = np.clip(raw, 0, None)   # a negative predicted share is meaningless
    total = raw.sum()
    if total <= 0:
        return np.full(24, np.nan)
    return raw / total
