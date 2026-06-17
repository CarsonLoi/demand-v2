"""Shared utilities for the hourly forecast module.

Reuses v2's holiday configuration so the daily and hourly pipelines agree on
which dates are holiday windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project root and reuse v2's holiday config
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from _shared import HOLIDAY_ANCHORS, HOLIDAY_WINDOWS  # noqa: E402

DATA_HOURLY = ROOT / "data" / "raw" / "hourly_demand.csv"
DERIVED = ROOT / "data" / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)

# Four DOW buckets used everywhere
DOW_BUCKETS = {
    "weekday":  [0, 1, 2, 3],   # Mon-Thu
    "friday":   [4],
    "saturday": [5],
    "sunday":   [6],
}

# TVD thresholds for the decision cascade
TVD_CLOSE = 0.05         # below = DOW pattern is essentially equivalent
TVD_BORDERLINE = 0.10    # below = DOW with manual-review flag
LOCAL_WINDOW_DAYS = 90   # prior-N-days local DOW baseline
MIN_DOW_SAMPLES = 5      # minimum non-holiday days per DOW bucket
MIN_HOLIDAY_OCCURRENCES = 2  # minimum historical samples per holiday-day cell


def load_hourly() -> pd.DataFrame:
    """Load data/raw/hourly_demand.csv with columns date, hour, demand."""
    df = pd.read_csv(DATA_HOURLY, parse_dates=["date"])
    df["hour"] = df["hour"].astype(int)
    df["demand"] = df["demand"].astype(float)
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def tvd(p, q) -> float:
    """Total Variation Distance between two probability vectors. Bounded [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def share_vector(hourly_demand_24) -> np.ndarray:
    """Normalize a 24-hour demand vector to shares summing to 1.0."""
    arr = np.asarray(hourly_demand_24, dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.full(24, np.nan)
    return arr / total


def get_holiday_window_dates() -> set:
    """All dates falling within ANY holiday window — used to exclude them
    when computing 'normal' DOW baselines."""
    dates = set()
    for name, anchors in HOLIDAY_ANCHORS.items():
        ws, we = HOLIDAY_WINDOWS[name]
        for a in anchors:
            for d in pd.date_range(a + pd.Timedelta(days=ws),
                                   a + pd.Timedelta(days=we)):
                dates.add(pd.Timestamp(d))
    return dates


def date_to_hourly_array(df: pd.DataFrame) -> dict:
    """date -> 24-vector of demand. Days with fewer than 24 rows get NaNs."""
    out = {}
    for d, g in df.groupby("date"):
        arr = np.full(24, np.nan)
        for _, r in g.iterrows():
            arr[int(r["hour"])] = float(r["demand"])
        out[pd.Timestamp(d)] = arr
    return out


def compute_local_dow_baselines(
    reference_date: pd.Timestamp,
    date_to_hours: dict,
    holiday_dates: set,
    days_back: int = LOCAL_WINDOW_DAYS,
) -> dict:
    """For a reference date, compute the four DOW share vectors using only
    non-holiday days from the prior `days_back` days. Returns a dict
    {bucket: 24-vector or None}."""
    window_start = reference_date - pd.Timedelta(days=days_back)
    window_end = reference_date - pd.Timedelta(days=1)

    bucket_shares = {b: [] for b in DOW_BUCKETS}
    for d, hours in date_to_hours.items():
        if d < window_start or d > window_end:
            continue
        if d in holiday_dates:
            continue
        if np.isnan(hours).any():
            continue
        dow = d.weekday()
        s = share_vector(hours)
        if np.isnan(s).any():
            continue
        for bucket, dows in DOW_BUCKETS.items():
            if dow in dows:
                bucket_shares[bucket].append(s)
                break

    out = {}
    for bucket, shares in bucket_shares.items():
        if len(shares) < MIN_DOW_SAMPLES:
            out[bucket] = None
        else:
            out[bucket] = np.mean(shares, axis=0)
    return out


def holiday_occurrence_dates(holiday_name: str, day_offset: int) -> list:
    """All historical dates matching a (holiday, day_offset) cell."""
    anchors = HOLIDAY_ANCHORS.get(holiday_name, [])
    return [a + pd.Timedelta(days=day_offset) for a in anchors]


def find_holiday_membership(date: pd.Timestamp) -> tuple:
    """For a given date, return (holiday_name, day_offset) if it falls in
    any holiday window, else (None, None). If multiple holidays overlap,
    returns the first match in HOLIDAY_ANCHORS' insertion order."""
    for name, anchors in HOLIDAY_ANCHORS.items():
        ws, we = HOLIDAY_WINDOWS[name]
        for a in anchors:
            offset = (date - a).days
            if ws <= offset <= we:
                return name, int(offset)
    return None, None
