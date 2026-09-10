from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "long_history"))

import config as C                           # noqa: E402
from engine import core as S                 # noqa: E402
from engine import features as F             # noqa: E402

CLEAN_YEARS = [2015, 2016, 2017, 2018, 2019, 2023, 2024, 2025, 2026]
COVID_YEARS = {2020, 2021, 2022}


def load_frames() -> pd.DataFrame:
    """Raw daily series + the derived per-table column and calendar parts."""
    d = F.load_long_demand()
    d = d[["date", "demand", "floortables"]].sort_values("date").reset_index(drop=True)
    d["demand_per_table"] = d["demand"] / d["floortables"].clip(lower=1)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["dow"] = d["date"].dt.weekday
    return d


def holiday_date_set() -> set[pd.Timestamp]:
    with F.extended_holidays():
        return set(S.holiday_date_set())


def normal_day_mask(df: pd.DataFrame) -> pd.Series:
    """demand>0, floortables>0, not a holiday-window day."""
    hol = holiday_date_set()
    return (
        (df["demand"] > 0)
        & (df["floortables"] > 0)
        & (~df["date"].isin(hol))
    )


def alignment_cv(values) -> float:
    a = np.asarray(values, dtype="float64")
    a = a[~np.isnan(a)]
    if a.size < 2:
        return float("nan")
    m = a.mean()
    if m == 0:
        return float("nan")
    return float(a.std() / m)


_GROUP_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("yearly_lookback", re.compile(r"^(lag_365|lag_728|lag_anchor_365|yoy_ratio|yoy_anchor_diff)$")),
    ("holiday_yoy",     re.compile(r"^(holiday_yoy_lag|holiday_yoy_lift|same_holiday_lastyear_lag)$")),
    ("recency_aggregate", re.compile(r"^(rolling_|ewma_|same_dow_mean_|trend_|same_dow_vs_overall$)")),
    ("capacity",        re.compile(r"^(floortables$|lag_anchor_\d+_per_table$)")),
    ("short_lag",       re.compile(r"^(lag_\d+$|lag_anchor_\d+$|lag_anchor_\d+_zscore$)")),
    ("mainland_block",  re.compile(r"^mainland_")),
    ("regime",          re.compile(r"^regime$")),
    ("holiday_flag",    re.compile(
        r"^(is_(CNY|MidAutumn|DragonBoat|ChingMing|Easter|GoldenWeek|Labour|NewYear|Christmas)$"
        r"|(CNY|GoldenWeek|Labour|DragonBoat)_d[mp]\d+$"
        r"|(CNY|GoldenWeek|Labour|MidAutumn)_dow\d$"
        r"|holiday_dow\d$"
        r"|days_(to_next|from_last)_holiday$"
        r"|(CNY|GoldenWeek|Labour|MidAutumn)_x_recent$)")),
    ("calendar",        re.compile(
        r"^(dow|day$|day_of_year|week_of_month|week_of_year|month|quarter|year$"
        r"|is_weekend|is_friday|is_saturday|is_sunday|is_month_start|is_month_end"
        r"|dow_sin|dow_cos|month_sin|month_cos|doy_sin|doy_cos|woy_sin|woy_cos)")),
]


def feature_group(name: str) -> str:
    for group, pat in _GROUP_PATTERNS:
        if pat.match(name):
            return group
    return "other"
