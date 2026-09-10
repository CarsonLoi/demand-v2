"""checks.py -- assert-based checks for the exploration library.

No pytest in this repo; matches long_history/hourly/checks.py style. Each
check returns (ok: bool, message: str). Run:

    uv run python research/explore_longhistory/checks.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eda_lib as L          # noqa: E402


def check_alignment_cv_zero_when_equal() -> tuple[bool, str]:
    v = L.alignment_cv([3.0, 3.0, 3.0])
    return v == 0.0, f"alignment_cv([3,3,3]) = {v} (want 0.0)"


def check_alignment_cv_known_spread() -> tuple[bool, str]:
    # values 1,2,3 -> mean 2, population std sqrt(2/3) ~ 0.8165 -> cv ~ 0.4082
    v = L.alignment_cv([1.0, 2.0, 3.0])
    return abs(v - 0.40825) < 1e-4, f"alignment_cv([1,2,3]) = {v:.5f} (want 0.40825)"


def check_alignment_cv_nan_when_thin() -> tuple[bool, str]:
    v = L.alignment_cv([5.0])
    return np.isnan(v), f"alignment_cv([5]) = {v} (want nan)"


def check_feature_group_known_names() -> tuple[bool, str]:
    cases = {
        "lag_365": "yearly_lookback", "yoy_ratio": "yearly_lookback",
        "lag_2": "short_lag", "lag_anchor_7": "short_lag",
        "rolling_max_28": "recency_aggregate", "ewma_14": "recency_aggregate",
        "same_dow_mean_8w": "recency_aggregate",
        "holiday_yoy_lag": "holiday_yoy",
        "is_CNY": "holiday_flag", "CNY_dm7": "holiday_flag", "holiday_dow3": "holiday_flag",
        "days_to_next_holiday": "holiday_flag", "CNY_x_recent": "holiday_flag",
        "dow_sin": "calendar", "month": "calendar", "week_of_year": "calendar",
        "mainland_block_pos_norm": "mainland_block",
        "regime": "regime",
        "floortables": "capacity", "lag_anchor_7_per_table": "capacity",
    }
    bad = {k: L.feature_group(k) for k, want in cases.items() if L.feature_group(k) != want}
    return not bad, ("misgrouped: " + str(bad) if bad else f"{len(cases)} feature names grouped correctly")


def check_load_frames_shape() -> tuple[bool, str]:
    d = L.load_frames()
    ok = (
        list(d.columns[:5]) == ["date", "demand", "floortables", "demand_per_table", "year"]
        and d["date"].is_monotonic_increasing
        and d["date"].duplicated().sum() == 0
    )
    return ok, f"load_frames: {len(d):,} rows, {d.date.min().date()}..{d.date.max().date()}"


def check_normal_day_mask_excludes_holidays() -> tuple[bool, str]:
    d = L.load_frames()
    hol = L.holiday_date_set()
    m = L.normal_day_mask(d)
    leaked = d.loc[m & d["date"].isin(hol)]
    return leaked.empty, (f"{len(leaked)} holiday days marked normal" if len(leaked)
                          else "no holiday-window day is marked normal")


CHECKS = [
    check_alignment_cv_zero_when_equal,
    check_alignment_cv_known_spread,
    check_alignment_cv_nan_when_thin,
    check_feature_group_known_names,
    check_load_frames_shape,
    check_normal_day_mask_excludes_holidays,
]


def run_all() -> int:
    failed = 0
    for fn in CHECKS:
        try:
            ok, msg = fn()
        except Exception as e:                       # noqa: BLE001
            ok, msg = False, f"raised {type(e).__name__}: {e}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {fn.__name__}: {msg}")
        failed += 0 if ok else 1
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all())
