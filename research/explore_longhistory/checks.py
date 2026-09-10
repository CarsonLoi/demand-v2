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


def check_describe_series_gaps_zero() -> tuple[bool, str]:
    d = L.load_frames()
    s = L.describe_series(d, "demand")
    return s["gaps"] == 0, f"gaps in daily series = {s['gaps']} (want 0)"


def check_closure_days_match_known() -> tuple[bool, str]:
    d = L.load_frames()
    cl = set(pd.to_datetime(L.closure_days(d)["date"]).dt.date)
    known = {pd.Timestamp("2018-09-16").date()}
    known |= {x.date() for x in pd.date_range("2020-02-05", "2020-02-19")}
    known |= {x.date() for x in pd.date_range("2022-07-11", "2022-07-22")}
    missing = known - cl
    return not missing, (f"known closures not detected: {sorted(missing)[:3]}..." if missing
                         else f"{len(cl)} closure/zero days, all known closures present")


def check_describe_by_year_all_years_present() -> tuple[bool, str]:
    d = L.load_frames()
    t = L.describe_by(d, "year", ["demand_per_table"])
    yrs = set(t.index.astype(int))
    return {2015, 2020, 2026}.issubset(yrs), f"years in table: {sorted(yrs)}"


def check_describe_by_dow_seven_rows() -> tuple[bool, str]:
    d = L.load_frames()
    t = L.describe_by(d, "dow", ["demand"])
    return len(t) == 7, f"dow table has {len(t)} rows (want 7)"


def check_monthly_index_flat_series_is_one() -> tuple[bool, str]:
    dates = pd.date_range("2015-01-01", "2016-12-31")
    df = pd.DataFrame({"date": dates, "demand": 100.0, "floortables": 10.0})
    df["demand_per_table"] = df["demand"] / df["floortables"].clip(lower=1)
    df["year"] = df["date"].dt.year
    t = L.monthly_index_by_year(df)
    return np.allclose(t.values, 1.0), \
        f"flat series -> index range [{t.values.min():.3f}, {t.values.max():.3f}] (want all 1.0)"


def check_monthly_index_doubled_month() -> tuple[bool, str]:
    dates = pd.date_range("2015-01-01", "2015-12-31")
    df = pd.DataFrame({"date": dates, "demand": 100.0, "floortables": 10.0})
    df.loc[df["date"].dt.month == 6, "demand"] = 200.0
    df["demand_per_table"] = df["demand"] / df["floortables"].clip(lower=1)
    df["year"] = df["date"].dt.year
    t = L.monthly_index_by_year(df)
    want_june = 20.0 / (df["demand_per_table"].mean())
    return abs(t.loc[2015, 6] - want_june) < 1e-6, \
        f"June index = {t.loc[2015,6]:.4f} (want {want_june:.4f})"


def check_monthly_no_covid_years() -> tuple[bool, str]:
    t = L.monthly_index_by_year(L.load_frames())
    bad = L.COVID_YEARS & set(t.index)
    return not bad, f"covid years in monthly table: {bad or 'none'}"


def _synth_holiday_frame():
    """A flat 100/day series with a known holiday window and a known dip
    just before it, so the baseline and multiplier are hand-checkable."""
    dates = pd.date_range("2015-01-01", "2015-12-31")
    df = pd.DataFrame({"date": dates, "demand": 100.0, "floortables": 10.0})
    win = pd.date_range("2015-06-10", "2015-06-16")
    df.loc[df["date"].isin(win), "demand"] = 200.0
    # POISON: inside the window and after it -- neither may enter the baseline
    df.loc[df["date"] == pd.Timestamp("2015-07-01"), "demand"] = 9999.0
    df["demand_per_table"] = df["demand"] / df["floortables"].clip(lower=1)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.weekday
    return df, list(win)


def check_pre_holiday_baseline_is_leadin_only() -> tuple[bool, str]:
    df, win = _synth_holiday_frame()
    orig = L.holiday_date_set
    L.holiday_date_set = lambda: set(win)                     # type: ignore
    try:
        b = L.pre_holiday_baseline(df, win)
    finally:
        L.holiday_date_set = orig                             # type: ignore
    ok = abs(b["demand"] - 100.0) < 1e-9 and b["n_days"] >= 5
    return ok, f"baseline demand = {b['demand']} (want 100.0 -- no window/after day leaked), n={b['n_days']}"


def check_pre_holiday_baseline_widens_and_nans() -> tuple[bool, str]:
    dates = pd.date_range("2015-05-01", "2015-06-16")
    df = pd.DataFrame({"date": dates, "demand": 0.0, "floortables": 10.0})
    win = list(pd.date_range("2015-06-10", "2015-06-16"))
    df["demand_per_table"] = 0.0
    orig = L.holiday_date_set
    L.holiday_date_set = lambda: set(win)                     # type: ignore
    try:
        b = L.pre_holiday_baseline(df, win)
    finally:
        L.holiday_date_set = orig                             # type: ignore
    return np.isnan(b["demand"]) and b["n_days"] < 5, \
        f"all-closure lead-in -> demand={b['demand']}, n={b['n_days']} (want nan, <5)"


def check_holiday_multiplier_no_covid() -> tuple[bool, str]:
    t = L.holiday_multiplier_by_year(L.load_frames())
    bad = L.COVID_YEARS & set(t["year"].astype(int))
    return not bad, f"covid years in holiday multiplier table: {bad or 'none'}"


def check_holiday_alignment_verdict_columns() -> tuple[bool, str]:
    t = L.holiday_alignment(L.holiday_multiplier_by_year(L.load_frames()))
    need = {"n_clean_years", "cv_per_table", "cv_raw", "verdict"}
    return need.issubset(t.columns), f"holiday_alignment columns: {list(t.columns)}"


def check_cny_trough_below_one() -> tuple[bool, str]:
    t = L.cny_trough_by_year(L.load_frames())
    vals = t["min_mult_value"].dropna()
    return (vals < 1.0).all() and len(vals) >= 3, \
        f"pre-CNY min multipliers: {vals.round(2).to_dict()}"


def check_covid_timeline_spans_2019_2024() -> tuple[bool, str]:
    t = L.covid_timeline(L.load_frames())
    lo, hi = str(t.index.min()), str(t.index.max())
    return lo == "2019-01" and hi == "2024-12", f"covid timeline span {lo}..{hi}"


def check_recovery_2023_climbed() -> tuple[bool, str]:
    r = L.recovery_2023_check(L.load_frames())
    return r["climbed"] and r["jan_per_table"] < r["dec_per_table"], \
        f"2023 per-table jan={r['jan_per_table']:.1f} dec={r['dec_per_table']:.1f} climbed={r['climbed']}"


CHECKS = [
    check_alignment_cv_zero_when_equal,
    check_alignment_cv_known_spread,
    check_alignment_cv_nan_when_thin,
    check_feature_group_known_names,
    check_load_frames_shape,
    check_normal_day_mask_excludes_holidays,
    check_describe_series_gaps_zero,
    check_closure_days_match_known,
    check_describe_by_year_all_years_present,
    check_describe_by_dow_seven_rows,
    check_monthly_index_flat_series_is_one,
    check_monthly_index_doubled_month,
    check_monthly_no_covid_years,
    check_pre_holiday_baseline_is_leadin_only,
    check_pre_holiday_baseline_widens_and_nans,
    check_holiday_multiplier_no_covid,
    check_holiday_alignment_verdict_columns,
    check_cny_trough_below_one,
    check_covid_timeline_spans_2019_2024,
    check_recovery_2023_climbed,
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
