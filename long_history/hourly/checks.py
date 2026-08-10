"""checks.py -- assert-based sanity checks for the hourly module.

Matches long_history/data/validate.py's PASS/WARN/FAIL style. There is no
pytest anywhere in this repo; this file grows across tasks as each new
module lands, and is the closest thing to a test suite this package has.

Usage:
    uv run python long_history/hourly/checks.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import patterns as P  # noqa: E402
import split as SPLIT  # noqa: E402


def check_share_vector() -> tuple[bool, str]:
    v = P.share_vector([10, 20, 30, 40])
    ok = abs(v.sum() - 1.0) < 1e-9 and np.allclose(v, [0.1, 0.2, 0.3, 0.4])
    return ok, f"share_vector([10,20,30,40]) = {v.tolist()}"


def check_share_vector_zero_total() -> tuple[bool, str]:
    v = P.share_vector(np.zeros(24))
    ok = np.isnan(v).all()
    return ok, "share_vector(all-zero) returns all-NaN (a zero day cannot be split)"


def check_tvd_bounds() -> tuple[bool, str]:
    same = P.tvd([0.5, 0.5], [0.5, 0.5])
    disjoint = P.tvd([1.0, 0.0], [0.0, 1.0])
    ok = abs(same - 0.0) < 1e-9 and abs(disjoint - 1.0) < 1e-9
    return ok, f"tvd(identical)={same:.4f} (want 0)  tvd(disjoint)={disjoint:.4f} (want 1)"


def check_local_dow_baseline_no_leakage() -> tuple[bool, str]:
    """The baseline for `reference_date` must never read reference_date or
    anything on/after it -- that would be using the future to predict itself."""
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="D")
    date_to_hours = {}
    for d in dates:
        # sentinel: hour 0 share == 999 only on/after a poison date, so if the
        # baseline ever includes a poisoned day, its hour-0 share is huge.
        arr = np.full(24, 1.0)
        if d >= pd.Timestamp("2024-05-01"):
            arr[0] = 999.0
        date_to_hours[d] = arr

    ref = pd.Timestamp("2024-05-01")  # the day the poison starts
    baselines = P.compute_local_dow_baselines(ref, date_to_hours, holiday_dates=set())
    bad = [b for b, v in baselines.items() if v is not None and v[0] > 10]
    ok = not bad
    return ok, (f"poisoned buckets leaked into baseline: {bad}" if bad
               else "no poisoned (on/after reference_date) day reached the baseline")


def check_split_conservation() -> tuple[bool, str]:
    """The 24 hourly p50s produced for a day must sum back to that day's
    daily p50 -- a split that doesn't conserve the total is silently wrong."""
    share_vec = np.array([1 / 24] * 24)  # a flat day, trivial to check by hand
    decisions_lookup, profile_lookup = {}, {}
    local_baselines = {"weekday": share_vec, "friday": share_vec,
                       "saturday": share_vec, "sunday": share_vec}
    rows, strategy = SPLIT.split_day(pd.Timestamp("2024-01-03"), 100.0, 200.0, 300.0,
                                     decisions_lookup, profile_lookup, local_baselines)
    total_p50 = sum(r["p50"] for r in rows)
    ok = abs(total_p50 - 200.0) < 0.01
    return ok, f"24-hour p50 sum = {total_p50:.2f} (want 200.00), strategy={strategy}"


def check_split_no_future_leakage() -> tuple[bool, str]:
    """split.py must build its local DOW baseline using run_date+1 as the
    reference -- i.e. it may see run_date, but nothing after it."""
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="D")
    date_to_hours = {}
    for d in dates:
        arr = np.full(24, 1.0)
        if d > pd.Timestamp("2024-05-01"):   # strictly AFTER the run date
            arr[0] = 999.0
        date_to_hours[d] = arr
    run_date = pd.Timestamp("2024-05-01")
    baselines = P.compute_local_dow_baselines(run_date + pd.Timedelta(days=1),
                                              date_to_hours, holiday_dates=set())
    bad = [b for b, v in baselines.items() if v is not None and v[0] > 10]
    ok = not bad
    return ok, (f"future-dated buckets leaked into the split baseline: {bad}" if bad
               else "no day after run_date reached the split's baseline")


def run_all() -> int:
    checks = [check_share_vector, check_share_vector_zero_total,
              check_tvd_bounds, check_local_dow_baseline_no_leakage,
              check_split_conservation, check_split_no_future_leakage]
    failed = 0
    for fn in checks:
        ok, msg = fn()
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {fn.__name__}: {msg}")
        failed += 0 if ok else 1
    print(f"\n{len(checks)-failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all())
