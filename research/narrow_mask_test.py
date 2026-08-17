"""narrow_mask_test.py — is a DEMAND-DRIVEN yearly-lag mask better than the
current all-or-nothing calendar-window mask?

STANDALONE / NON-DESTRUCTIVE. Imports read-only from v2/_shared.py and
monkeypatches the guard IN MEMORY ONLY for the duration of this process.
Nothing on disk in v2/ or long_history/ is modified.

THE PROBLEM
-----------
lag_365 / lag_anchor_365 / lag_728 / yoy_ratio all reach back a fixed number
of days. Moving holidays drift up to ~3 weeks year to year, so an ordinary
day this year can land on a holiday-collapsed day last year. Measured case:
forecasting from 2026-01-28, yoy_ratio compared an ordinary Tuesday against
2 days pre-CNY-2025 (51% of normal) -> one horizon model missed by 27.7%.

v2/_shared.guard_yearly_lags() fixes this by masking whenever the SOURCE date
falls inside a moving holiday's calendar window. For CNY that window is
(-7, +10) = 18 days, applied identically every year. Measured on a fresh
2026 YTD weekly backtest: pooled 5.08% -> 4.70%, BUT 10 of 21 origins got
WORSE. It is off in production for exactly that reason.

WHY IT OVER-MASKS (real numbers, CNY 2025 = Jan 29)
    2025-01-27   3,619   50% of normal   <- genuinely contaminated
    2025-01-28   3,331   46% of normal   <- genuinely contaminated
    2025-01-31   7,252   99% of normal   <- masked anyway, signal thrown away
    2025-02-05   7,150   98% of normal   <- masked anyway, signal thrown away

THE NARROWER IDEA
-----------------
Stop asking "is this date inside a holiday window?" (calendar) and start
asking "was demand on this date actually abnormal?" (data). A source day only
contaminates a yearly lookback if its demand was far from normal for that
weekday and season. Days at 98-99% of normal are fine to use even if the
calendar says CNY.

    ratio[d] = demand[d] / clean_same_weekday_baseline_around(d)
    contaminated  <=>  d is in a moving-holiday window AND ratio outside [LO, HI]

Restricting to moving-holiday windows is deliberate: it keeps the mask aimed
at the drift problem and stops it from eating ordinary volatility, typhoon
days, etc., which other machinery already handles.

LEAKAGE
-------
Safe by construction, and this was checked rather than assumed. The mask
decision for a source date d reads demand at d and in a +/-(21..56) day ring
around d, excluding holiday days. For a prediction row at target T, d = T-365,
so the newest value the mask can touch is T-309. Since T <= origin+28, that is
at worst origin-281 -- nearly a year before the forecast origin. No future
data participates in the mask decision, at any horizon.

PROTOCOL
--------
Identical to the guard re-test already run, so numbers are directly
comparable: evaluate.rolling_backtest over 2026 YTD, n=6 (7-day lead),
weekly cadence, use_cache=False (the cache key does NOT encode the guard
setting, so caching here would silently mix variants).

Verdict uses this project's standing rule: a variant is only adopted if it
beats the OFF baseline pooled AND does not regress any individual origin.

Usage:
    uv run python research/narrow_mask_test.py
    uv run python research/narrow_mask_test.py --thresholds 0.85,0.70
"""
from __future__ import annotations
import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v2"))

import _shared as S  # noqa: E402

OUT_DIR = ROOT / "research" / "narrow_mask_output"

START, END = "2026-01-01", "2026-05-28"
N = 6  # lead = 7 days

CNY_2026 = pd.Timestamp("2026-02-17")
CNY_WINDOW = {CNY_2026 + pd.Timedelta(days=k) for k in range(-7, 11)}


# ── the demand-driven abnormality measure ───────────────────────────────────
def build_ratio_map(demand: pd.DataFrame) -> dict:
    """date -> demand / (clean same-weekday baseline in a +/-(21..56) day ring).

    Holiday-window days are excluded from the baseline so the yardstick stays
    'normal trade', which matters most exactly when the centre day is not.
    """
    dm = dict(zip(demand["date"], demand["demand"].astype(float)))
    hol = S.holiday_date_set()
    ratio = {}
    for d, v in dm.items():
        if v is None or np.isnan(v) or v <= 0:
            continue
        vals = []
        for sign in (-1, 1):
            for k in range(21, 57, 7):  # same weekday, 3..8 weeks out
                t = d + pd.Timedelta(days=sign * k)
                if t in hol:
                    continue
                x = dm.get(t)
                if x is not None and not np.isnan(x) and x > 0:
                    vals.append(x)
        if len(vals) >= 4:
            ratio[d] = v / float(np.mean(vals))
    return ratio


def contaminated_dates(demand: pd.DataFrame, lo: float, hi: float) -> set:
    """Moving-holiday-window days whose demand was actually abnormal."""
    ratio = build_ratio_map(demand)
    moving = S.moving_holiday_date_set()
    return {d for d in moving if d in ratio and not (lo <= ratio[d] <= hi)}


def make_narrow_guard(bad: set):
    """Return a drop-in replacement for _shared.guard_yearly_lags that masks
    against `bad` (data-driven) instead of the full calendar window."""

    def narrow_guard(df: pd.DataFrame) -> pd.DataFrame:
        T = pd.to_datetime(df["target_date"])
        H = df["horizon"].astype(int)
        min_safe = H + 1

        v_recent_date = T - pd.to_timedelta(min_safe, unit="D")
        v_yoy_date = v_recent_date - pd.Timedelta(days=365)
        df.loc[v_recent_date.isin(bad) | v_yoy_date.isin(bad), "yoy_ratio"] = np.nan

        t365 = T - pd.Timedelta(days=365)
        bad_365 = t365.isin(bad)
        if "lag_365" in df.columns:
            df.loc[bad_365, "lag_365"] = np.nan
        if "lag_anchor_365" in df.columns:
            df.loc[bad_365, "lag_anchor_365"] = np.nan
        if "lag_728" in df.columns:
            df.loc[(T - pd.Timedelta(days=728)).isin(bad), "lag_728"] = np.nan
        return df

    return narrow_guard


# ── evaluation ──────────────────────────────────────────────────────────────
def mape(d):
    d = d[(d.actual > 0) & d.forecast.notna()]
    return float(np.mean(np.abs(d.actual - d.forecast) / d.actual)) * 100 if len(d) else np.nan


def run_variant(tag: str, guard_on: bool, guard_fn=None) -> pd.DataFrame:
    S.GUARD_YEARLY_LAGS = guard_on
    if guard_fn is not None:
        S.guard_yearly_lags = guard_fn
    else:
        S.guard_yearly_lags = _ORIGINAL_GUARD

    import evaluate
    importlib.reload(evaluate)

    t0 = time.time()
    # use_cache=False is REQUIRED: evaluate.py's cache key does not encode the
    # guard setting, so caching would silently serve another variant's result.
    df = evaluate.rolling_backtest(START, END, n=N, mode="rolling_window",
                                   use_cache=False, verbose=False)
    df["is_cny"] = df["target_date"].isin(CNY_WINDOW)
    df["is_jan"] = (df["target_date"] >= "2026-01-15") & (df["target_date"] <= "2026-01-31")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{tag}.csv", index=False)
    print(f"  [{tag}] {len(df)} rows in {time.time()-t0:.0f}s", flush=True)
    return df


_ORIGINAL_GUARD = S.guard_yearly_lags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", default="0.85,0.70",
                    help="comma-separated LO values; HI = 2-LO (symmetric)")
    a = ap.parse_args()
    los = [float(x) for x in a.thresholds.split(",")]

    print("=" * 78)
    print(f"NARROW MASK TEST -- {START}..{END}, n={N}, weekly rolling origins")
    print("=" * 78)

    demand = S.load_demand()

    # Diagnostic: how much does each rule actually mask?
    moving = S.moving_holiday_date_set()
    in_data = {d for d in moving if d <= demand.date.max()}
    print(f"\nmoving-holiday window days present in data: {len(in_data)}")
    ratio = build_ratio_map(demand)
    for lo in los:
        hi = 2 - lo
        bad = {d for d in in_data if d in ratio and not (lo <= ratio[d] <= hi)}
        print(f"  narrow [{lo:.2f},{hi:.2f}] masks {len(bad):3d} "
              f"({100*len(bad)/max(1,len(in_data)):.0f}% of the calendar window)")

    print("\nCNY-2025 window, day by day (what each rule would mask):")
    for k in range(-7, 11):
        d = pd.Timestamp("2025-01-29") + pd.Timedelta(days=k)
        r = ratio.get(d, np.nan)
        marks = " ".join(f"{'M' if not (lo <= r <= 2-lo) else '.'}" for lo in los) \
            if not np.isnan(r) else "?"
        print(f"  {d.date()}  offset {k:+3d}  ratio {r:5.2f}   full=M  narrow={marks}")

    variants = [("guard_off", False, None), ("guard_full", True, None)]
    for lo in los:
        hi = 2 - lo
        bad = contaminated_dates(demand, lo, hi)
        variants.append((f"narrow_{lo:.2f}", True, make_narrow_guard(bad)))

    print("\nrunning backtests (each ~10-16 min)...", flush=True)
    results = {}
    for tag, on, fn in variants:
        results[tag] = run_variant(tag, on, fn)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    base = results["guard_off"]
    for tag, df in results.items():
        print(f"\n{tag}")
        print(f"  overall  {mape(df):6.2f}%   CNY {mape(df[df.is_cny]):6.2f}%   "
              f"non-CNY {mape(df[~df.is_cny]):6.2f}%   Jan15-31 {mape(df[df.is_jan]):6.2f}%")

    print("\n" + "=" * 78)
    print("GATE (vs guard_off baseline): adopt only if pooled better AND no origin worse")
    print("=" * 78)
    origins = sorted(base.origin.unique())
    for tag, df in results.items():
        if tag == "guard_off":
            continue
        worse = []
        for o in origins:
            a_ = mape(base[base.origin == o])
            b_ = mape(df[df.origin == o])
            if b_ > a_ + 1e-9:
                worse.append((pd.Timestamp(o).date(), a_, b_))
        verdict = "PASS" if not worse else f"FAIL ({len(worse)}/{len(origins)} origins worse)"
        print(f"  {tag:14s} pooled {mape(df):5.2f}% "
              f"({mape(df)-mape(base):+.2f}pp)   {verdict}")

    print(f"\n-> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
