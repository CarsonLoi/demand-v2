"""split.py -- split a daily forecast into hourly demand using the
per-(holiday, day_offset) decisions from analyze.py and the most-recent
prior-90d local DOW baseline for non-holiday days.

Reads:
    long_history/output/run_<date>/predictions.csv
    long_history/output/hourly/decisions.csv
    long_history/output/hourly/holiday_profiles.csv
    long_history/data/raw/hourly_demand.csv

Writes:
    long_history/output/run_<date>/predictions_hourly.csv

Usage:
    uv run python long_history/hourly/split.py --run-date 2026-06-09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
import patterns as P  # noqa: E402


def _get_dow_baseline(date: pd.Timestamp, local_baselines: dict) -> tuple:
    """Return (24-vector, bucket_name) for this date's DOW bucket."""
    dow = date.weekday()
    for bucket, dows in P.DOW_BUCKETS.items():
        if dow in dows:
            return local_baselines.get(bucket), bucket
    return None, None


def split_day(date: pd.Timestamp, p10: float, p50: float, p90: float,
             decisions_lookup: dict, profile_lookup: dict,
             local_baselines: dict) -> tuple[list, str]:
    """Split one day's (p10, p50, p90) into 24 hourly rows.

    Returns (rows, strategy). rows is [] if no usable share vector was found
    (caller should warn and skip -- never silently emit garbage).
    """
    share_vec = None
    strategy = None

    holiday, offset = P.find_holiday_membership(date)
    if holiday is not None and (holiday, offset) in decisions_lookup:
        decision = decisions_lookup[(holiday, offset)]["decision"]
        if decision == "USE_HOLIDAY_PROFILE":
            share_vec = profile_lookup.get((holiday, offset))
            strategy = f"{holiday}_d{offset:+d}_holiday_profile"
        elif decision.startswith("USE_"):
            bucket = decision.replace("USE_", "").lower()
            share_vec = local_baselines.get(bucket)
            strategy = f"{holiday}_d{offset:+d}_use_{bucket}"

    if share_vec is None:
        share_vec, bucket = _get_dow_baseline(date, local_baselines)
        strategy = f"dow_{bucket}" if bucket else "missing"

    if share_vec is None or np.isnan(share_vec).any():
        return [], strategy

    rows = []
    for h in range(24):
        s = float(share_vec[h])
        rows.append({"date": date.date().isoformat(), "hour": h,
                     "p10": round(p10 * s, 4), "p50": round(p50 * s, 4),
                     "p90": round(p90 * s, 4), "strategy": strategy})
    return rows, strategy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-date", type=str, required=True,
                        help="ISO date matching output/run_<YYYYMMDD>/")
    args = parser.parse_args()

    run_date = pd.Timestamp(args.run_date).normalize()
    stamp = run_date.strftime("%Y%m%d")
    run_dir = ROOT / C.OUT_DIR / f"run_{stamp}"
    preds_path = run_dir / "predictions.csv"
    if not preds_path.exists():
        print(f"ERROR: {preds_path} not found. Run training/forecast.py first.")
        return 1

    preds = pd.read_csv(preds_path, parse_dates=["date"])
    print(f"  loaded daily forecast: {len(preds)} day(s) from {preds_path}")

    decisions_csv = P.DERIVED / "decisions.csv"
    profiles_csv = P.DERIVED / "holiday_profiles.csv"
    if not decisions_csv.exists() or not profiles_csv.exists():
        print(f"ERROR: missing {decisions_csv} or {profiles_csv}.")
        print("       Run hourly/analyze.py first.")
        return 1

    decisions = pd.read_csv(decisions_csv)
    decisions_lookup = decisions.set_index(["holiday", "day_offset"]).to_dict("index")
    print(f"  loaded {len(decisions)} decision row(s) from {decisions_csv.name}")

    profiles = pd.read_csv(profiles_csv)
    profile_lookup = {}
    for (hol, off), g in profiles.groupby(["holiday", "day_offset"]):
        profile_lookup[(hol, int(off))] = g.sort_values("hour")["share"].to_numpy()
    print(f"  loaded {len(profile_lookup)} holiday-profile cell(s)")

    try:
        df_hourly = P.load_hourly()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    date_to_hours = P.date_to_hourly_array(df_hourly)
    holiday_dates = P.get_holiday_window_dates()
    local_baselines = P.compute_local_dow_baselines(
        run_date + pd.Timedelta(days=1), date_to_hours, holiday_dates)

    missing = [b for b, v in local_baselines.items() if v is None]
    if missing:
        print(f"  [WARN] insufficient prior-90d samples for DOW bucket(s): {missing}")
    print(f"  computed local DOW baselines for {run_date.date()} reference")

    out_rows = []
    strategy_count = {}
    for _, row in preds.iterrows():
        rows, strategy = split_day(row["date"], float(row["p10"]), float(row["p50"]),
                                   float(row["p90"]), decisions_lookup, profile_lookup,
                                   local_baselines)
        if not rows:
            print(f"  [WARN] {row['date'].date()}: no share vector ({strategy}); skipping")
            continue
        strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
        out_rows.extend(rows)

    out_df = pd.DataFrame(out_rows)
    out_path = run_dir / "predictions_hourly.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  -> {out_path}: {len(out_df):,} row(s)")

    print("\n  Strategy summary (days assigned to each pattern):")
    for strat, n in sorted(strategy_count.items(), key=lambda kv: -kv[1]):
        print(f"    {strat:40s} {n:3d}")

    if not out_df.empty:
        check = (out_df.groupby("date")["p50"].sum().reset_index()
                 .merge(preds.assign(date=preds.date.dt.date.astype(str)),
                        on="date", suffixes=("_hourly_sum", "_daily")))
        diffs = (check["p50_hourly_sum"] - check["p50_daily"]).abs()
        bad = check[diffs > 1.0]
        if len(bad):
            print(f"\n  [WARN] {len(bad)} day(s) where hourly p50 sum != daily p50 "
                  f"(beyond rounding). Investigate.")

    print(f"\n=== Done -- open {out_path} ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
