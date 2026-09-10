"""forecast_from_clean.py -- run the REAL long_history forecast against a
demand frame with blank/NaT rows dropped.

WHY THIS EXISTS
---------------
rawdata_long.csv currently ends with a trailing blank line, which pandas
parses as a row with NaT date and NaN demand. That makes
long_history/data/validate.py FAIL (correctly), so run_pipeline.py stops
before the forecast step and you keep looking at a stale predictions.csv.
It also injects 28 junk NaT-target rows into the feature matrix.

This calls the SAME training/forecast.py:forecast() the pipeline calls --
same params, same CNY anchor, same config -- only with those rows removed.

Once you close the CSV in Excel and delete the trailing "," line, this file
is redundant: use run_pipeline.py instead.

Usage:
    uv run python research/forecast_from_clean.py --run-date 2026-09-09
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "long_history"))
sys.path.insert(0, str(ROOT / "long_history" / "training"))

import config as C                    # noqa: E402
from engine import core as S          # noqa: E402
from engine import features as F      # noqa: E402
import forecast as FC                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-date", default=None,
                    help="ISO date; default = last clean row in the data")
    ap.add_argument("--window", nargs=2, default=["2026-09-20", "2026-09-28"],
                    help="date range to summarise vs the recent mean")
    a = ap.parse_args()

    demand = F.load_long_demand()
    n0 = len(demand)
    demand = demand.dropna(subset=["date", "demand"]).reset_index(drop=True)
    dropped = n0 - len(demand)
    print(f"dropped {dropped} blank/NaT row(s)")
    print(f"data: {demand.date.min().date()} .. {demand.date.max().date()} "
          f"({len(demand):,} days)")
    print(f"EXCLUDE_FROM_TRAINING: {C.EXCLUDE_FROM_TRAINING}")
    print(f"GUARD_LAGS_CROSSING_EXCLUDED = {C.GUARD_LAGS_CROSSING_EXCLUDED}")
    print(f"comparable alignment in use  = "
          f"{'comparable' in S._holiday_alignment_map.__code__.co_varnames}")

    run_date = pd.Timestamp(a.run_date) if a.run_date else demand.date.max()
    preds = FC.forecast(demand, run_date, C.DEFAULT_HALF_LIFE, anchor=True, quiet=True)
    if preds.empty:
        print("no predictions produced")
        return 1
    preds["horizon"] = [(d - run_date).days for d in preds["date"]]

    out = ROOT / "research" / f"fc_{run_date:%Y%m%d}.csv"
    preds.to_csv(out, index=False)

    print(f"\n28-day forecast from {run_date.date()}:")
    print(preds[["date", "horizon", "p10", "p50", "p90"]].to_string(
        index=False, formatters={"p10": "{:,.0f}".format,
                                 "p50": "{:,.0f}".format,
                                 "p90": "{:,.0f}".format}))

    with F.extended_holidays():
        hol = S.holiday_date_set()
    recent = demand[demand.date > run_date - pd.Timedelta(days=14)]
    base = float(recent.demand.mean())
    lo, hi = pd.Timestamp(a.window[0]), pd.Timestamp(a.window[1])
    print(f"\nrecent 14-day mean ({recent.date.min().date()}..{run_date.date()}): {base:,.0f}")
    for _, r in preds.iterrows():
        d = r["date"]
        if lo <= d <= hi:
            print(f"  {d.date()} {d.strftime('%a')}  p50={r['p50']:>7,.0f}   "
                  f"{100*r['p50']/base-100:+6.1f}% vs recent mean"
                  f"   {'HOLIDAY' if d in hol else ''}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
