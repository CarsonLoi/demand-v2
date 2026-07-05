"""run_schedule.py — run a day-by-day forecasting cycle in one command.

Given an initial scheduling window (FROM TO) and a set of update run-dates, this
produces the frozen initial plan plus each re-forecast, all from the CURRENT
data/raw/rawdata.csv.

You do NOT need to trim rawdata.csv between runs: forecast.py's leakage guard
trains each run only on data <= its run-date. So keep every actual you have in
rawdata.csv, and this script replays the cycle by varying the run-date. Each run
still trains honestly (no future data leaks in).

After all runs it builds forecasts/tracking_comparison.png — actual demand vs
the frozen initial plan vs the latest forecast, focused on the initial window.

Examples
--------
    # Initial plan Jun 8-Jul 5 (run-date Jun 7), then re-forecast on Jun 14 & 19:
    uv run python run_schedule.py --initial 2026-06-08 2026-07-05 \
        --updates 2026-06-14 2026-06-19 --full --blend selection

    # Same, but re-forecast every 7 days from the start until the window ends:
    uv run python run_schedule.py --initial 2026-06-08 2026-07-05 \
        --update-every 7 --full --blend selection

    # See the planned runs without training anything:
    uv run python run_schedule.py --initial 2026-06-08 2026-07-05 \
        --updates 2026-06-14 2026-06-19 --dry-run
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import forecast as F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--initial", nargs=2, metavar=("FROM", "TO"), required=True,
                    help="Scheduling window for the frozen initial plan (ISO dates). "
                         "Initial run-date is FROM-1.")
    ap.add_argument("--updates", nargs="*", default=[], metavar="RUN_DATE",
                    help="Explicit run-dates for re-forecasts, e.g. 2026-06-14 2026-06-19.")
    ap.add_argument("--update-every", type=int, default=None, metavar="N",
                    help="Alternative to --updates: re-forecast every N days from the "
                         "initial run-date up to the window's TO date.")
    ap.add_argument("--full", action="store_true", help="Use the 6-model hybrid.")
    ap.add_argument("--blend", choices=["equal", "selection"], default="equal",
                    help="Hybrid blend method (implies --full when 'selection').")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned runs and exit without forecasting.")
    args = ap.parse_args()

    if args.blend == "selection":
        args.full = True

    init_from = pd.Timestamp(args.initial[0]).normalize()
    init_to = pd.Timestamp(args.initial[1]).normalize()
    if init_from > init_to:
        ap.error("--initial FROM must be on or before TO")
    init_run_date = init_from - pd.Timedelta(days=1)

    # Build the ordered run plan: (run_date, initial_range_or_None)
    runs = [(init_run_date, (init_from, init_to))]

    update_dates = [pd.Timestamp(u).normalize() for u in args.updates]
    if args.update_every:
        d = init_run_date + pd.Timedelta(days=args.update_every)
        while d <= init_to:
            update_dates.append(d)
            d += pd.Timedelta(days=args.update_every)
    for d in sorted(set(update_dates)):
        if d > init_run_date:
            runs.append((d, None))

    # Validate against available data
    demand = F.load_demand()
    last_data = demand.date.max()
    print("=== Forecast schedule ===")
    print(f"  rawdata.csv covers ... {last_data.date()}")
    print(f"  mode: {'hybrid-'+args.blend if args.full else 'lgbm_l2'}")
    print(f"  planned runs:")
    for run_date, irange in runs:
        tag = f"INITIAL plan {irange[0].date()}..{irange[1].date()}" if irange else "update"
        warn = "  <-- WARNING: run-date is beyond rawdata.csv" if run_date > last_data else ""
        print(f"    run-date {run_date.date()}  ({tag}){warn}")

    if any(rd > last_data for rd, _ in runs):
        print("\n  [error] one or more run-dates are beyond the data. Append actuals "
              "to rawdata.csv first.")
        return 1

    if args.dry_run:
        print("\n  (dry-run — nothing executed)")
        return 0

    latest_demand = demand
    for run_date, irange in runs:
        tag = "INITIAL" if irange else "UPDATE"
        print(f"\n{'#'*60}\n### {tag} run — run-date {run_date.date()}\n{'#'*60}")
        preds, latest_demand = F.make_forecast(run_date, use_hybrid=args.full, blend=args.blend)
        F.save_outputs(preds, run_date, latest_demand, use_hybrid=args.full,
                       blend=args.blend, initial_range=irange)

    print(f"\n{'#'*60}\n### Tracking + comparison charts\n{'#'*60}")
    F.make_tracking_chart(latest_demand)
    F.make_comparison_chart(runs[-1][0], latest_demand)
    print("\n=== Schedule complete ===")
    print("  -> forecasts/tracking_comparison.png  (actual vs initial plan vs latest)")
    print("  -> forecasts/tracking_comparison.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
