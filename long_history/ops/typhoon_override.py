"""typhoon_override.py — apply a manual demand adjustment to a saved forecast.

WHY THIS IS A MANUAL TOOL AND NOT A MODEL FEATURE
-------------------------------------------------
Across 2024-01-01..2026-05-28 there are nine T8+ days. Their realised impact,
measured against a same-weekday baseline, ranges from +10% to -87%:

    2024-11-13  TORAJI  T8   +10.1%      2025-09-07  TAPAH   T8   -13.3%
    2024-11-14  TORAJI  T8    +6.1%      2025-09-08  TAPAH   T8   -11.7%
    2024-09-05  YAGI    T8   -11.8%      2025-09-23  RAGASA  T8   -81.3%
    2024-09-06  YAGI    T8    -9.1%      2025-09-24  RAGASA  T10  -87.5%
    2025-07-20  WIPHA   T10  -22.5%

TORAJI (T8, 10.3 hours under signal) and RAGASA (T8, 9.7 hours) differ by 87
percentage points. Signal level does not separate them, and neither does
duration. What separates them is whether the casino SUSPENDED OPERATIONS —
RAGASA did, TORAJI did not. That is an operational decision, not a weather
variable, and nothing in the HKO feed predicts it.

With two closure days in the record, a learned coefficient would be fitting the
label of a single event. So `USE_WEATHER` stays False and the adjustment is
applied here, by a person who knows whether operations are being suspended.

USAGE
-----
    # See what the latest run predicts for the affected days
    uv run python typhoon_override.py --dates 2026-08-12 --show

    # Operations suspended for a day (the RAGASA case)
    uv run python typhoon_override.py --dates 2026-08-12 --closure

    # T8 expected, operations continuing (the YAGI/TAPAH case)
    uv run python typhoon_override.py --dates 2026-08-12 2026-08-13 --signal-t8

    # Your own multiplier
    uv run python typhoon_override.py --dates 2026-08-12 --factor 0.55

Writes predictions_adjusted.csv beside the run's predictions.csv and records
the override in metadata.json. The original predictions.csv is never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # long_history/
FORECASTS = ROOT / "output"

# Presets derived from the table above, as (low, mid, high) multipliers applied
# to the ORIGINAL p50. The band is set from the observed spread rather than
# scaled from the model's own interval: the model's P10-P90 reflects ordinary
# day-to-day variation, which is not the uncertainty that applies when a
# typhoon is inbound. Both presets are small-sample — the spread is the honest
# part of them.
PRESETS = {
    # RAGASA 2025-09-23/24 ran at 0.19x and 0.13x of a same-weekday baseline.
    # n = 2, so the band is widened beyond the observed pair on both sides.
    "closure": (0.08, 0.16, 0.28,
                "operations suspended (n=2: RAGASA 0.19x, 0.13x)"),
    # YAGI, TAPAH and TORAJI without closure: -11.8, -9.1, -13.3, -11.7,
    # +10.1, +6.1 -> mean -4.9%, spread straddles zero. n = 6.
    "t8": (0.85, 0.95, 1.11,
           "T8 signal, operations continuing (n=6, mean -4.9%, "
           "spread straddles zero)"),
}
# Relative band used for an explicit --factor, since no sample backs it.
MANUAL_BAND = 0.25


def latest_run() -> Path | None:
    runs = sorted(FORECASTS.glob("run_*/predictions.csv"))
    return runs[-1].parent if runs else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", required=True,
                    help="ISO date(s) to adjust, e.g. 2026-08-12 2026-08-13")
    ap.add_argument("--run-dir", default=None,
                    help="Run folder to adjust. Default: the most recent run.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--closure", action="store_true",
                   help=f"Preset {PRESETS['closure'][0]}x — {PRESETS['closure'][1]}")
    g.add_argument("--signal-t8", action="store_true",
                   help=f"Preset {PRESETS['t8'][0]}x — {PRESETS['t8'][1]}")
    g.add_argument("--factor", type=float, default=None,
                   help="Explicit multiplier applied to p10/p50/p90")
    ap.add_argument("--show", action="store_true",
                    help="Print the current forecast for those dates and exit")
    ap.add_argument("--note", default="", help="Free text recorded in metadata")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    if run_dir is None or not (run_dir / "predictions.csv").exists():
        print("No forecast run found. Run forecast.py first.", file=sys.stderr)
        return 1

    preds = pd.read_csv(run_dir / "predictions.csv", parse_dates=["date"])
    dates = [pd.Timestamp(d).normalize() for d in args.dates]
    hit = preds["date"].isin(dates)

    missing = [str(d.date()) for d in dates if d not in set(preds["date"])]
    if missing:
        print(f"  [warning] not in this forecast window: {', '.join(missing)}")
    if not hit.any():
        print(f"None of the requested dates are in {run_dir.name} "
              f"({preds.date.min().date()}..{preds.date.max().date()}).",
              file=sys.stderr)
        return 1

    print(f"\n=== {run_dir.name} ===")
    print(f"  forecast window: {preds.date.min().date()} .. {preds.date.max().date()}")
    print(f"\n  current forecast for the requested day(s):")
    for _, r in preds[hit].iterrows():
        print(f"    {r['date'].date()} ({r['date'].day_name()[:3]})  "
              f"p50={r['p50']:7.0f}   p10={r['p10']:7.0f}  p90={r['p90']:7.0f}")

    if args.show:
        print("\n  (--show: nothing written)")
        return 0

    if args.closure:
        lo, mid, hi, why = PRESETS["closure"]
    elif args.signal_t8:
        lo, mid, hi, why = PRESETS["t8"]
    elif args.factor is not None:
        f = args.factor
        lo, mid, hi = f * (1 - MANUAL_BAND), f, f * (1 + MANUAL_BAND)
        why = f"manual factor (band +/-{MANUAL_BAND:.0%}, no sample behind it)"
    else:
        print("\nPick one of --closure, --signal-t8 or --factor. "
              "Use --show to inspect first.", file=sys.stderr)
        return 1
    if min(lo, mid, hi) < 0:
        print("factor must be >= 0", file=sys.stderr)
        return 1

    # All three quantiles come off the ORIGINAL p50: on a typhoon day the
    # model's own interval describes the wrong kind of uncertainty.
    adj = preds.copy()
    base = preds.loc[hit, "p50"]
    adj.loc[hit, "p10"] = base * lo
    adj.loc[hit, "p50"] = base * mid
    adj.loc[hit, "p90"] = base * hi

    print(f"\n  applying x{mid:.2f} (P10 x{lo:.2f}, P90 x{hi:.2f}) — {why}")
    print(f"\n  adjusted:")
    for _, r in adj[hit].iterrows():
        print(f"    {r['date'].date()} ({r['date'].day_name()[:3]})  "
              f"p50={r['p50']:7.0f}   p10={r['p10']:7.0f}  p90={r['p90']:7.0f}")

    out = run_dir / "predictions_adjusted.csv"
    adj.to_csv(out, index=False)
    print(f"\n  -> {out}")
    print(f"     (predictions.csv left untouched)")

    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.setdefault("overrides", []).append({
            "dates": [str(d.date()) for d in dates],
            "factor": factor,
            "reason": why,
            "note": args.note,
            "applied_at": datetime.now().isoformat(timespec="seconds"),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  -> recorded in {meta_path.name}")

    print("\n  Remember: once the day has passed, append the ACTUAL demand to")
    print("  rawdata_long.csv as normal. If operations were suspended, also add the")
    print("  date to CLOSURE_DATES in engine/core.py so the shutdown does not")
    print("  drag the rolling features for the following four weeks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
