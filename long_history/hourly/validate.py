"""validate.py -- honest, leakage-free validation of the hourly-splitting
approach: does the empirical distribution method actually beat, or at least
match, a lightweight comparison model?

METHOD
------
Pick a holdout window (the last `--holdout-days` days with full 24-hour
data). Everything before the holdout trains the comparison model ONCE (not
refit per day -- see comparison_model.py's docstring for why). The empirical
method needs no training, so for EACH day inside the holdout it is
recomputed walk-forward, using only data strictly before that day:
    - the local DOW baseline (compute_local_dow_baselines, already leakage-
      safe by construction)
    - the holiday decision cascade (analyze.analyze_one_cell(..., as_of=
      day-1), which now supports an as_of cutoff for exactly this purpose)

Both methods' predicted share-vectors are applied to that day's ACTUAL daily
total (not a forecast total) -- this isolates hourly-split error from
daily-forecast error, which is a separately measured concern.

STATUS
------
Run today against long_history/hourly/sample_hourly_data.csv, this is a
MECHANISM CHECK ONLY -- 10 days, zero holiday coverage, no statistical
power. The numbers it prints are NOT evidence that the empirical method is
"good enough." Real validation requires long_history/data/raw/hourly_demand.csv
with real history (18-24+ months, multiple holiday occurrences per cell) --
this script does not change when that data lands; you just point it there
and the numbers become real.

Usage:
    uv run python long_history/hourly/validate.py
    uv run python long_history/hourly/validate.py --holdout-days 90
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

import patterns as P  # noqa: E402
import comparison_model as CM  # noqa: E402
from analyze import analyze_one_cell  # noqa: E402


def classify_day_type(is_holiday: bool, decision: str | None, dow: int) -> str:
    if not is_holiday:
        return "ordinary_weekday" if dow < 4 else "friday_saturday_sunday"
    if decision == "USE_HOLIDAY_PROFILE":
        return "holiday_specific_profile"
    return "holiday_inherited_dow"


def empirical_prediction(date: pd.Timestamp, date_to_hours: dict,
                         holiday_dates: set) -> tuple:
    """Walk-forward empirical-method prediction for `date`, using only data
    strictly before it. Returns (24-share-vector or None, day_type)."""
    local = P.compute_local_dow_baselines(date, date_to_hours, holiday_dates)
    hol_name, hol_offset = P.find_holiday_membership(date)

    if hol_name is None:
        bucket = "weekday" if date.weekday() < 4 else \
                 ["friday", "saturday", "sunday"][date.weekday() - 4]
        return local.get(bucket), classify_day_type(False, None, date.weekday())

    result = analyze_one_cell(hol_name, hol_offset, date_to_hours, holiday_dates,
                              as_of=date - pd.Timedelta(days=1))
    if result is None:
        # not enough PRIOR occurrences yet to decide -- fall through to DOW,
        # same fallback split.py uses
        bucket = "weekday" if date.weekday() < 4 else \
                 ["friday", "saturday", "sunday"][date.weekday() - 4]
        return local.get(bucket), classify_day_type(False, None, date.weekday())

    day_type = classify_day_type(True, result["decision"], date.weekday())
    if result["decision"] == "USE_HOLIDAY_PROFILE":
        return np.array(result["holiday_profile_shares"]), day_type
    bucket = result["modal_bucket"]
    return local.get(bucket), day_type


def mape_tvd(pred: np.ndarray, actual: np.ndarray) -> tuple:
    """`pred` is a 24-share vector (sums to ~1); `actual` is 24 raw hourly
    demand values. MAPE must compare like with like -- convert the share
    prediction to a predicted VALUE (share x the day's actual total) before
    comparing to actual, exactly what split.py does in production
    (`p50 * s`). TVD compares shapes directly, share vs share."""
    daily_total = actual.sum()
    pred_values = pred * daily_total
    mask = actual > 0
    if not mask.any():
        return np.nan, np.nan
    ape = np.abs(pred_values[mask] - actual[mask]) / actual[mask]
    return float(np.mean(ape)) * 100, P.tvd(pred, P.share_vector(actual))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-days", type=int, default=90,
                    help="how many of the most recent full days to validate on")
    a = ap.parse_args()

    print("=" * 78)
    print("HOURLY SPLIT VALIDATION")
    print("=" * 78)

    df = P.load_hourly()
    date_to_hours = P.date_to_hourly_array(df)
    holiday_dates = P.get_holiday_window_dates()
    full_dates = sorted(d for d, h in date_to_hours.items() if not np.isnan(h).any())

    holdout_days = min(a.holdout_days, max(1, len(full_dates) // 3))
    if holdout_days < a.holdout_days:
        print(f"  [note] only {len(full_dates)} full days available; shrinking "
              f"--holdout-days {a.holdout_days} -> {holdout_days}")
    holdout = full_dates[-holdout_days:]
    train_end = holdout[0]
    print(f"  data: {len(full_dates)} full days ({full_dates[0].date()} .. "
          f"{full_dates[-1].date()})")
    print(f"  holdout: {len(holdout)} days ({holdout[0].date()} .. {holdout[-1].date()})")

    if len(holiday_dates & set(holdout)) == 0:
        print("  [note] holdout window contains ZERO holiday days -- the "
              "holiday_* segments below will be empty. This is expected "
              "against the sample file; not an error.")

    # Train the comparison model ONCE, on everything before the holdout.
    feat = CM.build_hourly_feature_matrix(df, holiday_dates)
    train_mask = feat["date"] < train_end
    n_train = int(train_mask.sum())
    if n_train < 24 * 5:
        print(f"  [WARN] only {n_train} training rows for the comparison model "
              f"({n_train // 24} full days) -- far below what a real model "
              "needs. Proceeding anyway; treat its numbers as illustrative only.")
    model = CM.train_comparison_model(feat, train_mask) if n_train >= 24 else None

    rows = []
    for date in holdout:
        actual = date_to_hours[date]
        emp_shares, day_type = empirical_prediction(date, date_to_hours, holiday_dates)
        cmp_shares = (CM.predict_day_shares(model, feat, date)
                     if model is not None else np.full(24, np.nan))

        for method, shares in (("empirical", emp_shares), ("comparison_model", cmp_shares)):
            if shares is None or np.isnan(shares).any():
                rows.append({"date": date, "method": method, "day_type": day_type,
                            "mape": np.nan, "tvd": np.nan, "n_hours": 0})
                continue
            mape, tvd_ = mape_tvd(shares, actual)
            rows.append({"date": date, "method": method, "day_type": day_type,
                        "mape": mape, "tvd": tvd_, "n_hours": 24})

    report = pd.DataFrame(rows)
    P.DERIVED.mkdir(parents=True, exist_ok=True)
    report_path = P.DERIVED / "validation_report.csv"
    report.to_csv(report_path, index=False)

    summary = (report.dropna(subset=["mape"])
              .groupby(["day_type", "method"])["mape"]
              .agg(["mean", "count"]).reset_index()
              .rename(columns={"mean": "mape_pct", "count": "n_days"}))
    summary_path = P.DERIVED / "validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("RESULTS BY DAY TYPE  (MECHANISM CHECK ONLY IF RUN AGAINST THE SAMPLE "
          "FILE -- see module docstring)")
    print("=" * 78)
    if summary.empty:
        print("  No scoreable days -- every prediction was missing data. "
              "Expected on a very short/sparse hourly file.")
    else:
        for day_type in sorted(summary.day_type.unique()):
            sub = summary[summary.day_type == day_type]
            print(f"\n  {day_type}:")
            for _, r in sub.iterrows():
                print(f"    {r['method']:18s} MAPE {r['mape_pct']:6.2f}%  "
                      f"(n={int(r['n_days'])} day-method rows)")

    print(f"\n-> {report_path}")
    print(f"-> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
