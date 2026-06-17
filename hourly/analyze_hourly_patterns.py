"""Decide for each (holiday, day_offset) whether the hourly pattern should
inherit a local DOW pattern or use its own holiday-specific pattern.

For each historical occurrence of a holiday-day, compares its hourly share
distribution against the four DOW patterns computed from the PRIOR 90 DAYS
LOCAL TO THAT OCCURRENCE (excluding other holidays). Aggregates the
per-occurrence Total Variation Distance (TVD) across years and emits a
decision.

Outputs (under data/derived/):
    hourly_pattern_decisions.csv     # per-cell decisions
    hourly_holiday_profiles.csv      # recency-weighted holiday shares
    hourly_pattern_plots/*.png       # side-by-side charts for borderline /
                                      # holiday-specific cases

Usage:
    uv run python hourly/analyze_hourly_patterns.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    DERIVED, DOW_BUCKETS, HOLIDAY_ANCHORS, HOLIDAY_WINDOWS,
    MIN_HOLIDAY_OCCURRENCES, TVD_BORDERLINE, TVD_CLOSE,
    compute_local_dow_baselines, date_to_hourly_array, get_holiday_window_dates,
    holiday_occurrence_dates, load_hourly, share_vector, tvd,
)

# Which holidays to analyze (single-day low-impact ones get DOW patterns by default)
HOLIDAYS_TO_ANALYZE = [
    "CNY", "GoldenWeek", "Labour", "MidAutumn", "Christmas", "Easter",
]


def analyze_one_cell(holiday_name: str, day_offset: int,
                     date_to_hours: dict, holiday_dates: set) -> dict | None:
    """Returns a result dict with per-occurrence TVDs and an aggregated
    decision, or None if there isn't enough data to decide."""
    occ_dates = holiday_occurrence_dates(holiday_name, day_offset)
    occ_dates = [d for d in occ_dates if d in date_to_hours
                 and not np.isnan(date_to_hours[d]).any()]

    if len(occ_dates) < MIN_HOLIDAY_OCCURRENCES:
        return None

    per_occ = []
    p_hol_shares = []
    for occ_date in occ_dates:
        p_occ = share_vector(date_to_hours[occ_date])
        if np.isnan(p_occ).any():
            continue
        p_hol_shares.append(p_occ)

        local = compute_local_dow_baselines(occ_date, date_to_hours, holiday_dates)
        bucket_tvds = {b: tvd(p_occ, v) for b, v in local.items() if v is not None}
        if not bucket_tvds:
            continue

        best_bucket = min(bucket_tvds, key=bucket_tvds.get)
        per_occ.append({
            "date": occ_date.date().isoformat(),
            "p_hol": p_occ.tolist(),
            "tvds": bucket_tvds,
            "best_bucket": best_bucket,
            "best_tvd": float(bucket_tvds[best_bucket]),
        })

    if len(per_occ) < MIN_HOLIDAY_OCCURRENCES:
        return None

    tvd_values = [o["best_tvd"] for o in per_occ]
    tvd_mean = float(np.mean(tvd_values))
    tvd_max = float(np.max(tvd_values))

    bucket_votes = {}
    for o in per_occ:
        bucket_votes[o["best_bucket"]] = bucket_votes.get(o["best_bucket"], 0) + 1
    modal_bucket = max(bucket_votes, key=bucket_votes.get)
    modal_count = bucket_votes[modal_bucket]
    majority = max(2, len(per_occ) // 2 + 1)

    # Decision cascade
    if (tvd_mean < TVD_CLOSE and tvd_max < TVD_CLOSE + 0.03
            and modal_count >= majority):
        decision = f"USE_{modal_bucket.upper()}"
        reason = "consistent_close_match"
    elif tvd_mean < TVD_BORDERLINE and modal_count >= majority:
        decision = f"USE_{modal_bucket.upper()}"
        reason = "borderline_flag_review"
    elif tvd_mean >= TVD_BORDERLINE:
        decision = "USE_HOLIDAY_PROFILE"
        reason = "strong_difference"
    else:
        decision = "USE_HOLIDAY_PROFILE"
        reason = "inconsistent_match"

    # Recency-weighted holiday profile: more recent occurrences carry more weight
    n = len(p_hol_shares)
    weights = np.linspace(1.0, 1.5, n)
    weights /= weights.sum()
    holiday_profile = np.average(p_hol_shares, axis=0, weights=weights)

    return {
        "holiday": holiday_name,
        "day_offset": day_offset,
        "n_occurrences": len(per_occ),
        "per_occ": per_occ,
        "tvd_mean": tvd_mean,
        "tvd_max": tvd_max,
        "modal_bucket": modal_bucket,
        "modal_count": modal_count,
        "decision": decision,
        "reason": reason,
        "holiday_profile_shares": holiday_profile.tolist(),
    }


def plot_comparison(result: dict, date_to_hours: dict, holiday_dates: set,
                    out_path: Path) -> None:
    """Hourly chart showing each occurrence + recency-weighted holiday avg +
    the best DOW baseline (relative to the most recent occurrence)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for o in result["per_occ"]:
        ax.plot(range(24), o["p_hol"], "-", alpha=0.45, linewidth=1,
                label=f"{o['date']}")

    ax.plot(range(24), result["holiday_profile_shares"], "o-", linewidth=2.5,
            color="C3", label="Holiday avg (recency-weighted)")

    last_occ = pd.Timestamp(result["per_occ"][-1]["date"])
    local = compute_local_dow_baselines(last_occ, date_to_hours, holiday_dates)
    best = result["modal_bucket"]
    if local.get(best) is not None:
        ax.plot(range(24), local[best], "s-", linewidth=2, color="C0",
                label=f"{best} baseline (prior 90d before {last_occ.date()})")

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour")
    ax.set_ylabel("share of daily demand")
    ax.set_title(f"{result['holiday']} d{result['day_offset']:+d}  |  "
                 f"TVD_mean={result['tvd_mean']:.3f}  |  "
                 f"decision={result['decision']}  ({result['reason']})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def main() -> int:
    print("=== Hourly pattern analyzer ===\n")
    df = load_hourly()
    print(f"  Loaded {len(df):,} hourly rows  "
          f"({df.date.min().date()} .. {df.date.max().date()})")

    date_to_hours = date_to_hourly_array(df)
    holiday_dates = get_holiday_window_dates()
    print(f"  Identified {len(holiday_dates)} holiday-window dates to exclude "
          "from DOW baselines\n")

    results = []
    for holiday in HOLIDAYS_TO_ANALYZE:
        ws, we = HOLIDAY_WINDOWS[holiday]
        for offset in range(ws, we + 1):
            r = analyze_one_cell(holiday, offset, date_to_hours, holiday_dates)
            if r is None:
                continue
            results.append(r)
            print(f"  {r['holiday']:12s} d{r['day_offset']:+d}  "
                  f"n={r['n_occurrences']}  "
                  f"tvd_mean={r['tvd_mean']:.3f}  "
                  f"tvd_max={r['tvd_max']:.3f}  "
                  f"modal={r['modal_bucket']:8s} -> {r['decision']:24s} "
                  f"({r['reason']})")

    if not results:
        print("\n  [WARN] No (holiday, day_offset) cells had enough data to decide.")
        print(f"  Need at least {MIN_HOLIDAY_OCCURRENCES} occurrences with full "
              "24-hour data per cell.")
        return 1

    # 1) Decisions CSV
    rows = []
    max_occ = max(r["n_occurrences"] for r in results)
    for r in results:
        row = {
            "holiday": r["holiday"],
            "day_offset": r["day_offset"],
            "n_occurrences": r["n_occurrences"],
            "tvd_mean": round(r["tvd_mean"], 4),
            "tvd_max": round(r["tvd_max"], 4),
            "modal_bucket": r["modal_bucket"],
            "modal_count": r["modal_count"],
            "decision": r["decision"],
            "reason": r["reason"],
        }
        for i in range(max_occ):
            if i < len(r["per_occ"]):
                o = r["per_occ"][i]
                row[f"occ{i+1}_date"] = o["date"]
                row[f"occ{i+1}_best_bucket"] = o["best_bucket"]
                row[f"occ{i+1}_best_tvd"] = round(o["best_tvd"], 4)
            else:
                row[f"occ{i+1}_date"] = ""
                row[f"occ{i+1}_best_bucket"] = ""
                row[f"occ{i+1}_best_tvd"] = ""
        rows.append(row)

    decisions_csv = DERIVED / "hourly_pattern_decisions.csv"
    pd.DataFrame(rows).to_csv(decisions_csv, index=False)
    print(f"\n  -> {decisions_csv}")

    # 2) Holiday profiles CSV (long format: one row per hour per cell)
    prof_rows = []
    for r in results:
        for h, share in enumerate(r["holiday_profile_shares"]):
            prof_rows.append({
                "holiday": r["holiday"],
                "day_offset": r["day_offset"],
                "hour": h,
                "share": round(float(share), 6),
            })
    profiles_csv = DERIVED / "hourly_holiday_profiles.csv"
    pd.DataFrame(prof_rows).to_csv(profiles_csv, index=False)
    print(f"  -> {profiles_csv}")

    # 3) Side-by-side plots for non-trivial decisions
    plot_dir = DERIVED / "hourly_pattern_plots"
    plot_dir.mkdir(exist_ok=True)
    flagged = [r for r in results
               if r["reason"] in ("borderline_flag_review",
                                  "strong_difference",
                                  "inconsistent_match")]
    for r in flagged:
        fname = (f"{r['holiday']}_d{r['day_offset']:+d}.png"
                 .replace("+", "p").replace("-", "m"))
        plot_comparison(r, date_to_hours, holiday_dates, plot_dir / fname)
    print(f"  -> {len(flagged)} comparison plot(s) under {plot_dir}")

    print(f"\n=== Done — review {decisions_csv} and PNGs in {plot_dir} ===\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
