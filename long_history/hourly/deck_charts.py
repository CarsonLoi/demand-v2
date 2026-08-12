"""deck_charts.py -- presentation charts for the hourly-splitting slides.

Reads validate.py's output (validation_report.csv, validation_summary.csv)
and split.py's output (predictions_hourly.csv) and produces two PNGs sized
and styled to paste directly into the reserved chart slots in
docs/Demand_Forecast_LongHistory.pptx:

    deck_mape_by_daytype.png   MAPE by day-type, empirical vs comparison
                               model -- the "is the simple method good
                               enough" chart
    deck_actual_vs_split.png   actual hourly demand vs the empirically
                               split forecast, for one illustrative day

STATUS
------
Nothing this script produces is real evidence until
long_history/data/raw/hourly_demand.csv contains real history and
hourly/validate.py has been run against it. Run against the sample file (or
before validate.py has ever run), this script prints a clear warning and
either produces charts explicitly watermarked "MECHANISM CHECK ONLY -- NOT
REAL DATA" or refuses to produce the day-type chart at all when there is
nothing scoreable. Do not paste a mechanism-check chart into the deck as if
it were a real result -- the slide's reserved space says why.

Usage:
    uv run python long_history/hourly/validate.py --holdout-days 90
    uv run python long_history/hourly/split.py --run-date <a date in the holdout>
    uv run python long_history/hourly/deck_charts.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import patterns as P  # noqa: E402

NAVY, TEAL, GREY, GOLD, RED, GREEN = (
    "#0F2942", "#2E8BA8", "#5F7183", "#E8B547", "#C74B4B", "#2FA87C")

# Below this many holdout days, a chart from this data cannot be evidence --
# same threshold spirit as the rest of this project (a result from too few
# events is not a result). The 10-day sample file always falls under this.
MIN_DAYS_FOR_REAL_CHART = 30


def _style(ax):
    ax.grid(alpha=.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


def _watermark_if_mechanism_check(fig, n_days: int) -> None:
    if n_days < MIN_DAYS_FOR_REAL_CHART:
        fig.text(0.5, 0.5, "MECHANISM CHECK ONLY -- NOT REAL DATA",
                 fontsize=22, color="#C74B4B", alpha=0.35, ha="center",
                 va="center", rotation=28, weight="bold")


def chart_mape_by_daytype(report_path: Path, out_path: Path) -> None:
    if not report_path.exists():
        print(f"  [skip] {report_path} not found -- run validate.py first")
        return
    df = pd.read_csv(report_path, parse_dates=["date"]).dropna(subset=["mape"])
    if df.empty:
        print("  [skip] validation_report.csv has no scoreable rows")
        return

    n_days = df["date"].nunique()
    # Matches validate.py's bucket-level day_type labels exactly (Tier 1:
    # weekday = Mon-Thu pooled, friday/saturday/sunday each separate --
    # see validate.py's module docstring for why they're no longer pooled
    # together in reporting the way they used to be).
    order = ["weekday", "friday", "saturday", "sunday",
             "holiday_inherited_dow", "holiday_specific_profile"]
    present = [d for d in order if d in df.day_type.unique()]
    methods = [("empirical", "Historical distribution", TEAL),
              ("comparison_model", "Trained model", RED)]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(present))
    w = 0.35
    for i, (key, label, color) in enumerate(methods):
        vals = [df[(df.day_type == d) & (df.method == key)].mape.mean()
               for d in present]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=color, alpha=.9)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}",
                       ha="center", fontsize=9, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ") for d in present], fontsize=9)
    ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
    ax.set_title(f"Hourly split accuracy by day type  (n={n_days} holdout days)",
                fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.legend(fontsize=9)
    _style(ax)
    _watermark_if_mechanism_check(fig, n_days)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    tag = "MECHANISM CHECK ONLY" if n_days < MIN_DAYS_FOR_REAL_CHART else "real"
    print(f"  -> {out_path}  ({tag}, n={n_days} days)")


def chart_actual_vs_split(hourly_pred_path: Path, out_path: Path) -> None:
    if not hourly_pred_path.exists():
        print(f"  [skip] {hourly_pred_path} not found -- run split.py first, "
              "then point --hourly-pred at its predictions_hourly.csv")
        return
    df = pd.read_csv(hourly_pred_path, parse_dates=["date"])
    if df.empty:
        print("  [skip] predictions_hourly.csv is empty")
        return
    day = sorted(df.date.unique())[0]
    d = df[df.date == day].sort_values("hour")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(d.hour, d.p50, "o-", color=TEAL, lw=2.2, ms=5,
           label=f"split forecast ({d.strategy.iloc[0]})")
    ax.fill_between(d.hour, d.p10, d.p90, color=TEAL, alpha=.15, label="P10-P90")
    ax.set_xlabel("hour of day", fontsize=9, color=GREY)
    ax.set_ylabel("patron hours", fontsize=9, color=GREY)
    ax.set_title(f"Hourly split, {pd.Timestamp(day).date()}", fontsize=12,
               fontweight="bold", color=NAVY, loc="left")
    ax.legend(fontsize=9)
    _style(ax)
    fig.text(0.5, 0.5, "NO ACTUAL OVERLAY -- FORECAST ONLY", fontsize=18,
            color="#C74B4B", alpha=0.3, ha="center", va="center", rotation=28,
            weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}  (forecast only -- overlay real actuals by hand "
         "once that day's actual hourly demand is known)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hourly-pred", default=None,
                    help="path to a predictions_hourly.csv from split.py "
                         "(default: skip the actual-vs-split chart)")
    a = ap.parse_args()

    out_dir = P.DERIVED
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Hourly deck charts ===\n")
    chart_mape_by_daytype(out_dir / "validation_report.csv",
                          out_dir / "deck_mape_by_daytype.png")
    if a.hourly_pred:
        chart_actual_vs_split(Path(a.hourly_pred),
                              out_dir / "deck_actual_vs_split.png")
    else:
        print("  [skip] --hourly-pred not given -- pass the path to a "
             "predictions_hourly.csv (from split.py) to build this chart")
    print("\nPaste the resulting PNG(s) into the reserved chart slots on the "
         "hourly slides in docs/Demand_Forecast_LongHistory.pptx.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
