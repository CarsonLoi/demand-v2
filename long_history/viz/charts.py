"""charts.py — actual vs prediction charts for the long-history pipeline.

Reads the per-day backtest output written by experiment.py
(`long_history/output/experiment_raw.csv`) and produces four views. Every
point in them is an honest held-out prediction: trained only on data available
at that forecast origin.

    1. actual_vs_pred.png    the headline line chart, one panel per variant,
                             CNY windows shaded
    2. error_by_horizon.png  how accuracy decays with forecast distance
    3. error_distribution.png where the errors sit, and the tail
    4. window_comparison.png per-window MAPE, variant vs variant

Usage:
    uv run python long_history/experiment.py --quick     # produces the data
    uv run python long_history/charts.py                 # then the charts
    uv run python long_history/charts.py --raw path/to/other_raw.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/ -- this package owns its data
sys.path.insert(0, str(ROOT))
import config as C
from engine.core import HOLIDAY_ANCHORS

NAVY, TEAL, GREY, GOLD, RED, GREEN = (
    "#0F2942", "#2E8BA8", "#5F7183", "#E8B547", "#C74B4B", "#2FA87C")

CNY_WIN = set()
for _a in HOLIDAY_ANCHORS["CNY"]:
    for _k in range(-7, 11):
        CNY_WIN.add(_a + pd.Timedelta(days=_k))


def mape(d):
    d = d[(d.actual > 0) & d.pred.notna()]
    return float(np.mean(np.abs(d.actual - d.pred) / d.actual)) * 100 if len(d) else np.nan


def _style(ax):
    ax.grid(alpha=.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


def chart_actual_vs_pred(df, out):
    """One panel per variant: actual vs predicted through time."""
    variants = sorted(df.variant.unique())
    fig, axes = plt.subplots(len(variants), 1, figsize=(14, 3.5 * len(variants)),
                             sharex=True, squeeze=False)
    for ax, v in zip(axes[:, 0], variants):
        d = df[df.variant == v].sort_values("target").drop_duplicates("target")
        # shade each CNY window present in the range
        for a in HOLIDAY_ANCHORS["CNY"]:
            lo, hi = a - pd.Timedelta(days=7), a + pd.Timedelta(days=10)
            if lo <= d.target.max() and hi >= d.target.min():
                ax.axvspan(max(lo, d.target.min()), min(hi, d.target.max()),
                           color=GOLD, alpha=.16, zorder=0)
        ax.plot(d.target, d.actual, "-", color=NAVY, lw=1.9, label="actual", zorder=3)
        ax.plot(d.target, d.pred, "--", color=TEAL, lw=1.6, label="predicted", zorder=2)
        ax.set_title(f"{v}   ·   MAPE {mape(d):.2f}%", fontsize=11,
                     fontweight="bold", color=NAVY, loc="left")
        ax.set_ylabel("patron hours", fontsize=9, color=GREY)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=.9)
        _style(ax)
    axes[-1, 0].set_xlabel("date", fontsize=9, color=GREY)
    fig.suptitle("Actual vs predicted — held-out backtest (shaded = Chinese New Year)",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.999)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def chart_error_by_horizon(df, out):
    """Accuracy vs forecast distance, split CNY / non-CNY."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (title, sel) in zip(axes, [
            ("All days", lambda d: d),
            ("Split: CNY vs rest", None)]):
        for v in sorted(df.variant.unique()):
            d = df[df.variant == v].copy()
            d["ape"] = 100 * (d.pred - d.actual).abs() / d.actual
            if sel is not None:
                g = d.groupby("horizon").ape.mean()
                ax.plot(g.index, g.values, "o-", ms=3.5, lw=1.7, label=v)
            else:
                for lab, mask, ls in (("CNY", d.target.isin(CNY_WIN), "-"),
                                      ("non-CNY", ~d.target.isin(CNY_WIN), "--")):
                    g = d[mask].groupby("horizon").ape.mean()
                    if len(g):
                        ax.plot(g.index, g.values, ls, lw=1.7,
                                label=f"{v.split('/')[-1].strip()} · {lab}")
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, loc="left")
        ax.set_xlabel("forecast horizon (days ahead)", fontsize=9, color=GREY)
        ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, framealpha=.9)
        _style(ax)
    plt.tight_layout(); plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def chart_error_distribution(df, out):
    """Where the errors sit, and how heavy the tail is."""
    variants = sorted(df.variant.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for v in variants:
        d = df[df.variant == v].copy()
        d["pe"] = 100 * (d.pred - d.actual) / d.actual
        axes[0].hist(d.pe, bins=40, alpha=.45, label=f"{v}  (bias {d.pe.mean():+.1f}%)")
    axes[0].axvline(0, color=NAVY, lw=1.2, ls="--")
    axes[0].set_title("Signed % error  (right of 0 = over-forecast)",
                      fontsize=11, fontweight="bold", color=NAVY, loc="left")
    axes[0].set_xlabel("% error", fontsize=9, color=GREY)
    axes[0].set_ylabel("days", fontsize=9, color=GREY)
    axes[0].legend(fontsize=8)
    _style(axes[0])

    for v in variants:
        d = df[df.variant == v].copy()
        ape = np.sort((100 * (d.pred - d.actual).abs() / d.actual).values)
        axes[1].plot(ape, np.linspace(0, 100, len(ape)), lw=1.8, label=v)
    for q, c in ((5, GREEN), (10, GOLD)):
        axes[1].axvline(q, color=c, ls=":", lw=1.3)
        axes[1].text(q, 4, f" {q}%", color=c, fontsize=8.5)
    axes[1].set_title("Cumulative: what share of days are within X% error",
                      fontsize=11, fontweight="bold", color=NAVY, loc="left")
    axes[1].set_xlabel("absolute % error", fontsize=9, color=GREY)
    axes[1].set_ylabel("% of days at or below", fontsize=9, color=GREY)
    axes[1].set_xlim(0, 25); axes[1].legend(fontsize=8, loc="lower right")
    _style(axes[1])
    plt.tight_layout(); plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def chart_window_comparison(df, out):
    """Per-window MAPE, variant vs variant -- the view the gate is judged on."""
    rows = []
    for v in sorted(df.variant.unique()):
        for o, g in df[df.variant == v].groupby("origin"):
            rows.append({"variant": v, "origin": pd.Timestamp(o), "mape": mape(g)})
    p = pd.DataFrame(rows).pivot(index="origin", columns="variant", values="mape")
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(p)); w = 0.8 / max(1, len(p.columns))
    for i, c in enumerate(p.columns):
        ax.bar(x + i * w, p[c].values, w, label=c,
               color=[NAVY, TEAL, GOLD, GREEN, RED][i % 5], alpha=.88)
    ax.set_xticks(x + w * (len(p.columns) - 1) / 2)
    ax.set_xticklabels([str(d.date()) for d in p.index], rotation=30, fontsize=9)
    ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
    ax.set_title("Per-window MAPE — the gate is judged window by window, "
                 "not on the average", fontsize=12, fontweight="bold", color=NAVY,
                 loc="left")
    ax.legend(fontsize=8.5); _style(ax)
    plt.tight_layout(); plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=None,
                    help="per-day backtest CSV (default: output/experiment_raw.csv)")
    a = ap.parse_args()

    src = Path(a.raw) if a.raw else ROOT / C.OUT_DIR / "experiment_raw.csv"
    if not src.exists():
        print(f"FATAL: {src} not found.\n"
              f"       Run:  uv run python long_history/experiment.py --quick")
        return 1

    df = pd.read_csv(src, parse_dates=["origin", "target"])
    df = df[(df.actual > 0) & df.pred.notna()]
    out = ROOT / C.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print(f"source: {src}")
    print(f"  {len(df):,} predictions · {df.variant.nunique()} variant(s) · "
          f"{df.origin.nunique()} window(s)")
    print(f"  targets {df.target.min().date()} .. {df.target.max().date()}\n")

    chart_actual_vs_pred(df, out / "actual_vs_pred.png")
    chart_error_by_horizon(df, out / "error_by_horizon.png")
    chart_error_distribution(df, out / "error_distribution.png")
    chart_window_comparison(df, out / "window_comparison.png")

    print("\nsummary by variant:")
    for v in sorted(df.variant.unique()):
        d = df[df.variant == v]
        c, n = d[d.target.isin(CNY_WIN)], d[~d.target.isin(CNY_WIN)]
        print(f"  {v:34s} all {mape(d):5.2f}%   CNY {mape(c):5.2f}%   "
              f"non-CNY {mape(n):5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
