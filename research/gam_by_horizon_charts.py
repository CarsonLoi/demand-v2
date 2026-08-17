"""gam_by_horizon_charts.py — one graph per horizon, LightGBM vs GAM.

Reads research/gam_by_horizon_output/raw.csv and writes:
    h03.png .. h28.png   one chart per horizon: actual vs both models over
                         time, holiday windows shaded
    mape_vs_horizon.png  both models' MAPE as forecast distance grows,
                         split all / holiday / ordinary
    holiday_breakdown.png per-holiday MAPE, the hybrid question in one view

Standalone; writes only into that output folder.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "gam_by_horizon_output"

NAVY, TEAL, GOLD, GREY, RED, GREEN = (
    "#0F2942", "#2E8BA8", "#E8B547", "#5F7183", "#C74B4B", "#2FA87C")


def mape(d, col):
    d = d[d.actual > 0]
    return float(np.mean(np.abs(d[col] - d.actual) / d.actual)) * 100 if len(d) else np.nan


def _style(ax):
    ax.grid(alpha=.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


def per_horizon(df):
    for h in sorted(df.horizon.unique()):
        d = df[df.horizon == h].sort_values("target")
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]})

        for _, r in d[d.is_holiday].iterrows():
            ax.axvspan(r.target - pd.Timedelta(hours=12),
                       r.target + pd.Timedelta(hours=12),
                       color=GOLD, alpha=.22, zorder=0)
            ax2.axvspan(r.target - pd.Timedelta(hours=12),
                        r.target + pd.Timedelta(hours=12),
                        color=GOLD, alpha=.22, zorder=0)

        ax.plot(d.target, d.actual, "o-", color=NAVY, lw=2.1, ms=4.5,
                label="actual", zorder=5)
        ax.plot(d.target, d.lgbm, "s--", color=TEAL, lw=1.6, ms=3.6,
                label=f"LightGBM  ({mape(d,'lgbm'):.2f}%)", zorder=4)
        ax.plot(d.target, d.gam, "^--", color=RED, lw=1.6, ms=3.6,
                label=f"GAM  ({mape(d,'gam'):.2f}%)", zorder=3)
        ax.set_ylabel("demand (patron hours)", fontsize=9, color=GREY)
        ax.set_title(f"Horizon {h} days ahead — forecast made {h} day(s) before "
                     f"each target   ·   shaded = holiday window   ·   n={len(d)}",
                     fontsize=12, fontweight="bold", color=NAVY, loc="left")
        ax.legend(fontsize=9, loc="best")
        _style(ax)

        el = (d.lgbm - d.actual) / d.actual * 100
        eg = (d.gam - d.actual) / d.actual * 100
        ax2.axhline(0, color=NAVY, lw=1.1)
        ax2.plot(d.target, el, "s-", color=TEAL, lw=1.3, ms=3, label="LightGBM")
        ax2.plot(d.target, eg, "^-", color=RED, lw=1.3, ms=3, label="GAM")
        ax2.set_ylabel("% error", fontsize=9, color=GREY)
        ax2.set_xlabel("target date", fontsize=9, color=GREY)
        ax2.legend(fontsize=8)
        _style(ax2)

        plt.xticks(rotation=30)
        plt.tight_layout()
        p = OUT / f"h{h:02d}.png"
        plt.savefig(p, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  -> {p.name}")


def mape_curve(df):
    hs = sorted(df.horizon.unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    groups = [("All days", df), ("Holiday-window days", df[df.is_holiday]),
              ("Ordinary days", df[~df.is_holiday])]
    for ax, (lab, sel) in zip(axes, groups):
        for col, name, c, mk in (("lgbm", "LightGBM", TEAL, "s"),
                                 ("gam", "GAM", RED, "^")):
            vals = [mape(sel[sel.horizon == h], col) for h in hs]
            ax.plot(hs, vals, mk + "-", color=c, lw=1.9, ms=5, label=name)
        ax.set_title(f"{lab}  (n={len(sel)})", fontsize=11,
                     fontweight="bold", color=NAVY, loc="left")
        ax.set_xlabel("horizon (days ahead)", fontsize=9, color=GREY)
        ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        _style(ax)
    fig.suptitle("Accuracy vs forecast distance — where does each model win?",
                 fontsize=13, fontweight="bold", color=NAVY)
    plt.tight_layout(rect=[0, 0, 1, .94])
    p = OUT / "mape_vs_horizon.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  -> {p.name}")


def holiday_breakdown(df):
    names = sorted(x for x in df.holiday.unique() if x)
    if not names:
        return
    rows = [("ordinary", df[~df.is_holiday])] + [(n, df[df.holiday == n]) for n in names]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(rows))
    w = .38
    lv = [mape(d, "lgbm") for _, d in rows]
    gv = [mape(d, "gam") for _, d in rows]
    ax.bar(x - w / 2, lv, w, label="LightGBM", color=TEAL, alpha=.9)
    ax.bar(x + w / 2, gv, w, label="GAM", color=RED, alpha=.9)
    for i, (l, g) in enumerate(zip(lv, gv)):
        if not np.isnan(l):
            ax.text(i - w / 2, l + .1, f"{l:.1f}", ha="center", fontsize=8, color=GREY)
        if not np.isnan(g):
            ax.text(i + w / 2, g + .1, f"{g:.1f}", ha="center", fontsize=8, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={len(d)})" for n, d in rows], fontsize=8.5)
    ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
    ax.set_title("The hybrid question: is there any holiday where GAM beats "
                 "LightGBM?", fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout()
    p = OUT / "holiday_breakdown.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  -> {p.name}")


def main():
    src = OUT / "raw.csv"
    if not src.exists():
        print(f"FATAL: {src} not found. Run research/gam_by_horizon.py first.")
        return 1
    df = pd.read_csv(src, parse_dates=["origin", "target"])
    df["holiday"] = df["holiday"].fillna("")
    print(f"{len(df)} predictions · {df.horizon.nunique()} horizons · "
          f"{df.origin.nunique()} origins")
    per_horizon(df)
    mape_curve(df)
    holiday_breakdown(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
