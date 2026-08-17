"""narrow_mask_charts.py — visualize the narrow-mask backtest.

Standalone. Reads research/narrow_mask_output/*.csv (written by
narrow_mask_test.py) and writes PNGs back into the same folder. Modifies
nothing in v2/ or long_history/.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v2"))
import _shared as S  # noqa: E402
from research.narrow_mask_test import build_ratio_map, contaminated_dates  # noqa: E402

OUT = ROOT / "research" / "narrow_mask_output"
NAVY, TEAL, GOLD, GREY, RED, GREEN = (
    "#0F2942", "#2E8BA8", "#E8B547", "#5F7183", "#C74B4B", "#2FA87C")
TAGS = [("guard_off", "guard OFF (production today)", GREY),
        ("guard_full", "full calendar mask", RED),
        ("narrow_0.85", "narrow ±15%", GOLD),
        ("narrow_0.70", "narrow ±30%", GREEN)]


def mape(x):
    x = x[(x.actual > 0) & x.forecast.notna()]
    return float(np.mean(np.abs(x.actual - x.forecast) / x.actual)) * 100 if len(x) else np.nan


def _style(ax):
    ax.grid(alpha=.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


def main():
    demand = S.load_demand()
    ratio = build_ratio_map(demand)
    bad70 = contaminated_dates(demand, 0.70, 1.30)
    d = {t: pd.read_csv(OUT / f"{t}.csv", parse_dates=["origin", "target_date"])
         for t, _, _ in TAGS}

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, hspace=.35, wspace=.22)

    # ── 1. WHY: the CNY-2025 window, ratio vs what each rule masks ──────────
    ax = fig.add_subplot(gs[0, :])
    anchor = pd.Timestamp("2025-01-29")
    days = [anchor + pd.Timedelta(days=k) for k in range(-7, 11)]
    rr = [ratio.get(x, np.nan) for x in days]
    xs = np.arange(len(days))
    cols = [RED if x in bad70 else TEAL for x in days]
    ax.bar(xs, rr, color=cols, alpha=.9)
    ax.axhline(1.0, color=NAVY, lw=1.2, ls="--")
    ax.axhspan(0.70, 1.30, color=GREEN, alpha=.10)
    ax.axhline(0.70, color=GREEN, lw=1.1, ls=":")
    ax.axhline(1.30, color=GREEN, lw=1.1, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x.strftime('%b %d')}\n{k:+d}"
                        for x, k in zip(days, range(-7, 11))], fontsize=8)
    ax.set_ylabel("demand ÷ normal for that weekday", fontsize=9, color=GREY)
    ax.set_title("WHY narrow: CNY-2025 window. The full calendar mask discards ALL 18 days. "
                 "Only the red bars were actually abnormal.",
                 fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.text(.995, .05, "red = masked by narrow ±30%   ·   green band = kept",
            transform=ax.transAxes, ha="right", fontsize=9, color=GREY)
    _style(ax)

    # ── 2. per-origin MAPE ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    origins = sorted(d["guard_off"].origin.unique())
    x = np.arange(len(origins))
    w = .8 / len(TAGS)
    for i, (t, lab, c) in enumerate(TAGS):
        vals = [mape(d[t][d[t].origin == o]) for o in origins]
        ax.bar(x + i * w, vals, w, label=lab, color=c, alpha=.9)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([str(pd.Timestamp(o).date())[5:] for o in origins],
                       rotation=60, fontsize=7)
    ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
    ax.set_ylim(0, 12)
    ax.set_title("Per-origin MAPE (2026-02-11 clipped: 26–29%)",
                 fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax.legend(fontsize=7.5)
    _style(ax)

    # ── 3. the decisive slice: does the mask help where it fires ───────────
    ax = fig.add_subplot(gs[1, 1])
    base = d["guard_off"].copy()
    base["h"] = (base.target_date - base.origin).dt.days

    def fires(r):
        T, h = r.target_date, r.h
        return ((T - pd.Timedelta(days=365)) in bad70
                or (T - pd.Timedelta(days=728)) in bad70
                or (T - pd.Timedelta(days=h + 1)) in bad70
                or (T - pd.Timedelta(days=h + 1 + 365)) in bad70)

    m = base.apply(fires, axis=1).values
    groups = [("days the mask\nFIRES (n=13)", m), ("days it does\nNOT fire (n=135)", ~m)]
    xg = np.arange(len(groups))
    for i, (t, lab, c) in enumerate(TAGS):
        s = d[t].sort_values("target_date").reset_index(drop=True)
        vals = [mape(s[g]) for _, g in groups]
        bars = ax.bar(xg + i * w, vals, w, label=lab, color=c, alpha=.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .08, f"{v:.2f}",
                    ha="center", fontsize=8, color=GREY)
    ax.set_xticks(xg + w * 1.5)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
    ax.set_title("The decisive view: a targeted fix, judged where it applies",
                 fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax.legend(fontsize=7.5)
    _style(ax)

    fig.suptitle("Narrow demand-driven mask vs full calendar mask — honest held-out "
                 "backtest, 2026 YTD, 7-day lead",
                 fontsize=14, fontweight="bold", color=NAVY, y=.98)
    p = OUT / "narrow_mask_summary.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
