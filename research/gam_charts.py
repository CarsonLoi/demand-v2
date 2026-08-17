"""gam_charts.py — visualize the GAM vs LightGBM comparison from gam_compare.py.

Standalone. Reads research/gam_output/{lgbm,gam_plain,gam_interact}.csv (all
honest held-out backtest predictions -- see gam_compare.py's docstring for the
protocol). Writes PNGs back into the same folder. Touches nothing else.

    1. mape_by_origin.png       grouped bar, the "does GAM ever lose" view
    2. actual_vs_pred.png       small multiples, one panel per origin,
                                 actual vs each model across horizons
    3. summary_bars.png         overall / CNY / non-CNY MAPE, one glance
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
OUT_DIR = ROOT / "research" / "gam_output"
sys.path.insert(0, str(ROOT / "v2"))
from _shared import HOLIDAY_ANCHORS, HOLIDAY_WINDOWS  # noqa: E402  read-only

NAVY, TEAL, GOLD, GREY, RED = "#0F2942", "#2E8BA8", "#E8B547", "#5F7183", "#C74B4B"
MODELS = [("lgbm", "LightGBM", NAVY), ("gam_plain", "GAM (plain)", TEAL),
          ("gam_interact", "GAM (+holiday interactions)", GOLD)]

CNY_WINDOW = set()
for _a in HOLIDAY_ANCHORS["CNY"]:
    ws, we = HOLIDAY_WINDOWS["CNY"]
    for _k in range(ws, we + 1):
        CNY_WINDOW.add(_a + pd.Timedelta(days=_k))


def mape(d):
    d = d[(d.actual > 0) & d.pred.notna()]
    return float(np.mean(np.abs(d.actual - d.pred) / d.actual)) * 100 if len(d) else np.nan


def _style(ax):
    ax.grid(alpha=.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


def load():
    out = {}
    for key, _, _ in MODELS:
        d = pd.read_csv(OUT_DIR / f"{key}.csv", parse_dates=["origin", "target"])
        out[key] = d
    return out


def chart_mape_by_origin(data, out):
    origins = sorted(data["lgbm"].origin.unique())
    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(origins))
    w = 0.8 / len(MODELS)
    for i, (key, label, color) in enumerate(MODELS):
        d = data[key]
        vals = [mape(d[d.origin == o]) for o in origins]
        bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}",
                    ha="center", fontsize=7.5, color=GREY)
    ax.set_xticks(x + w * (len(MODELS) - 1) / 2)
    ax.set_xticklabels([str(pd.Timestamp(o).date()) for o in origins],
                       rotation=25, fontsize=9)
    ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
    ax.set_title("MAPE by origin — held-out, horizons 1/3/7/14/21/28 days ahead. "
                 "GAM only wins if EVERY bar is lower.", fontsize=12,
                 fontweight="bold", color=NAVY, loc="left")
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout(); plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def chart_actual_vs_pred(data, out):
    origins = sorted(data["lgbm"].origin.unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    for ax, o in zip(axes.flat, origins):
        d0 = data["lgbm"][data["lgbm"].origin == o].sort_values("horizon")
        ax.plot(d0.horizon, d0.actual, "o-", color=NAVY, lw=2.2, ms=6,
                label="actual", zorder=5)
        for key, label, color in MODELS:
            d = data[key][data[key].origin == o].sort_values("horizon")
            ax.plot(d.horizon, d.pred, "s--", color=color, lw=1.5, ms=4,
                    alpha=.85, label=label)
        ax.set_title(f"origin {pd.Timestamp(o).date()}", fontsize=10,
                     fontweight="bold", color=NAVY)
        ax.set_xlabel("horizon (days ahead)", fontsize=8, color=GREY)
        ax.set_ylabel("demand", fontsize=8, color=GREY)
        _style(ax)
    axes.flat[0].legend(fontsize=7.5, loc="best")
    fig.suptitle("Actual vs predicted, by forecast horizon — each panel is one "
                 "held-out origin", fontsize=13, fontweight="bold", color=NAVY)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def chart_summary(data, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    slices = ["overall", "CNY", "non-CNY"]
    x = np.arange(len(slices))
    w = 0.8 / len(MODELS)
    for i, (key, label, color) in enumerate(MODELS):
        d = data[key]
        cny = d[d.target.isin(CNY_WINDOW)]
        non_cny = d[~d.target.isin(CNY_WINDOW)]
        vals = [mape(d), mape(cny), mape(non_cny)]
        bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=.9)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}",
                        ha="center", fontsize=9, color=GREY)
    ax.set_xticks(x + w * (len(MODELS) - 1) / 2)
    ax.set_xticklabels(slices, fontsize=10)
    ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
    ax.set_title("Pooled MAPE across all 8 origins (n=48)", fontsize=12,
                 fontweight="bold", color=NAVY, loc="left")
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout(); plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def main():
    data = load()
    print(f"loaded: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()))
    chart_mape_by_origin(data, OUT_DIR / "mape_by_origin.png")
    chart_actual_vs_pred(data, OUT_DIR / "actual_vs_pred.png")
    chart_summary(data, OUT_DIR / "summary_bars.png")


if __name__ == "__main__":
    main()
