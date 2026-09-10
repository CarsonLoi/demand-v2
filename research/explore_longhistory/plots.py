from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

NAVY, TEAL, GREY, GOLD, RED, GREEN = (
    "#0F2942", "#2E8BA8", "#5F7183", "#E8B547", "#C74B4B", "#2FA87C")

# a stable colour per clean year for the year-over-year line charts
YEAR_COLOURS = {
    2015: "#6B8CAE", 2016: "#4E7A9E", 2017: "#2E8BA8", 2018: "#1F6F86",
    2019: "#0F2942", 2023: "#E8B547", 2024: "#D98C3A", 2025: "#C74B4B", 2026: "#2FA87C",
}


def style_ax(ax) -> None:
    ax.grid(alpha=0.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ── Part A1 ─────────────────────────────────────────────────────────────
def plot_per_table_by_year(df: pd.DataFrame, out: Path) -> None:
    """Monthly mean demand-per-table, one line per year (all years shown)."""
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    g = d.groupby([d["date"].dt.year, d["date"].dt.month])["demand_per_table"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for yr in sorted({i[0] for i in g.index}):
        sub = g.loc[yr]
        ax.plot(sub.index, sub.values, "o-", ms=3, lw=1.5,
                color=YEAR_COLOURS.get(yr, GREY), label=str(yr),
                alpha=0.55 if yr in (2020, 2021, 2022) else 1.0)
    ax.set_xlabel("month", color=GREY, fontsize=9)
    ax.set_ylabel("mean demand / table", color=GREY, fontsize=9)
    ax.set_title("Demand per table by month and year (faded = COVID years)",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=4)
    style_ax(ax)
    _save(fig, out)


# ── Part A2 ─────────────────────────────────────────────────────────────
def plot_monthly_seasonality(index_table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for yr in index_table.index:
        ax.plot(range(1, 13), index_table.loc[yr].values, "o-", ms=3.5, lw=1.6,
                color=YEAR_COLOURS.get(int(yr), GREY), label=str(int(yr)))
    ax.axhline(1.0, color=GREY, lw=1, ls="--")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month", color=GREY, fontsize=9)
    ax.set_ylabel("monthly index (1.0 = year's average)", color=GREY, fontsize=9)
    ax.set_title("Monthly seasonality by year — clean years only",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=5)
    style_ax(ax)
    _save(fig, out)


def plot_monthly_shape_corr(shape_corr: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(shape_corr.values, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(shape_corr)))
    ax.set_yticks(range(len(shape_corr)))
    ax.set_xticklabels([str(int(y)) for y in shape_corr.columns], rotation=45, fontsize=8)
    ax.set_yticklabels([str(int(y)) for y in shape_corr.index], fontsize=8)
    for i in range(len(shape_corr)):
        for j in range(len(shape_corr)):
            ax.text(j, i, f"{shape_corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if shape_corr.values[i, j] > 0.6 else "black")
    ax.set_title("Year-to-year correlation of the 12-month shape",
                 color=NAVY, fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out)
