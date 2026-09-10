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


# ── Part A3 ─────────────────────────────────────────────────────────────
def plot_holiday_impact(mult_table: pd.DataFrame, out: Path) -> None:
    hols = sorted(mult_table["holiday"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, name in zip(axes.flat, hols):
        sub = mult_table[mult_table["holiday"] == name].sort_values("year")
        clean = sub[sub["year"].isin(YEAR_COLOURS)]
        ax.bar(clean["year"].astype(str), clean["mult_per_table"],
               color=[YEAR_COLOURS.get(int(y), GREY) for y in clean["year"]])
        ax.axhline(1.0, color=GREY, lw=1, ls="--")
        ax.set_title(name, color=NAVY, fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        style_ax(ax)
    for ax in axes.flat[len(hols):]:
        ax.set_visible(False)
    fig.suptitle("Holiday demand multiplier vs pre-holiday baseline (per table), clean years",
                 color=NAVY, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out)


def plot_holiday_offset_shape(shape_by_holiday: dict, out: Path) -> None:
    names = list(shape_by_holiday)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.4 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        tbl = shape_by_holiday[name]
        for yr in tbl.index:
            ax.plot(tbl.columns, tbl.loc[yr].values, "o-", ms=3, lw=1.5,
                    color=YEAR_COLOURS.get(int(yr), GREY), label=str(int(yr)))
        ax.axhline(1.0, color=GREY, lw=1, ls="--")
        ax.set_title(f"{name} - demand/table by day offset, / pre-holiday baseline",
                     color=NAVY, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("days from anchor", color=GREY, fontsize=8)
        ax.legend(fontsize=7, ncol=5)
        style_ax(ax)
    fig.tight_layout()
    _save(fig, out)


def plot_cny_trough(trough_table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    yrs = [int(y) for y in trough_table.index]
    ax.bar([str(y) for y in yrs], trough_table["min_mult_value"],
           color=[YEAR_COLOURS.get(y, GREY) for y in yrs])
    ax.axhline(1.0, color=GREY, lw=1, ls="--")
    for x, (v, o) in enumerate(zip(trough_table["min_mult_value"], trough_table["min_mult_offset"])):
        ax.text(x, v + 0.01, f"d{int(o)}", ha="center", fontsize=8, color=GREY)
    ax.set_ylabel("min demand/table in CNY d-7..d-1  / baseline", color=GREY, fontsize=9)
    ax.set_title("Pre-CNY trough depth by year - the mid-January failure mode",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    style_ax(ax)
    _save(fig, out)


# ── Part A4 ─────────────────────────────────────────────────────────────
def plot_covid_timeline(timeline: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = range(len(timeline))
    ax.plot(x, timeline["mean_per_table"], "o-", color=NAVY, lw=1.7, ms=3)
    idx = list(timeline.index)
    ax.axvspan(idx.index("2020-01"), idx.index("2022-12"), color=RED, alpha=0.10)
    ax.set_xticks(list(x)[::3])
    ax.set_xticklabels(idx[::3], rotation=45, fontsize=8)
    ax.set_ylabel("mean demand / table", color=GREY, fontsize=9)
    ax.set_title("COVID collapse and recovery - monthly demand per table, 2019-2024",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    style_ax(ax)
    _save(fig, out)


# ── Part B2 ─────────────────────────────────────────────────────────────
def plot_group_corr(group_corr: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(group_corr.values.astype(float), vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(group_corr))); ax.set_yticks(range(len(group_corr)))
    ax.set_xticklabels(group_corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(group_corr.index, fontsize=8)
    for i in range(len(group_corr)):
        for j in range(len(group_corr)):
            v = group_corr.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "black")
    ax.set_title("Mean |correlation| between feature groups", color=NAVY,
                 fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out)


def plot_target_corr(target_corr: pd.DataFrame, out: Path, top: int = 30) -> None:
    s = target_corr["spearman"].dropna()
    top_s = s.head(top).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top_s))))
    ax.barh(range(len(top_s)), top_s.values,
            color=[TEAL if v >= 0 else RED for v in top_s.values])
    ax.set_yticks(range(len(top_s))); ax.set_yticklabels(top_s.index, fontsize=8)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("Spearman correlation with y", color=GREY, fontsize=9)
    ax.set_title(f"Top {top} features by |correlation| with the target",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    style_ax(ax)
    _save(fig, out)


# ── Part B3 ─────────────────────────────────────────────────────────────
def plot_shap_summary(mat, feats, global_table, out: Path, top: int = 25) -> None:
    import eda_lib as L
    as_of = mat["target_date"].max()
    m = L.train_horizon_model(mat, feats, 7, as_of)
    pool = mat[(mat["horizon"] == 7) & (mat["y"].notna())]
    take = pool.sample(min(2000, len(pool)), random_state=0)
    contribs = m.predict(take[feats], pred_contrib=True)[:, :-1]
    sh = pd.DataFrame(contribs, columns=feats)
    order = list(global_table.head(top).index)
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(order) + 1))
    for i, f in enumerate(order[::-1]):
        vals = take[f].to_numpy(dtype="float64")
        rank = pd.Series(vals).rank(pct=True).to_numpy()
        ax.scatter(sh[f], np.full(len(sh), i) + (rank - 0.5) * 0.7,
                   c=rank, cmap="coolwarm", s=6, alpha=0.5, linewidths=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1], fontsize=8)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("SHAP value (impact on predicted patron-hours)", color=GREY, fontsize=9)
    ax.set_title("SHAP summary - horizon 7, top features (colour = feature value)",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    style_ax(ax)
    _save(fig, out)


def plot_shap_waterfall(local_df: pd.DataFrame, meta: pd.DataFrame, label: str, out: Path) -> None:
    top = local_df.head(15).iloc[::-1]
    base = float(meta["base_value"].iloc[0]); pred = float(meta["prediction"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 6))
    running = base
    for i, (feat, r) in enumerate(top.iterrows()):
        ax.barh(i, r["shap"], left=running, color=TEAL if r["shap"] >= 0 else RED)
        running += r["shap"]
    ax.axvline(base, color=GREY, ls="--", lw=1, label=f"base {base:,.0f}")
    ax.axvline(pred, color=NAVY, ls="-", lw=1.4, label=f"prediction {pred:,.0f}")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{f} = {v:,.2f}" for f, v in zip(top.index, top['feature_value'])],
                       fontsize=8)
    ax.set_title(f"Why the {label} forecast landed where it did (horizon 7)",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    style_ax(ax)
    _save(fig, out)
