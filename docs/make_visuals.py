"""Generate infographic assets for the deck (run: uv run python docs/make_visuals.py)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "v2"))

NAVY = "#0F2942"; TEAL = "#2E8BA8"; GOLD = "#E8B547"
GREEN = "#2FA87C"; RED = "#C74B4B"; GREY = "#5F7183"


# ── 1. Boosting: how adding trees closes the gap ──────────────────────────
def boosting_illustration():
    import lightgbm as lgb
    rng = np.random.default_rng(7)
    x = np.linspace(0, 30, 220)
    y = (7000 + 700 * np.sin(x / 30 * 4 * np.pi)          # weekly rhythm
         + 500 * np.sin(x / 30 * 1.2 * np.pi)              # slow drift
         - 2200 * np.exp(-((x - 12) ** 2) / 2.0)           # a holiday trough
         + rng.normal(0, 90, len(x)))
    X = x.reshape(-1, 1)

    stages = [1, 10, 300]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
    for ax, n in zip(axes, stages):
        m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.25, num_leaves=8,
                              min_child_samples=5, verbose=-1, random_state=0)
        m.fit(X, y)
        p = m.predict(X)
        err = np.mean(np.abs(y - p) / y) * 100
        ax.plot(x, y, color="black", lw=1.6, label="Actual", zorder=2)
        ax.plot(x, p, color=TEAL, lw=2.6, label="Model", zorder=3)
        ax.fill_between(x, y, p, color=RED, alpha=0.18, zorder=1)
        ax.set_title(f"{n} tree{'s' if n > 1 else ''}   ·   error {err:.1f}%",
                     fontsize=13, fontweight="bold", color=NAVY)
        ax.set_xticks([]); ax.grid(alpha=0.25, axis="y")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Demand", fontsize=11, color=GREY)
    axes[0].legend(loc="lower left", fontsize=10, framealpha=0.9)
    fig.suptitle("Shaded area = what the model still gets wrong",
                 fontsize=12, color=GREY, y=0.02)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUT / "boosting_stages.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> boosting_stages.png")


# ── 2. Feature composition ────────────────────────────────────────────────
def feature_donut():
    from _shared import load_demand, build_matrix
    mat = build_matrix(load_demand())
    cols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]

    def fam(c):
        if c.startswith(("lag_anchor_",)):        return "Anchor lags"
        if c.startswith("lag_"):                  return "Lags"
        if c.startswith("rolling_"):              return "Rolling stats"
        if c.startswith("ewma_") or "same_dow" in c: return "EWMA / same-DOW"
        if c.startswith("mainland_"):             return "Mainland blocks"
        if ("holiday" in c or c.startswith("is_") and c not in
            ("is_weekend", "is_friday", "is_saturday", "is_sunday",
             "is_month_start", "is_month_end", "is_typhoon_t8plus")
            or any(c.startswith(h) for h in
                   ("CNY", "GoldenWeek", "Labour", "MidAutumn", "DragonBoat"))):
            return "Holiday"
        if c in ("is_typhoon_t8plus",) or c.startswith("res_"): return "External"
        if c in ("dow", "day", "week_of_month", "month", "quarter", "year",
                 "day_of_year", "is_weekend", "is_friday", "is_saturday",
                 "is_sunday", "is_month_start", "is_month_end") or \
           c.endswith(("_sin", "_cos")):          return "Calendar"
        return "Interactions"

    counts = pd.Series([fam(c) for c in cols]).value_counts()
    order = ["Holiday", "Lags", "Rolling stats", "Calendar", "Interactions",
             "EWMA / same-DOW", "Anchor lags", "Mainland blocks", "External"]
    counts = counts.reindex([o for o in order if o in counts.index])
    palette = [GOLD, TEAL, "#4E9CB5", NAVY, "#7A8FA3", "#9CC6D6", "#1E6E8C",
               "#B8860B", GREEN][:len(counts)]

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    wedges, _ = ax.pie(counts.values, colors=palette, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.text(0, 0.10, str(len(cols)), ha="center", va="center",
            fontsize=42, fontweight="bold", color=NAVY)
    ax.text(0, -0.20, "features", ha="center", va="center",
            fontsize=13, color=GREY)
    ax.legend(wedges, [f"{n}   {v}" for n, v in counts.items()],
              loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=11.5)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT / "feature_donut.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> feature_donut.png  ({len(cols)} features)")
    return len(cols)


# ── 3. Model family comparison ────────────────────────────────────────────
def family_bars():
    fams = [("Gradient-boosted trees\n(chosen)", 2.52, GREEN),
            ("Linear — best case", 4.01, GOLD),
            ("Prophet / NeuralProphet", 5.56, RED),
            ("Linear — worst case", 10.65, RED)]
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    names = [f[0] for f in fams]; vals = [f[1] for f in fams]; cols = [f[2] for f in fams]
    yy = np.arange(len(fams))[::-1]
    ax.barh(yy, vals, color=cols, height=0.6)
    for y_, v in zip(yy, vals):
        ax.text(v + 0.2, y_, f"{v:.2f}%", va="center", fontsize=12,
                fontweight="bold", color=NAVY)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=11.5)
    ax.set_xlabel("MAPE (%) — lower is better", fontsize=11, color=GREY)
    ax.set_xlim(0, 12.5)
    ax.grid(alpha=0.25, axis="x")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "family_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> family_bars.png")


# ── 4. Error concentration: where the 3.98% comes from ────────────────────
def error_concentration():
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    segs = [("Ordinary days  (138 of 148)", 3.52, TEAL),
            ("Chinese New Year  (10 of 148)", 6.16, RED)]
    yy = [1, 0]
    for y_, (lab, v, c) in zip(yy, segs):
        ax.barh(y_, v, color=c, height=0.52)
        ax.text(v + 0.12, y_, f"{v:.2f}%", va="center", fontsize=13,
                fontweight="bold", color=NAVY)
    ax.axvline(3.98, color=NAVY, ls="--", lw=2)
    ax.text(3.98, 1.62, "  Blended average 3.98%", color=NAVY,
            fontsize=11.5, fontweight="bold", va="center")
    ax.set_yticks(yy); ax.set_yticklabels([s[0] for s in segs], fontsize=12)
    ax.set_xlabel("MAPE (%) at 1-day lead", fontsize=11, color=GREY)
    ax.set_xlim(0, 7.6); ax.set_ylim(-0.55, 1.95)
    ax.grid(alpha=0.25, axis="x")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "error_concentration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> error_concentration.png")


if __name__ == "__main__":
    print("Generating deck visuals...")
    boosting_illustration()
    family_bars()
    error_concentration()
    feature_donut()
    print("Done.")
