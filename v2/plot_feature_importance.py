"""Feature importance chart from the LightGBM-L2 v2 models.

Loads the trained models saved by simple_lightgbm_l2.py and aggregates
importance across all 28 horizons. Produces:
  - output/feature_importance_top30.png  — top 30 features bar chart
  - output/feature_importance_by_horizon.png  — heatmap of top features × horizons
  - output/feature_importance.csv  — all features with mean/std/max importance

Run AFTER simple_lightgbm_l2.py:
  uv run python v2/plot_feature_importance.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _shared import OUT

MODELS_PKL = OUT / "simple_lightgbm_l2_models.pkl"


def main():
    if not MODELS_PKL.exists():
        raise FileNotFoundError(
            f"{MODELS_PKL} not found — run simple_lightgbm_l2.py first."
        )

    with open(MODELS_PKL, "rb") as f:
        bundle = pickle.load(f)
    models = bundle["models"]
    feature_cols = bundle["feature_cols"]

    # Build importance matrix: rows=features, cols=horizons
    horizons = sorted(models.keys())
    imp = np.zeros((len(feature_cols), len(horizons)))
    for j, h in enumerate(horizons):
        m = models[h]
        imp[:, j] = m.feature_importances_

    df = pd.DataFrame(imp, index=feature_cols, columns=[f"h={h}" for h in horizons])
    df["mean"] = df.mean(axis=1)
    df["std"] = df.iloc[:, :-1].std(axis=1)
    df["max"] = df.iloc[:, :-2].max(axis=1)
    df = df.sort_values("mean", ascending=False)
    df.to_csv(OUT / "feature_importance.csv")
    print(f"  -> {OUT / 'feature_importance.csv'}")

    # === Chart 1: Top 30 bar chart (mean importance across horizons) ===
    top = df.head(30).iloc[::-1]  # reverse for horizontal bar (top on top)
    colors = []
    for feat in top.index:
        if any(k in feat for k in ("mainland_", "block_")):
            colors.append("#d62728")          # red: new Tier 1 features
        elif "Labour" in feat or "CNY" in feat or "GoldenWeek" in feat or "MidAutumn" in feat \
                or "is_New" in feat or "is_Christmas" in feat or "is_Easter" in feat \
                or "is_ChingMing" in feat or "holiday" in feat:
            colors.append("#ff7f0e")          # orange: holiday features
        elif "lag" in feat or "rolling" in feat or "ewma" in feat or "same_dow" in feat \
                or "yoy" in feat:
            colors.append("#2ca02c")          # green: temporal lag features
        elif "dow" in feat or "month" in feat or "day" in feat or "week" in feat \
                or "year" in feat or "weekend" in feat or "friday" in feat or "saturday" in feat \
                or "sunday" in feat or "doy" in feat or "quarter" in feat:
            colors.append("#1f77b4")          # blue: calendar features
        elif "res_" in feat:
            colors.append("#9467bd")          # purple: reservation features
        else:
            colors.append("#7f7f7f")          # grey: other / interactions

    fig, ax = plt.subplots(figsize=(11, 12))
    ax.barh(range(len(top)), top["mean"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("Mean importance (averaged across 28 horizons)")
    ax.set_title("Top 30 Feature Importance — LightGBM-L2 v2 (Tier 1 enhanced)")
    ax.grid(alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#d62728", label="Mainland holiday block (NEW Tier 1)"),
        Patch(facecolor="#ff7f0e", label="Holiday flags"),
        Patch(facecolor="#2ca02c", label="Temporal lags / rolling / EWMA"),
        Patch(facecolor="#1f77b4", label="Calendar (DOW, month, day, ...)"),
        Patch(facecolor="#9467bd", label="Reservation (would appear if toggled on)"),
        Patch(facecolor="#7f7f7f", label="Interactions / other"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance_top30.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUT / 'feature_importance_top30.png'}")

    # === Chart 2: Heatmap top 20 × horizon ===
    top20 = df.head(20).iloc[:, :-3]  # drop mean/std/max columns
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(top20.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_xticks(range(len(top20.columns)))
    ax.set_xticklabels(top20.columns, rotation=90, fontsize=7)
    ax.set_title("Top 20 Feature Importance × Horizon — LightGBM-L2 v2")
    ax.set_xlabel("Horizon (h=1 .. h=28)")
    cbar = plt.colorbar(im, ax=ax, label="Importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance_by_horizon.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUT / 'feature_importance_by_horizon.png'}")

    # Print top 20 to console
    print(f"\n=== Top 20 features (by mean importance) ===")
    print(df.head(20)[["mean", "std", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
