"""v2: LightGBM L2 + holiday upweight + Tier 1 features."""
from __future__ import annotations

import logging, pickle, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("lightgbm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import lightgbm as lgb

from _shared import (
    HOLDOUT_DAYS, OUT, build_matrix, compute_metrics, holiday_mask_from_matrix,
    load_demand, make_sample_weights, print_metrics, split_train_test, USE_RESERVATIONS,
)


def train_and_predict(mat, train_dates_set, test_dates):
    feature_cols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    train_mat = mat[mat["target_date"].isin(train_dates_set)].dropna(subset=["y"])

    models = {}; residuals = {}
    for h in range(1, HOLDOUT_DAYS + 1):
        sub = train_mat[train_mat["horizon"] == h]
        if len(sub) < 30: continue
        X, y = sub[feature_cols], sub["y"]
        is_hol = holiday_mask_from_matrix(sub)
        weights = make_sample_weights(sub["target_date"], is_holiday=is_hol)
        m = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500, learning_rate=0.02, num_leaves=15, max_depth=5,
            min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, verbose=-1, random_state=123,
        )
        m.fit(X, y, sample_weight=weights)
        models[h] = m
        residuals[h] = (y - m.predict(X)).values

    preds = {"target_date": [], "y_true": [], "y_p10": [], "y_p50": [], "y_p90": []}
    date_to_y = dict(zip(mat["target_date"], mat["y"]))
    for i, td in enumerate(test_dates, start=1):
        row = mat[(mat["target_date"] == td) & (mat["horizon"] == i)]
        if row.empty or i not in models:
            preds["target_date"].append(td); preds["y_true"].append(date_to_y.get(td, np.nan))
            preds["y_p10"].append(np.nan); preds["y_p50"].append(np.nan); preds["y_p90"].append(np.nan)
            continue
        X = row[feature_cols]
        p50 = max(0.0, float(models[i].predict(X)[0]))
        res = residuals.get(i, np.array([0.0]))
        p10 = max(0.0, p50 + float(np.quantile(res, 0.10)))
        p90 = max(p50, p50 + float(np.quantile(res, 0.90)))
        preds["target_date"].append(td); preds["y_true"].append(date_to_y.get(td, np.nan))
        preds["y_p10"].append(p10); preds["y_p50"].append(p50); preds["y_p90"].append(p90)
    return preds, models, feature_cols


def main():
    print(f"=== v2 LightGBM-L2 (Tier 1 + reservations={USE_RESERVATIONS}) ===\n")
    demand = load_demand()
    train, test = split_train_test(demand)
    print(f"  train: {len(train)} days,  test: {len(test)} days")

    mat = build_matrix(demand)
    print(f"  matrix: {len(mat):,} rows x {len(mat.columns)} cols")

    print(f"\n  training {HOLDOUT_DAYS} LightGBM models...")
    preds, models, feature_cols = train_and_predict(
        mat, set(train["date"]), sorted(test["date"].to_list())
    )
    metrics = compute_metrics(preds["y_true"], preds["y_p50"])
    print_metrics("LightGBM-L2 v2", metrics)

    pd.DataFrame(preds).assign(
        abs_error=lambda d: np.abs(d.y_true - d.y_p50),
        pct_error=lambda d: (d.y_true - d.y_p50) / d.y_true,
    ).to_csv(OUT / "simple_lightgbm_l2.csv", index=False)
    with open(OUT / "simple_lightgbm_l2_preds.pkl", "wb") as f:
        pickle.dump(preds, f)
    # Save the models for feature-importance plotting
    with open(OUT / "simple_lightgbm_l2_models.pkl", "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols}, f)

    test_dates = sorted(test["date"].to_list())
    recent = train.tail(60)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(recent["date"], recent["demand"], "o-", color="black", markersize=3, alpha=0.6,
            label="actual (train, trailing 60d)")
    ax.plot(test_dates, preds["y_true"], "o-", color="black", markersize=5, label="actual")
    ax.plot(test_dates, preds["y_p50"], "s-", color="C4", markersize=4, label="P50")
    ax.fill_between(test_dates, preds["y_p10"], preds["y_p90"], color="C4", alpha=0.2)
    ax.axvline(train["date"].max(), color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"v2 LightGBM-L2 | MAPE={metrics['mape']*100:.2f}%  (reservations={USE_RESERVATIONS})")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT / "simple_lightgbm_l2.png", dpi=120, bbox_inches="tight"); plt.close()
    return metrics["mape"]


if __name__ == "__main__":
    main()
