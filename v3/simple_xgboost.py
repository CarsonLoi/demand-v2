"""v2: XGBoost quantile + Tier 1 features."""
from __future__ import annotations

import pickle, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import xgboost as xgb

from _shared import (
    HOLDOUT_DAYS, OUT, build_matrix, compute_metrics, load_demand,
    print_metrics, split_train_test, USE_RESERVATIONS,
)

QUANTILES = [0.10, 0.50, 0.90]


def train_and_predict(mat, train_dates_set, test_dates):
    feature_cols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    train_mat = mat[mat["target_date"].isin(train_dates_set)].dropna(subset=["y"])
    models = {}
    for h in range(1, HOLDOUT_DAYS + 1):
        sub = train_mat[train_mat["horizon"] == h]
        if len(sub) < 30: continue
        X, y = sub[feature_cols], sub["y"]
        for q in QUANTILES:
            m = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=q,
                n_estimators=400, learning_rate=0.03, max_depth=6,
                min_child_weight=1, reg_alpha=0.1, reg_lambda=0.5,
                subsample=0.9, colsample_bytree=0.9,
                tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
            )
            m.fit(X, y); models[(h, q)] = m

    preds = {"target_date": [], "y_true": [], "y_p10": [], "y_p50": [], "y_p90": []}
    date_to_y = dict(zip(mat["target_date"], mat["y"]))
    for i, td in enumerate(test_dates, start=1):
        row = mat[(mat["target_date"] == td) & (mat["horizon"] == i)]
        if row.empty or (i, 0.50) not in models:
            preds["target_date"].append(td); preds["y_true"].append(date_to_y.get(td, np.nan))
            preds["y_p10"].append(np.nan); preds["y_p50"].append(np.nan); preds["y_p90"].append(np.nan)
            continue
        X = row[feature_cols]
        p10 = max(0.0, float(models[(i, 0.10)].predict(X)[0]))
        p50 = max(0.0, float(models[(i, 0.50)].predict(X)[0]))
        p90 = max(p50, float(models[(i, 0.90)].predict(X)[0]))
        p10 = min(p10, p50)
        preds["target_date"].append(td); preds["y_true"].append(date_to_y.get(td, np.nan))
        preds["y_p10"].append(p10); preds["y_p50"].append(p50); preds["y_p90"].append(p90)
    return preds


def main():
    print(f"=== v2 XGBoost (Tier 1 + reservations={USE_RESERVATIONS}) ===\n")
    demand = load_demand()
    train, test = split_train_test(demand)
    mat = build_matrix(demand)
    print(f"  matrix: {len(mat):,} rows x {len(mat.columns)} cols")
    preds = train_and_predict(mat, set(train["date"]), sorted(test["date"].to_list()))
    metrics = compute_metrics(preds["y_true"], preds["y_p50"])
    print_metrics("XGBoost v2", metrics)

    pd.DataFrame(preds).assign(
        abs_error=lambda d: np.abs(d.y_true - d.y_p50),
        pct_error=lambda d: (d.y_true - d.y_p50) / d.y_true,
    ).to_csv(OUT / "simple_xgboost.csv", index=False)
    with open(OUT / "simple_xgboost_preds.pkl", "wb") as f:
        pickle.dump(preds, f)

    test_dates = sorted(test["date"].to_list())
    recent = train.tail(60)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(recent["date"], recent["demand"], "o-", color="black", markersize=3, alpha=0.6)
    ax.plot(test_dates, preds["y_true"], "o-", color="black", markersize=5, label="actual")
    ax.plot(test_dates, preds["y_p50"], "s-", color="C1", markersize=4, label="P50")
    ax.fill_between(test_dates, preds["y_p10"], preds["y_p90"], color="C1", alpha=0.2)
    ax.axvline(train["date"].max(), color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"v2 XGBoost | MAPE={metrics['mape']*100:.2f}%")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT / "simple_xgboost.png", dpi=120, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
