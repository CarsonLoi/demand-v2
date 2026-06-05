"""v2: LightGBM Bagged (15 seeds + col subsample) + holiday upweight + Tier 1."""
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

N_BAGS = 15
SEEDS = list(range(N_BAGS))


def train_and_predict(mat, train_dates_set, test_dates):
    feature_cols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    train_mat = mat[mat["target_date"].isin(train_dates_set)].dropna(subset=["y"])

    per_h_pred = {h: [] for h in range(1, HOLDOUT_DAYS + 1)}
    residuals_all = {h: [] for h in range(1, HOLDOUT_DAYS + 1)}

    for h in range(1, HOLDOUT_DAYS + 1):
        sub = train_mat[train_mat["horizon"] == h]
        if len(sub) < 30: continue
        X_all, y = sub[feature_cols], sub["y"]
        is_hol = holiday_mask_from_matrix(sub)
        weights = make_sample_weights(sub["target_date"], is_holiday=is_hol)

        test_row = None
        for i, td in enumerate(test_dates, start=1):
            if i != h: continue
            test_row = mat[(mat["target_date"] == td) & (mat["horizon"] == h)]
        if test_row is None or test_row.empty: continue
        X_test_all = test_row[feature_cols]

        rng = np.random.default_rng(42 + h)
        for seed in SEEDS:
            keep_n = int(0.85 * len(feature_cols))
            sub_cols = rng.choice(feature_cols, size=keep_n, replace=False)
            X_sub = X_all[sub_cols]; X_test_sub = X_test_all[sub_cols]
            m = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=600, learning_rate=0.02, num_leaves=15, max_depth=5,
                min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
                subsample=0.85, colsample_bytree=1.0,
                bagging_fraction=0.85, bagging_freq=5,
                n_jobs=-1, verbose=-1, random_state=seed,
            )
            m.fit(X_sub, y, sample_weight=weights)
            per_h_pred[h].append(float(m.predict(X_test_sub)[0]))
            residuals_all[h].append((y - m.predict(X_sub)).values)

    preds = {"target_date": [], "y_true": [], "y_p10": [], "y_p50": [], "y_p90": []}
    date_to_y = dict(zip(mat["target_date"], mat["y"]))
    for i, td in enumerate(test_dates, start=1):
        if i not in per_h_pred or len(per_h_pred[i]) == 0:
            preds["target_date"].append(td); preds["y_true"].append(date_to_y.get(td, np.nan))
            preds["y_p10"].append(np.nan); preds["y_p50"].append(np.nan); preds["y_p90"].append(np.nan)
            continue
        bag = np.array(per_h_pred[i])
        p50 = max(0.0, float(np.mean(bag)))
        bag_std = float(np.std(bag))
        res = np.concatenate(residuals_all[i])
        p10 = max(0.0, p50 + float(np.quantile(res, 0.10)) - bag_std)
        p90 = max(p50, p50 + float(np.quantile(res, 0.90)) + bag_std)
        preds["target_date"].append(td); preds["y_true"].append(date_to_y.get(td, np.nan))
        preds["y_p10"].append(p10); preds["y_p50"].append(p50); preds["y_p90"].append(p90)
    return preds


def main():
    print(f"=== v2 LightGBM-Bagged x{N_BAGS} (Tier 1 + reservations={USE_RESERVATIONS}) ===\n")
    demand = load_demand()
    train, test = split_train_test(demand)
    mat = build_matrix(demand)
    print(f"  matrix: {len(mat):,} rows x {len(mat.columns)} cols")
    print(f"\n  training {HOLDOUT_DAYS} x {N_BAGS} = {HOLDOUT_DAYS * N_BAGS} models...")
    preds = train_and_predict(mat, set(train["date"]), sorted(test["date"].to_list()))
    metrics = compute_metrics(preds["y_true"], preds["y_p50"])
    print_metrics("LightGBM-Bagged v2", metrics)

    pd.DataFrame(preds).assign(
        abs_error=lambda d: np.abs(d.y_true - d.y_p50),
        pct_error=lambda d: (d.y_true - d.y_p50) / d.y_true,
    ).to_csv(OUT / "simple_lightgbm_bagged.csv", index=False)
    with open(OUT / "simple_lightgbm_bagged_preds.pkl", "wb") as f:
        pickle.dump(preds, f)

    test_dates = sorted(test["date"].to_list())
    recent = train.tail(60)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(recent["date"], recent["demand"], "o-", color="black", markersize=3, alpha=0.6)
    ax.plot(test_dates, preds["y_true"], "o-", color="black", markersize=5, label="actual")
    ax.plot(test_dates, preds["y_p50"], "s-", color="C5", markersize=4, label="P50")
    ax.fill_between(test_dates, preds["y_p10"], preds["y_p90"], color="C5", alpha=0.2)
    ax.axvline(train["date"].max(), color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"v2 LightGBM-Bagged | MAPE={metrics['mape']*100:.2f}%")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT / "simple_lightgbm_bagged.png", dpi=120, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
