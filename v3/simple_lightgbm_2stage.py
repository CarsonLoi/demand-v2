"""v3: Two-stage residual model (Point 5).

Stage 1: train LightGBM on ALL training days → baseline prediction y_base
Stage 2: train a SECOND LightGBM on holiday-window training days only,
         predicting (y - y_base) — the holiday residual

At inference:
    if test day is in any holiday window:
        final = y_base + residual_predicted
    else:
        final = y_base

The residual model focuses entirely on "what's different about holidays"
without diluting its signal across normal days.
"""
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


def train_baseline(X, y, weights):
    m = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500, learning_rate=0.02, num_leaves=15, max_depth=5,
        min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1, random_state=123,
    )
    m.fit(X, y, sample_weight=weights)
    return m


def train_residual(X_hol, residuals_hol, weights_hol):
    """Trained on holiday-window rows only."""
    if len(X_hol) < 20:
        return None
    m = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=4,
        min_child_samples=3, reg_alpha=0.3, reg_lambda=2.0,
        subsample=0.85, colsample_bytree=0.85,
        n_jobs=-1, verbose=-1, random_state=321,
    )
    m.fit(X_hol, residuals_hol, sample_weight=weights_hol)
    return m


def train_and_predict(mat, train_dates_set, test_dates):
    feature_cols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    train_mat = mat[mat["target_date"].isin(train_dates_set)].dropna(subset=["y"])

    baseline_models = {}
    residual_models = {}
    baseline_resid_quantiles = {}  # for interval estimation

    for h in range(1, HOLDOUT_DAYS + 1):
        sub = train_mat[train_mat["horizon"] == h]
        if len(sub) < 30: continue
        X, y = sub[feature_cols], sub["y"]
        is_hol = holiday_mask_from_matrix(sub)
        weights = make_sample_weights(sub["target_date"], is_holiday=is_hol)

        # Stage 1: baseline on ALL days
        bm = train_baseline(X, y, weights)
        baseline_models[h] = bm
        y_base_pred = bm.predict(X)
        residuals_full = y.values - y_base_pred
        baseline_resid_quantiles[h] = (
            float(np.quantile(residuals_full, 0.10)),
            float(np.quantile(residuals_full, 0.90)),
        )

        # Stage 2: residual on holiday days only
        X_hol = X[is_hol.values]
        residuals_hol = pd.Series(residuals_full, index=X.index)[is_hol.values]
        weights_hol = make_sample_weights(sub.loc[is_hol.values, "target_date"],
                                            is_holiday=pd.Series([True]*int(is_hol.sum()),
                                                                  index=X_hol.index))
        rm = train_residual(X_hol, residuals_hol, weights_hol)
        if rm is not None:
            residual_models[h] = rm

    preds = {"target_date": [], "y_true": [], "y_p10": [], "y_p50": [], "y_p90": []}
    date_to_y = dict(zip(mat["target_date"], mat["y"]))
    for i, td in enumerate(test_dates, start=1):
        row = mat[(mat["target_date"] == td) & (mat["horizon"] == i)]
        if row.empty or i not in baseline_models:
            preds["target_date"].append(td); preds["y_true"].append(date_to_y.get(td, np.nan))
            preds["y_p10"].append(np.nan); preds["y_p50"].append(np.nan); preds["y_p90"].append(np.nan)
            continue
        X = row[feature_cols]
        y_base = float(baseline_models[i].predict(X)[0])

        # Check if test day is in any holiday window
        is_hol_row = bool(holiday_mask_from_matrix(row).values[0])
        residual = 0.0
        if is_hol_row and i in residual_models:
            residual = float(residual_models[i].predict(X)[0])
        p50 = max(0.0, y_base + residual)

        q10, q90 = baseline_resid_quantiles.get(i, (0.0, 0.0))
        p10 = max(0.0, p50 + q10)
        p90 = max(p50, p50 + q90)

        preds["target_date"].append(td); preds["y_true"].append(date_to_y.get(td, np.nan))
        preds["y_p10"].append(p10); preds["y_p50"].append(p50); preds["y_p90"].append(p90)
    return preds


def main():
    print(f"=== v3 LightGBM 2-stage (baseline + holiday residual) ===\n")
    demand = load_demand()
    train, test = split_train_test(demand)
    print(f"  train: {len(train)} days,  test: {len(test)} days")

    mat = build_matrix(demand)
    print(f"  matrix: {len(mat):,} rows x {len(mat.columns)} cols")

    print(f"\n  training {HOLDOUT_DAYS} baseline + residual models...")
    preds = train_and_predict(mat, set(train["date"]), sorted(test["date"].to_list()))
    metrics = compute_metrics(preds["y_true"], preds["y_p50"])
    print_metrics("LightGBM 2-stage", metrics)

    pd.DataFrame(preds).assign(
        abs_error=lambda d: np.abs(d.y_true - d.y_p50),
        pct_error=lambda d: (d.y_true - d.y_p50) / d.y_true,
    ).to_csv(OUT / "simple_lightgbm_2stage.csv", index=False)
    with open(OUT / "simple_lightgbm_2stage_preds.pkl", "wb") as f:
        pickle.dump(preds, f)

    test_dates = sorted(test["date"].to_list())
    recent = train.tail(60)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(recent["date"], recent["demand"], "o-", color="black", markersize=3, alpha=0.6,
            label="actual (train, trailing 60d)")
    ax.plot(test_dates, preds["y_true"], "o-", color="black", markersize=5, label="actual")
    ax.plot(test_dates, preds["y_p50"], "s-", color="C7", markersize=4, label="P50 (2-stage)")
    ax.fill_between(test_dates, preds["y_p10"], preds["y_p90"], color="C7", alpha=0.2)
    ax.axvline(train["date"].max(), color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"v3 LightGBM 2-stage | MAPE={metrics['mape']*100:.2f}%")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT / "simple_lightgbm_2stage.png", dpi=120, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
