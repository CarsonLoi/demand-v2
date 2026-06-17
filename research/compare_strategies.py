"""Compare equal-weighted vs (DOW x horizon)-selected hybrid blends.

Two training rounds (each trains all 6 model variants per horizon h=1..28):

  Round 1 (SELECTION):
    Train on demand <= 2025-10-31
    Predict validation slice 2025-11-01 .. 2026-04-30
    For each (DOW, horizon), pick top 2 models by validation MAPE
    -> selection table

  Round 2 (PRODUCTION):
    Train on demand <= 2026-04-30
    Predict holdout 2026-05-01 .. 2026-05-28
    For each holdout row, apply:
      equal:     mean of all 6 model predictions  (baseline = forecast.py --full)
      selection: mean of top-2 picked for that (DOW, horizon)

Outputs per-strategy MAPE/WAPE/RMSE/Bias on the holdout for direct comparison.
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # project root (scripts live in research/)
sys.path.insert(0, str(ROOT / "v2"))

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from _shared import (  # noqa: E402
    HOLDOUT_DAYS, build_matrix, compute_metrics, holiday_mask_from_matrix,
    load_demand, make_sample_weights,
)

MODEL_NAMES = ["lgbm_l2", "lgbm_q", "xgb", "cat", "lgbm_bag", "v3_2stage"]


def _train_lgbm_l2(X, y, w):
    m = lgb.LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.02,
        num_leaves=15, max_depth=5, min_child_samples=5,
        reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1, random_state=123,
    )
    m.fit(X, y, sample_weight=w)
    return m


def _train_lgbm_q(X, y, w):
    m = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03,
        num_leaves=31, max_depth=7, min_child_samples=3,
        reg_alpha=0.1, reg_lambda=0.5, subsample=0.9, colsample_bytree=0.9,
        n_jobs=-1, verbose=-1, random_state=42,
    )
    m.fit(X, y, sample_weight=w)
    return m


def _train_xgb(X, y):
    m = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=0.5,
        n_estimators=400, learning_rate=0.03, max_depth=6,
        min_child_weight=1, reg_alpha=0.1, reg_lambda=0.5,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
    )
    m.fit(X, y)
    return m


def _train_cat(X, y, w):
    m = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function="MAE", random_seed=42, verbose=False,
        allow_writing_files=False,
    )
    m.fit(X, y, sample_weight=w)
    return m


def _train_lgbm_bag(X, y, w):
    preds_func = []
    for seed in [42, 123, 777]:
        m = lgb.LGBMRegressor(
            objective="regression", n_estimators=400, learning_rate=0.02,
            num_leaves=15, max_depth=5, min_child_samples=5,
            reg_alpha=0.5, reg_lambda=1.0,
            subsample=0.85, colsample_bytree=0.85,
            bagging_fraction=0.85, bagging_freq=5,
            n_jobs=-1, verbose=-1, random_state=seed,
        )
        m.fit(X, y, sample_weight=w)
        preds_func.append(m)
    return preds_func   # list of 3 models


def _train_2stage(X, y, w, is_hol):
    bm = _train_lgbm_l2(X, y, w)
    y_base = bm.predict(X)
    residuals = y.values - y_base
    rm = None
    if is_hol.sum() >= 20:
        X_hol = X[is_hol.values]
        r_hol = residuals[is_hol.values]
        w_hol = w[is_hol.values]
        rm = lgb.LGBMRegressor(
            objective="regression", n_estimators=300, learning_rate=0.03,
            num_leaves=7, max_depth=4, min_child_samples=3,
            reg_alpha=0.3, reg_lambda=2.0,
            subsample=0.85, colsample_bytree=0.85,
            n_jobs=-1, verbose=-1, random_state=321,
        )
        rm.fit(X_hol, r_hol, sample_weight=w_hol)
    return bm, rm


def round_train_and_predict(mat, train_dates_set, predict_dates_set, label):
    """For each horizon h, train all 6 models on training rows, predict pred rows.
    Returns dict[model_name] -> DataFrame[target_date, horizon, pred, y]."""
    feature_cols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    train_mat = mat[mat["target_date"].isin(train_dates_set)].dropna(subset=["y"])
    pred_mat_all = mat[mat["target_date"].isin(predict_dates_set)]

    print(f"\n  ROUND: {label}")
    print(f"    train rows: {len(train_mat):,}, predict rows: {len(pred_mat_all):,}")

    out = {n: [] for n in MODEL_NAMES}
    t_round = time.time()

    for h in range(1, HOLDOUT_DAYS + 1):
        sub_train = train_mat[train_mat["horizon"] == h]
        sub_pred = pred_mat_all[pred_mat_all["horizon"] == h]
        if len(sub_train) < 30 or len(sub_pred) == 0:
            continue

        X, y = sub_train[feature_cols], sub_train["y"]
        Xp = sub_pred[feature_cols]
        td = sub_pred["target_date"].values
        y_true = sub_pred["y"].values
        is_hol = holiday_mask_from_matrix(sub_train)
        w = make_sample_weights(sub_train["target_date"], is_holiday=is_hol)

        # 1. LGBM-L2
        m = _train_lgbm_l2(X, y, w)
        for i, p in enumerate(np.maximum(0.0, m.predict(Xp))):
            out["lgbm_l2"].append((td[i], h, p, y_true[i]))

        # 2. LGBM-Q
        m = _train_lgbm_q(X, y, w)
        for i, p in enumerate(np.maximum(0.0, m.predict(Xp))):
            out["lgbm_q"].append((td[i], h, p, y_true[i]))

        # 3. XGBoost
        m = _train_xgb(X, y)
        for i, p in enumerate(np.maximum(0.0, m.predict(Xp))):
            out["xgb"].append((td[i], h, p, y_true[i]))

        # 4. CatBoost
        m = _train_cat(X, y, w)
        for i, p in enumerate(np.maximum(0.0, m.predict(Xp))):
            out["cat"].append((td[i], h, p, y_true[i]))

        # 5. LGBM bagged
        ms = _train_lgbm_bag(X, y, w)
        preds = np.mean([mm.predict(Xp) for mm in ms], axis=0)
        for i, p in enumerate(np.maximum(0.0, preds)):
            out["lgbm_bag"].append((td[i], h, p, y_true[i]))

        # 6. v3 2-stage
        bm, rm = _train_2stage(X, y, w, is_hol)
        y_base = bm.predict(Xp)
        is_hol_p = holiday_mask_from_matrix(sub_pred)
        if rm is not None:
            residual = np.where(is_hol_p.values, rm.predict(Xp), 0.0)
        else:
            residual = np.zeros(len(Xp))
        for i, p in enumerate(np.maximum(0.0, y_base + residual)):
            out["v3_2stage"].append((td[i], h, p, y_true[i]))

        if h % 4 == 0 or h == HOLDOUT_DAYS:
            elapsed = time.time() - t_round
            print(f"    h={h:2d}/{HOLDOUT_DAYS}: cumulative {elapsed:.0f}s")

    return {
        n: pd.DataFrame(out[n], columns=["target_date", "horizon", "pred", "y"])
        for n in MODEL_NAMES
    }


def build_selection_table(val_preds):
    """For each (DOW, horizon), pick top 2 models by validation MAPE."""
    selection = {}
    diagnostic = []
    for h in range(1, HOLDOUT_DAYS + 1):
        for dow in range(7):
            mapes = {}
            for n, df in val_preds.items():
                sub = df[
                    (df["horizon"] == h)
                    & (pd.to_datetime(df["target_date"]).dt.weekday == dow)
                ]
                sub = sub.dropna(subset=["pred", "y"])
                if len(sub) < 3 or sub["y"].abs().sum() == 0:
                    continue
                mape = (np.abs(sub["y"] - sub["pred"]) / sub["y"].clip(lower=1)).mean()
                mapes[n] = mape
            if not mapes:
                continue
            sorted_models = sorted(mapes.items(), key=lambda kv: kv[1])
            top2 = [m[0] for m in sorted_models[:2]]
            selection[(dow, h)] = top2
            diagnostic.append({
                "dow": dow, "horizon": h,
                "best1": sorted_models[0][0], "best1_mape": sorted_models[0][1],
                "best2": sorted_models[1][0] if len(sorted_models) > 1 else "",
                "best2_mape": sorted_models[1][1] if len(sorted_models) > 1 else np.nan,
            })
    return selection, pd.DataFrame(diagnostic)


def main() -> int:
    print("=== Strategy comparison ===\n")
    demand = load_demand()
    print(f"  Data: {demand.date.min().date()} .. {demand.date.max().date()} "
          f"({len(demand)} days)")

    SELECTION_TRAIN_END = pd.Timestamp("2025-10-31")
    VAL_START = pd.Timestamp("2025-11-01")
    VAL_END = pd.Timestamp("2026-04-30")
    HOLDOUT_START = pd.Timestamp("2026-05-01")
    HOLDOUT_END = pd.Timestamp("2026-05-28")

    print(f"\n  Selection train: <= {SELECTION_TRAIN_END.date()}")
    print(f"  Validation:      {VAL_START.date()} .. {VAL_END.date()}")
    print(f"  Production train: <= {VAL_END.date()}")
    print(f"  Holdout test:    {HOLDOUT_START.date()} .. {HOLDOUT_END.date()}")

    print("\n  Building feature matrix...")
    t0 = time.time()
    mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS)
    print(f"  done in {time.time()-t0:.1f}s ({len(mat):,} rows)")

    sel_train = set(d for d in demand.date if d <= SELECTION_TRAIN_END)
    val_dates = set(d for d in demand.date if VAL_START <= d <= VAL_END)
    prod_train = set(d for d in demand.date if d <= VAL_END)
    holdout_dates = set(d for d in demand.date if HOLDOUT_START <= d <= HOLDOUT_END)

    # Round 1: selection
    t1 = time.time()
    val_preds = round_train_and_predict(mat, sel_train, val_dates, "SELECTION")
    print(f"  Round 1 done in {(time.time()-t1)/60:.1f} min")

    selection_table, diag = build_selection_table(val_preds)
    print(f"\n  Selection table: {len(selection_table)} (DOW, horizon) cells")
    print("  Top 1-pick frequency across all cells:")
    pick_freq = diag["best1"].value_counts().to_dict()
    for k, v in pick_freq.items():
        print(f"    {k:10s} {v:3d} cells")

    # Round 2: production
    t2 = time.time()
    test_preds = round_train_and_predict(mat, prod_train, holdout_dates, "PRODUCTION")
    print(f"  Round 2 done in {(time.time()-t2)/60:.1f} min")

    # Build wide combined frame
    combined = None
    for n, df in test_preds.items():
        c = df.rename(columns={"pred": n})[["target_date", "horizon", n, "y"]]
        if combined is None:
            combined = c
        else:
            combined = combined.merge(c.drop(columns=["y"]),
                                       on=["target_date", "horizon"], how="outer")

    # Apply strategies on the natural test-date schedule (test_dates[h-1] = holdout day h)
    test_dates_sorted = sorted(holdout_dates)
    pairs = pd.DataFrame(
        {"target_date": test_dates_sorted, "horizon": list(range(1, len(test_dates_sorted) + 1))}
    )
    result = pairs.merge(combined, on=["target_date", "horizon"], how="left")

    # Equal-weighted
    result["equal"] = result[MODEL_NAMES].mean(axis=1)

    # Selection-based
    sel_preds = []
    for _, row in result.iterrows():
        dow = pd.Timestamp(row["target_date"]).weekday()
        h = int(row["horizon"])
        key = (dow, h)
        if key in selection_table:
            top2 = selection_table[key]
            vals = [row[n] for n in top2 if not pd.isna(row[n])]
            sel_preds.append(float(np.mean(vals)) if vals else float(row[MODEL_NAMES].mean()))
        else:
            sel_preds.append(float(row[MODEL_NAMES].mean()))
    result["selection"] = sel_preds

    print("\n=== Holdout MAPE on May 1-28 ===\n")
    print(f"  {'strategy':18s}  MAPE      WAPE      RMSE     Bias")
    print(f"  {'-'*55}")
    for name in MODEL_NAMES + ["equal", "selection"]:
        m = compute_metrics(result["y"], result[name])
        print(f"  {name:18s} {m['mape']*100:6.2f}%   {m['wape']*100:6.2f}%   "
              f"{m['rmse']:6.0f}   {m['bias']:+6.0f}")

    out_dir = ROOT / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out_dir / "model_selection_diagnostic.csv", index=False)
    result.to_csv(out_dir / "strategy_comparison_predictions.csv", index=False)
    print(f"\n  -> {out_dir/'model_selection_diagnostic.csv'}")
    print(f"  -> {out_dir/'strategy_comparison_predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
