"""Quick honest probe: predict demand directly vs predict ratio-to-baseline.

ratio target = y / baseline, where baseline = same_dow_mean_4w (horizon-safe
mean of the last 4 same-weekday values), with fallback to rolling_mean_28.
forecast = predicted_ratio * baseline.

Single LightGBM-L2 per horizon (fast) on two holdout windows, so we can see
whether the ratio formulation helps before scaling to the full ensemble.
"""
from __future__ import annotations
import logging, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("lightgbm").setLevel(logging.ERROR)
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent  # project root (scripts live in research/)
sys.path.insert(0, str(ROOT / "v2"))
import lightgbm as lgb
from _shared import (HOLDOUT_DAYS, build_matrix, compute_metrics,
                     holiday_mask_from_matrix, load_demand, make_sample_weights)


def lgbm(X, y, w):
    m = lgb.LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.02,
        num_leaves=15, max_depth=5, min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=123)
    m.fit(X, y, sample_weight=w); return m


def baseline_col(df):
    """Horizon-safe per-row baseline: same_dow_mean_4w, fallback rolling_mean_28,
    fallback rolling_mean_7."""
    b = df["same_dow_mean_4w"].copy()
    b = b.fillna(df["rolling_mean_28"]).fillna(df["rolling_mean_7"])
    return b


def run(mat, train_dates, hold_dates, mode):
    """mode in {'abs','ratio'}. Returns DataFrame[target_date,horizon,pred,y]."""
    fcols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    tr = mat[mat["target_date"].isin(train_dates)].dropna(subset=["y"])
    pr = mat[mat["target_date"].isin(hold_dates)]
    rows = []
    for h in range(1, HOLDOUT_DAYS + 1):
        st = tr[tr["horizon"] == h]; sp = pr[pr["horizon"] == h]
        if len(st) < 30 or len(sp) == 0:
            continue
        X = st[fcols]; Xp = sp[fcols]
        is_hol = holiday_mask_from_matrix(st)
        w = make_sample_weights(st["target_date"], is_holiday=is_hol)
        if mode == "abs":
            m = lgbm(X, st["y"], w)
            pred = np.maximum(0, m.predict(Xp))
        else:
            b_tr = baseline_col(st)
            ok = b_tr.notna() & (b_tr > 0) & st["y"].notna()
            ratio = (st.loc[ok, "y"] / b_tr[ok]).clip(0.3, 3.0)
            m = lgbm(X[ok.values], ratio, w[ok.values])
            b_pr = baseline_col(sp).values
            r = m.predict(Xp)
            pred = np.maximum(0, r * b_pr)
            # where baseline missing, fall back to abs model trained above? keep NaN
        sp2 = sp.reset_index(drop=True)
        for i in range(len(sp2)):
            rows.append((sp2.loc[i, "target_date"], h, float(pred[i]), sp2.loc[i, "y"]))
    return pd.DataFrame(rows, columns=["target_date", "horizon", "pred", "y"])


def natural(df, hold_dates):
    hd = sorted(hold_dates)
    sched = pd.DataFrame({"target_date": hd, "horizon": range(1, len(hd) + 1)})
    return sched.merge(df, on=["target_date", "horizon"], how="left")


def main():
    demand = load_demand()
    mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS)
    for hold_month in ["2026-05", "2025-10"]:
        hm = pd.Timestamp(hold_month + "-01")
        HS, HE = hm, hm + pd.Timedelta(days=27)
        train = set(d for d in demand.date if d < HS)
        hold = set(d for d in demand.date if HS <= d <= HE)
        print(f"\n=== {hold_month} ===")
        for mode in ["abs", "ratio"]:
            r = natural(run(mat, train, hold, mode), hold)
            m = compute_metrics(r["y"], r["pred"])
            print(f"  lgbm_l2 [{mode:5s}]  MAPE={m['mape']*100:5.2f}%  "
                  f"WAPE={m['wape']*100:5.2f}%  RMSE={m['rmse']:5.0f}  Bias={m['bias']:+5.0f}")


if __name__ == "__main__":
    sys.exit(main())
