"""Honest holiday-anchored correction.

Root cause found: base models under-predict fixed-date holiday spikes because
they regress toward surrounding lower-demand days and under-use the year-ago
signal. For holiday-window days, blend the model prediction with an anchor:

    anchor = lag_365(target) * recent_YoY_growth_scalar

Both inputs are known before the forecast (last year's same-date demand, and
trailing YoY growth from data before the forecast) -> fully honest.

The blend weight alpha is calibrated on the VALIDATION window's holidays
(CNY, New Year, Ching Ming all fall in Nov-Apr), then applied to the holdout's
holidays (Labour Day for May).

Saves base-model predictions to data/derived/ so blend experiments are cheap.

Usage:
    uv run python holiday_anchor.py --holdout 2026-05
"""
from __future__ import annotations

import argparse
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
    HOLDOUT_DAYS, HOLIDAY_ANCHORS, HOLIDAY_WINDOWS, build_matrix,
    compute_metrics, holiday_mask_from_matrix, load_demand, make_sample_weights,
)

MODEL_NAMES = ["lgbm_l2", "lgbm_q", "xgb", "cat", "lgbm_bag", "v3_2stage"]
DERIVED = ROOT / "data" / "derived"


def _m_lgbm_l2(X, y, w):
    m = lgb.LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.02,
        num_leaves=15, max_depth=5, min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=123)
    m.fit(X, y, sample_weight=w); return m

def _m_lgbm_q(X, y, w):
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03,
        num_leaves=31, max_depth=7, min_child_samples=3, reg_alpha=0.1, reg_lambda=0.5,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1, verbose=-1, random_state=42)
    m.fit(X, y, sample_weight=w); return m

def _m_xgb(X, y):
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
        n_estimators=400, learning_rate=0.03, max_depth=6, min_child_weight=1,
        reg_alpha=0.1, reg_lambda=0.5, subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", n_jobs=-1, random_state=42, verbosity=0)
    m.fit(X, y); return m

def _m_cat(X, y, w):
    m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function="MAE", random_seed=42, verbose=False, allow_writing_files=False)
    m.fit(X, y, sample_weight=w); return m

def _m_bag(X, y, w):
    ms = []
    for seed in [42, 123, 777]:
        m = lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=0.02,
            num_leaves=15, max_depth=5, min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
            subsample=0.85, colsample_bytree=0.85, bagging_fraction=0.85, bagging_freq=5,
            n_jobs=-1, verbose=-1, random_state=seed)
        m.fit(X, y, sample_weight=w); ms.append(m)
    return ms

def _m_2stage(X, y, w, is_hol):
    bm = _m_lgbm_l2(X, y, w)
    resid = y.values - bm.predict(X)
    rm = None
    if is_hol.sum() >= 20:
        rm = lgb.LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.03,
            num_leaves=7, max_depth=4, min_child_samples=3, reg_alpha=0.3, reg_lambda=2.0,
            subsample=0.85, colsample_bytree=0.85, n_jobs=-1, verbose=-1, random_state=321)
        rm.fit(X[is_hol.values], resid[is_hol.values], sample_weight=w[is_hol.values])
    return bm, rm


def round_predict(mat, train_dates, predict_dates, label):
    fcols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    tr = mat[mat["target_date"].isin(train_dates)].dropna(subset=["y"])
    pr = mat[mat["target_date"].isin(predict_dates)]
    print(f"  ROUND {label}: train {len(tr):,}, predict {len(pr):,}")
    rows = []
    t0 = time.time()
    for h in range(1, HOLDOUT_DAYS + 1):
        st = tr[tr["horizon"] == h]; sp = pr[pr["horizon"] == h]
        if len(st) < 30 or len(sp) == 0:
            continue
        X, y = st[fcols], st["y"]; Xp = sp[fcols]
        is_hol = holiday_mask_from_matrix(st)
        w = make_sample_weights(st["target_date"], is_holiday=is_hol)
        preds = {}
        m = _m_lgbm_l2(X, y, w); preds["lgbm_l2"] = np.maximum(0, m.predict(Xp))
        m = _m_lgbm_q(X, y, w);  preds["lgbm_q"]  = np.maximum(0, m.predict(Xp))
        m = _m_xgb(X, y);        preds["xgb"]     = np.maximum(0, m.predict(Xp))
        m = _m_cat(X, y, w);     preds["cat"]     = np.maximum(0, m.predict(Xp))
        ms = _m_bag(X, y, w);    preds["lgbm_bag"]= np.maximum(0, np.mean([mm.predict(Xp) for mm in ms], axis=0))
        bm, rm = _m_2stage(X, y, w, is_hol)
        is_hol_p = holiday_mask_from_matrix(sp)
        resid = np.where(is_hol_p.values, rm.predict(Xp), 0.0) if rm is not None else np.zeros(len(Xp))
        preds["v3_2stage"] = np.maximum(0, bm.predict(Xp) + resid)
        sp2 = sp.reset_index(drop=True)
        for i in range(len(sp2)):
            row = {"target_date": sp2.loc[i, "target_date"], "horizon": h,
                   "y": sp2.loc[i, "y"], "lag_365": sp2.loc[i, "lag_365"],
                   "is_hol": int(is_hol_p.values[i])}
            for n in MODEL_NAMES:
                row[n] = preds[n][i]
            rows.append(row)
        if h % 7 == 0 or h == HOLDOUT_DAYS:
            print(f"    h={h:2d}: {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    df["dow"] = pd.to_datetime(df["target_date"]).dt.weekday
    return df


FIXED_HOLIDAYS = {"NewYear", "Christmas", "ChingMing", "Labour", "GoldenWeek"}


def yoy_growth(demand, before_date, window=28):
    """Trailing-window YoY growth ratio using only data strictly before
    `before_date`."""
    d = demand[demand.date < before_date]
    end = d.date.max()
    recent = d[d.date > end - pd.Timedelta(days=window)]["demand"].mean()
    yr_ago_end = end - pd.Timedelta(days=365)
    prior = d[(d.date > yr_ago_end - pd.Timedelta(days=window)) &
              (d.date <= yr_ago_end)]["demand"].mean()
    if prior and prior > 0 and not np.isnan(recent):
        return float(recent / prior)
    return 1.0


def fixed_influence_mask(dates, post):
    """True for dates within [ws, we+post] of any FIXED-date holiday anchor.
    lag_365 aligns to the same holiday phase only for fixed-date holidays."""
    out = []
    for date in pd.to_datetime(dates):
        hit = False
        for name in FIXED_HOLIDAYS:
            ws, we = HOLIDAY_WINDOWS[name]
            for a in HOLIDAY_ANCHORS[name]:
                if ws <= (date - a).days <= we + post:
                    hit = True
                    break
            if hit:
                break
        out.append(hit)
    return np.array(out)


def top2_blend(cal_df, apply_df):
    sel = {}
    for h in range(1, HOLDOUT_DAYS + 1):
        for d in range(7):
            sub = cal_df[(cal_df["horizon"] == h) & (cal_df["dow"] == d)]
            if len(sub) < 3:
                continue
            mp = {n: (np.abs(sub["y"] - sub[n]) / sub["y"].clip(lower=1)).mean() for n in MODEL_NAMES}
            sel[(d, h)] = [k for k, _ in sorted(mp.items(), key=lambda kv: kv[1])[:2]]
    out = []
    for _, r in apply_df.iterrows():
        names = sel.get((int(r["dow"]), int(r["horizon"])), MODEL_NAMES)
        out.append(float(np.mean([r[n] for n in names])))
    return np.array(out)


def equal_blend(apply_df):
    return apply_df[MODEL_NAMES].mean(axis=1).values


def apply_anchor(df, blend, growth, alpha):
    """For holiday rows, final = (1-alpha)*blend + alpha*(lag_365*growth)."""
    anchor = df["lag_365"].values * growth
    out = blend.copy()
    hol = (df["is_hol"].values == 1) & ~np.isnan(anchor)
    out[hol] = (1 - alpha) * blend[hol] + alpha * anchor[hol]
    return out


def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = ~(np.isnan(y) | np.isnan(p)) & (y > 0)
    return np.mean(np.abs(y[m] - p[m]) / y[m]) if m.sum() else np.nan


def apply_fixed_anchor(df, blend, growth, alpha, post):
    """Anchor fixed-date holiday-influence rows toward lag_365*growth."""
    mask = fixed_influence_mask(df["target_date"], post)
    anchor = df["lag_365"].values * growth
    out = blend.copy()
    m = mask & ~np.isnan(anchor)
    out[m] = (1 - alpha) * blend[m] + alpha * anchor[m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=str, default="2026-05")
    ap.add_argument("--use-cache", action="store_true",
                    help="Reuse saved base_preds_*.csv instead of retraining")
    args = ap.parse_args()
    hm = pd.Timestamp(args.holdout + "-01")
    HOLDOUT_START, HOLDOUT_END = hm, hm + pd.Timedelta(days=27)
    VAL_END = HOLDOUT_START - pd.Timedelta(days=1)
    VAL_START = VAL_END - pd.Timedelta(days=180)

    demand = load_demand()
    print(f"=== Holiday anchor | holdout {HOLDOUT_START.date()}..{HOLDOUT_END.date()} ===")
    print(f"  validation {VAL_START.date()}..{VAL_END.date()}\n")

    vpath = DERIVED / f"base_preds_val_{args.holdout}.csv"
    hpath = DERIVED / f"base_preds_hold_{args.holdout}.csv"
    if args.use_cache and vpath.exists() and hpath.exists():
        print("  using cached base predictions")
        vp = pd.read_csv(vpath, parse_dates=["target_date"])
        hp = pd.read_csv(hpath, parse_dates=["target_date"])
    else:
        mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS)
        sel_train = set(d for d in demand.date if d < VAL_START)
        val_dates = set(d for d in demand.date if VAL_START <= d <= VAL_END)
        prod_train = set(d for d in demand.date if d < HOLDOUT_START)
        hold_dates = set(d for d in demand.date if HOLDOUT_START <= d <= HOLDOUT_END)
        vp = round_predict(mat, sel_train, val_dates, "VAL")
        hp = round_predict(mat, prod_train, hold_dates, "HOLD")
        DERIVED.mkdir(parents=True, exist_ok=True)
        vp.to_csv(vpath, index=False)
        hp.to_csv(hpath, index=False)
    for df in (vp, hp):
        if "dow" not in df.columns:
            df["dow"] = pd.to_datetime(df["target_date"]).dt.weekday

    hd = sorted(hp["target_date"].unique())
    sched = pd.DataFrame({"target_date": hd, "horizon": range(1, len(hd) + 1)})
    hold = sched.merge(hp.drop(columns=["dow"]), on=["target_date", "horizon"], how="left")
    hold["dow"] = pd.to_datetime(hold["target_date"]).dt.weekday

    base_top2 = top2_blend(vp, hold)
    val_top2 = top2_blend(vp, vp)

    # Calibrate (growth_window, post, alpha) ONLY on validation fixed-influence rows
    best = (1e9, None)
    for window in [28, 42, 56, 90]:
        Gv = yoy_growth(demand, VAL_START, window)
        for post in [0, 1, 2, 3]:
            vmask = fixed_influence_mask(vp["target_date"], post)
            if vmask.sum() < 10:
                continue
            anc = vp["lag_365"].values * Gv
            for a in np.linspace(0, 1, 21):
                p = val_top2.copy()
                m = vmask & ~np.isnan(anc)
                p[m] = (1 - a) * val_top2[m] + a * anc[m]
                sc = mape(vp["y"].values[vmask], p[vmask])
                if sc < best[0]:
                    best = (sc, (window, post, a))
    vw, vpost, va = best[1]
    print(f"\n  Calibrated on validation fixed-influence days:")
    print(f"    growth_window={vw}d  post={vpost}  alpha={va:.2f}  "
          f"(val fixed-infl MAPE={best[0]*100:.2f}%)")

    G_hold = yoy_growth(demand, HOLDOUT_START, vw)
    anchored = apply_fixed_anchor(hold, base_top2, G_hold, va, vpost)
    n_anchored = (fixed_influence_mask(hold["target_date"], vpost)).sum()

    print(f"\n=== Holdout results ===\n")
    print(f"  {'strategy':24s}  MAPE     WAPE     RMSE    Bias")
    print(f"  {'-'*56}")
    for name, p in [("top2 (baseline)", base_top2),
                    ("top2+fixed_anchor", anchored)]:
        m = compute_metrics(hold["y"], p)
        print(f"  {name:24s} {m['mape']*100:6.2f}%  {m['wape']*100:6.2f}%  "
              f"{m['rmse']:6.0f}  {m['bias']:+6.0f}")
    print(f"  (G_hold={G_hold:.3f}, {n_anchored} days anchored)")


if __name__ == "__main__":
    sys.exit(main())
