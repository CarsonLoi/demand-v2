"""Honest MAPE optimization: try log-transform + learned blends + bias
correction, all calibrated ONLY on a validation window, evaluated on a holdout.

For a given holdout window [HOLDOUT_START, HOLDOUT_END]:
  Round 1 (SELECTION/CALIBRATION):
      train models on demand < VAL_START
      predict validation window [VAL_START, VAL_END]  -> val_preds
  Round 2 (PRODUCTION):
      train models on demand < HOLDOUT_START
      predict holdout window -> holdout_preds

Base models trained twice: once on raw y, once on log1p(y).  All blend
strategies are calibrated on val_preds and applied to holdout_preds.

Strategies compared (all honest):
  equal                : mean of 6 models
  top2                 : per (DOW,horizon) top-2 by validation MAPE, averaged
  nnls_h               : per-horizon non-negative least squares weights
  nnls_h + dowscale    : nnls_h then per-DOW multiplicative bias correction
  top2  + dowscale     : top2 then per-DOW bias correction

Usage:
  uv run python optimize_mape.py --holdout 2026-05            # May 1-28
  uv run python optimize_mape.py --holdout 2026-02            # a 2nd window
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
from scipy.optimize import nnls

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


# ----- base learners -----------------------------------------------------
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


def round_predict(mat, train_dates, predict_dates, label, use_log):
    """Train 6 models per horizon, predict. Returns dict[name]->DataFrame."""
    fcols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    tr = mat[mat["target_date"].isin(train_dates)].dropna(subset=["y"])
    pr = mat[mat["target_date"].isin(predict_dates)]
    print(f"  ROUND {label} (log={use_log}): train {len(tr):,}, predict {len(pr):,}")

    out = {n: [] for n in MODEL_NAMES}
    t0 = time.time()
    for h in range(1, HOLDOUT_DAYS + 1):
        st = tr[tr["horizon"] == h]; sp = pr[pr["horizon"] == h]
        if len(st) < 30 or len(sp) == 0:
            continue
        X = st[fcols]; Xp = sp[fcols]
        y_raw = st["y"]
        y = np.log1p(y_raw) if use_log else y_raw
        if use_log:
            y = pd.Series(y, index=y_raw.index)
        td = sp["target_date"].values; y_true = sp["y"].values
        is_hol = holiday_mask_from_matrix(st)
        w = make_sample_weights(st["target_date"], is_holiday=is_hol)

        def back(p):
            return np.maximum(0.0, np.expm1(p) if use_log else p)

        preds = {}
        m = _m_lgbm_l2(X, y, w); preds["lgbm_l2"] = back(m.predict(Xp))
        m = _m_lgbm_q(X, y, w);  preds["lgbm_q"]  = back(m.predict(Xp))
        m = _m_xgb(X, y);        preds["xgb"]     = back(m.predict(Xp))
        m = _m_cat(X, y, w);     preds["cat"]     = back(m.predict(Xp))
        ms = _m_bag(X, y, w);    preds["lgbm_bag"]= back(np.mean([mm.predict(Xp) for mm in ms], axis=0))
        bm, rm = _m_2stage(X, y, w, is_hol)
        base = bm.predict(Xp)
        is_hol_p = holiday_mask_from_matrix(sp)
        resid = np.where(is_hol_p.values, rm.predict(Xp), 0.0) if rm is not None else np.zeros(len(Xp))
        preds["v3_2stage"] = back(base + resid)

        for n in MODEL_NAMES:
            for i in range(len(td)):
                out[n].append((td[i], h, preds[n][i], y_true[i]))
        if h % 7 == 0 or h == HOLDOUT_DAYS:
            print(f"    h={h:2d}: {time.time()-t0:.0f}s")
    return {n: pd.DataFrame(out[n], columns=["target_date", "horizon", "pred", "y"]) for n in MODEL_NAMES}


def wide(preds):
    comb = None
    for n, df in preds.items():
        c = df.rename(columns={"pred": n})[["target_date", "horizon", n, "y"]]
        comb = c if comb is None else comb.merge(c.drop(columns=["y"]), on=["target_date", "horizon"], how="outer")
    comb["dow"] = pd.to_datetime(comb["target_date"]).dt.weekday
    return comb


def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    mask = ~(np.isnan(y) | np.isnan(p)) & (y > 0)
    return np.mean(np.abs(y[mask] - p[mask]) / y[mask])


# ----- blend strategies (calibrated on val, applied to holdout) ----------
def strat_equal(val, hold):
    return hold[MODEL_NAMES].mean(axis=1).values

def strat_top2(val, hold):
    sel = {}
    for h in range(1, HOLDOUT_DAYS + 1):
        for d in range(7):
            sub = val[(val["horizon"] == h) & (val["dow"] == d)]
            if len(sub) < 3:
                continue
            mp = {n: mape(sub["y"], sub[n]) for n in MODEL_NAMES}
            sel[(d, h)] = [k for k, _ in sorted(mp.items(), key=lambda kv: kv[1])[:2]]
    res = []
    for _, r in hold.iterrows():
        key = (int(r["dow"]), int(r["horizon"]))
        names = sel.get(key, MODEL_NAMES)
        res.append(np.mean([r[n] for n in names]))
    return np.array(res)

def _nnls_weights(sub):
    A = sub[MODEL_NAMES].values; b = sub["y"].values
    msk = ~np.isnan(A).any(axis=1) & ~np.isnan(b)
    if msk.sum() < 8:
        return np.ones(len(MODEL_NAMES)) / len(MODEL_NAMES)
    w, _ = nnls(A[msk], b[msk])
    if w.sum() == 0:
        return np.ones(len(MODEL_NAMES)) / len(MODEL_NAMES)
    return w / w.sum()

def strat_nnls_h(val, hold):
    wts = {}
    for h in range(1, HOLDOUT_DAYS + 1):
        sub = val[val["horizon"] == h]
        if len(sub) >= 8:
            wts[h] = _nnls_weights(sub)
    res = []
    for _, r in hold.iterrows():
        w = wts.get(int(r["horizon"]), np.ones(len(MODEL_NAMES)) / len(MODEL_NAMES))
        res.append(float(np.dot(w, [r[n] for n in MODEL_NAMES])))
    return np.array(res)

def _dow_scale(val, hold, val_blend, hold_blend):
    """Multiplicative per-DOW bias correction calibrated on validation."""
    vb = val.copy(); vb["_b"] = val_blend
    scale = {}
    for d in range(7):
        s = vb[vb["dow"] == d]
        denom = s["_b"].sum()
        scale[d] = float(np.clip(s["y"].sum() / denom, 0.9, 1.1)) if denom > 0 else 1.0
    return np.array([hold_blend[i] * scale[int(hold.iloc[i]["dow"])] for i in range(len(hold))])


def evaluate(val, hold, tag):
    rows = []
    # individuals
    for n in MODEL_NAMES:
        m = compute_metrics(hold["y"], hold[n])
        rows.append((f"{tag}:{n}", m))
    # blends
    eq = strat_equal(val, hold)
    rows.append((f"{tag}:equal", compute_metrics(hold["y"], eq)))
    t2 = strat_top2(val, hold)
    rows.append((f"{tag}:top2", compute_metrics(hold["y"], t2)))
    nn = strat_nnls_h(val, hold)
    rows.append((f"{tag}:nnls_h", compute_metrics(hold["y"], nn)))

    # + dow scale (need validation blend for same strategy)
    val_eq = strat_equal(val, val); val_t2 = strat_top2(val, val); val_nn = strat_nnls_h(val, val)
    rows.append((f"{tag}:top2+dowscale",
                 compute_metrics(hold["y"], _dow_scale(val, hold, val_t2, t2))))
    rows.append((f"{tag}:nnls_h+dowscale",
                 compute_metrics(hold["y"], _dow_scale(val, hold, val_nn, nn))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=str, default="2026-05",
                    help="Holdout month YYYY-MM (uses days 1..28 of that month)")
    args = ap.parse_args()

    hm = pd.Timestamp(args.holdout + "-01")
    HOLDOUT_START = hm
    HOLDOUT_END = hm + pd.Timedelta(days=27)
    VAL_END = HOLDOUT_START - pd.Timedelta(days=1)
    VAL_START = VAL_END - pd.Timedelta(days=180)

    demand = load_demand()
    print(f"=== Optimize MAPE | holdout {HOLDOUT_START.date()}..{HOLDOUT_END.date()} ===")
    print(f"  validation {VAL_START.date()}..{VAL_END.date()}")
    print(f"  data ends {demand.date.max().date()}\n")

    mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS)

    sel_train = set(d for d in demand.date if d < VAL_START)
    val_dates = set(d for d in demand.date if VAL_START <= d <= VAL_END)
    prod_train = set(d for d in demand.date if d < HOLDOUT_START)
    hold_dates = set(d for d in demand.date if HOLDOUT_START <= d <= HOLDOUT_END)

    all_rows = []
    for use_log in (False, True):
        tag = "log" if use_log else "raw"
        print(f"\n----- transform: {tag} -----")
        vp = round_predict(mat, sel_train, val_dates, "VAL", use_log)
        hp = round_predict(mat, prod_train, hold_dates, "HOLD", use_log)
        val_w = wide(vp); hold_w = wide(hp)
        # natural horizon schedule for holdout
        hd = sorted(hold_dates)
        sched = pd.DataFrame({"target_date": hd, "horizon": range(1, len(hd) + 1)})
        hold_nat = sched.merge(hold_w.drop(columns=["dow"]), on=["target_date", "horizon"], how="left")
        hold_nat["dow"] = pd.to_datetime(hold_nat["target_date"]).dt.weekday
        all_rows += evaluate(val_w, hold_nat, tag)

    print("\n=== RESULTS (holdout MAPE, sorted) ===\n")
    all_rows.sort(key=lambda r: r[1]["mape"])
    print(f"  {'strategy':28s}  MAPE     WAPE     RMSE    Bias")
    print(f"  {'-'*60}")
    for name, m in all_rows:
        print(f"  {name:28s} {m['mape']*100:6.2f}%  {m['wape']*100:6.2f}%  "
              f"{m['rmse']:6.0f}  {m['bias']:+6.0f}")


if __name__ == "__main__":
    sys.exit(main())
