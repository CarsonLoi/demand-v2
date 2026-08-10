"""Honest model tuning: feature pruning, hyperparameters, horizon pooling.

Protocol (no leakage, no test-set peeking):
  TUNE windows   — configs are ranked on these
  TEST windows   — never used for selection, reported once at the end
  For every window, training uses ONLY dates strictly before that window.

Run:  uv run python research/tune_model.py
"""
from __future__ import annotations
import io, contextlib, logging, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
import _shared as S
from _shared import (HOLDOUT_DAYS, compute_metrics, holiday_mask_from_matrix,
                     load_demand, make_sample_weights)

TUNE = [("2026-02-01", "2026-02-28"), ("2026-03-01", "2026-03-28")]
TEST = [("2026-04-01", "2026-04-28"), ("2026-05-01", "2026-05-28")]

BASE = dict(objective="regression", n_estimators=500, learning_rate=0.02,
            num_leaves=15, max_depth=5, min_child_samples=5,
            reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, verbose=-1)


def build_matrix_quiet(demand):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return S.build_matrix(demand)


def rank_features(mat, demand, cutoff, topk_probe=200):
    """Feature importance from models trained ONLY on data before `cutoff`.
    Returns feature names ordered most- to least-important."""
    fc = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    tr = mat[mat["target_date"] < cutoff].dropna(subset=["y"])
    gains = pd.Series(0.0, index=fc)
    for h in (1, 4, 8, 14, 21, 28):          # sample horizons; full sweep is slow
        sub = tr[tr["horizon"] == h]
        if len(sub) < 30:
            continue
        w = make_sample_weights(sub["target_date"],
                                is_holiday=holiday_mask_from_matrix(sub))
        m = lgb.LGBMRegressor(**BASE, random_state=123)
        m.fit(sub[fc], sub["y"], sample_weight=w)
        gains += pd.Series(m.booster_.feature_importance("gain"), index=fc)
    return list(gains.sort_values(ascending=False).index)


def evaluate(mat, demand, start, end, feats=None, params=None, pool=0, seed=123):
    """Train on data strictly before `start`; predict [start, end] at the
    natural horizon schedule (day i of the window is an i-day-ahead forecast)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    all_fc = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    fc = list(feats) if feats else all_fc
    if pool and "horizon" not in fc:
        fc = fc + ["horizon"]                # pooled models need to know h
    p = {**BASE, **(params or {}), "random_state": seed}

    tr_all = mat[mat["target_date"] < start].dropna(subset=["y"])
    te = sorted(d for d in demand["date"] if start <= d <= end)

    rows = []
    for h in range(1, min(HOLDOUT_DAYS, len(te)) + 1):
        lo, hi = h - pool, h + pool
        sub = tr_all[(tr_all["horizon"] >= lo) & (tr_all["horizon"] <= hi)]
        if len(sub) < 30:
            continue
        w = make_sample_weights(sub["target_date"],
                                is_holiday=holiday_mask_from_matrix(sub))
        m = lgb.LGBMRegressor(**p)
        m.fit(sub[fc], sub["y"], sample_weight=w)
        row = mat[(mat["target_date"] == te[h - 1]) & (mat["horizon"] == h)]
        if row.empty:
            continue
        rows.append((max(0.0, float(m.predict(row[fc])[0])), float(row["y"].iloc[0])))
    if not rows:
        return np.nan
    d = pd.DataFrame(rows, columns=["pred", "y"])
    return compute_metrics(d["y"], d["pred"])["mape"] * 100


def score(mat, demand, windows, **kw):
    vals = [evaluate(mat, demand, s, e, **kw) for s, e in windows]
    return float(np.nanmean(vals)), vals


def main():
    demand = load_demand()
    S.USE_WEATHER = False          # measured harmful; see _shared.py
    mat = build_matrix_quiet(demand)
    all_fc = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    print(f"matrix: {len(mat):,} rows x {len(all_fc)} features\n")

    # Feature ranking uses only data before the earliest tune window.
    order = rank_features(mat, demand, pd.Timestamp(TUNE[0][0]))
    print(f"top 12 by gain: {', '.join(order[:12])}\n")

    results = []

    def trial(label, **kw):
        t0 = time.time()
        mean, vals = score(mat, demand, TUNE, **kw)
        results.append((label, mean, kw))
        print(f"  {label:34s} tune={mean:5.2f}%  "
              f"({', '.join(f'{v:.2f}' for v in vals)})  [{time.time()-t0:.0f}s]",
              flush=True)

    print("=== A. feature count ===")
    trial("all features", feats=None)
    for k in (30, 50, 80, 120):
        trial(f"top {k} features", feats=order[:k])

    best_feats_label, _, best_feats_kw = min(results, key=lambda r: r[1])
    feats = best_feats_kw.get("feats")
    print(f"  -> best: {best_feats_label}\n")

    print("=== B. horizon pooling (on best feature set) ===")
    pool_res = []
    for pool in (0, 1, 2, 3):
        t0 = time.time()
        mean, vals = score(mat, demand, TUNE, feats=feats, pool=pool)
        pool_res.append((pool, mean))
        print(f"  pool +/-{pool:<28d} tune={mean:5.2f}%  "
              f"({', '.join(f'{v:.2f}' for v in vals)})  [{time.time()-t0:.0f}s]",
              flush=True)
    best_pool = min(pool_res, key=lambda r: r[1])[0]
    print(f"  -> best pool: +/-{best_pool}\n")

    print("=== C. hyperparameters (on best features + pooling) ===")
    grid = [
        ("baseline", {}),
        ("deeper (leaves 31, depth 7)", dict(num_leaves=31, max_depth=7)),
        ("shallower (leaves 7, depth 4)", dict(num_leaves=7, max_depth=4)),
        ("slower+longer (lr .01, 1200)", dict(learning_rate=0.01, n_estimators=1200)),
        ("stronger reg (a2, l5)", dict(reg_alpha=2.0, reg_lambda=5.0)),
        ("less colsample (0.5)", dict(colsample_bytree=0.5)),
    ]
    hp_res = []
    for label, params in grid:
        t0 = time.time()
        mean, vals = score(mat, demand, TUNE, feats=feats, pool=best_pool, params=params)
        hp_res.append((label, mean, params))
        print(f"  {label:34s} tune={mean:5.2f}%  "
              f"({', '.join(f'{v:.2f}' for v in vals)})  [{time.time()-t0:.0f}s]",
              flush=True)
    best_hp = min(hp_res, key=lambda r: r[1])
    print(f"  -> best hp: {best_hp[0]}\n")

    print("=== FINAL: held-out TEST windows (never used for selection) ===")
    print(f"  config: {best_feats_label}, pool +/-{best_pool}, {best_hp[0]}\n")
    for tag, kw in (("ORIGINAL default", dict(feats=None, pool=0, params={})),
                    ("TUNED", dict(feats=feats, pool=best_pool, params=best_hp[2]))):
        line = []
        for s, e in TEST:
            seeds = [evaluate(mat, demand, s, e, seed=sd, **kw) for sd in (123, 7, 999)]
            line.append(f"{s[:7]}: {np.mean(seeds):.2f}% (+/-{(max(seeds)-min(seeds))/2:.2f})")
        print(f"  {tag:18s} " + "   ".join(line), flush=True)


if __name__ == "__main__":
    main()
