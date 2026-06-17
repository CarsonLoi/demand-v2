"""Honest probe of non-GBM model families for ensemble diversity.

Trains Ridge, ElasticNet, ExtraTrees, RandomForest per horizon (in addition to
the 6 cached GBM predictions) and reports:
  - each new model's standalone holdout MAPE
  - whether adding it to the blend improves the top2 selection

Tested on May 2026 and Oct 2025. Linear models get median-imputed + scaled
features; tree baggers get median-imputed features (sklearn can't take NaN).
"""
from __future__ import annotations
import logging, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("lightgbm").setLevel(logging.ERROR)
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent  # project root (scripts live in research/)
sys.path.insert(0, str(ROOT / "v2"))
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from _shared import (HOLDOUT_DAYS, build_matrix, compute_metrics,
                     holiday_mask_from_matrix, load_demand, make_sample_weights)

GBM = ["lgbm_l2", "lgbm_q", "xgb", "cat", "lgbm_bag", "v3_2stage"]
NEW = ["ridge", "enet", "xtrees", "rf"]


def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); m = (y > 0) & ~np.isnan(p)
    return np.mean(np.abs(y[m] - p[m]) / y[m])


def train_new_models(mat, train_dates, hold_dates):
    fcols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    tr = mat[mat["target_date"].isin(train_dates)].dropna(subset=["y"])
    pr = mat[mat["target_date"].isin(hold_dates)]
    rows = []
    for h in range(1, HOLDOUT_DAYS + 1):
        st = tr[tr["horizon"] == h]; sp = pr[pr["horizon"] == h]
        if len(st) < 30 or len(sp) == 0:
            continue
        X = st[fcols].values; y = st["y"].values; Xp = sp[fcols].values
        w = make_sample_weights(st["target_date"],
                                is_holiday=holiday_mask_from_matrix(st))
        imp = SimpleImputer(strategy="median").fit(X)
        Xi, Xpi = imp.transform(X), imp.transform(Xp)
        sc = StandardScaler().fit(Xi)
        Xs, Xps = sc.transform(Xi), sc.transform(Xpi)

        preds = {}
        m = Ridge(alpha=10.0).fit(Xs, y, sample_weight=w); preds["ridge"] = np.maximum(0, m.predict(Xps))
        m = ElasticNet(alpha=1.0, l1_ratio=0.3, max_iter=5000).fit(Xs, y); preds["enet"] = np.maximum(0, m.predict(Xps))
        m = ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                n_jobs=-1, random_state=42).fit(Xi, y, sample_weight=w)
        preds["xtrees"] = np.maximum(0, m.predict(Xpi))
        m = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                  n_jobs=-1, random_state=42).fit(Xi, y, sample_weight=w)
        preds["rf"] = np.maximum(0, m.predict(Xpi))

        sp2 = sp.reset_index(drop=True)
        for i in range(len(sp2)):
            row = {"target_date": sp2.loc[i, "target_date"], "horizon": h}
            for n in NEW:
                row[n] = preds[n][i]
            rows.append(row)
    return pd.DataFrame(rows)


def top2(cal, app, pool):
    sel = {}
    for kv, s in cal.groupby(["dow", "horizon"]):
        if len(s) < 3:
            continue
        mp = {n: mape(s.y, s[n]) for n in pool}
        sel[kv] = [k for k, _ in sorted(mp.items(), key=lambda x: x[1])[:2]]
    out = []
    for _, r in app.iterrows():
        names = sel.get((r["dow"], r["horizon"]), pool)
        out.append(np.mean([r[n] for n in names]))
    return np.array(out)


def main():
    demand = load_demand()
    mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS)
    for month, vstart in [("2026-05", "2025-11-01"), ("2025-10", "2025-04-01")]:
        HS = pd.Timestamp(month + "-01"); HE = HS + pd.Timedelta(days=27)
        VS = pd.Timestamp(vstart); VE = HS - pd.Timedelta(days=1)
        # cached GBM preds
        vp = pd.read_csv(f"data/derived/base_preds_val_{month}.csv", parse_dates=["target_date"])
        hp = pd.read_csv(f"data/derived/base_preds_hold_{month}.csv", parse_dates=["target_date"])
        # train new models for val + holdout
        vnew = train_new_models(mat, set(d for d in demand.date if d < VS),
                                set(d for d in demand.date if VS <= d <= VE))
        hnew = train_new_models(mat, set(d for d in demand.date if d < HS),
                                set(d for d in demand.date if HS <= d <= HE))
        vp = vp.merge(vnew, on=["target_date", "horizon"], how="left")
        hp = hp.merge(hnew, on=["target_date", "horizon"], how="left")
        for df in (vp, hp):
            df["dow"] = df["target_date"].dt.weekday
        vp.to_csv(f"data/derived/zoo_val_{month}.csv", index=False)
        hp.to_csv(f"data/derived/zoo_hold_{month}.csv", index=False)
        hd = sorted(hp.target_date.unique())
        sched = pd.DataFrame({"target_date": hd, "horizon": range(1, len(hd) + 1)})
        hold = sched.merge(hp, on=["target_date", "horizon"], how="left")
        hold["dow"] = hold.target_date.dt.weekday

        print(f"\n=== {month} ===")
        print("  standalone holdout MAPE:")
        for n in NEW:
            print(f"    {n:8s} {mape(hold.y, hold[n])*100:5.2f}%")
        print(f"  blend top2 (DOW,h) GBM-6        {mape(hold.y, top2(vp, hold, GBM))*100:5.2f}%")
        print(f"  blend top2 (DOW,h) GBM-6 + all4 {mape(hold.y, top2(vp, hold, GBM+NEW))*100:5.2f}%")
        # which new models actually beat the worst GBM standalone?
        gbm_best = min(mape(hold.y, hold[n]) for n in GBM)
        print(f"    (best GBM standalone = {gbm_best*100:.2f}%)")


if __name__ == "__main__":
    sys.exit(main())
