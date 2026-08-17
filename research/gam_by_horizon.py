"""gam_by_horizon.py — LightGBM vs GAM, every 3 days since 2026, per horizon.

Groundwork for the hybrid question: "could a GAM take the holidays while the
existing model keeps ordinary days?" To answer that we first need to see WHERE
each model wins -- by horizon, and by holiday vs not.

STANDALONE. Read-only import from v2/_shared.py and research/gam_compare.py.
Writes only to research/gam_by_horizon_output/. Nothing in v2/ or
long_history/ is modified.

SETUP
  origins   every 3 days, 2026-01-01 .. 2026-04-30 (40 origins; the end date
            is set so even horizon 28 still has a real actual to score)
  horizons  3, 5, 7, 10, 14, 18, 21, 24, 28 -- identical origin set for every
            horizon, so the per-horizon charts are directly comparable
  models    LightGBM (production learner) vs GAM (plain, pyGAM)
            Same feature matrix, same recency+holiday sample weights, same
            per-horizon architecture. Only the learner changes.

NO CNY ANCHOR IS APPLIED -- deliberately. Production blends CNY-window days
80% toward a profile, which would overwrite BOTH models with the same numbers
and hide the very difference this run is meant to measure. These are raw model
outputs. Production CNY-window accuracy is better than what you see here for
both models.

Usage:
    uv run python research/gam_by_horizon.py
    uv run python research/gam_by_horizon.py --step 7 --jobs 8
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from pygam import LinearGAM, s, f

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v2"))

from _shared import (  # noqa: E402  read-only
    HOLDOUT_DAYS, HOLIDAY_ANCHORS, HOLIDAY_WINDOWS,
    load_demand, build_matrix, make_sample_weights, holiday_mask_from_matrix,
)
from research.gam_compare import GAM_CONTINUOUS, GAM_FACTOR  # noqa: E402

OUT = ROOT / "research" / "gam_by_horizon_output"
HORIZONS = [3, 5, 7, 10, 14, 18, 21, 24, 28]
START, END = "2026-01-01", "2026-04-30"


def holiday_lookup():
    """target date -> holiday name (window-aware), else None."""
    out = {}
    for name, anchors in HOLIDAY_ANCHORS.items():
        ws, we = HOLIDAY_WINDOWS[name]
        for a in anchors:
            for k in range(ws, we + 1):
                out.setdefault(a + pd.Timedelta(days=k), name)
    return out


def fit_lgbm(X, y, w, jobs):
    m = lgb.LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.02,
        num_leaves=15, max_depth=5, min_child_samples=5,
        reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        n_jobs=jobs, verbose=-1, random_state=123,
    )
    m.fit(X, y, sample_weight=w)
    return m


def fit_gam(X, y, w, cols):
    terms = None
    for i, c in enumerate(cols):
        t = f(i) if c in GAM_FACTOR else s(i, n_splines=10)
        terms = t if terms is None else terms + t
    g = LinearGAM(terms, max_iter=150)
    g.fit(X, y, weights=w)
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=3, help="origin stride in days")
    ap.add_argument("--jobs", type=int, default=4,
                    help="LightGBM threads (kept low to share the box)")
    a = ap.parse_args()

    print("=" * 78)
    print("LightGBM vs GAM — per horizon, origins every "
          f"{a.step} day(s) since {START}")
    print("=" * 78)

    demand = load_demand()
    dm = dict(zip(demand["date"], demand["demand"].astype(float)))
    hol = holiday_lookup()

    origins = list(pd.date_range(START, END, freq=f"{a.step}D"))
    print(f"origins  : {len(origins)}  ({origins[0].date()} .. {origins[-1].date()})")
    print(f"horizons : {HORIZONS}")
    print(f"fits     : ~{len(origins)*len(HORIZONS)*2:,}")

    t0 = time.time()
    mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS, as_of=demand.date.max())
    print(f"[matrix] {mat.shape[0]:,} x {mat.shape[1]}  [{time.time()-t0:.0f}s]")

    feat_full = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    feat_gam = [c for c in (GAM_CONTINUOUS + GAM_FACTOR) if c in mat.columns]
    print(f"[features] lgbm {len(feat_full)}   gam {len(feat_gam)}")

    rows, t0 = [], time.time()
    for hi, h in enumerate(HORIZONS, 1):
        hrows = 0
        for o in origins:
            T = o + pd.Timedelta(days=h)
            actual = dm.get(T)
            if actual is None or np.isnan(actual) or actual <= 0:
                continue
            sub = mat[(mat["target_date"] <= o) &
                      (mat["horizon"] == h)].dropna(subset=["y"] + feat_gam)
            if len(sub) < 60:
                continue
            pr = mat[(mat["target_date"] == T) & (mat["horizon"] == h)]
            if pr.empty or pr[feat_gam].isna().any(axis=1).iloc[0]:
                continue

            w = make_sample_weights(sub["target_date"],
                                    is_holiday=holiday_mask_from_matrix(sub))
            wv = w.to_numpy() if hasattr(w, "to_numpy") else np.asarray(w)

            p_l = max(0.0, float(fit_lgbm(sub[feat_full], sub["y"], w,
                                          a.jobs).predict(pr[feat_full])[0]))
            g = fit_gam(sub[feat_gam].to_numpy("float64"),
                        sub["y"].to_numpy(), wv, feat_gam)
            p_g = max(0.0, float(g.predict(pr[feat_gam].to_numpy("float64"))[0]))

            rows.append({"origin": o, "target": T, "horizon": h, "actual": actual,
                         "lgbm": p_l, "gam": p_g,
                         "holiday": hol.get(T) or "", "is_holiday": T in hol})
            hrows += 1
        print(f"  h={h:2d}  {hrows:3d} points  [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "raw.csv", index=False)

    def mape(d, col):
        d = d[d.actual > 0]
        return float(np.mean(np.abs(d[col] - d.actual) / d.actual)) * 100 if len(d) else np.nan

    print("\n" + "=" * 78)
    print("MAPE BY HORIZON")
    print("=" * 78)
    print(f"{'h':>3s} {'n':>4s} {'LightGBM':>10s} {'GAM':>8s} {'winner':>9s}   "
          f"{'| holiday days':>16s} {'lgbm':>7s} {'gam':>7s}")
    for h in HORIZONS:
        d = df[df.horizon == h]
        dh = d[d.is_holiday]
        l, g = mape(d, "lgbm"), mape(d, "gam")
        win = "GAM" if g < l else "LightGBM"
        hl = f"{mape(dh,'lgbm'):7.2f}" if len(dh) else "     --"
        hg = f"{mape(dh,'gam'):7.2f}" if len(dh) else "     --"
        print(f"{h:3d} {len(d):4d} {l:9.2f}% {g:7.2f}% {win:>9s}   "
              f"{'| n=%d'%len(dh):>16s} {hl} {hg}")

    print("\nPOOLED, split by holiday:")
    for lab, sel in (("all days", df), ("holiday-window days", df[df.is_holiday]),
                     ("ordinary days", df[~df.is_holiday])):
        print(f"  {lab:22s} n={len(sel):4d}   LightGBM {mape(sel,'lgbm'):5.2f}%   "
              f"GAM {mape(sel,'gam'):5.2f}%")

    print("\nBY HOLIDAY (the hybrid question):")
    for name in sorted(x for x in df.holiday.unique() if x):
        sel = df[df.holiday == name]
        l, g = mape(sel, "lgbm"), mape(sel, "gam")
        print(f"  {name:12s} n={len(sel):4d}   LightGBM {l:6.2f}%   GAM {g:6.2f}%   "
              f"{'GAM better' if g < l else 'LightGBM better'} ({g-l:+.2f}pp)")

    print(f"\n-> {OUT/'raw.csv'}")
    print("\nNOTE: no CNY anchor applied — raw model outputs, so the two models "
          "are actually distinguishable on CNY days. Production is better than "
          "this on those days for BOTH models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
