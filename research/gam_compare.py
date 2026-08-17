"""gam_compare.py — does a GAM produce lower MAPE than LightGBM on this data?

STANDALONE. Imports READ-ONLY from v2/_shared.py -- the same pattern every
other research/ script already uses (build_matrix, load_demand, etc.). This
file does not write to, modify, or get imported by any production file, and
has no dependency on long_history/ at all.

Design, so the comparison is actually fair:
  - Same feature matrix (v2/_shared.build_matrix), same recency+holiday
    sample weights, same per-horizon architecture, same held-out origins for
    every model. Only the learner changes.
  - GAMs can't take 170 columns the way a tree can -- they'd be unstable on
    per-horizon slices of a few hundred rows. The reduced feature set below
    was chosen from this project's own measured LightGBM importance numbers
    (holidays ~47% of total gain; short lags/rolling next).
  - Two GAM variants:
      gam_plain     -- smooth/factor terms only, no interactions. Tests
                        "does an additive model just work here."
      gam_interact  -- same, plus the ready-made {holiday}_x_recent columns
                        v2/_shared.py already engineers (is_CNY * rolling_
                        mean_28, etc). Tests whether the gap is "GAM vs
                        tree" or "nobody told the GAM the one interaction
                        that matters."
  - LightGBM here is refit on the SAME row set the GAMs use (NaNs in the
    reduced feature set dropped) so training data is identical across all
    three models -- not the full-feature production configuration. This
    isolates the model as the only variable; it is not a reproduction of
    production's own MAPE.
  - lam (GAM regularization) is left at pyGAM's default, not grid-searched.
    That's a real caveat: if a GAM variant looks competitive, gridsearch
    should be run before trusting the number.
  - Verdict follows the same rule used everywhere else in this project: a
    model only "wins" if it beats LightGBM on pooled MAPE AND does not lose
    on any individual held-out window.

Usage:
    uv run python research/gam_compare.py
    uv run python research/gam_compare.py --quick     # 3 origins instead of 8
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from pygam import LinearGAM, s, f

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))

from _shared import (   # noqa: E402  -- read-only import, nothing written back
    HOLDOUT_DAYS, HOLIDAY_ANCHORS, HOLIDAY_WINDOWS,
    load_demand, build_matrix, make_sample_weights, holiday_mask_from_matrix,
)

OUT_DIR = ROOT / "research" / "gam_output"

# ── reduced feature set for the GAMs ────────────────────────────────────────
# continuous -> smooth term s(); low-cardinality integer/binary -> factor f()
GAM_CONTINUOUS = [
    "lag_anchor_7", "lag_anchor_14", "lag_anchor_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_28",
    "ewma_7", "ewma_28",
    "days_to_next_holiday", "days_from_last_holiday",
    "doy_sin", "doy_cos",
]
GAM_FACTOR = [
    "dow", "month", "is_weekend",
    "is_CNY", "is_GoldenWeek", "is_Labour", "is_MidAutumn",
    "is_DragonBoat", "is_ChingMing", "is_Christmas", "is_NewYear",
]
GAM_INTERACT_EXTRA = [   # gam_interact only -- already computed by _shared.py
    "CNY_x_recent", "GoldenWeek_x_recent", "Labour_x_recent", "MidAutumn_x_recent",
]

CNY_WINDOW = set()
for _a in HOLIDAY_ANCHORS["CNY"]:
    ws, we = HOLIDAY_WINDOWS["CNY"]
    for _k in range(ws, we + 1):
        CNY_WINDOW.add(_a + pd.Timedelta(days=_k))


def mape(actual, pred):
    m = (actual > 0) & pred.notna() if hasattr(pred, "notna") else (actual > 0)
    a, p = np.asarray(actual)[m], np.asarray(pred)[m]
    return float(np.mean(np.abs(a - p) / a)) * 100 if len(a) else np.nan


def make_origins(demand: pd.DataFrame, n: int, stride: int = 28) -> list:
    """Same spirit as every other backtest harness in this project: N origins
    stepping back from the end of the data, each needing a full 28 days of
    actuals after it plus enough history before it to train at all."""
    last = demand.date.max()
    origins, o = [], last - pd.Timedelta(days=HOLDOUT_DAYS)
    for _ in range(n):
        if o <= demand.date.min() + pd.Timedelta(days=300):
            break
        origins.append(o)
        o -= pd.Timedelta(days=stride)
    return sorted(origins)


def fit_lgbm(X, y, w):
    m = lgb.LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.02,
        num_leaves=15, max_depth=5, min_child_samples=5,
        reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1, random_state=123,
    )
    m.fit(X, y, sample_weight=w)
    return m


def fit_gam(X, y, w, cols, interact: bool):
    terms = None
    for i, c in enumerate(cols):
        t = f(i) if (c in GAM_FACTOR) else s(i, n_splines=10)
        terms = t if terms is None else terms + t
    gam = LinearGAM(terms, max_iter=150)
    gam.fit(X, y, weights=w)
    return gam


def run_window(mat, feat_full, feat_gam, demand_map, origin, horizons):
    train_all = mat[mat["target_date"] <= origin]
    rows_lgbm, rows_gam_plain, rows_gam_int = [], [], []

    for h in horizons:
        sub = train_all[train_all["horizon"] == h].dropna(subset=["y"] + feat_gam)
        if len(sub) < 60:
            continue
        T = origin + pd.Timedelta(days=h)
        if T not in demand_map or np.isnan(demand_map[T]):
            continue
        pr = mat[(mat["target_date"] == T) & (mat["horizon"] == h)]
        if pr.empty or pr[feat_gam].isna().any(axis=1).iloc[0]:
            continue

        is_hol = holiday_mask_from_matrix(sub)
        w = make_sample_weights(sub["target_date"], is_holiday=is_hol)
        actual = demand_map[T]

        # LightGBM, same rows as the GAMs (see module docstring)
        m_lgbm = fit_lgbm(sub[feat_full], sub["y"], w)
        p_lgbm = max(0.0, float(m_lgbm.predict(pr[feat_full])[0]))
        rows_lgbm.append({"origin": origin, "target": T, "horizon": h,
                          "actual": actual, "pred": p_lgbm})

        Xg = sub[feat_gam].to_numpy(dtype="float64")
        Xp = pr[feat_gam].to_numpy(dtype="float64")

        g1 = fit_gam(Xg, sub["y"].to_numpy(), w.to_numpy() if hasattr(w, "to_numpy") else w,
                     feat_gam, interact=False)
        p1 = max(0.0, float(g1.predict(Xp)[0]))
        rows_gam_plain.append({"origin": origin, "target": T, "horizon": h,
                               "actual": actual, "pred": p1})

    return (pd.DataFrame(rows_lgbm), pd.DataFrame(rows_gam_plain))


def run_window_interact(mat, feat_gam_int, demand_map, origin, horizons):
    train_all = mat[mat["target_date"] <= origin]
    rows = []
    for h in horizons:
        sub = train_all[train_all["horizon"] == h].dropna(subset=["y"] + feat_gam_int)
        if len(sub) < 60:
            continue
        T = origin + pd.Timedelta(days=h)
        if T not in demand_map or np.isnan(demand_map[T]):
            continue
        pr = mat[(mat["target_date"] == T) & (mat["horizon"] == h)]
        if pr.empty or pr[feat_gam_int].isna().any(axis=1).iloc[0]:
            continue
        is_hol = holiday_mask_from_matrix(sub)
        w = make_sample_weights(sub["target_date"], is_holiday=is_hol)
        Xg = sub[feat_gam_int].to_numpy(dtype="float64")
        Xp = pr[feat_gam_int].to_numpy(dtype="float64")
        g = fit_gam(Xg, sub["y"].to_numpy(), w.to_numpy() if hasattr(w, "to_numpy") else w,
                    feat_gam_int, interact=True)
        p = max(0.0, float(g.predict(Xp)[0]))
        rows.append({"origin": origin, "target": T, "horizon": h,
                     "actual": demand_map[T], "pred": p})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 origins instead of 8")
    ap.add_argument("--horizons", default="1,3,7,14,21,28",
                    help="comma-separated horizons to test (default: sampled)")
    a = ap.parse_args()

    horizons = [int(x) for x in a.horizons.split(",")]
    n_origins = 3 if a.quick else 8

    print("=" * 78)
    print("GAM vs LightGBM -- standalone comparison (production code untouched)")
    print("=" * 78)

    demand = load_demand()
    demand_map = dict(zip(demand["date"], demand["demand"].astype(float)))
    origins = make_origins(demand, n_origins)
    print(f"origins ({len(origins)}): {[o.date() for o in origins]}")
    print(f"horizons: {horizons}")

    t0 = time.time()
    mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS, as_of=demand.date.max())
    print(f"[matrix] {mat.shape[0]:,} x {mat.shape[1]}  [{time.time()-t0:.0f}s]")

    feat_full = [c for c in mat.columns if c not in
                 {"target_date", "horizon", "y"}]
    feat_gam = list(GAM_CONTINUOUS) + list(GAM_FACTOR)
    feat_gam = [c for c in feat_gam if c in mat.columns]
    missing = set(GAM_CONTINUOUS + GAM_FACTOR) - set(feat_gam)
    if missing:
        print(f"[warn] columns not found, skipped: {missing}")
    feat_gam_int = feat_gam + [c for c in GAM_INTERACT_EXTRA if c in mat.columns]

    all_lgbm, all_plain, all_int = [], [], []
    for o in origins:
        print(f"\n--- origin {o.date()} ---", flush=True)
        t0 = time.time()
        df_l, df_p = run_window(mat, feat_full, feat_gam, demand_map, o, horizons)
        df_i = run_window_interact(mat, feat_gam_int, demand_map, o, horizons)
        print(f"  lgbm MAPE={mape(df_l.actual, df_l.pred):6.2f}%  "
              f"gam_plain MAPE={mape(df_p.actual, df_p.pred):6.2f}%  "
              f"gam_interact MAPE={mape(df_i.actual, df_i.pred):6.2f}%  "
              f"[{time.time()-t0:.0f}s]")
        all_lgbm.append(df_l); all_plain.append(df_p); all_int.append(df_i)

    lgbm = pd.concat(all_lgbm, ignore_index=True)
    plain = pd.concat(all_plain, ignore_index=True)
    inter = pd.concat(all_int, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lgbm.to_csv(OUT_DIR / "lgbm.csv", index=False)
    plain.to_csv(OUT_DIR / "gam_plain.csv", index=False)
    inter.to_csv(OUT_DIR / "gam_interact.csv", index=False)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, df in (("LightGBM (reference)", lgbm),
                     ("GAM plain (no interactions)", plain),
                     ("GAM + holiday_x_recent", inter)):
        cny = df[df.target.isin(CNY_WINDOW)]
        non_cny = df[~df.target.isin(CNY_WINDOW)]
        print(f"{name:32s} overall {mape(df.actual, df.pred):6.2f}%   "
              f"CNY {mape(cny.actual, cny.pred):6.2f}%   "
              f"non-CNY {mape(non_cny.actual, non_cny.pred):6.2f}%   (n={len(df)})")

    print("\nPer-origin (does either GAM ever lose to LightGBM on a window it "
          "would need to win to be worth adopting):")
    for o in origins:
        a = mape(lgbm[lgbm.origin == o].actual, lgbm[lgbm.origin == o].pred)
        b = mape(plain[plain.origin == o].actual, plain[plain.origin == o].pred)
        c = mape(inter[inter.origin == o].actual, inter[inter.origin == o].pred)
        print(f"  {o.date()}   lgbm={a:6.2f}%   gam_plain={b:6.2f}%"
              f"{'  <-- worse' if b > a + 1e-9 else ''}"
              f"   gam_interact={c:6.2f}%{'  <-- worse' if c > a + 1e-9 else ''}")

    print(f"\n-> {OUT_DIR}")
    print("\nNote: lam not grid-searched (see module docstring). If a GAM "
          "variant looks competitive here, gridsearch before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
