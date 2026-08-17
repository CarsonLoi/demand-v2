"""alpha_test.py — the last blocker: how hard should the CNY anchor pull when
the model is trained on long history?

Run3 established that `full history / hl=1095` improves 7 of 8 held-out
windows but regresses the CNY window by +0.81pp, which fails the per-window
gate. The cause is arithmetic rather than mysterious:

    anchored = (1 - alpha) * model + alpha * profile        alpha = 0.80

The `profile` term is computed from raw demand and is IDENTICAL in both
variants. Only the `model` term differs -- and on CNY days the long-history
model is worse (9.07% vs 6.28% unanchored). At alpha=0.80 a fifth of that
worse prediction still leaks through, which accounts for almost exactly the
observed gap.

So the question is whether leaning harder on the profile (which does not
depend on the model at all) closes it. Tests alpha in {0.80, 0.90, 1.00} on
the long-history model, against the production baseline at its calibrated 0.80.

alpha = 1.00 means "ignore the model entirely on CNY-window days and use the
profile alone" -- defensible precisely because the profile is model-independent.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/ -- this package owns its data
sys.path.insert(0, str(ROOT))
import config as C
from engine import features as F
from training import experiment as E
from engine.core import (HOLDOUT_DAYS, holiday_mask_from_matrix, make_sample_weights,
                     apply_moving_anchor, HOLIDAY_ANCHORS)

ALPHAS = [0.80, 0.90, 1.00]
HL = 1095


def raw_preds(mat, demand, origin, half_life, horizons):
    """Unanchored model predictions for one window (anchor applied separately
    so several alphas share one expensive fit)."""
    origin = pd.Timestamp(origin)
    feats = F.feature_columns(mat)
    train_all = mat[mat["target_date"] <= origin].dropna(subset=["y"])
    dm = dict(zip(demand.date, demand.demand.astype(float)))
    rows = []
    for h in horizons:
        sub = train_all[train_all["horizon"] == h]
        if len(sub) < 30:
            continue
        T = origin + pd.Timedelta(days=h)
        if T not in dm or np.isnan(dm[T]):
            continue
        pr = mat[(mat["target_date"] == T) & (mat["horizon"] == h)]
        if pr.empty:
            continue
        w = make_sample_weights(sub["target_date"],
                                is_holiday=holiday_mask_from_matrix(sub),
                                half_life_days=half_life)
        m = lgb.LGBMRegressor(**C.LGBM_PARAMS)
        m.fit(sub[feats], sub["y"], sample_weight=w)
        rows.append({"target": T, "pred": max(0.0, float(m.predict(pr[feats])[0])),
                     "actual": dm[T]})
    return pd.DataFrame(rows)


def anchored(df, demand, origin, alpha):
    if df.empty:
        return df
    p = pd.DataFrame({"date": df["target"], "p10": df["pred"],
                      "p50": df["pred"], "p90": df["pred"]})
    with F.extended_holidays():
        a = apply_moving_anchor(p, demand, pd.Timestamp(origin), alpha=alpha)
    out = df.copy(); out["pred"] = a["p50"].values
    return out


def mape(d):
    d = d[(d.actual > 0) & d.pred.notna()]
    return float(np.mean(np.abs(d.actual - d.pred) / d.actual)) * 100 if len(d) else np.nan


def main():
    demand = F.load_long_demand()
    origins = E.make_origins(demand, C.N_EVAL_WINDOWS, C.EVAL_WINDOW_STRIDE)
    horizons = [1, 3, 7, 14, 21, 28]
    cny = set()
    for a in HOLIDAY_ANCHORS["CNY"]:
        for k in range(-7, 11):
            cny.add(a + pd.Timedelta(days=k))

    print(f"origins: {len(origins)}  ({origins[0].date()} .. {origins[-1].date()})\n",
          flush=True)

    t0 = time.time()
    base_recent = F.build_base(demand, history_start="2024-01-01")
    mat_recent = F.apply_exclusions(base_recent, demand, [])
    print(f"[matrix] recent_only ready [{time.time()-t0:.0f}s]", flush=True)
    t0 = time.time()
    base_full = F.build_base(demand, history_start=None)
    mat_full = F.apply_exclusions(base_full, demand, [])
    print(f"[matrix] full ready [{time.time()-t0:.0f}s]\n", flush=True)

    raw = {}
    for tag, mat, hl in (("baseline", mat_recent, 240), ("full1095", mat_full, HL)):
        for o in origins:
            raw[(tag, o)] = raw_preds(mat, demand, o, hl, horizons)
        print(f"  fitted {tag}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'variant':34s} {'pooled':>8s}  " +
          "  ".join(str(o.date())[5:] for o in origins))
    print("=" * 92)

    rows = {}
    b = pd.concat([anchored(raw[("baseline", o)], demand, o, 0.80) for o in origins],
                  ignore_index=True)
    b_per = [mape(anchored(raw[("baseline", o)], demand, o, 0.80)) for o in origins]
    rows["baseline (hl=240, alpha=.80)"] = (mape(b), b_per)

    for al in ALPHAS:
        per = [mape(anchored(raw[("full1095", o)], demand, o, al)) for o in origins]
        allr = pd.concat([anchored(raw[("full1095", o)], demand, o, al)
                          for o in origins], ignore_index=True)
        rows[f"full hl=1095, alpha={al:.2f}"] = (mape(allr), per)

    for k, (pool, per) in rows.items():
        print(f"{k:34s} {pool:8.3f}  " + "  ".join(f"{x:5.2f}" for x in per))

    print("\nGATE (no window worse than baseline):")
    bp = rows["baseline (hl=240, alpha=.80)"][1]
    for k, (pool, per) in rows.items():
        if k.startswith("baseline"):
            continue
        worse = [(str(o.date()), p, q) for o, p, q in zip(origins, bp, per) if q > p + 1e-9]
        tag = "PASS" if not worse else f"FAIL ({len(worse)}/{len(origins)})"
        print(f"  {k:34s} {pool-rows['baseline (hl=240, alpha=.80)'][0]:+.2f}pp   {tag}")
        for w in worse:
            print(f"      worse: {w[0]}  {w[1]:.2f}% -> {w[2]:.2f}%")

    pd.DataFrame({k: v[1] for k, v in rows.items()},
                 index=[str(o.date()) for o in origins]).to_csv(
        ROOT / C.OUT_DIR / "alpha_test.csv")
    print(f"\n-> {ROOT / C.OUT_DIR / 'alpha_test.csv'}")


if __name__ == "__main__":
    main()
