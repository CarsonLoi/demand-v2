"""experiment.py — does longer history actually help, and at what weighting?

Runs a grid of variants over the SAME held-out windows and reports per-window
MAPE with a no-regression gate.

Protocol (identical to every honest test in this project):
  For origin O, target O+h is taken at horizon h, so features resolve to O-1.
  Training uses only target_date <= O. Scored against true actuals.
  The model is held FIXED across variants, so any difference is attributable
  to the data range and the weighting -- not to the model.

The variant that matters most is `full history x long half-life`. At the
production default of 240 days a 2016 row carries ~1/35,000th the weight of a
recent one -- so the half-life change is not a follow-up to adding history,
it IS the experiment. Note that `full / hl=240` will NOT match `recent_only`:
adding history also makes lag_365/lag_728/holiday-YoY features AVAILABLE where
they were previously NaN. See README section 0.

Usage:
    uv run python long_history/experiment.py                # full grid
    uv run python long_history/experiment.py --quick        # sampled horizons
    uv run python long_history/experiment.py --baseline-only
"""
from __future__ import annotations
import argparse, io, contextlib, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/ -- this package owns its data
sys.path.insert(0, str(ROOT))

import config as C
from engine import features as F
from engine.core import HOLDOUT_DAYS, holiday_mask_from_matrix, make_sample_weights
from engine.holidays import verify_anchors


# ── evaluation windows ────────────────────────────────────────────────────
def make_origins(demand: pd.DataFrame, n: int, stride: int) -> list:
    """N held-out origins stepping back from the end of the data. Each origin
    needs a full 28 days of actuals after it."""
    last = demand.date.max()
    origins = []
    o = last - pd.Timedelta(days=HOLDOUT_DAYS)
    for _ in range(n):
        if o <= demand.date.min() + pd.Timedelta(days=400):
            break                      # need enough history to train at all
        origins.append(o)
        o -= pd.Timedelta(days=stride)
    return sorted(origins)


def run_window(mat, demand, origin, half_life, horizons=None, anchor=True):
    """One production-shaped 28-day forecast from `origin`.

    `anchor` applies the CNY correction (core.apply_moving_anchor),
    which blends CNY-window days 80% toward a position-matched, weekday-corrected
    prior-year profile. Production runs this in EVERY forecast mode, so leaving
    it out measures something the business never actually sees -- and it happens
    to be exactly the days where long history struggles.
    """
    origin = pd.Timestamp(origin)
    feats = F.feature_columns(mat)
    horizons = horizons or list(range(1, HOLDOUT_DAYS + 1))
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
        rows.append({"origin": origin, "target": T, "horizon": h,
                     "pred": max(0.0, float(m.predict(pr[feats])[0])),
                     "actual": dm[T]})
    df = pd.DataFrame(rows)
    if df.empty or not anchor:
        return df

    # Apply the CNY anchor exactly as forecast.py does. Extended anchors must be
    # active so pre-2024 CNYs resolve as prior occurrences.
    from engine.core import apply_moving_anchor
    p = pd.DataFrame({"date": df["target"], "p10": df["pred"],
                      "p50": df["pred"], "p90": df["pred"]})
    with F.extended_holidays():
        a = apply_moving_anchor(p, demand, origin)
    df["pred"] = a["p50"].values
    return df


def mape(d):
    d = d[(d.actual > 0) & d.pred.notna()]
    return float(np.mean(np.abs(d.actual - d.pred) / d.actual)) * 100 if len(d) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="sample horizons (1,3,7,14,21,28) instead of all 28")
    ap.add_argument("--baseline-only", action="store_true",
                    help="run only the recent-only baseline, then stop")
    a = ap.parse_args()

    out = ROOT / C.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    horizons = [1, 3, 7, 14, 21, 28] if a.quick else None

    print("=" * 78)
    print("LONG-HISTORY EXPERIMENT")
    print("=" * 78)

    # 1. validate before anything else
    from data.validate import validate
    n_err, _ = validate(ROOT / C.DATA_FILE)
    if n_err:
        print("\nABORTED: fix the data errors above first.")
        return 1

    # 2. verify holiday anchors
    print()
    if verify_anchors():
        print("\nABORTED: holiday anchors did not verify.")
        return 1

    demand = F.load_long_demand()
    origins = make_origins(demand, C.N_EVAL_WINDOWS, C.EVAL_WINDOW_STRIDE)
    print(f"\nheld-out origins ({len(origins)}): "
          f"{origins[0].date()} .. {origins[-1].date()}")
    print(f"horizons: {'sampled 1,3,7,14,21,28' if a.quick else 'all 1..28'}")

    # 3. build the variant grid: history x exclusion-set x half-life
    exclusion_sets = getattr(C, "EXCLUSION_SETS", [("covid_only", [])])
    variants = []
    for hname, hstart in C.HISTORY_STARTS:
        for ename, eranges in exclusion_sets:
            for hl in C.HALF_LIVES:
                # the baseline only needs one cell
                if hname == "recent_only" and (hl != C.HALF_LIVES[0]
                                               or ename != exclusion_sets[0][0]):
                    continue
                label = (f"{hname} / {ename} / hl={hl}" if hname != "recent_only"
                         else f"{hname} / hl={hl}  (BASELINE)")
                variants.append((label, hstart, eranges, hl))
    if a.baseline_only:
        variants = variants[:1]

    # 4. run. The expensive matrix build is shared across exclusion sets and
    #    half-lives; only the cheap y-blanking step is repeated.
    results, all_rows = [], []
    base_cache, mat_cache = {}, {}
    for label, hstart, eranges, hl in variants:
        if hstart not in base_cache:
            t0 = time.time()
            base_cache[hstart] = F.build_base(demand, history_start=hstart)
            print(f"\n  [matrix] history_start={hstart or 'ALL'}  "
                  f"{base_cache[hstart].shape[0]:,} rows x "
                  f"{base_cache[hstart].shape[1]} cols  [{time.time()-t0:.0f}s]")
        key = (hstart, tuple(map(tuple, eranges)))
        if key not in mat_cache:
            mat_cache[key] = F.apply_exclusions(base_cache[hstart], demand, eranges)
            n_train = int(mat_cache[key]["y"].notna().sum())
            print(f"  [exclusions] {eranges or 'covid+closures only'} "
                  f"-> {n_train:,} trainable rows")
        mat = mat_cache[key]

        print(f"\n### {label}")
        per_window = []
        for o in origins:
            r = run_window(mat, demand, o, hl, horizons)
            if r.empty:
                continue
            r["variant"] = label
            all_rows.append(r)
            per_window.append((o, mape(r)))
            print(f"    origin {o.date()}  MAPE={mape(r):6.2f}%", flush=True)
        if per_window:
            pooled = mape(pd.concat([x for x in all_rows if
                                     (x.variant == label).all()], ignore_index=True))
            results.append({"variant": label, "pooled": pooled,
                            **{str(o.date()): m for o, m in per_window}})

    if not results:
        print("\nNo results -- check that your data covers the evaluation range.")
        return 1

    df = pd.DataFrame(results)
    pd.concat(all_rows, ignore_index=True).to_csv(out / "experiment_raw.csv", index=False)
    df.to_csv(out / "experiment_summary.csv", index=False)

    # 5. report, with the per-window gate
    print("\n" + "=" * 78)
    print("SUMMARY — MAPE % (lower is better)")
    print("=" * 78)
    print(df.to_string(index=False))

    base = df.iloc[0]
    print(f"\nbaseline = {base['variant']}  ({base['pooled']:.2f}%)")
    print("\nGATE: a variant is only adopted if NO window is worse than baseline.")
    wincols = [c for c in df.columns if c not in ("variant", "pooled")]
    for _, r in df.iloc[1:].iterrows():
        worse = [c for c in wincols
                 if pd.notna(r[c]) and pd.notna(base[c]) and r[c] > base[c] + 1e-9]
        verdict = "PASS" if not worse else f"FAIL ({len(worse)}/{len(wincols)} windows worse)"
        print(f"  {r['variant']:28s} pooled {r['pooled']:6.2f}%  "
              f"({r['pooled']-base['pooled']:+.2f}pp)   {verdict}")

    print(f"\n-> {out/'experiment_summary.csv'}")
    print("\nSanity check: 'full / hl=240' should land within ~0.1pp of "
          "'recent_only / hl=240'.\nIf it does not, the harness is mis-wired — "
          "where they were previously NaN. See README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
