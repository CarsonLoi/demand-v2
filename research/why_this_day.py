"""why_this_day.py -- decompose the forecast for specific target dates.

Answers "why is this day predicted low/high" with exact SHAP contributions
from a model trained the way the pipeline trains, plus a dump of the yearly
lookback features and what calendar date each one actually reads.

Standalone / read-only: imports long_history/{config,engine}. Writes nothing.

Usage:
    uv run python research/why_this_day.py --dates 2026-09-23 2026-09-24 2026-09-25 \
        --run-date 2026-09-01
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "long_history"))

import config as C                      # noqa: E402
from engine import core as S            # noqa: E402
from engine import features as F        # noqa: E402

YEARLY = ["lag_365", "lag_anchor_365", "lag_728", "yoy_ratio", "yoy_anchor_diff",
          "holiday_yoy_lag", "holiday_yoy_lift", "same_holiday_lastyear_lag"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--run-date", required=True,
                    help="the origin the forecast was made from")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    run_date = pd.Timestamp(a.run_date).normalize()
    targets = [pd.Timestamp(x).normalize() for x in a.dates]

    print("=" * 78)
    print("WHY THIS DAY -- forecast decomposition")
    print("=" * 78)
    print("config.EXCLUDE_FROM_TRAINING:")
    for r in C.EXCLUDE_FROM_TRAINING:
        print(f"   {r}")
    print(f"GUARD_LAGS_CROSSING_EXCLUDED = {C.GUARD_LAGS_CROSSING_EXCLUDED}")
    print(f"GUARD_YEARLY_LAGS (core)     = {S.GUARD_YEARLY_LAGS}")

    demand = F.load_long_demand()
    dm = dict(zip(demand["date"], demand["demand"].astype(float)))
    print(f"\ndata: {demand.date.min().date()} .. {demand.date.max().date()}")
    if demand.date.max() < run_date:
        print(f"  ERROR: data ends before run_date {run_date.date()}")
        return 1

    # pad so future targets have matrix rows, exactly like forecast.py does
    last = demand.date.max()
    pad = [d for d in pd.date_range(run_date + pd.Timedelta(days=1), periods=S.HOLDOUT_DAYS)
           if d > last]
    padded = demand
    if pad:
        padded = pd.concat([demand, pd.DataFrame(
            {"date": pad, "demand": np.nan,
             "floortables": float(demand.floortables.iloc[-1])})],
            ignore_index=True).sort_values("date").reset_index(drop=True)

    base = F.build_base(padded, history_start=None, quiet=True)
    mat = F.apply_exclusions(base, demand, [])
    feats = F.feature_columns(mat)

    for T in targets:
        h = (T - run_date).days
        if h < 1 or h > S.HOLDOUT_DAYS:
            print(f"\n{T.date()}: horizon {h} outside 1..{S.HOLDOUT_DAYS}, skipped")
            continue
        row = mat[(mat["target_date"] == T) & (mat["horizon"] == h)]
        if row.empty:
            print(f"\n{T.date()}: no matrix row at horizon {h}")
            continue

        print("\n" + "=" * 78)
        print(f"{T.date()}   horizon {h}   (run {run_date.date()})")
        print("=" * 78)

        # 1) what each yearly lookback reads, and whether it survived the guard
        print("  yearly lookbacks -- source date, value in data, value in matrix:")
        src_of = {"lag_365": T - pd.Timedelta(days=365),
                  "lag_anchor_365": T - pd.Timedelta(days=365),
                  "lag_728": T - pd.Timedelta(days=728),
                  "same_holiday_lastyear_lag": T - pd.Timedelta(days=364)}
        with F.extended_holidays():
            align = S._holiday_alignment_map(comparable=True) \
                if "comparable" in S._holiday_alignment_map.__code__.co_varnames \
                else S._holiday_alignment_map()
        hit = align.get(T)
        if hit:
            src_of["holiday_yoy_lag"] = hit[2] + pd.Timedelta(days=hit[1])
        for c in YEARLY:
            if c not in row.columns:
                continue
            v = row[c].iloc[0]
            s = src_of.get(c)
            sv = dm.get(s) if s is not None else None
            print(f"    {c:26s} src={str(s.date()) if s is not None else '-':>12}"
                  f"  data={sv if sv is not None else '-':>8}"
                  f"  matrix={'NaN (guarded)' if pd.isna(v) else f'{v:,.3f}'}")

        # 2) SHAP: what actually moved this prediction
        train = mat[(mat["horizon"] == h) & (mat["target_date"] <= run_date)].dropna(subset=["y"])
        w = S.make_sample_weights(train["target_date"],
                                  is_holiday=S.holiday_mask_from_matrix(train),
                                  half_life_days=C.DEFAULT_HALF_LIFE)
        m = lgb.LGBMRegressor(**C.LGBM_PARAMS)
        m.fit(train[feats], train["y"], sample_weight=w)
        contrib = m.predict(row[feats], pred_contrib=True)
        bias, sh = float(contrib[0, -1]), contrib[0, :-1]
        pred = float(m.predict(row[feats])[0])
        s = pd.Series(sh, index=feats).sort_values(key=np.abs, ascending=False)

        print(f"\n  base {bias:,.0f}  ->  prediction {pred:,.0f}"
              f"   (trained on {len(train):,} rows <= {run_date.date()})")
        print(f"  top {a.top} contributions:")
        for f_, v in s.head(a.top).items():
            fv = row[f_].iloc[0]
            fvs = "NaN" if pd.isna(fv) else f"{fv:,.2f}"
            print(f"    {v:+10.1f}   {f_:28s} = {fvs}")
        yl = s[[c for c in YEARLY if c in s.index]]
        print(f"\n  yearly-lookback block net contribution: {yl.sum():+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
