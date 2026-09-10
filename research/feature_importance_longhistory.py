"""feature_importance_longhistory.py -- what is the long-history model
actually paying attention to?

Trains real LightGBM models on the long-history matrix (today's shipped
"off" feature set, config.py defaults) at a handful of sampled horizons,
using only data strictly before a cutoff (no leakage), and sums gain-based
feature importance across them. Same method as research/tune_model.py's
rank_features(), which already established this pattern in this repo.

READ-ONLY -- imports long_history/{config,engine}.py read-only, writes
nothing back into long_history/ or v2/.

Usage:
    uv run python research/feature_importance_longhistory.py
    uv run python research/feature_importance_longhistory.py --cutoff 2025-01-01 --top 30
"""
from __future__ import annotations
import argparse
import re
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

import config as C          # noqa: E402
from engine import core as S      # noqa: E402
from engine import features as F  # noqa: E402

HORIZONS = (1, 4, 8, 14, 21, 28)

GROUP_PATTERNS = [
    ("lag_365 / lag_728 / yoy / anchor_365", re.compile(r"^(lag_365|lag_728|lag_anchor_365|yoy_ratio)$")),
    ("short lags (<365)",                     re.compile(r"^lag_\d+$")),
    ("holiday flags/offsets",                 re.compile(r"^(HOLIDAY_|is_holiday|holiday_)", re.I)),
    ("regime",                                re.compile(r"^regime")),
    ("calendar (dow/month/week/quarter/doy)", re.compile(r"^(dow|month|week_of|quarter|day_of_year)")),
    ("other",                                 re.compile(r".*")),
]


def group_of(col: str) -> str:
    for name, pat in GROUP_PATTERNS:
        if pat.match(col):
            return name
    return "other"


def rank_features(mat: pd.DataFrame, feats: list, cutoff: pd.Timestamp,
                  horizons=HORIZONS) -> pd.Series:
    tr_all = mat[mat["target_date"] < cutoff].dropna(subset=["y"])
    gains = pd.Series(0.0, index=feats)
    for h in horizons:
        sub = tr_all[tr_all["horizon"] == h]
        if len(sub) < 60:
            print(f"  horizon {h}: only {len(sub)} trainable rows, skipped")
            continue
        w = S.make_sample_weights(sub["target_date"],
                                  is_holiday=S.holiday_mask_from_matrix(sub),
                                  half_life_days=C.DEFAULT_HALF_LIFE)
        m = lgb.LGBMRegressor(**C.LGBM_PARAMS)
        m.fit(sub[feats], sub["y"], sample_weight=w)
        gains += pd.Series(m.booster_.feature_importance("gain"), index=feats)
    return gains.sort_values(ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=None,
                    help="train only on data before this date (default: "
                         "data max - 90d, so the ranking reflects a realistic "
                         "recent training window, not the full history)")
    ap.add_argument("--top", type=int, default=25, help="rows to print (default 25)")
    a = ap.parse_args()

    print("=" * 78)
    print("FEATURE IMPORTANCE -- LONG-HISTORY MODEL (shipped 'off' feature set)")
    print("=" * 78)

    demand = F.load_long_demand()
    cutoff = pd.Timestamp(a.cutoff) if a.cutoff else demand.date.max() - pd.Timedelta(days=90)
    print(f"data: {demand.date.min().date()} .. {demand.date.max().date()}  "
          f"cutoff: {cutoff.date()}  horizons sampled: {HORIZONS}")

    base = F.build_base(demand, history_start=None, quiet=True)
    mat = F.apply_exclusions(base, demand, [])
    feats = F.feature_columns(mat)
    print(f"matrix {mat.shape[0]:,} x {mat.shape[1]}  ({len(feats)} candidate features)\n")

    gains = rank_features(mat, feats, cutoff)
    total = gains.sum()
    if total <= 0:
        print("No gain recorded -- check cutoff / trainable rows.")
        return 1

    print(f"\nTop {a.top} features by summed gain (share of total):")
    for name, g in gains.head(a.top).items():
        print(f"  {name:35s} {g:14,.0f}   {100*g/total:5.1f}%")

    grouped = gains.groupby([group_of(c) for c in gains.index]).sum().sort_values(ascending=False)
    print("\nBy group (share of total gain):")
    for name, g in grouped.items():
        print(f"  {name:40s} {100*g/total:5.1f}%")

    out = ROOT / "research" / "lag_holiday_features_output" / "feature_importance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    gains.rename("gain").to_frame().assign(share=gains/total).to_csv(out)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
