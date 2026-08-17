"""make_model_demo.py — real numbers for the "how the model works" slides.

Produces two things, both from a genuinely trained model rather than invented
for illustration:

  1. BOOSTING CONVERGENCE. One real forecast, re-read after 1, 2, 5, 10 ...
     500 trees, so you can show the prediction walking toward the actual as
     each new tree corrects the last one's mistake.

  2. A REAL DECISION RULE. The first tree's top split, in plain language, so
     the "it asks yes/no questions" explanation can point at something the
     model actually did.

Usage:
    uv run python long_history/make_model_demo.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/ -- this package owns its data
sys.path.insert(0, str(ROOT))

import config as C
from engine import features as F
from engine.core import holiday_mask_from_matrix, make_sample_weights

OUT = ROOT / "docs" / "deck_model_demo.json"

ORIGIN = pd.Timestamp("2026-05-21")   # forecast made here
HORIZON = 7                            # for 2026-05-28
STAGES = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 175, 300, 500]


def main() -> int:
    demand = F.load_long_demand()
    dm = dict(zip(demand.date, demand.demand.astype(float)))
    target = ORIGIN + pd.Timedelta(days=HORIZON)
    actual = float(dm[target])

    base = F.build_base(demand, history_start=None, quiet=True)
    mat = F.apply_exclusions(base, demand, [])
    feats = F.feature_columns(mat)

    sub = mat[(mat.target_date <= ORIGIN) & (mat.horizon == HORIZON)].dropna(subset=["y"])
    pr = mat[(mat.target_date == target) & (mat.horizon == HORIZON)]
    print(f"target {target.date()}  actual {actual:,.0f}   training rows {len(sub):,}")

    w = make_sample_weights(sub.target_date,
                            is_holiday=holiday_mask_from_matrix(sub),
                            half_life_days=C.DEFAULT_HALF_LIFE)
    m = lgb.LGBMRegressor(**C.LGBM_PARAMS)
    m.fit(sub[feats], sub["y"], sample_weight=w)

    # prediction after N trees
    preds = [float(m.predict(pr[feats], num_iteration=n)[0]) for n in STAGES]
    errs = [round(abs(p - actual) / actual * 100, 2) for p in preds]

    # the very first tree's top split, in words
    td = m.booster_.dump_model()["tree_info"][0]["tree_structure"]
    feat_name = feats[td["split_feature"]]
    thresh = float(td["threshold"])
    first = {
        "feature": feat_name,
        "threshold": round(thresh, 2),
        "left_leaf": round(float(td["left_child"].get("leaf_value", np.nan)), 1)
        if "leaf_value" in td.get("left_child", {}) else None,
        "right_leaf": round(float(td["right_child"].get("leaf_value", np.nan)), 1)
        if "leaf_value" in td.get("right_child", {}) else None,
    }

    out = {
        "origin": str(ORIGIN.date()),
        "target": str(target.date()),
        "horizon": HORIZON,
        "actual": round(actual),
        "stages": STAGES,
        "preds": [round(p) for p in preds],
        "errs": errs,
        "n_train": int(len(sub)),
        "first_split": first,
        "final": round(preds[-1]),
        "final_err": errs[-1],
    }
    OUT.write_text(json.dumps(out, indent=1))

    print(f"\nfirst tree splits on: {feat_name} <= {thresh:,.1f}")
    print("\n  trees      prediction    miss")
    for n, p, e in zip(STAGES, preds, errs):
        print(f"  {n:5d}      {p:9,.0f}   {e:5.2f}%")
    print(f"\nactual {actual:,.0f}   final {preds[-1]:,.0f}   ({errs[-1]}%)")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
