"""make_deck_data.py — generate the actual-vs-forecast series used by the deck.

Runs the long-history model over several periods and lead times, all through
the same honest backtest used everywhere else (evaluate_long.rolling_backtest),
and writes one JSON the slide generator reads.

Every series here is held-out: for each origin the model trained only on data
up to that origin, so nothing it predicts was ever seen during training.

Usage:
    uv run python long_history/make_deck_data.py
    uv run python long_history/make_deck_data.py --skip-2025    # faster
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/ -- this package owns its data
sys.path.insert(0, str(ROOT))

import config as C
from training import evaluate as E

OUT = ROOT / "docs" / "deck_data_long.json"

# (key, start, end, n, mode) -- n+1 is the lead time in days
RUNS = [
    ("lead01_2026", "2026-01-01", "2026-05-28", 0, "fixed_lead"),
    ("lead07_2026", "2026-01-01", "2026-05-28", 6, "rolling_window"),
    ("lead14_2026", "2026-01-01", "2026-05-28", 13, "rolling_window"),
    ("lead28_2026", "2026-01-01", "2026-05-28", 27, "rolling_window"),
]
RUN_2025 = ("lead07_2025", "2025-01-01", "2025-12-31", 6, "rolling_window")


def series(df: pd.DataFrame) -> dict:
    d = df[(df.actual > 0) & df.forecast.notna()].sort_values("target_date")
    d = d.drop_duplicates("target_date")
    ape = np.abs(d.forecast - d.actual) / d.actual * 100
    return {
        "labels": [x.strftime("%d %b") for x in d.target_date],
        "iso": [x.strftime("%Y-%m-%d") for x in d.target_date],
        "actual": [round(float(v)) for v in d.actual],
        "forecast": [round(float(v)) for v in d.forecast],
        "holiday": [str(h) if isinstance(h, str) else "" for h in d.holiday],
        "mape": round(float(ape.mean()), 2),
        "bias": round(float(((d.forecast - d.actual) / d.actual * 100).mean()), 2),
        "n": int(len(d)),
        "within5": round(float((ape <= 5).mean() * 100), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-2025", action="store_true")
    a = ap.parse_args()

    runs = list(RUNS) + ([] if a.skip_2025 else [RUN_2025])
    out = {}

    for key, start, end, n, mode in runs:
        print(f"\n=== {key}: {start}..{end}, lead {n+1}d, {mode} ===", flush=True)
        t0 = time.time()
        df = E.rolling_backtest(start, end, n=n, mode=mode, verbose=True)
        out[key] = series(df)
        out[key]["lead"] = n + 1
        print(f"  {out[key]['n']} days, MAPE {out[key]['mape']}%  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # Production vs long-history on the identical protocol (2026, lead 7,
    # rolling_window). The production side was produced earlier by the same
    # rolling-window backtest, so the two are directly comparable.
    prod = ROOT / "research" / "narrow_mask_output" / "guard_off.csv"
    if prod.exists() and "lead07_2026" in out:
        p = pd.read_csv(prod, parse_dates=["target_date"])
        p = p[(p.actual > 0) & p.forecast.notna()].sort_values("target_date")
        p = p.drop_duplicates("target_date")
        lh = pd.DataFrame({"iso": out["lead07_2026"]["iso"],
                           "lh": out["lead07_2026"]["forecast"],
                           "actual": out["lead07_2026"]["actual"]})
        p["iso"] = p.target_date.dt.strftime("%Y-%m-%d")
        m = lh.merge(p[["iso", "forecast"]], on="iso", how="inner")
        if len(m):
            ape = lambda c: round(float((np.abs(m[c] - m.actual) / m.actual * 100).mean()), 2)
            out["compare_2026"] = {
                "labels": [pd.Timestamp(x).strftime("%d %b") for x in m.iso],
                "actual": [int(v) for v in m.actual],
                "long_history": [round(float(v)) for v in m.lh],
                "production": [round(float(v)) for v in m.forecast],
                "mape_long": ape("lh"), "mape_prod": ape("forecast"),
                "n": int(len(m)),
            }
            print(f"\ncomparison built: n={len(m)}  "
                  f"long-history {out['compare_2026']['mape_long']}%  "
                  f"production {out['compare_2026']['mape_prod']}%")
        else:
            print("\n[warn] no overlapping dates for the production comparison")
    else:
        print("\n[warn] production baseline not found -- comparison slide skipped")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT}")
    for k, v in out.items():
        if "mape" in v:
            print(f"   {k:14s} n={v['n']:4d}  MAPE {v['mape']:5.2f}%  "
                  f"bias {v['bias']:+5.2f}%  within 5%: {v['within5']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
