"""backtest.py — Validate MAPE for production models on the last 28 days.

Holds out the most recent 28 days from data/raw/rawdata.csv, trains on
everything earlier, predicts the held-out 28 days using the SAME functions
forecast.py uses in production, and reports WAPE / MAPE / RMSE / Bias.

By default runs both:
    - LightGBM-L2 (the default in forecast.py)
    - 6-model Hybrid (the --full mode of forecast.py)

Usage:
    uv run python backtest.py                # both models (~20 min)
    uv run python backtest.py --fast         # LGBM-L2 only (~3 min)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # project root (scripts live in research/)
sys.path.insert(0, str(ROOT / "v2"))
from _shared import (  # noqa: E402
    HOLDOUT_DAYS, build_matrix, compute_metrics, load_demand, print_metrics,
)
sys.path.insert(0, str(ROOT))
from forecast import forecast_hybrid, forecast_lgbm_l2  # noqa: E402


def run_backtest(fast: bool, holdout_end: str | None = None) -> int:
    demand = load_demand()
    print(f"\n=== Backtest harness ===\n")
    print(f"  Data: {demand.date.min().date()} .. {demand.date.max().date()}  "
          f"({len(demand)} days)")

    if holdout_end:
        end_ts = pd.Timestamp(holdout_end).normalize()
        end_idx = demand.index[demand["date"] <= end_ts].max() + 1
        demand = demand.iloc[:end_idx].reset_index(drop=True)
        print(f"  Holdout end overridden to {end_ts.date()} -> "
              f"using {len(demand)} days of history")

    train_demand = demand.iloc[:-HOLDOUT_DAYS]
    test_demand = demand.iloc[-HOLDOUT_DAYS:].copy()
    test_dates = sorted(test_demand["date"].tolist())

    print(f"  Train: {train_demand.date.min().date()} .. "
          f"{train_demand.date.max().date()}  ({len(train_demand)} days)")
    print(f"  Test:  {test_demand.date.min().date()} .. "
          f"{test_demand.date.max().date()}  ({len(test_demand)} days)\n")

    print("  Building feature matrix...")
    t0 = time.time()
    mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS)
    print(f"  done in {time.time() - t0:.1f}s "
          f"({len(mat):,} rows × {len(mat.columns):,} cols)\n")

    train_dates_set = set(train_demand["date"])
    actuals = test_demand[["date", "demand"]].rename(columns={"demand": "y"})

    results = {}

    # --- LightGBM-L2 (forecast.py default) ---
    print("  [1/2] Training LightGBM-L2 (~3 min)...")
    t0 = time.time()
    preds = forecast_lgbm_l2(mat, train_dates_set, test_dates)
    merged = preds.merge(actuals, on="date", how="inner")
    m = compute_metrics(merged["y"], merged["p50"])
    m["seconds"] = time.time() - t0
    m["coverage"] = float(((merged["p10"] <= merged["y"]) &
                           (merged["y"] <= merged["p90"])).mean())
    results["LightGBM-L2 (default)"] = m
    print(f"  done in {m['seconds']:.0f}s")

    # --- 6-model Hybrid (forecast.py --full) ---
    if not fast:
        print("\n  [2/2] Training 6-model Hybrid (~15-20 min)...")
        t0 = time.time()
        preds = forecast_hybrid(mat, train_dates_set, test_dates)
        merged = preds.merge(actuals, on="date", how="inner")
        m = compute_metrics(merged["y"], merged["p50"])
        m["seconds"] = time.time() - t0
        m["coverage"] = float(((merged["p10"] <= merged["y"]) &
                               (merged["y"] <= merged["p90"])).mean())
        results["6-model Hybrid (--full)"] = m
        print(f"  done in {m['seconds']:.0f}s")

    print("\n=== Backtest results ===\n")
    for name, m in results.items():
        print(f"  {name}")
        print(f"    WAPE  = {m['wape']*100:6.2f}%")
        print(f"    MAPE  = {m['mape']*100:6.2f}%")
        print(f"    RMSE  = {m['rmse']:7.0f}")
        print(f"    Bias  = {m['bias']:+7.0f}")
        print(f"    P10-P90 coverage = {m['coverage']*100:5.1f}% "
              f"(target 80%)")
        print(f"    runtime = {m['seconds']:.0f}s\n")

    if not fast and len(results) == 2:
        lgbm_mape = results["LightGBM-L2 (default)"]["mape"]
        hybrid_mape = results["6-model Hybrid (--full)"]["mape"]
        delta = (lgbm_mape - hybrid_mape) * 100
        print(f"  Hybrid improvement over LGBM-L2: {delta:+.2f}pp MAPE")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="LGBM-L2 only (~3 min); skip the slow hybrid.")
    p.add_argument("--holdout-end", type=str, default=None,
                   help="ISO date; use this as the holdout's last day "
                        "(default: last date in rawdata.csv).")
    args = p.parse_args()
    return run_backtest(fast=args.fast, holdout_end=args.holdout_end)


if __name__ == "__main__":
    sys.exit(main())
