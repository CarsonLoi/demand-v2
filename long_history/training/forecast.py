"""forecast_long.py — produce a 28-day forecast using the long-history setup.

The long-history counterpart of the production `forecast.py`. Fully
self-contained: reads `core.py` (this folder's vendored feature engine), never
touches production code or production output.

Differences from production forecast.py:
  - trains on ALL available history, not just 2024+
  - recency half-life from config (default 1095 days, not 240)
  - COVID and full-closure days excluded from training targets
  - yearly lookbacks guarded where their source falls in an excluded period
  - holiday anchors cover 2016-2030, with Ching Ming corrected
  - adds week-of-year and regime features

Usage:
    uv run python long_history/forecast_long.py
    uv run python long_history/forecast_long.py --run-date 2026-04-30
    uv run python long_history/forecast_long.py --half-life 720 --no-anchor

Outputs to long_history/output/run_<YYYYMMDD>/:
    predictions.csv   date, p10, p50, p90
    forecast.png      trailing actuals + the 28-day band
    metadata.json     full run configuration, for auditability
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/ -- this package owns its data
sys.path.insert(0, str(ROOT))

import config as C
from engine import features as F
from engine import core as S
from engine.core import HOLDOUT_DAYS, holiday_mask_from_matrix, make_sample_weights

NAVY, TEAL, GREY, GOLD = "#0F2942", "#2E8BA8", "#5F7183", "#E8B547"


def forecast(demand, run_date, half_life, anchor=True, quiet=False):
    """28-day forecast from run_date+1, with residual-based P10/P90."""
    run_date = pd.Timestamp(run_date).normalize()
    last = demand.date.max()
    if last < run_date:
        raise ValueError(f"data ends {last.date()} but run_date is {run_date.date()}")
    if last > run_date:
        n = int((demand.date > run_date).sum())
        print(f"  [guard] {n} row(s) after run_date will be EXCLUDED from training")

    targets = pd.date_range(run_date + pd.Timedelta(days=1), periods=HOLDOUT_DAYS)
    print(f"  history : {demand.date.min().date()} .. {last.date()} ({len(demand):,} days)")
    print(f"  run date: {run_date.date()}")
    print(f"  forecast: {targets[0].date()} .. {targets[-1].date()}")

    # pad future dates so the matrix has rows for them
    pad = [d for d in targets if d > last]
    if pad:
        padded = pd.concat([demand, pd.DataFrame({
            "date": pad, "demand": np.nan,
            "floortables": float(demand.floortables.iloc[-1])})],
            ignore_index=True).sort_values("date").reset_index(drop=True)
    else:
        padded = demand.copy()

    t0 = time.time()
    base = F.build_base(padded, history_start=None, quiet=quiet)
    mat = F.apply_exclusions(base, demand, [])
    print(f"  matrix  : {mat.shape[0]:,} x {mat.shape[1]} "
          f"({int(mat.y.notna().sum()):,} trainable)  [{time.time()-t0:.0f}s]")

    feats = F.feature_columns(mat)
    train = mat[mat["target_date"] <= run_date].dropna(subset=["y"])

    rows = []
    for h in range(1, HOLDOUT_DAYS + 1):
        sub = train[train["horizon"] == h]
        if len(sub) < 30:
            continue
        T = targets[h - 1]
        pr = mat[(mat["target_date"] == T) & (mat["horizon"] == h)]
        if pr.empty:
            continue
        w = make_sample_weights(sub["target_date"],
                                is_holiday=holiday_mask_from_matrix(sub),
                                half_life_days=half_life)
        m = lgb.LGBMRegressor(**C.LGBM_PARAMS)
        m.fit(sub[feats], sub["y"], sample_weight=w)
        p50 = max(0.0, float(m.predict(pr[feats])[0]))
        # NOTE: in-sample residuals -- these intervals are known to be too
        # narrow (~46-54% coverage vs a nominal 80%). Same limitation as
        # production. Treat the band as relative confidence, not a guarantee.
        res = sub["y"].values - m.predict(sub[feats])
        p10 = min(max(0.0, p50 + float(np.quantile(res, .10))), p50)
        p90 = max(p50 + float(np.quantile(res, .90)), p50)
        rows.append({"date": T, "p10": p10, "p50": p50, "p90": p90})

    preds = pd.DataFrame(rows)
    if anchor and not preds.empty:
        with F.extended_holidays():
            preds = S.apply_moving_anchor(preds, demand, run_date)
    return preds


def chart(preds, demand, run_date, path, half_life):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    recent = demand[demand.date <= run_date].tail(75)
    ax.plot(recent.date, recent.demand, "o-", color=NAVY, ms=3, lw=1.5,
            alpha=.85, label="actual (trailing 75d)")
    ax.plot(preds.date, preds.p50, "s--", color=TEAL, ms=4, lw=1.8, label="P50 forecast")
    ax.fill_between(preds.date, preds.p10, preds.p90, color=TEAL, alpha=.18,
                    label="P10–P90 (see caveat)")
    ax.axvline(run_date, color=GOLD, ls="--", lw=1.6, alpha=.9,
               label=f"run date ({run_date.date()})")
    ax.set_xlabel("date"); ax.set_ylabel("demand (patron hours)")
    ax.set_title(f"Long-history 28-day forecast  ·  run {run_date.date()}  ·  "
                 f"half-life {half_life}d", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-date", default=None, help="ISO date; default = last row")
    ap.add_argument("--half-life", type=int, default=None,
                    help=f"recency half-life in days (default {C.DEFAULT_HALF_LIFE})")
    ap.add_argument("--no-anchor", action="store_true", help="skip the CNY correction")
    a = ap.parse_args()

    from data.validate import validate
    if validate(ROOT / C.DATA_FILE, verbose=True)[0]:
        print("\nABORTED: fix the data errors above first.")
        return 1

    demand = F.load_long_demand()
    run_date = pd.Timestamp(a.run_date) if a.run_date else demand.date.max()
    hl = a.half_life or C.DEFAULT_HALF_LIFE

    print(f"\n=== Long-history forecast ===")
    preds = forecast(demand, run_date, hl, anchor=not a.no_anchor)
    if preds.empty:
        print("no predictions produced"); return 1

    outdir = ROOT / C.OUT_DIR / f"run_{pd.Timestamp(run_date):%Y%m%d}"
    outdir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(outdir / "predictions.csv", index=False)
    chart(preds, demand, pd.Timestamp(run_date), outdir / "forecast.png", hl)
    (outdir / "metadata.json").write_text(json.dumps({
        "run_date": str(pd.Timestamp(run_date).date()),
        "pipeline": "long_history",
        "data_file": C.DATA_FILE,
        "history": [str(demand.date.min().date()), str(demand.date.max().date())],
        "half_life_days": hl,
        "cny_anchor": not a.no_anchor,
        "excluded_from_training": C.EXCLUDE_FROM_TRAINING,
        "exclude_closures": C.EXCLUDE_CLOSURES,
        "guard_yearly_lags": C.GUARD_LAGS_CROSSING_EXCLUDED,
        "week_of_year": C.ADD_WEEK_OF_YEAR,
        "regime_feature": C.ADD_REGIME_FEATURE,
        "p50": {"min": float(preds.p50.min()), "max": float(preds.p50.max()),
                "mean": float(preds.p50.mean()), "sum": float(preds.p50.sum())},
        "interval_caveat": "P10-P90 from in-sample residuals; measured coverage "
                           "46-54% vs nominal 80%. Relative signal only.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    print(f"\n  -> {outdir/'predictions.csv'}")
    print(f"  -> {outdir/'forecast.png'}")
    print(f"  -> {outdir/'metadata.json'}")
    print(f"\nP50: min {preds.p50.min():,.0f}  max {preds.p50.max():,.0f}  "
          f"mean {preds.p50.mean():,.0f}  sum {preds.p50.sum():,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
