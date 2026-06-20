"""forecast.py — produce a 28-day demand forecast and track it over time.

This is the PRODUCTION-style script:
  - Trains on ALL historical data (no holdout)
  - Predicts the next 28 days starting from --run-date + 1
  - Saves each run under forecasts/run_<YYYYMMDD>/
  - Builds a comparison chart showing how forecasts have evolved over runs

Default: uses LightGBM-L2 (best single model, ~3 min). Pass --full to run the
6-model hybrid ensemble of 5 v2 base GBMs + an inlined v3-style 2-stage
holiday specialist (~15 min). Note: the v3-style model is reimplemented inside
this script — the v3/ sub-folder is not imported at runtime.

Usage:
    uv run python forecast.py                            # run_date = last_data; forecasts last_data+1 .. +28
    uv run python forecast.py --run-date 2026-06-04
    uv run python forecast.py --run-date 2026-06-04 --full
    uv run python forecast.py --full --blend selection   # most accurate (selection + holiday anchor)

Day-by-day scheduling cycle:
    # Day 1 of the cycle — freeze the initial plan for a date range:
    uv run python forecast.py --full --blend selection --initial-forecast 2026-06-17 2026-07-14
    # Each following day — append yesterday's actual to rawdata.csv, then update:
    uv run python forecast.py --full --blend selection
    # Produces forecasts/tracking_comparison.png: actual vs initial plan vs latest forecast,
    # focused on the scheduling window.

Outputs to forecasts/run_<YYYYMMDD>/:
    predictions.csv     # date, p10, p50, p90
    forecast.png        # chart: 60-day actuals + 28-day forecast band
    metadata.json       # run info (model, train range, etc.)
Plus:
    forecasts/comparison.png  # overlay of last ~6 runs vs trailing actuals
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("lightgbm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import lightgbm as lgb

ROOT = Path(__file__).parent
FORECASTS = ROOT / "forecasts"
FORECASTS.mkdir(exist_ok=True)

# Use v2's feature engineering (Tier 1 enhanced, no broken trend features)
sys.path.insert(0, str(ROOT / "v2"))
from _shared import (
    HOLDOUT_DAYS,
    USE_RESERVATIONS,
    USE_WEATHER,
    build_matrix,
    holiday_mask_from_matrix,
    load_demand,
    make_sample_weights,
)


# =============================================================================
# Single-model forecaster (LGBM-L2)
# =============================================================================
def forecast_lgbm_l2(mat: pd.DataFrame, train_dates_set: set, test_dates: list):
    """Train LGBM-L2 per horizon on all historical, predict the future test_dates."""
    feature_cols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    train_mat = mat[mat["target_date"].isin(train_dates_set)].dropna(subset=["y"])

    rows = []
    for h in range(1, HOLDOUT_DAYS + 1):
        sub = train_mat[train_mat["horizon"] == h]
        if len(sub) < 30:
            continue
        X, y = sub[feature_cols], sub["y"]
        is_hol = holiday_mask_from_matrix(sub)
        weights = make_sample_weights(sub["target_date"], is_holiday=is_hol)

        m = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500, learning_rate=0.02, num_leaves=15, max_depth=5,
            min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, verbose=-1, random_state=123,
        )
        m.fit(X, y, sample_weight=weights)
        residuals = (y - m.predict(X)).values

        # Predict the matching future date for this horizon
        if h - 1 >= len(test_dates):
            continue
        td = test_dates[h - 1]
        row = mat[(mat["target_date"] == td) & (mat["horizon"] == h)]
        if row.empty:
            continue
        Xt = row[feature_cols]
        p50 = max(0.0, float(m.predict(Xt)[0]))
        q10 = float(np.quantile(residuals, 0.10))
        q90 = float(np.quantile(residuals, 0.90))
        p10 = max(0.0, p50 + q10)
        p90 = max(0.0, p50 + q90)
        # Enforce p10 <= p50 <= p90 symmetrically (residual quantiles can be
        # skewed enough to violate the ordering for well-fit models)
        p10, p90 = min(p10, p50), max(p50, p90)
        rows.append({"date": td, "p10": p10, "p50": p50, "p90": p90})

    return pd.DataFrame(rows)


# =============================================================================
# Optional: full hybrid (slower but slightly better)
# =============================================================================
def _train_2stage(X, y, weights, is_hol_mask):
    """v3 two-stage approach: baseline + holiday-residual model.

    Returns (baseline_model, residual_model_or_None).
    Predict via: y_base + (residual if is_holiday else 0)
    """
    # Stage 1: baseline on all days
    bm = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500, learning_rate=0.02, num_leaves=15, max_depth=5,
        min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1, random_state=123,
    )
    bm.fit(X, y, sample_weight=weights)
    y_base = bm.predict(X)
    residuals = y.values - y_base

    # Stage 2: residual model trained only on holiday-window rows
    rm = None
    if is_hol_mask.sum() >= 20:
        X_hol = X[is_hol_mask.values]
        r_hol = residuals[is_hol_mask.values]
        w_hol = weights[is_hol_mask.values]
        rm = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=4,
            min_child_samples=3, reg_alpha=0.3, reg_lambda=2.0,
            subsample=0.85, colsample_bytree=0.85,
            n_jobs=-1, verbose=-1, random_state=321,
        )
        rm.fit(X_hol, r_hol, sample_weight=w_hol)
    return bm, rm


def _train_all_models_one_horizon(X, y, weights, is_hol, Xt, is_hol_test):
    """Train the 6 base models for one horizon; return dict of point predictions
    for the prediction rows Xt, plus the LGBM-L2 in-sample residuals."""
    import xgboost as xgb
    from catboost import CatBoostRegressor

    preds = {}

    # 1. LGBM-L2
    m = lgb.LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.02,
        num_leaves=15, max_depth=5, min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=123)
    m.fit(X, y, sample_weight=weights)
    preds["lgbm_l2"] = np.maximum(0.0, m.predict(Xt))
    resid = y.values - m.predict(X)

    # 2. LGBM quantile P50
    m = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03,
        num_leaves=31, max_depth=7, min_child_samples=3, reg_alpha=0.1, reg_lambda=0.5,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1, verbose=-1, random_state=42)
    m.fit(X, y, sample_weight=weights)
    preds["lgbm_q"] = np.maximum(0.0, m.predict(Xt))

    # 3. XGBoost quantile
    m = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=0.5, n_estimators=400,
        learning_rate=0.03, max_depth=6, min_child_weight=1, reg_alpha=0.1, reg_lambda=0.5,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist", n_jobs=-1,
        random_state=42, verbosity=0)
    m.fit(X, y)
    preds["xgb"] = np.maximum(0.0, m.predict(Xt))

    # 4. CatBoost
    from catboost import CatBoostRegressor as _CB
    m = _CB(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
            loss_function="MAE", random_seed=42, verbose=False, allow_writing_files=False)
    m.fit(X, y, sample_weight=weights)
    preds["cat"] = np.maximum(0.0, m.predict(Xt))

    # 5. LGBM bagged (3 seeds)
    bag = []
    for seed in (42, 123, 777):
        m = lgb.LGBMRegressor(
            objective="regression", n_estimators=400, learning_rate=0.02,
            num_leaves=15, max_depth=5, min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
            subsample=0.85, colsample_bytree=0.85, bagging_fraction=0.85, bagging_freq=5,
            n_jobs=-1, verbose=-1, random_state=seed)
        m.fit(X, y, sample_weight=weights)
        bag.append(m.predict(Xt))
    preds["lgbm_bag"] = np.maximum(0.0, np.mean(bag, axis=0))

    # 6. v3 two-stage
    bm, rm = _train_2stage(X, y, weights, is_hol)
    base = bm.predict(Xt)
    residual = np.where(is_hol_test, rm.predict(Xt), 0.0) if rm is not None else np.zeros(len(Xt))
    preds["v3_2stage"] = np.maximum(0.0, base + residual)

    return preds, resid


def predict_all_models(mat: pd.DataFrame, train_dates_set: set,
                       targets: pd.DataFrame) -> tuple:
    """Train the 6 models per horizon on train_dates_set and predict the rows in
    `targets` (columns target_date, horizon). Returns (wide_df, residuals_per_h).

    wide_df has: target_date, horizon, dow, y, lag_365, is_hol, + one column per model.
    """
    from blend import MODEL_NAMES
    feature_cols = [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
    train_mat = mat[mat["target_date"].isin(train_dates_set)].dropna(subset=["y"])

    rows = []
    residuals_per_h = {}
    for h in range(1, HOLDOUT_DAYS + 1):
        sub = train_mat[train_mat["horizon"] == h]
        tgt = targets[targets["horizon"] == h]
        if len(sub) < 30 or len(tgt) == 0:
            continue
        X, y = sub[feature_cols], sub["y"]
        is_hol = holiday_mask_from_matrix(sub)
        weights = make_sample_weights(sub["target_date"], is_holiday=is_hol)

        pr = mat.merge(tgt[["target_date", "horizon"]], on=["target_date", "horizon"])
        pr = pr[pr["horizon"] == h]
        if pr.empty:
            continue
        Xt = pr[feature_cols]
        is_hol_test = holiday_mask_from_matrix(pr).values

        preds, resid = _train_all_models_one_horizon(X, y, weights, is_hol, Xt, is_hol_test)
        residuals_per_h[h] = resid

        pr2 = pr.reset_index(drop=True)
        for i in range(len(pr2)):
            row = {"target_date": pr2.loc[i, "target_date"], "horizon": h,
                   "y": pr2.loc[i, "y"], "lag_365": pr2.loc[i, "lag_365"],
                   "is_hol": int(is_hol_test[i])}
            for n in MODEL_NAMES:
                row[n] = float(preds[n][i])
            rows.append(row)

    wide = pd.DataFrame(rows)
    if not wide.empty:
        wide["dow"] = pd.to_datetime(wide["target_date"]).dt.weekday
    return wide, residuals_per_h


def _intervals_from_residuals(blend, residuals_per_h, horizons):
    """P10/P90 from per-horizon LGBM-L2 residual quantiles (in-sample proxy)."""
    p10_arr, p90_arr = [], []
    for i, h in enumerate(horizons):
        if h in residuals_per_h:
            r = residuals_per_h[h]
            base = blend[i]
            p10 = max(0.0, base + float(np.quantile(r, 0.10)))
            p90 = max(0.0, base + float(np.quantile(r, 0.90)))
            p10, p90 = min(p10, base), max(base, p90)
            p10_arr.append(p10); p90_arr.append(p90)
        else:
            p10_arr.append(np.nan); p90_arr.append(np.nan)
    return p10_arr, p90_arr


def forecast_hybrid(mat: pd.DataFrame, train_dates_set: set, test_dates: list):
    """6-model hybrid with EQUAL-weight blending (the original --full behavior)."""
    natural = pd.DataFrame({"target_date": test_dates,
                            "horizon": range(1, len(test_dates) + 1)})
    wide, residuals_per_h = predict_all_models(mat, train_dates_set, natural)
    from blend import MODEL_NAMES
    wide = wide.sort_values("horizon").reset_index(drop=True)
    blend = wide[MODEL_NAMES].mean(axis=1).values
    p10, p90 = _intervals_from_residuals(blend, residuals_per_h, wide["horizon"].tolist())
    return pd.DataFrame({"date": wide["target_date"].values,
                         "p10": p10, "p50": blend, "p90": p90})


def forecast_hybrid_selection(mat: pd.DataFrame, demand: pd.DataFrame,
                              train_dates_set: set, test_dates: list,
                              run_date: pd.Timestamp, val_days: int = 180):
    """6-model hybrid with honest per-(DOW,horizon) selection + holiday anchor.

    Two training rounds:
      1. Train on data before a recent VALIDATION window; predict that window to
         build the selection table and calibrate the holiday anchor.
      2. Train on ALL data; predict the future; apply selection + anchor.
    """
    import blend as B

    train_sorted = sorted(train_dates_set)
    val_end = train_sorted[-1]
    val_start = val_end - pd.Timedelta(days=val_days - 1)
    val_dates = [d for d in train_sorted if d >= val_start]
    sel_train = set(d for d in train_sorted if d < val_start)

    # Round 1: validation predictions at ALL horizons (for a robust table)
    print(f"    [selection] round 1: calibrating on validation "
          f"{val_dates[0].date()}..{val_dates[-1].date()}")
    val_targets = pd.DataFrame(
        [(d, h) for d in val_dates for h in range(1, HOLDOUT_DAYS + 1)],
        columns=["target_date", "horizon"])
    val_wide, _ = predict_all_models(mat, sel_train, val_targets)
    table = B.build_selection_table(val_wide)
    val_blend = B.apply_selection(val_wide, table)
    anchor_params = B.calibrate_anchor(val_wide, val_blend, demand, pd.Timestamp(val_start))
    print(f"    [selection] anchor params (growth_window, post, alpha) = {anchor_params}")

    # Round 2: production predictions at natural horizon
    print(f"    [selection] round 2: training on all data, predicting future")
    natural = pd.DataFrame({"target_date": test_dates,
                            "horizon": range(1, len(test_dates) + 1)})
    prod_wide, residuals_per_h = predict_all_models(mat, train_dates_set, natural)
    prod_wide = prod_wide.sort_values("horizon").reset_index(drop=True)

    blend = B.apply_selection(prod_wide, table)
    forecast_start = run_date + pd.Timedelta(days=1)
    blend = B.apply_anchor(prod_wide, blend, demand, forecast_start, anchor_params)

    p10, p90 = _intervals_from_residuals(blend, residuals_per_h, prod_wide["horizon"].tolist())
    return pd.DataFrame({"date": prod_wide["target_date"].values,
                         "p10": p10, "p50": blend, "p90": p90})


# =============================================================================
# Orchestration
# =============================================================================
def make_forecast(run_date: pd.Timestamp, use_hybrid: bool = False,
                  blend: str = "equal") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce 28-day forecast starting from run_date + 1. Returns (predictions, demand)."""
    demand = load_demand()
    last_data = demand.date.max()
    print(f"  Historical data:    {demand.date.min().date()} .. {last_data.date()} ({len(demand)} days)")
    print(f"  Run date:           {run_date.date()}")

    # Sanity: data must extend to at least run_date
    if last_data < run_date:
        raise ValueError(
            f"rawdata.csv ends {last_data.date()} but run_date is {run_date.date()}. "
            "Update rawdata.csv before forecasting from this run date."
        )
    # If data extends BEYOND run_date (historical/backtest rerun), warn — the
    # train-set filter below will exclude those future rows to prevent leakage.
    if last_data > run_date:
        future_rows = int((demand.date > run_date).sum())
        print(f"  [warning] rawdata.csv extends {future_rows} day(s) past run_date — "
              f"those rows will be EXCLUDED from training to prevent leakage.")

    future_targets = pd.date_range(
        run_date + pd.Timedelta(days=1), periods=HOLDOUT_DAYS, freq="D"
    )
    print(f"  Forecasting:        {future_targets[0].date()} .. {future_targets[-1].date()}")

    # Pad demand with NaN-target rows for future dates so build_matrix has rows for them
    pad_dates = [d for d in future_targets if d > last_data]
    if pad_dates:
        pad = pd.DataFrame({
            "date": pad_dates,
            "demand": np.nan,
            "floortables": float(demand.floortables.iloc[-1]),
        })
        padded = pd.concat([demand, pad], ignore_index=True).sort_values("date").reset_index(drop=True)
    else:
        padded = demand.copy()

    print("  Building feature matrix...")
    mat = build_matrix(padded, holdout_days=HOLDOUT_DAYS)

    # Train on all historical data up to and including run_date.
    # In production this is a no-op (last_data == run_date); on backtest reruns
    # it prevents future-data leakage into training.
    train_dates_set = set(d for d in demand.date if d <= run_date)
    test_dates = sorted(future_targets.tolist())

    if use_hybrid and blend == "selection":
        print("  Training 6-model HYBRID + selection/anchor (2 rounds, ~30-40 min)...")
        preds = forecast_hybrid_selection(mat, demand, train_dates_set, test_dates, run_date)
    elif use_hybrid:
        print("  Training 6-model HYBRID, equal blend (~15-20 min)...")
        preds = forecast_hybrid(mat, train_dates_set, test_dates)
    else:
        print("  Training LightGBM-L2 (~3 min)...")
        preds = forecast_lgbm_l2(mat, train_dates_set, test_dates)

    return preds, demand


def save_outputs(preds: pd.DataFrame, run_date: pd.Timestamp, demand: pd.DataFrame,
                  use_hybrid: bool, blend: str = "equal",
                  initial_range: tuple | None = None) -> Path:
    """Save predictions.csv + forecast.png + metadata.json to forecasts/run_<YYYYMMDD>/."""
    stamp = run_date.strftime("%Y%m%d")
    run_dir = FORECASTS / f"run_{stamp}"
    run_dir.mkdir(exist_ok=True)

    csv_path = run_dir / "predictions.csv"
    preds.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    if use_hybrid and blend == "selection":
        model_name = "hybrid_6model_selection_anchor"
    elif use_hybrid:
        model_name = "hybrid_6model_equal"
    else:
        model_name = "lightgbm_l2_v2"
    init_range = ([str(initial_range[0].date()), str(initial_range[1].date())]
                  if initial_range else None)
    meta = {
        "run_date": str(run_date.date()),
        "model": model_name,
        "initial_forecast": initial_range is not None,
        "initial_forecast_range": init_range,
        "horizon_days": HOLDOUT_DAYS,
        "historical_data_range": [str(demand.date.min().date()),
                                    str(demand.date.max().date())],
        "forecast_range": [str(preds.date.min().date()),
                            str(preds.date.max().date())],
        "p50_min": float(preds.p50.min()),
        "p50_max": float(preds.p50.max()),
        "p50_mean": float(preds.p50.mean()),
        "p50_sum": float(preds.p50.sum()),
        "use_reservations": bool(USE_RESERVATIONS),
        "use_weather": bool(USE_WEATHER),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  -> {run_dir / 'metadata.json'}")

    # Chart
    fig, ax = plt.subplots(figsize=(13, 5))
    recent = demand.tail(60)
    ax.plot(recent.date, recent.demand, "o-", color="black", markersize=3, alpha=0.6,
            label="actual (trailing 60d)")
    ax.plot(preds.date, preds.p50, "s-", color="C0", markersize=4, label="P50 forecast")
    ax.fill_between(preds.date, preds.p10, preds.p90, color="C0", alpha=0.2,
                    label="P10-P90 interval")
    ax.axvline(run_date, color="red", linestyle="--", alpha=0.5,
               label=f"run date ({run_date.date()})")
    ax.set_xlabel("date"); ax.set_ylabel("demand (patron hours)")
    ax.set_title(f"28-day forecast | run date: {run_date.date()} | "
                 f"model: {meta['model']}")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); plt.xticks(rotation=30); plt.tight_layout()
    chart_path = run_dir / "forecast.png"
    plt.savefig(chart_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  -> {chart_path}")

    return run_dir


def make_comparison_chart(current_run_date: pd.Timestamp, demand: pd.DataFrame) -> None:
    """Overlay current + up to 5 prior forecasts on one chart."""
    runs = sorted(FORECASTS.glob("run_*/predictions.csv"))
    if len(runs) <= 1:
        print("  (no prior runs to compare — skipping comparison chart)")
        return

    # Take the latest 6 runs (including current)
    runs = runs[-6:]
    fig, ax = plt.subplots(figsize=(14, 6))
    recent = demand.tail(60)
    ax.plot(recent.date, recent.demand, "o-", color="black", markersize=3, alpha=0.6,
            label="actual (trailing 60d)")

    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(runs)))
    for color, run_csv in zip(colors, runs):
        stamp = run_csv.parent.name.replace("run_", "")
        run_date = pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}")
        p = pd.read_csv(run_csv, parse_dates=["date"])
        is_current = (run_date == current_run_date)
        lw = 2.5 if is_current else 1
        alpha = 1.0 if is_current else 0.45
        label = f"{run_date.date()}" + (" ⭐ current" if is_current else "")
        ax.plot(p.date, p.p50, "s-", color=color, markersize=3, linewidth=lw,
                alpha=alpha, label=label)

    ax.set_xlabel("date"); ax.set_ylabel("demand (patron hours)")
    ax.set_title(f"Forecast comparison — {len(runs)} most recent runs")
    ax.legend(loc="upper left", fontsize=9, title="Run date")
    ax.grid(alpha=0.3); plt.xticks(rotation=30); plt.tight_layout()

    cmp_path = FORECASTS / "comparison.png"
    plt.savefig(cmp_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  -> {cmp_path}")


def _load_runs() -> list:
    """Scan forecasts/run_*/metadata.json. Return list of dicts sorted by run_date."""
    runs = []
    for md in FORECASTS.glob("run_*/metadata.json"):
        try:
            meta = json.loads(md.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        pred = md.parent / "predictions.csv"
        if not pred.exists():
            continue
        rng = meta.get("initial_forecast_range")
        runs.append({
            "run_date": pd.Timestamp(meta["run_date"]),
            "initial": bool(meta.get("initial_forecast", False)),
            "range": (pd.Timestamp(rng[0]), pd.Timestamp(rng[1])) if rng else None,
            "pred_path": pred,
        })
    return sorted(runs, key=lambda r: r["run_date"])


def make_tracking_chart(demand: pd.DataFrame) -> None:
    """Overlay actual demand, the frozen INITIAL scheduling plan, and the most
    recent forecast — the day-by-day tracking view for one planning cycle.

    The 'initial' plan is the most recent run saved with an --initial-forecast
    FROM TO range; the chart focuses on that scheduling window.
    """
    runs = _load_runs()
    if not runs:
        print("  (no runs found — skipping tracking chart)")
        return
    initials = [r for r in runs if r["initial"] and r["range"] is not None]
    if not initials:
        print("  (no run with an --initial-forecast date range yet — skipping tracking chart)")
        return

    initial = initials[-1]            # current cycle's frozen baseline
    latest = runs[-1]                 # most recent forecast overall
    win_from, win_to = initial["range"]   # the scheduling window

    init_p = pd.read_csv(initial["pred_path"], parse_dates=["date"])
    late_p = pd.read_csv(latest["pred_path"], parse_dates=["date"])

    # Clip every series to the scheduling window
    init_p = init_p[(init_p.date >= win_from) & (init_p.date <= win_to)]
    late_p = late_p[(late_p.date >= win_from) & (late_p.date <= win_to)]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Initial plan: P50 line + P10-P90 band (the planning uncertainty)
    if not init_p.empty:
        ax.fill_between(init_p.date, init_p.p10, init_p.p90, color="C1", alpha=0.12,
                        label="initial P10–P90")
        ax.plot(init_p.date, init_p.p50, "s--", color="C1", markersize=4, linewidth=2,
                label=f"initial plan ({initial['run_date'].date()})")

    # Latest forecast (only if it's a different run and overlaps the window)
    if latest["run_date"] != initial["run_date"] and not late_p.empty:
        ax.plot(late_p.date, late_p.p50, "^-", color="C0", markersize=4, linewidth=2,
                label=f"latest forecast ({latest['run_date'].date()})")

    # Actual demand within the window (plus a short lead-in for context)
    act = demand[(demand.date >= win_from - pd.Timedelta(days=5)) &
                 (demand.date <= win_to)]
    ax.plot(act.date, act.demand, "o-", color="black", markersize=4, linewidth=1.8,
            label="actual demand")

    # Mark "today" (latest actual) for orientation
    if not act.empty:
        last_actual = act.date.max()
        ax.axvline(last_actual, color="grey", linestyle=":", alpha=0.7,
                   label=f"latest actual ({last_actual.date()})")

    ax.set_xlabel("date"); ax.set_ylabel("demand (patron hours)")
    ax.set_title(f"Scheduling tracking {win_from.date()} .. {win_to.date()} — "
                 "actual vs initial plan vs latest forecast")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3); plt.xticks(rotation=30); plt.tight_layout()

    out = FORECASTS / "tracking_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")

    # CSV joining the three series across the scheduling window
    window_dates = pd.date_range(win_from, win_to)
    merged = pd.DataFrame({"date": window_dates})
    merged = merged.merge(init_p[["date", "p50"]].rename(columns={"p50": "initial_p50"}),
                          on="date", how="left")
    merged = merged.merge(late_p[["date", "p50"]].rename(columns={"p50": "latest_p50"}),
                          on="date", how="left")
    merged = merged.merge(demand[["date", "demand"]].rename(columns={"demand": "actual"}),
                          on="date", how="left")
    track_csv = FORECASTS / "tracking_comparison.csv"
    merged.to_csv(track_csv, index=False)
    print(f"  -> {track_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-date", type=str, default=None,
                        help="ISO date, e.g. 2026-06-04. Default: rawdata.csv max date.")
    parser.add_argument("--full", action="store_true",
                        help="Use the 6-model hybrid blend (slower; default is LGBM-L2 only)")
    parser.add_argument("--blend", choices=["equal", "selection"], default="equal",
                        help="Hybrid blend method (only with --full). 'equal' = mean of "
                             "6 models (~15-20 min). 'selection' = per-(DOW,horizon) top-2 "
                             "+ holiday anchor, calibrated on a validation window "
                             "(~30-40 min; more accurate on holiday windows).")
    parser.add_argument("--initial-forecast", "--initial_forecast", dest="initial_forecast",
                        nargs=2, metavar=("FROM", "TO"), default=None,
                        help="Two ISO dates FROM TO marking this run as the frozen INITIAL "
                             "scheduling plan covering that date range (the tracking chart "
                             "focuses on this window). Omit for a regular daily update. "
                             "e.g. --initial-forecast 2026-06-17 2026-07-14")
    args = parser.parse_args()

    if args.blend == "selection" and not args.full:
        print("[note] --blend selection requires --full; enabling --full.")
        args.full = True

    demand = load_demand()

    initial_range = None
    if args.initial_forecast:
        initial_range = (pd.Timestamp(args.initial_forecast[0]).normalize(),
                         pd.Timestamp(args.initial_forecast[1]).normalize())
        if initial_range[0] > initial_range[1]:
            parser.error("--initial-forecast FROM must be on or before TO")
        span = (initial_range[1] - initial_range[0]).days + 1
        if span != HOLDOUT_DAYS:
            print(f"  [warning] initial-forecast range spans {span} days, but the model "
                  f"forecasts {HOLDOUT_DAYS} days from run-date+1; only the overlap is charted.")

    # run-date precedence: explicit --run-date, else derive from initial FROM (FROM-1),
    # else the last row of rawdata.csv.
    if args.run_date:
        run_date = pd.Timestamp(args.run_date).normalize()
    elif initial_range:
        run_date = (initial_range[0] - pd.Timedelta(days=1)).normalize()
    else:
        run_date = demand.date.max().normalize()

    print(f"\n=== Forecast run ===")
    if initial_range:
        print(f"  [initial-forecast] FROZEN scheduling plan for "
              f"{initial_range[0].date()} .. {initial_range[1].date()}")
    preds, demand = make_forecast(run_date, use_hybrid=args.full, blend=args.blend)
    print(f"\n=== Saving outputs ===")
    run_dir = save_outputs(preds, run_date, demand, use_hybrid=args.full, blend=args.blend,
                           initial_range=initial_range)
    print(f"\n=== Building comparison chart ===")
    make_comparison_chart(run_date, demand)
    print(f"\n=== Building scheduling tracking chart ===")
    make_tracking_chart(demand)
    print(f"\n=== Done — open {run_dir / 'forecast.png'} ===\n")

    # Quick summary
    print(f"P50 statistics:")
    print(f"  min:  {preds.p50.min():.0f}")
    print(f"  max:  {preds.p50.max():.0f}")
    print(f"  mean: {preds.p50.mean():.0f}")
    print(f"  sum:  {preds.p50.sum():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
