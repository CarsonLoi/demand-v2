"""evaluate.py — actual vs forecast comparison over a selectable date range
and a selectable forecast lead time.

The core idea
-------------
For a target date `t`, a forecast made using data ONLY up to `t-1` is a
1-day-ahead forecast. Using data up to `t-1` to predict `t .. t+n` means the
last day of that window (`t+n`) is an (n+1)-day-ahead forecast.

`n` therefore controls HOW FAR AHEAD the forecast was made:
    n = 0   -> forecast made with data up to the day before  (1 day ahead)
    n = 10  -> forecast made with data up to 11 days before  (11 days ahead)

Two evaluation modes
--------------------
fixed_lead      (default) Every plotted point uses the SAME lead time (n+1
                days). Answers: "how accurate are we at N days out?" Because
                only one horizon is needed per origin, this is fast.

rolling_window  Origins step by (n+1) days; each origin contributes a full
                `t .. t+n` segment, tiling the range. Answers: "what did our
                operational forecast actually look like?" — this mirrors
                re-forecasting every (n+1) days.

Every run is honest: for each origin O the model trains ONLY on demand dates
<= O, so nothing after the origin leaks into its forecast.

Usage
-----
    # 1-day-ahead accuracy across June
    uv run python evaluate.py --start 2026-06-01 --end 2026-06-30 --n 0

    # 11-day-ahead accuracy across the same range
    uv run python evaluate.py --start 2026-06-01 --end 2026-06-30 --n 10

    # Operational view: re-forecast every 11 days, plot the tiled segments
    uv run python evaluate.py --start 2026-06-01 --end 2026-06-30 --n 10 \
        --mode rolling_window

    # From Python
    from evaluate import rolling_backtest, plot_actual_vs_forecast
    df = rolling_backtest("2026-06-01", "2026-06-30", n=0)
    plot_actual_vs_forecast(df, out_path="june_1day.png")
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "v2"))

import lightgbm as lgb  # noqa: E402
from _shared import (  # noqa: E402
    HOLDOUT_DAYS, USE_RESERVATIONS, build_matrix, compute_metrics,
    holiday_mask_from_matrix, load_demand, make_sample_weights,
)

CACHE_DIR = ROOT / "data" / "derived" / "eval_cache"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _fit_lgbm(X, y, w):
    m = lgb.LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.02,
        num_leaves=15, max_depth=5, min_child_samples=5,
        reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1, random_state=123,
    )
    m.fit(X, y, sample_weight=w)
    return m


def _train_and_predict(mat, feature_cols, train_dates, horizon, target_dates):
    """Train one model for `horizon` on data <= origin, predict `target_dates`.

    Returns (predictions dict {target_date: p50}, residual array) or ({}, None)
    when there isn't enough training data.
    """
    sub = mat[(mat["target_date"].isin(train_dates)) &
              (mat["horizon"] == horizon)].dropna(subset=["y"])
    if len(sub) < 30:
        return {}, None

    X, y = sub[feature_cols], sub["y"]
    is_hol = holiday_mask_from_matrix(sub)
    w = make_sample_weights(sub["target_date"], is_holiday=is_hol)
    model = _fit_lgbm(X, y, w)
    residuals = y.values - model.predict(X)

    rows = mat[(mat["target_date"].isin(target_dates)) &
               (mat["horizon"] == horizon)]
    if rows.empty:
        return {}, residuals

    preds = np.maximum(0.0, model.predict(rows[feature_cols]))
    out = {pd.Timestamp(d): float(p)
           for d, p in zip(rows["target_date"].values, preds)}
    return out, residuals


# ---------------------------------------------------------------------------
# Rolling-origin backtest
# ---------------------------------------------------------------------------
def rolling_backtest(start, end, n: int = 0, mode: str = "fixed_lead",
                     step: int | None = None, verbose: bool = True,
                     use_cache: bool = True) -> pd.DataFrame:
    """Honest rolling-origin backtest over [start, end].

    Parameters
    ----------
    start, end : ISO date strings — the TARGET date range to evaluate.
    n          : forecast window depth. Data up to t-1 predicts t .. t+n, so
                 the evaluated lead time is n+1 days.
    mode       : 'fixed_lead' (all points at lead n+1) or 'rolling_window'
                 (origins step by n+1, each contributing a t..t+n segment).
    step       : origin stride in days. Defaults to 1 for fixed_lead and
                 n+1 for rolling_window. Raise it to trade resolution for speed.
    use_cache  : reuse a previously computed result with identical settings.

    Returns
    -------
    DataFrame with columns:
        origin, target_date, horizon, lead_days, forecast, p10, p90, actual
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start > end:
        raise ValueError("start must be on or before end")
    if mode not in ("fixed_lead", "rolling_window"):
        raise ValueError("mode must be 'fixed_lead' or 'rolling_window'")
    if n < 0 or n + 1 > HOLDOUT_DAYS:
        raise ValueError(f"n must be between 0 and {HOLDOUT_DAYS - 1}")

    if step is None:
        step = 1 if mode == "fixed_lead" else n + 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start.date()}_{end.date()}_n{n}_{mode}_s{step}"
    cache_path = CACHE_DIR / f"eval_{tag}.csv"
    if use_cache and cache_path.exists():
        if verbose:
            print(f"  using cached result: {cache_path.name}")
        return pd.read_csv(cache_path, parse_dates=["origin", "target_date"])

    demand = load_demand()
    last_actual = demand["date"].max()
    if end > last_actual:
        raise ValueError(
            f"end={end.date()} is beyond the last actual in rawdata.csv "
            f"({last_actual.date()}). Backtesting needs actuals to compare against."
        )

    lead = n + 1
    if verbose:
        print(f"\n=== Rolling backtest ===")
        print(f"  targets     : {start.date()} .. {end.date()}")
        print(f"  window      : data up to t-1 predicts t .. t+{n}  "
              f"(lead = {lead} day{'s' if lead > 1 else ''})")
        print(f"  mode        : {mode}   step: {step} day(s)")

    # Build the (origin, horizon, targets) work plan
    plan = []           # list of (origin, horizon, [target_dates])
    if mode == "fixed_lead":
        # Every target evaluated at the same lead: origin = target - lead
        targets = pd.date_range(start, end, freq=f"{step}D")
        by_origin = {}
        for t in targets:
            by_origin.setdefault(t - pd.Timedelta(days=lead), []).append(t)
        for origin, tgts in sorted(by_origin.items()):
            plan.append((origin, lead, tgts))
    else:
        # Origins tile the range; each covers t .. t+n
        origin = start - pd.Timedelta(days=1)
        while origin < end:
            tgts, horizons = [], []
            for h in range(1, lead + 1):
                t = origin + pd.Timedelta(days=h)
                if start <= t <= end:
                    tgts.append(t)
                    horizons.append(h)
            for h, t in zip(horizons, tgts):
                plan.append((origin, h, [t]))
            origin += pd.Timedelta(days=step)

    # Feature matrix. When reservations are OFF the matrix is origin-independent
    # so we build it once; otherwise it must be rebuilt per origin so the
    # as_of snapshot filter stays honest.
    rebuild_per_origin = USE_RESERVATIONS
    mat = None
    if not rebuild_per_origin:
        if verbose:
            print("  building feature matrix (once)...")
        t0 = time.time()
        mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS, as_of=end)
        if verbose:
            print(f"  matrix ready in {time.time() - t0:.0f}s "
                  f"({len(mat):,} rows)")

    feature_cols = None
    actual_map = dict(zip(demand["date"], demand["demand"].astype(float)))

    records = []
    t_start = time.time()
    n_fits = len(plan)
    for i, (origin, horizon, tgts) in enumerate(plan, 1):
        if rebuild_per_origin:
            mat = build_matrix(demand, holdout_days=HOLDOUT_DAYS, as_of=origin)
        if feature_cols is None:
            feature_cols = [c for c in mat.columns
                            if c not in {"target_date", "horizon", "y"}]

        train_dates = set(d for d in demand["date"] if d <= origin)
        preds, residuals = _train_and_predict(
            mat, feature_cols, train_dates, horizon, tgts)

        if residuals is not None and len(residuals) > 0:
            q10 = float(np.quantile(residuals, 0.10))
            q90 = float(np.quantile(residuals, 0.90))
        else:
            q10 = q90 = np.nan

        for t in tgts:
            p50 = preds.get(pd.Timestamp(t), np.nan)
            p10 = min(p50 + q10, p50) if np.isfinite(p50) and np.isfinite(q10) else np.nan
            p90 = max(p50 + q90, p50) if np.isfinite(p50) and np.isfinite(q90) else np.nan
            records.append({
                "origin": origin,
                "target_date": pd.Timestamp(t),
                "horizon": horizon,
                "lead_days": horizon,
                "forecast": p50,
                "p10": max(0.0, p10) if np.isfinite(p10) else np.nan,
                "p90": p90,
                "actual": actual_map.get(pd.Timestamp(t), np.nan),
            })

        if verbose and (i % 10 == 0 or i == n_fits):
            el = time.time() - t_start
            print(f"    {i}/{n_fits} fits  ({el:.0f}s elapsed, "
                  f"~{el / i * (n_fits - i):.0f}s left)")

    df = pd.DataFrame(records).sort_values(["target_date", "horizon"])
    df = df.reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    if verbose:
        print(f"  -> cached to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Metrics + plot
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """WAPE / MAPE / RMSE / Bias plus interval coverage."""
    ok = df.dropna(subset=["forecast", "actual"])
    m = compute_metrics(ok["actual"], ok["forecast"])
    band = ok.dropna(subset=["p10", "p90"])
    m["coverage"] = (float(((band["p10"] <= band["actual"]) &
                            (band["actual"] <= band["p90"])).mean())
                     if len(band) else np.nan)
    m["n_days"] = int(ok["target_date"].nunique())
    return m


def plot_actual_vs_forecast(df: pd.DataFrame, out_path: str | Path | None = None,
                            title: str | None = None, show_band: bool = True,
                            figsize=(14, 6)):
    """Line chart: actual demand vs forecast, with metrics annotated.

    `df` is the output of rolling_backtest(). If it contains several horizons
    per target date (rolling_window mode), each origin's segment is drawn so
    you can see the operational re-forecast pattern.
    """
    ok = df.dropna(subset=["forecast", "actual"]).sort_values("target_date")
    if ok.empty:
        raise ValueError("nothing to plot — no rows with both forecast and actual")

    m = summarize(ok)
    lead_min = int(ok["lead_days"].min())
    lead_max = int(ok["lead_days"].max())
    # A single fixed lead vs a window of leads (rolling_window tiles 1..n+1).
    lead_txt = (f"{lead_max}-day lead" if lead_min == lead_max
                else f"lead {lead_min}–{lead_max}d, re-forecast every {lead_max}d")
    multi_origin = ok["origin"].nunique() > 1 and ok.groupby("target_date").size().max() > 1

    fig, ax = plt.subplots(figsize=figsize)

    # Actual — one line, deduplicated by date
    act = ok.drop_duplicates("target_date")[["target_date", "actual"]]
    ax.plot(act["target_date"], act["actual"], "o-", color="black",
            markersize=4, linewidth=2, label="Actual demand", zorder=3)

    if multi_origin:
        # rolling_window: draw each origin's forecast segment separately
        origins = sorted(ok["origin"].unique())
        colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(origins)))
        for color, origin in zip(colors, origins):
            seg = ok[ok["origin"] == origin].sort_values("target_date")
            ax.plot(seg["target_date"], seg["forecast"], "s--", color=color,
                    markersize=3.5, linewidth=1.8, alpha=0.95,
                    label=f"forecast from {pd.Timestamp(origin).date()}")
            if show_band and seg["p10"].notna().any():
                ax.fill_between(seg["target_date"], seg["p10"], seg["p90"],
                                color=color, alpha=0.10)
    else:
        fc = ok.drop_duplicates("target_date").sort_values("target_date")
        if show_band and fc["p10"].notna().any():
            ax.fill_between(fc["target_date"], fc["p10"], fc["p90"],
                            color="C0", alpha=0.15, label="P10–P90")
        ax.plot(fc["target_date"], fc["forecast"], "s--", color="C0",
                markersize=4, linewidth=2,
                label=f"Forecast ({lead_txt})", zorder=2)

    if title is None:
        title = (f"Actual vs Forecast — {ok['target_date'].min().date()} to "
                 f"{ok['target_date'].max().date()}  |  {lead_txt}")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (patron hours)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    box = (f"MAPE  {m['mape']*100:5.2f}%\n"
           f"WAPE  {m['wape']*100:5.2f}%\n"
           f"RMSE  {m['rmse']:6.0f}\n"
           f"Bias  {m['bias']:+6.0f}\n"
           f"Days  {m['n_days']}")
    ax.text(0.985, 0.03, box, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      edgecolor="#888", alpha=0.9))

    plt.xticks(rotation=30)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  -> {out_path}")
    plt.close()
    return m


def accuracy_by_lead(start, end, leads=(1, 3, 7, 14, 21, 28),
                     step: int = 3, verbose: bool = True) -> pd.DataFrame:
    """MAPE as a function of forecast lead time — the classic degradation curve.

    Runs one fixed_lead backtest per lead in `leads`. Useful as a headline
    slide: accuracy is strong near-term and decays with horizon.
    """
    rows = []
    for lead in leads:
        n = lead - 1
        if verbose:
            print(f"\n--- lead {lead} day(s) ---")
        df = rolling_backtest(start, end, n=n, mode="fixed_lead",
                              step=step, verbose=verbose)
        m = summarize(df)
        rows.append({"lead_days": lead, "mape": m["mape"], "wape": m["wape"],
                     "rmse": m["rmse"], "bias": m["bias"], "n_days": m["n_days"]})
    return pd.DataFrame(rows)


def plot_accuracy_by_lead(curve: pd.DataFrame, out_path=None, figsize=(10, 5)):
    """MAPE vs lead time. The y-axis is anchored at zero deliberately: on a
    zoomed axis a flat curve looks like a dramatic trend."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(curve["lead_days"], curve["mape"] * 100, "o-", color="C0",
            linewidth=2.5, markersize=8)
    for _, r in curve.iterrows():
        ax.annotate(f"{r['mape']*100:.2f}%",
                    (r["lead_days"], r["mape"] * 100),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Forecast lead time (days ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Forecast accuracy by lead time", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(6.0, curve["mape"].max() * 100 * 1.4))
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  -> {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="First TARGET date (ISO)")
    ap.add_argument("--end", required=True, help="Last TARGET date (ISO)")
    ap.add_argument("--n", type=int, default=0,
                    help="Window depth: data up to t-1 predicts t..t+n "
                         "(lead = n+1 days). Default 0.")
    ap.add_argument("--mode", choices=["fixed_lead", "rolling_window"],
                    default="fixed_lead")
    ap.add_argument("--step", type=int, default=None,
                    help="Origin stride in days (speed knob).")
    ap.add_argument("--out", default=None, help="Output PNG path")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--by-lead", action="store_true",
                    help="Also produce the MAPE-vs-lead-time curve")
    args = ap.parse_args()

    df = rolling_backtest(args.start, args.end, n=args.n, mode=args.mode,
                          step=args.step, use_cache=not args.no_cache)
    out = args.out or f"forecasts/eval_{args.start}_{args.end}_n{args.n}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    m = plot_actual_vs_forecast(df, out_path=out)

    print("\n=== Accuracy ===")
    print(f"  MAPE      {m['mape']*100:6.2f}%")
    print(f"  WAPE      {m['wape']*100:6.2f}%")
    print(f"  RMSE      {m['rmse']:7.0f}")
    print(f"  Bias      {m['bias']:+7.0f}")
    print(f"  Coverage  {m['coverage']*100:5.1f}%  (P10-P90, target 80%)")
    print(f"  Days      {m['n_days']}")

    if args.by_lead:
        curve = accuracy_by_lead(args.start, args.end)
        print("\n=== MAPE by lead time ===")
        print(curve.to_string(index=False))
        plot_accuracy_by_lead(curve, out_path="forecasts/eval_accuracy_by_lead.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
