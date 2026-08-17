"""evaluate_long.py — how accurate is the long-history model, really?

This is the long-history counterpart of the production `evaluate.py`, and it
answers a different question from `experiment.py`:

    experiment.py    "which SETTINGS are best?"        (compares variants)
    evaluate_long.py "how good is the CHOSEN setup?"   (measures one config)

THE CORE IDEA
-------------
To forecast target date `t` honestly, the model may only see data up to some
earlier day. If it may see up to `t-1`, that is a 1-day-ahead forecast. If it
may only see up to `t-8`, that is an 8-day-ahead forecast. The gap is the
LEAD TIME, and accuracy gets worse as it grows -- so a single "MAPE" number
is meaningless unless you say at what lead it was measured.

    n = 0   -> data up to t-1  predicts t          (lead 1)
    n = 6   -> data up to t-1  predicts t .. t+6   (lead 1..7)

TWO MODES
---------
fixed_lead      Every plotted point is measured at the SAME lead (n+1 days).
                Answers "how accurate are we 7 days out?" Fast: one horizon
                per origin.

rolling_window  Origins step by (n+1) days; each contributes a full t..t+n
                segment, tiling the range. Answers "what would our actual
                operating forecast have looked like?" -- it mirrors
                re-forecasting every (n+1) days.

HONESTY
-------
For each origin O the model trains ONLY on target dates <= O. Features are
horizon-gated inside core.py (min_safe = horizon + 1), so a lead-14 forecast
physically cannot read the previous 13 days. Nothing after O participates.

The CNY anchor is applied by default because production applies it in every
forecast mode -- measuring without it reports a number the business never
actually sees.

CACHING
-------
The cache key encodes the FULL configuration -- dates, mode, lead, half-life,
anchor, exclusions, guard, model params AND a hash of the feature engine
(engine/core.py). Both halves matter:

  - Production's evaluate.py keys only on dates+mode+lead, so two runs with
    different model settings silently shared one cache entry. That bug
    produced a mislabeled result earlier in this project.
  - Without the engine hash, editing core.py would serve results computed by
    the OLD feature code under the new code's name.

Do not "simplify" the key.

USAGE
-----
    # 7-day-ahead accuracy across 2026 so far
    uv run python long_history/evaluate_long.py --start 2026-01-01 --end 2026-05-28 --n 6

    # 1-day-ahead, with the MAPE-vs-lead curve as well
    uv run python long_history/evaluate_long.py --start 2026-01-01 --end 2026-05-28 --by-lead

    # Operational view: re-forecast every 7 days and plot the tiled segments
    uv run python long_history/evaluate_long.py --start 2026-01-01 --end 2026-05-28 \
        --n 6 --mode rolling_window

    # Same thing without the CNY correction, to see what it is worth
    uv run python long_history/evaluate_long.py --start 2026-01-01 --end 2026-05-28 --no-anchor
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import warnings
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

CACHE_DIR = ROOT / C.OUT_DIR / "eval_cache"
NAVY, TEAL, GOLD, GREY, RED = "#0F2942", "#2E8BA8", "#E8B547", "#5F7183", "#C74B4B"


# ── helpers ────────────────────────────────────────────────────────────────
def holiday_lookup() -> dict:
    """target date -> holiday name (window-aware), else not present.

    core.py already ships anchors for 2016-2030, so this covers the whole
    long-history range, not just 2024+.
    """
    out = {}
    for name, anchors in S.HOLIDAY_ANCHORS.items():
        ws, we = S.HOLIDAY_WINDOWS[name]
        for a in anchors:
            for k in range(ws, we + 1):
                out.setdefault(a + pd.Timedelta(days=k), name)
    return out


def _cache_key(**kw) -> str:
    """Full-configuration hash. See the CACHING note in the module docstring."""
    blob = json.dumps(kw, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _style(ax):
    ax.grid(alpha=.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


# ── the backtest ───────────────────────────────────────────────────────────
def rolling_backtest(start, end, n: int = 0, mode: str = "fixed_lead",
                     half_life: int | None = None, anchor: bool = True,
                     step: int | None = None, use_cache: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """Honest rolling-origin backtest over [start, end].

    Returns one row per (origin, target) with forecast, p10, p90 and actual.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    half_life = half_life if half_life is not None else C.DEFAULT_HALF_LIFE
    if step is None:
        step = 1 if mode == "fixed_lead" else n + 1
    lead = n + 1

    key = _cache_key(start=start, end=end, n=n, mode=mode, step=step,
                     half_life=half_life, anchor=anchor,
                     data=C.DATA_FILE, excl=C.EXCLUDE_FROM_TRAINING,
                     closures=C.EXCLUDE_CLOSURES,
                     guard=C.GUARD_LAGS_CROSSING_EXCLUDED,
                     woy=C.ADD_WEEK_OF_YEAR, reg=C.ADD_REGIME_FEATURE,
                     lgbm=C.LGBM_PARAMS,
                     engine=F.engine_fingerprint())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"eval_{start.date()}_{end.date()}_n{n}_{mode}_{key}.csv"
    if use_cache and cache_path.exists():
        if verbose:
            print(f"  using cached result: {cache_path.name}")
        cached = pd.read_csv(cache_path, parse_dates=["origin", "target_date"])
        # A round-trip through CSV turns the empty-string holiday marker into
        # NaN, and NaN is truthy -- which silently poisons every downstream
        # `if holiday` test. Normalise on the way back in.
        cached["holiday"] = cached["holiday"].fillna("").astype(str)
        cached["is_holiday"] = cached["holiday"] != ""
        return cached

    demand = F.load_long_demand()
    last_actual = demand["date"].max()
    if end > last_actual:
        raise ValueError(
            f"end={end.date()} is past the last actual in {C.DATA_FILE} "
            f"({last_actual.date()}). Backtesting needs actuals to score against.")

    if verbose:
        print(f"\n=== Long-history rolling backtest ===")
        print(f"  targets   : {start.date()} .. {end.date()}")
        print(f"  window    : data up to t-1 predicts t .. t+{n}  (lead {lead}d)")
        print(f"  mode      : {mode}   step: {step} day(s)")
        print(f"  half-life : {half_life}d    CNY anchor: {'on' if anchor else 'OFF'}")

    # ── work plan: (origin, horizon, [targets]) ───────────────────────────
    plan = []
    if mode == "fixed_lead":
        by_origin = {}
        for t in pd.date_range(start, end, freq=f"{step}D"):
            by_origin.setdefault(t - pd.Timedelta(days=lead), []).append(t)
        for o, tgts in sorted(by_origin.items()):
            plan.append((o, lead, tgts))
    else:
        o = start - pd.Timedelta(days=1)
        while o < end:
            for h in range(1, n + 2):
                t = o + pd.Timedelta(days=h)
                if start <= t <= end:
                    plan.append((o, h, [t]))
            o += pd.Timedelta(days=step)

    t0 = time.time()
    if verbose:
        print(f"  building feature matrix (once, ~5-8 min on 11 years)...", flush=True)
    base = F.build_base(demand, history_start=None, quiet=True)
    mat = F.apply_exclusions(base, demand, [])
    feats = F.feature_columns(mat)
    if verbose:
        print(f"  matrix {mat.shape[0]:,} x {mat.shape[1]} "
              f"({int(mat.y.notna().sum()):,} trainable)  [{time.time()-t0:.0f}s]")

    dm = dict(zip(demand["date"], demand["demand"].astype(float)))
    rows, t0 = [], time.time()
    for i, (origin, horizon, tgts) in enumerate(plan, 1):
        sub = mat[(mat["target_date"] <= origin) &
                  (mat["horizon"] == horizon)].dropna(subset=["y"])
        if len(sub) < 60:
            continue
        pr = mat[(mat["target_date"].isin(tgts)) & (mat["horizon"] == horizon)]
        if pr.empty:
            continue
        w = make_sample_weights(sub["target_date"],
                                is_holiday=holiday_mask_from_matrix(sub),
                                half_life_days=half_life)
        m = lgb.LGBMRegressor(**C.LGBM_PARAMS)
        m.fit(sub[feats], sub["y"], sample_weight=w)
        res = sub["y"].values - m.predict(sub[feats])
        q10, q90 = float(np.quantile(res, .10)), float(np.quantile(res, .90))

        for d, p in zip(pr["target_date"].values, m.predict(pr[feats])):
            T = pd.Timestamp(d)
            actual = dm.get(T, np.nan)
            p50 = max(0.0, float(p))
            rows.append({"origin": origin, "target_date": T, "horizon": horizon,
                         "lead_days": (T - origin).days,
                         "forecast": p50,
                         "p10": min(max(0.0, p50 + q10), p50),
                         "p90": max(p50 + q90, p50),
                         "actual": actual})
        if verbose and i % 25 == 0:
            el = time.time() - t0
            print(f"    {i}/{len(plan)} fits  ({el:.0f}s elapsed, "
                  f"~{el/i*(len(plan)-i):.0f}s left)", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # CNY anchor, applied per origin exactly as forecast_long.py does
    if anchor:
        out = []
        for o, g in df.groupby("origin"):
            p = g[["target_date", "p10", "forecast", "p90"]].rename(
                columns={"target_date": "date", "forecast": "p50"})
            with F.extended_holidays():
                a = S.apply_moving_anchor(p, demand, pd.Timestamp(o))
            g = g.copy()
            g[["p10", "forecast", "p90"]] = a[["p10", "p50", "p90"]].values
            out.append(g)
        df = pd.concat(out, ignore_index=True)

    hol = holiday_lookup()
    df["holiday"] = df["target_date"].map(lambda d: hol.get(pd.Timestamp(d), ""))
    df["is_holiday"] = df["holiday"] != ""
    df = df.sort_values(["target_date", "lead_days"]).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    if verbose:
        print(f"  -> cached to {cache_path.name}")
    return df


# ── metrics ────────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict:
    d = df[(df.actual > 0) & df.forecast.notna()]
    if d.empty:
        return {}
    err = d.forecast - d.actual
    return {
        "mape": float(np.mean(np.abs(err) / d.actual)),
        "wape": float(np.sum(np.abs(err)) / np.sum(d.actual)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "coverage": float(np.mean((d.actual >= d.p10) & (d.actual <= d.p90))),
        "n_days": int(len(d)),
    }


def accuracy_by_lead(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead, g in df.groupby("lead_days"):
        g = g[(g.actual > 0) & g.forecast.notna()]
        if g.empty:
            continue
        hol, ord_ = g[g.is_holiday], g[~g.is_holiday]
        ape = lambda x: float(np.mean(np.abs(x.forecast - x.actual) / x.actual)) if len(x) else np.nan
        rows.append({"lead_days": int(lead), "n": len(g), "mape_all": ape(g),
                     "n_holiday": len(hol), "mape_holiday": ape(hol),
                     "mape_ordinary": ape(ord_)})
    return pd.DataFrame(rows).sort_values("lead_days")


def holiday_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ape = lambda x: float(np.mean(np.abs(x.forecast - x.actual) / x.actual)) if len(x) else np.nan
    ordinary = df[~df.is_holiday & (df.actual > 0)]
    rows.append({"segment": "ordinary", "n": len(ordinary), "mape": ape(ordinary)})
    names = {str(x) for x in df.holiday.dropna().unique() if str(x)}
    for name in sorted(names):
        g = df[(df.holiday == name) & (df.actual > 0)]
        rows.append({"segment": name, "n": len(g), "mape": ape(g)})
    return pd.DataFrame(rows)


# ── charts ─────────────────────────────────────────────────────────────────
def plot_actual_vs_forecast(df: pd.DataFrame, out_path: Path, title_extra=""):
    d = df[(df.actual > 0) & df.forecast.notna()]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2.3, 1]})
    for _, r in d[d.is_holiday].iterrows():
        for a in (ax, ax2):
            a.axvspan(r.target_date - pd.Timedelta(hours=12),
                      r.target_date + pd.Timedelta(hours=12),
                      color=GOLD, alpha=.20, zorder=0)

    multi = d.origin.nunique() > 1 and d.groupby("target_date").size().max() > 1
    ax.plot(d.target_date, d.actual, "o-", color=NAVY, lw=2, ms=4,
            label="actual", zorder=5)
    if multi:
        origins = sorted(d.origin.unique())
        cols = plt.cm.viridis(np.linspace(.25, .85, len(origins)))
        for c, o in zip(cols, origins):
            seg = d[d.origin == o].sort_values("target_date")
            ax.plot(seg.target_date, seg.forecast, "s--", color=c, lw=1.4, ms=3)
        ax.plot([], [], "s--", color=cols[len(cols) // 2], label="forecast segments")
    else:
        ax.plot(d.target_date, d.forecast, "s--", color=TEAL, lw=1.6, ms=3.5,
                label="forecast", zorder=4)
        ax.fill_between(d.target_date, d.p10, d.p90, color=TEAL, alpha=.16,
                        label="P10–P90")
    m = compute_metrics(d)
    ax.set_title(f"Actual vs forecast — MAPE {m['mape']*100:.2f}%   "
                 f"shaded = holiday window{title_extra}",
                 fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.set_ylabel("demand (patron hours)", fontsize=9, color=GREY)
    ax.legend(fontsize=9)
    _style(ax)

    pe = (d.forecast - d.actual) / d.actual * 100
    ax2.axhline(0, color=NAVY, lw=1.1)
    ax2.plot(d.target_date, pe, "o-", color=RED, lw=1.2, ms=3)
    ax2.set_ylabel("% error", fontsize=9, color=GREY)
    ax2.set_xlabel("target date", fontsize=9, color=GREY)
    _style(ax2)
    plt.xticks(rotation=30)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")
    return m


def plot_accuracy_by_lead(curve: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(curve.lead_days, curve.mape_all * 100, "o-", color=NAVY, lw=2,
            ms=5, label="all days")
    if curve.mape_ordinary.notna().any():
        ax.plot(curve.lead_days, curve.mape_ordinary * 100, "s--", color=TEAL,
                lw=1.5, ms=4, label="ordinary days")
    if curve.mape_holiday.notna().any():
        ax.plot(curve.lead_days, curve.mape_holiday * 100, "^--", color=GOLD,
                lw=1.5, ms=4, label="holiday-window days")
    ax.set_xlabel("lead time (days ahead)", fontsize=9, color=GREY)
    ax.set_ylabel("MAPE (%)", fontsize=9, color=GREY)
    ax.set_ylim(bottom=0)
    ax.set_title("Accuracy vs how far ahead the forecast was made",
                 fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first TARGET date (ISO)")
    ap.add_argument("--end", required=True, help="last TARGET date (ISO)")
    ap.add_argument("--n", type=int, default=0,
                    help="window depth: data to t-1 predicts t..t+n (lead n+1). Default 0")
    ap.add_argument("--mode", choices=["fixed_lead", "rolling_window"],
                    default="fixed_lead")
    ap.add_argument("--half-life", type=int, default=None,
                    help=f"recency half-life (default {C.DEFAULT_HALF_LIFE})")
    ap.add_argument("--no-anchor", action="store_true",
                    help="skip the CNY correction (production applies it)")
    ap.add_argument("--step", type=int, default=None, help="origin stride (speed knob)")
    ap.add_argument("--by-lead", action="store_true",
                    help="also produce the MAPE-vs-lead curve (slower)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None, help="output PNG path")
    a = ap.parse_args()

    from data.validate import validate
    if validate(ROOT / C.DATA_FILE, verbose=True)[0]:
        print("\nABORTED: fix the data errors above first.")
        return 1

    df = rolling_backtest(a.start, a.end, n=a.n, mode=a.mode,
                          half_life=a.half_life, anchor=not a.no_anchor,
                          step=a.step, use_cache=not a.no_cache)
    if df.empty:
        print("No predictions produced — check that your range has actuals.")
        return 1

    outdir = ROOT / C.OUT_DIR / "evaluation"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.start}_{a.end}_n{a.n}_{a.mode}"
    out = Path(a.out) if a.out else outdir / f"eval_{tag}.png"
    m = plot_actual_vs_forecast(df, out)
    df.to_csv(outdir / f"eval_{tag}.csv", index=False)

    print("\n=== Accuracy ===")
    print(f"  MAPE      {m['mape']*100:6.2f}%   (average % miss)")
    print(f"  WAPE      {m['wape']*100:6.2f}%   (total error ÷ total demand)")
    print(f"  RMSE      {m['rmse']:7.0f}    (patron hours)")
    print(f"  Bias      {m['bias']:+7.0f}    (+ = over-forecast)")
    print(f"  Coverage  {m['coverage']*100:5.1f}%   (P10-P90, nominal 80% — see caveat)")
    print(f"  Days      {m['n_days']}")

    hb = holiday_breakdown(df)
    print("\n=== By segment ===")
    for _, r in hb.iterrows():
        if r["n"]:
            print(f"  {r['segment']:12s} n={int(r['n']):4d}   MAPE {r['mape']*100:6.2f}%")
    hb.to_csv(outdir / f"segments_{tag}.csv", index=False)

    if a.by_lead:
        curve = accuracy_by_lead(df)
        if len(curve) > 1:
            print("\n=== MAPE by lead time ===")
            print(curve.assign(**{c: (curve[c] * 100).round(2)
                                  for c in curve.columns if c.startswith("mape")}
                               ).to_string(index=False))
            plot_accuracy_by_lead(curve, outdir / f"by_lead_{tag}.png")
            curve.to_csv(outdir / f"by_lead_{tag}.csv", index=False)
        else:
            print("\n(--by-lead needs several leads; use --mode rolling_window --n 6)")

    print(f"\n-> {outdir}")
    print("\nNOTE on coverage: P10-P90 comes from in-sample residuals and is "
          "known to be too narrow (~46-54% actual vs 80% nominal). Treat the "
          "band as relative confidence, not a guarantee.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
