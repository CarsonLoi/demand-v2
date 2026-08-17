"""narrow_mask_longhistory.py — does the narrow mask hold up across MANY
Chinese New Years, or was the 2026 win luck?

WHY THIS EXISTS
---------------
research/narrow_mask_test.py showed the narrow demand-driven mask beats the
full calendar mask on every metric over 2026 YTD, but nothing reached
significance (pooled p=0.15). Root cause: that backtest contains exactly ONE
CNY event, so only 13 of 148 evaluated days were ever affected. A rare-event
fix cannot be proven on one occurrence.

The long-history file (2015-2026) carries 11 CNY events. After removing the
COVID regime (2020-01-23..2022-12-31, where demand is not a valid signal),
SIX usable contamination events remain -- targets in 2017, 2018, 2019, 2024,
2025 and 2026, each contaminated by the PREVIOUS year's CNY collapse.

READ-ONLY / NON-DESTRUCTIVE
---------------------------
Imports long_history/{core,features,config}.py read-only and masks COPIES of
the built matrix in memory. Nothing in v2/ or long_history/ is modified on
disk, consistent with the standing instruction to keep those untouched.

METHOD
------
  - Build the long-history matrix ONCE (expensive), apply the COVID/closure
    exclusions, then derive three cheap masked copies:
        off     no holiday-drift mask (long_history default today)
        full    v2-style all-or-nothing calendar-window mask
        narrow  demand-driven: mask only moving-holiday days whose demand was
                outside [LO, HI] x normal for that weekday
  - Origins are placed where the masks can actually differ (the affected
    window of each event), PLUS clean control origins each year to detect
    collateral damage.
  - Weekly origins, horizons 1..7, so target dates tile with no gaps.
  - The CNY anchor is applied exactly as production does, because leaving it
    out measures something the business never sees.
  - Half-life held at the production default (240d) so the MASK is the only
    variable under test.

Usage:
    uv run python research/narrow_mask_longhistory.py
    uv run python research/narrow_mask_longhistory.py --lo 0.70 --controls 2
"""
from __future__ import annotations
import argparse
import contextlib
import io
import sys
import time
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
from engine import core as S      # long_history's VENDORED engine -- read-only
from engine import features as F  # noqa: E402

OUT = ROOT / "research" / "narrow_mask_output"
HALF_LIFE = 240             # production default; held fixed
HORIZONS = list(range(1, 8))
COVID_LO, COVID_HI = pd.Timestamp("2020-01-23"), pd.Timestamp("2022-12-31")


# ── the demand-driven abnormality measure (same rule as the 2026 test) ──────
def build_ratio_map(demand: pd.DataFrame) -> dict:
    dm = dict(zip(demand["date"], demand["demand"].astype(float)))
    hol = S.holiday_date_set()
    ratio = {}
    for d, v in dm.items():
        if v is None or np.isnan(v) or v <= 0:
            continue
        vals = []
        for sign in (-1, 1):
            for k in range(21, 57, 7):
                t = d + pd.Timedelta(days=sign * k)
                if t in hol:
                    continue
                x = dm.get(t)
                if x is not None and not np.isnan(x) and x > 0:
                    vals.append(x)
        if len(vals) >= 4:
            ratio[d] = v / float(np.mean(vals))
    return ratio


def mask_columns(mat: pd.DataFrame, bad: set) -> pd.DataFrame:
    """Return a COPY with yearly-lookback features NaN'd where their SOURCE
    date is in `bad`. Same four features the production guard touches."""
    df = mat.copy()
    T = pd.to_datetime(df["target_date"])
    min_safe = df["horizon"].astype(int) + 1

    v_recent = T - pd.to_timedelta(min_safe, unit="D")
    if "yoy_ratio" in df.columns:
        df.loc[v_recent.isin(bad) |
               (v_recent - pd.Timedelta(days=365)).isin(bad), "yoy_ratio"] = np.nan
    bad365 = (T - pd.Timedelta(days=365)).isin(bad)
    for c in ("lag_365", "lag_anchor_365"):
        if c in df.columns:
            df.loc[bad365, c] = np.nan
    if "lag_728" in df.columns:
        df.loc[(T - pd.Timedelta(days=728)).isin(bad), "lag_728"] = np.nan
    return df


def affected_targets(bad: set, demand: pd.DataFrame) -> set:
    """Target dates whose yearly lookback reads a contaminated source."""
    have = set(demand["date"])
    out = set()
    for d in bad:
        for off in (365, 728):
            t = d + pd.Timedelta(days=off)
            if t in have:
                out.add(t)
    return out


def usable(d: pd.Timestamp) -> bool:
    return not (COVID_LO <= d <= COVID_HI)


# ── backtest ────────────────────────────────────────────────────────────────
def run_origin(mat, demand_map, feats, origin, anchor_demand):
    train = mat[mat["target_date"] <= origin].dropna(subset=["y"])
    rows = []
    for h in HORIZONS:
        sub = train[train["horizon"] == h]
        if len(sub) < 60:
            continue
        T = origin + pd.Timedelta(days=h)
        v = demand_map.get(T)
        if v is None or np.isnan(v) or v <= 0:
            continue
        pr = mat[(mat["target_date"] == T) & (mat["horizon"] == h)]
        if pr.empty:
            continue
        w = S.make_sample_weights(sub["target_date"],
                                  is_holiday=S.holiday_mask_from_matrix(sub),
                                  half_life_days=HALF_LIFE)
        m = lgb.LGBMRegressor(**C.LGBM_PARAMS)
        m.fit(sub[feats], sub["y"], sample_weight=w)
        rows.append({"origin": origin, "target": T, "horizon": h,
                     "pred": max(0.0, float(m.predict(pr[feats])[0])), "actual": v})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    p = pd.DataFrame({"date": df["target"], "p10": df["pred"],
                      "p50": df["pred"], "p90": df["pred"]})
    with F.extended_holidays():
        a = S.apply_moving_anchor(p, anchor_demand, origin)
    df["pred"] = a["p50"].values
    return df


def mape(d):
    d = d[(d.actual > 0) & d.pred.notna()]
    return float(np.mean(np.abs(d.actual - d.pred) / d.actual)) * 100 if len(d) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lo", type=float, default=0.70,
                    help="narrow-mask lower bound; HI = 2-LO (default 0.70)")
    ap.add_argument("--controls", type=int, default=2,
                    help="clean control origins per event year (default 2)")
    a = ap.parse_args()
    LO, HI = a.lo, 2 - a.lo

    print("=" * 78)
    print(f"NARROW MASK -- LONG-HISTORY VALIDATION  (narrow band [{LO:.2f},{HI:.2f}])")
    print("=" * 78)

    demand = F.load_long_demand()
    print(f"data: {demand.date.min().date()} .. {demand.date.max().date()} "
          f"({len(demand):,} days)")

    ratio = build_ratio_map(demand)
    moving = S.moving_holiday_date_set()
    bad_narrow = {d for d in moving if d in ratio and not (LO <= ratio[d] <= HI)}
    bad_full = set(moving)
    bad_narrow_u = {d for d in bad_narrow if usable(d)}
    print(f"moving-holiday window days     : {len(moving)}")
    print(f"narrow flags as contaminated   : {len(bad_narrow)} "
          f"({100*len(bad_narrow)/len(moving):.0f}%)")

    # affected target dates, grouped by year -> these define where to look
    aff = {t for t in affected_targets(bad_narrow_u, demand) if usable(t)}
    by_year = {}
    for t in sorted(aff):
        by_year.setdefault(t.year, []).append(t)
    by_year = {y: v for y, v in by_year.items() if len(v) >= 2}
    print(f"\nusable contamination events ({len(by_year)}):")
    for y, v in by_year.items():
        print(f"   {y}: {len(v):2d} affected target days "
              f"({min(v).date()} .. {max(v).date()})")

    # ── origins: cover each affected window, plus clean controls ───────────
    origins = []
    for y, v in by_year.items():
        lo_t, hi_t = min(v), max(v)
        o = lo_t - pd.Timedelta(days=7)
        while o <= hi_t:
            origins.append(o)
            o += pd.Timedelta(days=7)
        for i in range(a.controls):                 # controls: mid-year, clean
            c = pd.Timestamp(f"{y}-06-15") + pd.Timedelta(days=28 * i)
            if c <= demand.date.max() - pd.Timedelta(days=8) and usable(c):
                origins.append(c)
    origins = sorted({o for o in origins
                      if o >= demand.date.min() + pd.Timedelta(days=420)
                      and o <= demand.date.max() - pd.Timedelta(days=8)})
    print(f"\norigins: {len(origins)}  ({origins[0].date()} .. {origins[-1].date()})")
    print(f"fits: ~{len(origins)*len(HORIZONS)*3:,} across 3 variants")

    # ── build the matrix once, then cheap masked copies ────────────────────
    t0 = time.time()
    print("\nbuilding long-history matrix (one time, ~5-8 min)...", flush=True)
    base = F.build_base(demand, history_start=None, quiet=True)
    mat_off = F.apply_exclusions(base, demand, [])
    print(f"  matrix {mat_off.shape[0]:,} x {mat_off.shape[1]} "
          f"({int(mat_off.y.notna().sum()):,} trainable)  [{time.time()-t0:.0f}s]")

    variants = {
        "off": mat_off,
        "full": mask_columns(mat_off, bad_full),
        f"narrow_{LO:.2f}": mask_columns(mat_off, bad_narrow),
    }
    feats = F.feature_columns(mat_off)
    dm = dict(zip(demand.date, demand.demand.astype(float)))

    results = {}
    for tag, mat in variants.items():
        print(f"\n### {tag}", flush=True)
        t0, rows = time.time(), []
        for o in origins:
            r = run_origin(mat, dm, feats, o, demand)
            if not r.empty:
                rows.append(r)
        df = pd.concat(rows, ignore_index=True)
        df["variant"] = tag
        results[tag] = df
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / f"lh_{tag}.csv", index=False)
        print(f"  {len(df)} predictions, MAPE {mape(df):.2f}%  [{time.time()-t0:.0f}s]")

    # ── report ─────────────────────────────────────────────────────────────
    base_df = results["off"].sort_values(["origin", "horizon"]).reset_index(drop=True)
    fire = base_df.target.isin(aff).values

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'variant':16s} {'overall':>9s} {'FIRES':>9s} {'clean':>9s}")
    print(f"{'':16s} {'(n=%d)'%len(base_df):>9s} {'(n=%d)'%fire.sum():>9s} "
          f"{'(n=%d)'%(~fire).sum():>9s}")
    for tag in variants:
        d = results[tag].sort_values(["origin", "horizon"]).reset_index(drop=True)
        print(f"{tag:16s} {mape(d):8.2f}% {mape(d[fire]):8.2f}% {mape(d[~fire]):8.2f}%")

    print("\nPer-event MAPE on affected days (the whole point of this run):")
    hdr = "  " + f"{'event year':12s}" + "".join(f"{t:>14s}" for t in variants)
    print(hdr)
    for y in sorted(by_year):
        line = f"  {y:<12d}"
        for tag in variants:
            d = results[tag].sort_values(["origin", "horizon"]).reset_index(drop=True)
            sel = fire & (d.target.dt.year == y).values
            line += f"{mape(d[sel]):13.2f}%" if sel.sum() else f"{'--':>14s}"
        print(line)

    print("\nPaired per-day test on affected rows (vs off):")
    from scipy.stats import wilcoxon
    a_ape = (np.abs(base_df.actual - base_df.pred) / base_df.actual * 100)[fire]
    for tag in variants:
        if tag == "off":
            continue
        d = results[tag].sort_values(["origin", "horizon"]).reset_index(drop=True)
        b_ape = (np.abs(d.actual - d.pred) / d.actual * 100)[fire]
        better, worse = int((b_ape < a_ape).sum()), int((b_ape > a_ape).sum())
        try:
            _, p = wilcoxon(b_ape, a_ape)
        except ValueError:
            p = np.nan
        print(f"  {tag:16s} {better:3d} better / {worse:3d} worse   "
              f"mean {a_ape.mean():.2f}% -> {b_ape.mean():.2f}%   p={p:.4f}")

    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
