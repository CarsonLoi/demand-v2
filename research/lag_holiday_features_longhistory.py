"""lag_holiday_features_longhistory.py -- does INFORMING the model about a
contaminated yearly lookback beat leaving it alone, on the same class of
problem the narrow-mask experiment already tried and failed at (fixing by
deletion)?

WHY THIS EXISTS
---------------
The narrow-mask experiment (research/narrow_mask_longhistory.py) tried
MASKING lag_365/lag_728/yoy_ratio when their source looked contaminated.
Measured on 6 real CNY events: masking LOST to doing nothing (8.02% MAPE on
affected days for "off" vs 8.18% for both "full" and "narrow" masks).

This tests a different idea: instead of deleting the contaminated value,
give the model two new columns describing it -- whether the source date
fell in a holiday window, and how depressed/elevated it was relative to its
own normal baseline. The model itself decides how much to discount, instead
of a human-picked threshold deciding for it.

READ-ONLY / NON-DESTRUCTIVE
----------------------------
Imports long_history/{config,engine}.py read-only. New feature columns are
added to a COPY of the matrix in memory. Nothing in v2/ or long_history/ is
modified on disk.

WHICH LAGS -- "all post-year lags" turns out to be 3 distinct source dates,
not 1: lag_365 and lag_anchor_365 share one source (T-365) because
lag_anchor_365's eff=max(365,min_safe) always resolves to 365 in this
dataset (horizons only go to 28). lag_728 has its own source (T-728).
yoy_ratio has a THIRD, horizon-dependent source (T-(h+1)-365, drifting up
to 27 days from T-365) and needs its own pair.

    lag365_src_is_holiday / lag365_src_ratio   informs lag_365, lag_anchor_365
    lag728_src_is_holiday / lag728_src_ratio   informs lag_728
    yoyratio_src_is_holiday / yoyratio_src_ratio   informs yoy_ratio

WHICH HOLIDAYS -- the FULL holiday_date_set() (all 9: CNY, GoldenWeek,
Labour, MidAutumn, NewYear, Christmas, ChingMing, Easter, DragonBoat), not
just the 4 moving ones. build_ratio_map already used the full set (see its
`hol = S.holiday_date_set()` line) so this needed no change to reuse.

METHOD
------
  - Build the long-history matrix ONCE, derive two variants:
        off       today's shipped default, unchanged
        informed  off + the 6 new columns above, added to the feature list
  - Origins are NOT a manual guess at "Jan is bad" -- they are computed from
    affected_targets_by_holiday(), which finds, per holiday, the target
    dates whose lag_365, lag_728, OR yoy_ratio source lands in that
    holiday's window. CNY gets full weekly coverage of every affected week
    (it's the dominant, ~50%-collapse case); the other 8 holidays get
    --other-holiday-weeks (default 2) representative weeks per year each,
    since their swings are much milder and exhaustive coverage would spend
    most of the compute budget on the least informative cases. Plus 2 clean
    control origins per year to catch collateral damage.
  - Horizons 1..7 (matches the narrow-mask harness this reuses; wider
    horizons drift yoy_ratio's source further and could be tested as a
    follow-up).
  - The CNY anchor is applied exactly as production does.
  - Half-life held at the production default (240d) so the FEATURES are the
    only variable under test.

Usage:
    uv run python research/lag_holiday_features_longhistory.py
    uv run python research/lag_holiday_features_longhistory.py --controls 3
"""
from __future__ import annotations
import argparse
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

OUT = ROOT / "research" / "lag_holiday_features_output"
HALF_LIFE = 240             # production default; held fixed
HORIZONS = list(range(1, 8))
COVID_LO, COVID_HI = pd.Timestamp("2020-01-23"), pd.Timestamp("2022-12-31")

NEW_COLS = ["lag365_src_is_holiday", "lag365_src_ratio",
           "lag728_src_is_holiday", "lag728_src_ratio",
           "yoyratio_src_is_holiday", "yoyratio_src_ratio"]


def usable(d: pd.Timestamp) -> bool:
    return not (COVID_LO <= d <= COVID_HI)


# ── reused as-is from narrow_mask_longhistory.py (same ratio construction,
#    already validated this session) ─────────────────────────────────────
def build_ratio_map(demand: pd.DataFrame) -> dict:
    dm = dict(zip(demand["date"], demand["demand"].astype(float)))
    hol = S.holiday_date_set()   # full 9-holiday set, not moving-only
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


def add_informed_features(mat: pd.DataFrame, ratio: dict, hol: set) -> pd.DataFrame:
    """Return a COPY with the 6 new columns added. Always populated (no
    threshold, no masking) -- the model learns how much to weigh them."""
    df = mat.copy()
    T = pd.to_datetime(df["target_date"])
    h = df["horizon"].astype(int)

    src365 = T - pd.Timedelta(days=365)
    src728 = T - pd.Timedelta(days=728)
    src_yoy = T - pd.to_timedelta(h + 1, unit="D") - pd.Timedelta(days=365)

    df["lag365_src_is_holiday"] = src365.isin(hol).astype("int8")
    df["lag365_src_ratio"] = src365.map(ratio).astype("float64")
    df["lag728_src_is_holiday"] = src728.isin(hol).astype("int8")
    df["lag728_src_ratio"] = src728.map(ratio).astype("float64")
    df["yoyratio_src_is_holiday"] = src_yoy.isin(hol).astype("int8")
    df["yoyratio_src_ratio"] = src_yoy.map(ratio).astype("float64")
    return df


def holiday_window_dates_by_name() -> dict:
    """holiday_name -> set of all its own window dates, across all anchors.
    Split by name (not the flat 9-holiday union) so CNY can get full
    coverage while the other 8 get sampled -- see main()."""
    out = {}
    for name, anchors in S.HOLIDAY_ANCHORS.items():
        ws, we = S.HOLIDAY_WINDOWS[name]
        dates = set()
        for a in anchors:
            for k in range(ws, we + 1):
                dates.add(a + pd.Timedelta(days=k))
        out[name] = dates
    return out


def affected_targets_by_holiday(demand: pd.DataFrame, hol_by_name: dict,
                                horizons: list) -> dict:
    """holiday_name -> set of target dates whose lag_365, lag_728, OR
    yoy_ratio source lands in THAT holiday's window. This is what decides
    where the origins go -- not a manual guess."""
    have = set(demand["date"])
    out = {}
    for name, dates in hol_by_name.items():
        s = set()
        for d in dates:
            if not usable(d):
                continue
            for off in (365, 728):
                t = d + pd.Timedelta(days=off)
                if t in have and usable(t):
                    s.add(t)
            for hh in horizons:
                t = d + pd.Timedelta(days=365 + hh + 1)
                if t in have and usable(t):
                    s.add(t)
        out[name] = s
    return out


def week_bucket_origin(t: pd.Timestamp) -> pd.Timestamp:
    """The origin whose horizons 1..7 include t -- buckets t into a stable
    weekly grid anchored at 2015-01-01, so nearby affected dates collapse
    onto the same origin instead of each getting their own."""
    return t - pd.Timedelta(days=((t - pd.Timestamp("2015-01-01")).days % 7) + 1)


# ── backtest, same shape as narrow_mask_longhistory.py's run_origin ───────
def run_origin(mat, feats, demand_map, origin, anchor_demand):
    # Only drop rows missing the TARGET -- not rows missing some feature.
    # LightGBM handles NaN features natively (this is by design: long-lag
    # columns like lag_728 are legitimately NaN for the first 2 years, and
    # short lags are legitimately NaN whenever horizon exceeds them). An
    # earlier version of this line required every one of the ~180 feature
    # columns to be simultaneously non-null, which no row ever satisfies --
    # it silently zeroed out every origin. Caught by actually running it:
    # the first live run crashed with "No objects to concatenate" because
    # every single origin produced zero trainable rows.
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
    ap.add_argument("--controls", type=int, default=2,
                    help="clean control origins per event year (default 2)")
    ap.add_argument("--other-holiday-weeks", type=int, default=2,
                    help="representative affected-weeks per year for each "
                         "NON-CNY holiday (default 2). CNY itself always "
                         "gets full weekly coverage of every affected week -- "
                         "it is the dominant, highest-magnitude case and the "
                         "one this experiment exists to check first.")
    a = ap.parse_args()

    print("=" * 78)
    print("LAG-HOLIDAY INFORMED FEATURES -- LONG-HISTORY VALIDATION")
    print("=" * 78)

    demand = F.load_long_demand()
    print(f"data: {demand.date.min().date()} .. {demand.date.max().date()} "
          f"({len(demand):,} days)")

    hol_by_name = holiday_window_dates_by_name()
    hol = set().union(*hol_by_name.values())
    print(f"holiday-window days (all {len(hol_by_name)} holidays): {len(hol)}")

    aff_by_name = affected_targets_by_holiday(demand, hol_by_name, HORIZONS)
    aff = set().union(*aff_by_name.values())
    by_year = {}
    for t in sorted(aff):
        by_year.setdefault(t.year, []).append(t)
    by_year = {y: v for y, v in by_year.items() if len(v) >= 2}
    print(f"\naffected targets by holiday:")
    for name in sorted(aff_by_name, key=lambda n: -len(aff_by_name[n])):
        print(f"   {name:12s} {len(aff_by_name[name]):4d} affected target days")

    # CNY: full weekly coverage of every affected week -- it is the
    # dominant, highest-magnitude case (~50% demand collapse) and the one
    # this experiment exists to check first, per explicit instruction.
    # Other 8 holidays: only `other_holiday_weeks` representative weeks per
    # year (the highest-density ones), not exhaustive coverage -- their
    # swings are much milder (a few percent, not ~50%), so exhaustively
    # tiling the whole year for each of them the way CNY warrants would
    # spend most of the compute budget on the least informative holidays.
    origins = set()
    for t in aff_by_name.get("CNY", set()):
        origins.add(week_bucket_origin(t))

    for name, targets in aff_by_name.items():
        if name == "CNY" or not targets:
            continue
        by_year_h = {}
        for t in targets:
            by_year_h.setdefault(t.year, []).append(t)
        for y, ts in by_year_h.items():
            buckets = {}
            for t in ts:
                buckets.setdefault(week_bucket_origin(t), []).append(t)
            top = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
            for b, _ in top[:a.other_holiday_weeks]:
                origins.add(b)

    for y in by_year:
        for i in range(a.controls):
            cdate = pd.Timestamp(f"{y}-06-15") + pd.Timedelta(days=28 * i)
            if cdate <= demand.date.max() - pd.Timedelta(days=8) and usable(cdate):
                origins.add(cdate)
    origins = sorted({o for o in origins
                      if o >= demand.date.min() + pd.Timedelta(days=420)
                      and o <= demand.date.max() - pd.Timedelta(days=8)})
    print(f"\norigins: {len(origins)}  ({origins[0].date()} .. {origins[-1].date()})")
    print(f"fits: ~{len(origins)*len(HORIZONS)*2:,} across 2 variants")

    t0 = time.time()
    print("\nbuilding long-history matrix (one time, ~5-8 min)...", flush=True)
    base = F.build_base(demand, history_start=None, quiet=True)
    mat_off = F.apply_exclusions(base, demand, [])
    print(f"  matrix {mat_off.shape[0]:,} x {mat_off.shape[1]} "
          f"({int(mat_off.y.notna().sum()):,} trainable)  [{time.time()-t0:.0f}s]")

    ratio = build_ratio_map(demand)
    mat_informed = add_informed_features(mat_off, ratio, hol)

    feats_off = F.feature_columns(mat_off)
    feats_informed = F.feature_columns(mat_informed)
    print(f"  feature count: off={len(feats_off)}  informed={len(feats_informed)} "
          f"(+{len(feats_informed)-len(feats_off)})")

    variants = {"off": (mat_off, feats_off), "informed": (mat_informed, feats_informed)}
    dm = dict(zip(demand.date, demand.demand.astype(float)))

    results = {}
    for tag, (mat, feats) in variants.items():
        print(f"\n### {tag}", flush=True)
        t0, rows = time.time(), []
        for o in origins:
            r = run_origin(mat, feats, dm, o, demand)
            if not r.empty:
                rows.append(r)
        df = pd.concat(rows, ignore_index=True)
        df["variant"] = tag
        results[tag] = df
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / f"lh_{tag}.csv", index=False)
        print(f"  {len(df)} predictions, MAPE {mape(df):.2f}%  [{time.time()-t0:.0f}s]")

    base_df = results["off"].sort_values(["origin", "horizon"]).reset_index(drop=True)
    fire = base_df.target.isin(aff).values

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'variant':10s} {'overall':>9s} {'FIRES':>9s} {'clean':>9s}")
    print(f"{'':10s} {'(n=%d)'%len(base_df):>9s} {'(n=%d)'%fire.sum():>9s} "
          f"{'(n=%d)'%(~fire).sum():>9s}")
    for tag in variants:
        d = results[tag].sort_values(["origin", "horizon"]).reset_index(drop=True)
        print(f"{tag:10s} {mape(d):8.2f}% {mape(d[fire]):8.2f}% {mape(d[~fire]):8.2f}%")

    print("\nPer-holiday MAPE on that holiday's affected days:")
    hdr2 = "  " + f"{'holiday':12s}" + "".join(f"{t:>12s}" for t in variants)
    print(hdr2)
    for name in sorted(aff_by_name, key=lambda n: -len(aff_by_name[n])):
        if not aff_by_name[name]:
            continue
        line = f"  {name:<12s}"
        for tag in variants:
            d = results[tag].sort_values(["origin", "horizon"]).reset_index(drop=True)
            sel = d.target.isin(aff_by_name[name]).values
            line += f"{mape(d[sel]):11.2f}%" if sel.sum() else f"{'--':>12s}"
        print(line)

    print("\nPer-year MAPE on affected days:")
    hdr = "  " + f"{'event year':12s}" + "".join(f"{t:>12s}" for t in variants)
    print(hdr)
    for y in sorted(by_year):
        line = f"  {y:<12d}"
        for tag in variants:
            d = results[tag].sort_values(["origin", "horizon"]).reset_index(drop=True)
            sel = fire & (d.target.dt.year == y).values
            line += f"{mape(d[sel]):11.2f}%" if sel.sum() else f"{'--':>12s}"
        print(line)

    print("\nPer-origin comparison (does informed ever lose to off):")
    d_off = results["off"]
    d_inf = results["informed"]
    worse = 0
    for o in origins:
        a_ = mape(d_off[d_off.origin == o])
        b_ = mape(d_inf[d_inf.origin == o])
        if np.isnan(a_) or np.isnan(b_):
            continue
        flag = " <-- WORSE" if b_ > a_ + 1e-9 else ""
        if flag:
            worse += 1
        print(f"  {o.date()}   off={a_:6.2f}%   informed={b_:6.2f}%   "
              f"diff={b_-a_:+.2f}pp{flag}")

    print(f"\nGATE: {worse}/{len(origins)} origins worse -- "
          f"{'FAIL' if worse else 'PASS'}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
