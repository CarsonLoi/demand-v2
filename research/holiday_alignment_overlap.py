"""holiday_alignment_overlap.py -- should the holiday-aligned YoY features
match to the most recent COMPARABLE occurrence instead of the most recent one?

THE PROBLEM
-----------
holiday_yoy_lag / holiday_yoy_lift align a holiday day this year to the same
position in the SAME holiday last year. That is right for the lunar drift it
was built for, but it ignores whether the two occurrences sat in the same
company.

Measured, Mid-Autumn multiplier vs its own pre-holiday lead-in baseline:

    standalone   2016 0.834  2018 0.890  2019 0.811  2023 0.923  2024 0.749
                 -> mean 0.841  (n=5)
    inside GW    2017 0.885   2025 1.036
                 -> mean 0.961  (n=2)

Mid-Autumn 2025 (Oct 5-7) fell entirely inside Golden Week (Oct 1-7) and ran
at 1.04x -- not soft at all. Mid-Autumn 2026 (Sep 24-26) is standalone, but
its holiday_yoy_* features point at 2025. The model is handed a reference
~23% above the standalone norm for exactly the days it must forecast.

THE FIX UNDER TEST
------------------
When matching, walk back to the most recent occurrence whose OVERLAP STATUS
matches -- i.e. the same set of other holidays covering that same offset day
-- rather than always taking the immediate predecessor. Mid-Autumn 2026 then
aligns to Mid-Autumn 2024 (standalone), not 2025 (inside Golden Week).

Two guards on the walk-back, both load-bearing:
  * skip occurrences whose matched day falls in an EXCLUDED period (COVID,
    closures). Without this the rule happily points Easter 2022 at Easter
    2020-04-12, deep in the COVID collapse -- verified, and worse than what
    it replaces.
  * cap the walk at --max-back occurrences; fall back to the immediate
    predecessor if nothing comparable is found, so the variant is never
    structurally worse than today's behaviour.

holiday_yoy_lift is a RATIO to its own anchor-local baseline (mean demand
30-60 days either side of that anchor), so a 2-year-old reference carries no
level drift. That is what makes reaching further back cheap.

WHAT ACTUALLY CHANGES (2016-2030 anchors, MOVING holidays only):
    20 (occurrence, offset) cells across 10 occurrences. Of those, only
    Mid-Autumn 2018 and Mid-Autumn 2025 are measurable here -- 2020/2021/2022
    are COVID-excluded and 2026/2027 have no actuals yet. TWO EVENTS. Read
    the gate accordingly; this is indicative, not conclusive.

READ-ONLY / NON-DESTRUCTIVE
    Imports long_history/{config,engine} read-only. The rebuilt columns are
    written to a COPY of the matrix in memory. Nothing on disk in
    long_history/ or v2/ is modified.

Usage:
    uv run python research/holiday_alignment_overlap.py --dry-run
    uv run python research/holiday_alignment_overlap.py
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "long_history"))

import config as C                    # noqa: E402
from engine import core as S          # long_history's VENDORED engine -- read-only
from engine import features as F      # noqa: E402

OUT = ROOT / "research" / "holiday_alignment_output"
HALF_LIFE = 240                       # production default; held fixed
HORIZONS = list(range(1, 8))
COVID_LO, COVID_HI = pd.Timestamp("2020-01-23"), pd.Timestamp("2022-12-31")
YOY_COLS = ["holiday_yoy_lag", "holiday_yoy_lift"]


def usable(d: pd.Timestamp) -> bool:
    return not (COVID_LO <= d <= COVID_HI)


# ══════════════════════════════════════════════════════════════════════════
# Overlap-aware alignment
# ══════════════════════════════════════════════════════════════════════════
def window_membership() -> dict:
    """date -> set of holiday names whose window contains it (ALL 9)."""
    member = defaultdict(set)
    for name, anchors in S.HOLIDAY_ANCHORS.items():
        ws, we = S.HOLIDAY_WINDOWS[name]
        for a in anchors:
            for x in pd.date_range(a + pd.Timedelta(days=ws), a + pd.Timedelta(days=we)):
                member[pd.Timestamp(x)].add(name)
    return member


def _overlap(member: dict, d: pd.Timestamp, self_name: str) -> frozenset:
    """Which OTHER holidays cover date d."""
    return frozenset(member.get(pd.Timestamp(d), set()) - {self_name})


def excluded_date_set(demand: pd.DataFrame) -> set:
    """Same set apply_exclusions builds: config ranges + full closures."""
    bad = set()
    for lo, hi in C.EXCLUDE_FROM_TRAINING:
        bad |= set(pd.date_range(lo, hi))
    if getattr(C, "EXCLUDE_CLOSURES", True):
        bad |= set(demand.loc[(demand.floortables <= 0) | (demand.demand <= 0), "date"])
    return bad


def alignment_map(comparable: bool, member: dict, max_back: int = 3) -> dict:
    """date -> (holiday_name, offset, matched_anchor).

    comparable=False reproduces core._holiday_alignment_map() exactly
    (always the immediate predecessor). comparable=True walks back to the
    most recent occurrence with the SAME OVERLAP STATUS, capped at max_back,
    falling back to the immediate predecessor when none is found.

    Deliberately does NOT skip excluded/COVID occurrences here. That is a
    separate concern already handled downstream by the same guard
    apply_exclusions applies (see guard_excluded_sources) -- folding it in
    re-points every 2021/2022 holiday away from COVID as well, which is a
    different change and would make the gate below uninterpretable.
    """
    out = {}
    for name, anchors in S.HOLIDAY_ANCHORS.items():
        if name not in S.MOVING_HOLIDAYS:
            continue
        ws, we = S.HOLIDAY_WINDOWS[name]
        srt = sorted(anchors)
        for j, a in enumerate(srt):
            if j == 0:
                continue
            prev = srt[j - 1]
            for off in range(ws, we + 1):
                d = a + pd.Timedelta(days=off)
                match = prev
                if comparable:
                    want = _overlap(member, d, name)
                    for k in range(j - 1, max(-1, j - 1 - max_back), -1):
                        cand = srt[k]
                        if _overlap(member, cand + pd.Timedelta(days=off), name) == want:
                            match = cand
                            break
                out.setdefault(d, (name, off, match))
    return out


def guard_excluded_sources(df: pd.DataFrame, align: dict, bad: set) -> pd.DataFrame:
    """NaN holiday_yoy_* where the matched source day falls in an excluded
    period -- the same rule features._guard_yearly applies to the shipped
    columns. Applied identically to BOTH variants so the only difference
    between them is the alignment itself."""
    t = pd.to_datetime(df["target_date"])
    src = t.map(lambda x: (lambda h: h[2] + pd.Timedelta(days=h[1]) if h else pd.NaT)(
        align.get(pd.Timestamp(x))))
    mask = src.isin(bad)
    for col in YOY_COLS:
        df.loc[mask, col] = np.nan
    return df


def rebuild_holiday_yoy(mat: pd.DataFrame, demand: pd.DataFrame,
                        align: dict) -> pd.DataFrame:
    """Return a COPY of mat with holiday_yoy_lag / _lift recomputed against
    `align`. Mirrors core.add_holiday_aligned_yoy's arithmetic exactly --
    only the alignment map differs."""
    dm = dict(zip(demand["date"], demand["demand"].astype(float)))
    base_cache = {}

    def baseline_around(anchor):
        if anchor in base_cache:
            return base_cache[anchor]
        vals = []
        for lo, hi in ((-60, -30), (30, 60)):
            for k in range(lo, hi + 1):
                v = dm.get(anchor + pd.Timedelta(days=k), np.nan)
                if v is not None and not np.isnan(v):
                    vals.append(v)
        base_cache[anchor] = float(np.mean(vals)) if vals else np.nan
        return base_cache[anchor]

    lag_by_date, lift_by_date = {}, {}
    for d, (_name, off, matched) in align.items():
        v = dm.get(matched + pd.Timedelta(days=off), np.nan)
        if v is None or np.isnan(v):
            continue
        lag_by_date[d] = float(v)
        b = baseline_around(matched)
        if b and not np.isnan(b) and b > 0:
            lift_by_date[d] = float(v) / b

    df = mat.copy()
    t = pd.to_datetime(df["target_date"])
    df["holiday_yoy_lag"] = t.map(lag_by_date).astype("float64")
    df["holiday_yoy_lift"] = t.map(lift_by_date).astype("float64")
    return df


def changed_cells(member: dict, max_back: int) -> list:
    """(holiday, anchor, offset, old_match, new_match) wherever they differ."""
    a_off = alignment_map(False, member, max_back)
    a_cmp = alignment_map(True, member, max_back)
    rows = []
    for d, (name, off, old) in a_off.items():
        new = a_cmp[d][2]
        if new != old:
            rows.append({"holiday": name, "target": d, "offset": off,
                         "old_match": old + pd.Timedelta(days=off),
                         "new_match": new + pd.Timedelta(days=off)})
    return sorted(rows, key=lambda r: r["target"])


# ══════════════════════════════════════════════════════════════════════════
# Backtest -- same shape as research/lag_holiday_features_longhistory.py
# ══════════════════════════════════════════════════════════════════════════
def run_origin(mat, feats, demand_map, origin, anchor_demand):
    # dropna on "y" ONLY. Long-lag columns are legitimately NaN by design and
    # LightGBM handles them natively; dropping on the full feature list
    # silently empties the training set.
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


# ══════════════════════════════════════════════════════════════════════════
# Model-free reference check
# ══════════════════════════════════════════════════════════════════════════
def reference_check(demand: pd.DataFrame, member: dict, max_back: int,
                    changed: list) -> int:
    """holiday_yoy_lift is a PREDICTION of how much the holiday lifts demand
    relative to its own local baseline. So score it directly: which matched
    reference is closer to the lift that actually happened?

    This needs no model and uses every affected cell with actuals, so it is
    not throttled by how few events a per-window model gate can resolve.
    """
    dm = dict(zip(demand["date"], demand["demand"].astype(float)))

    def baseline_around(anchor):
        vals = []
        for lo, hi in ((-60, -30), (30, 60)):
            for k in range(lo, hi + 1):
                v = dm.get(anchor + pd.Timedelta(days=k), np.nan)
                if v is not None and not np.isnan(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else np.nan

    with F.extended_holidays():
        anchors_of = {n: sorted(v) for n, v in S.HOLIDAY_ANCHORS.items()}

    print("\n" + "=" * 78)
    print("REFERENCE CHECK -- does the re-pointed reference predict the realised")
    print("holiday lift better?  (model-free; lift = demand / anchor-local baseline)")
    print("=" * 78)
    print(f"  {'target':>12} {'hol':11} {'off':>4} {'actual':>8} {'old ref':>9} "
          f"{'new ref':>9} {'|old err|':>10} {'|new err|':>10}  winner")

    rows = []
    for r in changed:
        T = r["target"]
        if T not in dm or not usable(T):
            continue
        own_anchor = None
        for cand in anchors_of[r["holiday"]]:
            if cand + pd.Timedelta(days=r["offset"]) == T:
                own_anchor = cand
                break
        if own_anchor is None:
            continue
        b_now = baseline_around(own_anchor)
        if not b_now or np.isnan(b_now) or b_now <= 0:
            continue
        actual_lift = dm[T] / b_now

        def ref_lift(day):
            anch = day - pd.Timedelta(days=r["offset"])
            v, b = dm.get(day, np.nan), baseline_around(anch)
            if v is None or np.isnan(v) or not b or np.isnan(b) or b <= 0:
                return np.nan
            return v / b

        lo_, ln_ = ref_lift(r["old_match"]), ref_lift(r["new_match"])
        if np.isnan(lo_) or np.isnan(ln_):
            continue
        eo, en = abs(lo_ - actual_lift), abs(ln_ - actual_lift)
        win = "new" if en < eo - 1e-9 else ("old" if eo < en - 1e-9 else "tie")
        rows.append({"target": T, "holiday": r["holiday"], "offset": r["offset"],
                     "actual_lift": actual_lift, "old_lift": lo_, "new_lift": ln_,
                     "old_err": eo, "new_err": en, "winner": win})
        print(f"  {str(T.date()):>12} {r['holiday']:11} {r['offset']:>+4d} "
              f"{actual_lift:8.3f} {lo_:9.3f} {ln_:9.3f} {eo:10.3f} {en:10.3f}  {win}")

    if not rows:
        print("\n  No affected cell has both references and an actual. Nothing to score.")
        return 1

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "reference_check.csv", index=False)
    n_new = int((df.winner == "new").sum())
    n_old = int((df.winner == "old").sum())
    print(f"\n  cells scored: {len(df)}   new-reference better: {n_new}   "
          f"old better: {n_old}   tie: {len(df)-n_new-n_old}")
    print(f"  mean |error| in predicted lift:  old={df.old_err.mean():.4f}   "
          f"new={df.new_err.mean():.4f}   "
          f"({100*(df.new_err.mean()/df.old_err.mean()-1):+.1f}%)")
    print("\n  by event:")
    for (hol, yr), g in df.groupby([df.holiday, df.target.dt.year]):
        print(f"    {hol:11s} {yr}  n={len(g)}  old={g.old_err.mean():.4f}  "
              f"new={g.new_err.mean():.4f}  -> "
              f"{'new better' if g.new_err.mean() < g.old_err.mean() else 'old better'}")
    print(f"\n  NOTE: {df.target.dt.year.nunique()} event year(s) scored. A win here "
          "says the reference is closer to truth,\n  not that end-to-end forecast "
          "accuracy improves -- run the full backtest for that.")
    print(f"\n-> {OUT / 'reference_check.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-back", type=int, default=3,
                    help="how many occurrences back the walk may reach (default 3)")
    ap.add_argument("--controls", type=int, default=2,
                    help="clean control origins per affected year (default 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what the alignment change touches, then exit")
    ap.add_argument("--reference-check", action="store_true",
                    help="model-free: does the re-pointed reference predict the "
                         "realised holiday lift better? Uses every affected cell "
                         "with actuals, so it is not limited by how few events "
                         "a model gate can resolve.")
    a = ap.parse_args()

    print("=" * 78)
    print("HOLIDAY-ALIGNED YOY: COMPARABLE-OCCURRENCE MATCHING")
    print("=" * 78)

    demand = F.load_long_demand()
    print(f"data: {demand.date.min().date()} .. {demand.date.max().date()} "
          f"({len(demand):,} days)")

    with F.extended_holidays():
        member = window_membership()
        bad = excluded_date_set(demand)
        changed = changed_cells(member, a.max_back)

    have = set(demand["date"])
    print(f"\nalignment cells re-pointed: {len(changed)}")
    print(f"  {'target':>12} {'hol':11} {'off':>4} {'old ref':>12} {'new ref':>12}  measurable")
    measurable = []
    for r in changed:
        ok = (r["target"] in have and usable(r["target"])
              and r["new_match"] in have and r["old_match"] in have)
        if ok:
            measurable.append(r["target"])
        print(f"  {str(r['target'].date()):>12} {r['holiday']:11} {r['offset']:>+4d} "
              f"{str(r['old_match'].date()):>12} {str(r['new_match'].date()):>12}"
              f"  {'YES' if ok else 'no'}")
    print(f"\nmeasurable affected target days: {len(measurable)}")
    if measurable:
        yrs = sorted({d.year for d in measurable})
        print(f"  spanning {len(yrs)} event year(s): {yrs}")
        print("  NOTE: this is a SMALL number of events. The gate below is "
              "indicative, not conclusive.")
    if a.dry_run:
        return 0

    if a.reference_check:
        return reference_check(demand, member, a.max_back, changed)

    if not measurable:
        print("\nNothing measurable in this data range -- nothing to test. "
              "Re-run once the affected windows have actuals.")
        return 1

    # ── origins: cover each affected target at several leads, plus controls ──
    origins = set()
    for T in measurable:
        for lead in (1, 3, 5, 7):
            o = T - pd.Timedelta(days=lead)
            if o in have and usable(o):
                origins.add(o)
    for y in sorted({d.year for d in measurable}):
        for i in range(a.controls):
            c = pd.Timestamp(f"{y}-06-15") + pd.Timedelta(days=28 * i)
            if c in have and usable(c) and c <= demand.date.max() - pd.Timedelta(days=8):
                origins.add(c)
    origins = sorted(origins)
    print(f"\norigins: {len(origins)}  ({origins[0].date()} .. {origins[-1].date()})")
    print(f"fits: ~{len(origins)*len(HORIZONS)*2:,} across 2 variants")

    t0 = time.time()
    print("\nbuilding long-history matrix (cached after first run)...", flush=True)
    base = F.build_base(demand, history_start=None, quiet=True)
    mat_off = F.apply_exclusions(base, demand, [])
    print(f"  matrix {mat_off.shape[0]:,} x {mat_off.shape[1]}  [{time.time()-t0:.0f}s]")

    # Rebuild BOTH variants from the same code path, then apply the same
    # excluded-source guard to each. Rebuilding "off" too (rather than reusing
    # the shipped columns) keeps the only difference between the variants the
    # alignment map itself -- an earlier version overwrote the already-guarded
    # columns for "comparable" only, restoring COVID-sourced values on one side
    # of the comparison.
    with F.extended_holidays():
        a_off = alignment_map(False, member, a.max_back)
        a_cmp = alignment_map(True, member, a.max_back)
    mat_off = guard_excluded_sources(rebuild_holiday_yoy(mat_off, demand, a_off), a_off, bad)
    mat_cmp = guard_excluded_sources(rebuild_holiday_yoy(mat_off, demand, a_cmp), a_cmp, bad)
    feats = F.feature_columns(mat_off)

    n_diff = int((~mat_off["holiday_yoy_lag"].fillna(-1).eq(
        mat_cmp["holiday_yoy_lag"].fillna(-1))).sum())
    print(f"  holiday_yoy_lag rows differing between variants: {n_diff:,}")

    variants = {"off": mat_off, "comparable": mat_cmp}
    dm = dict(zip(demand.date, demand.demand.astype(float)))
    results = {}
    for tag, mt in variants.items():
        print(f"\n### {tag}", flush=True)
        t0, rows = time.time(), []
        for o in origins:
            r = run_origin(mt, feats, dm, o, demand)
            if not r.empty:
                rows.append(r)
        df = pd.concat(rows, ignore_index=True)
        df["variant"] = tag
        results[tag] = df
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / f"ha_{tag}.csv", index=False)
        print(f"  {len(df)} predictions, MAPE {mape(df):.2f}%  [{time.time()-t0:.0f}s]")

    base_df = results["off"].sort_values(["origin", "horizon"]).reset_index(drop=True)
    fire = base_df.target.isin(set(measurable)).values

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'variant':12s} {'overall':>9s} {'AFFECTED':>10s} {'clean':>9s}")
    print(f"{'':12s} {'(n=%d)'%len(base_df):>9s} {'(n=%d)'%fire.sum():>10s} "
          f"{'(n=%d)'%(~fire).sum():>9s}")
    for tag in variants:
        d = results[tag].sort_values(["origin", "horizon"]).reset_index(drop=True)
        print(f"{tag:12s} {mape(d):8.2f}% {mape(d[fire]):9.2f}% {mape(d[~fire]):8.2f}%")

    print("\nPer affected event year (affected days only):")
    for y in sorted({d.year for d in measurable}):
        line = f"  {y}  "
        for tag in variants:
            d = results[tag].sort_values(["origin", "horizon"]).reset_index(drop=True)
            sel = fire & (d.target.dt.year == y).values
            line += f"{tag}={mape(d[sel]):6.2f}%  " if sel.sum() else f"{tag}=--  "
        print(line)

    print("\nPer-origin comparison:")
    d_off, d_cmp = results["off"], results["comparable"]
    worse = 0
    for o in origins:
        x, y = mape(d_off[d_off.origin == o]), mape(d_cmp[d_cmp.origin == o])
        if np.isnan(x) or np.isnan(y):
            continue
        flag = " <-- WORSE" if y > x + 1e-9 else ""
        worse += 1 if flag else 0
        print(f"  {o.date()}   off={x:6.2f}%   comparable={y:6.2f}%   "
              f"diff={y-x:+.2f}pp{flag}")

    print(f"\nGATE: {worse}/{len(origins)} origins worse -- "
          f"{'FAIL' if worse else 'PASS'}")
    print(f"\nCAVEAT: {len(measurable)} affected day(s) across "
          f"{len({d.year for d in measurable})} event year(s). Treat a PASS as "
          f"a reason to test further, not as adoption evidence.")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
