"""analog_days_longhistory.py -- can "find similar days in past years, reuse
their lift over their own normal baseline" beat the LightGBM model?

THE IDEA UNDER TEST
-------------------
Instead of learning a regression, describe each day by CALENDAR + HOLIDAY
facts only (things known years in advance -- no demand anywhere in the
similarity space), find its nearest historical analogs, and reuse how much
those analogs deviated from *their own* recent normal level:

    ratio(S)  = demand(S) / normal_baseline(S, S - h)
    pred(T)   = normal_baseline(T, O) x aggregate{ ratio(S) : S in analogs(T) }

`normal_baseline(D, cutoff)` = mean demand over non-holiday, non-closure days
in the 28 days ending at `cutoff`. Using a RATIO rather than raw demand is
what makes 2016 analogs usable at all: the Macau market level changed
structurally (COVID, then the 2022 junket crackdown), but "CNY day +1 runs at
0.55x the preceding normal weeks" is a shape that may survive the level shift.
That is precisely the bet this script measures.

Note the baselines are HORIZON-PARALLEL. The target is forecast from origin O
at lead h, so its baseline can only use data up to O. The analog is therefore
scored from ITS parallel origin S-h, not from S-1 -- otherwise a 14-day-ahead
forecast would be compared against analogs that had 13 extra days of
information. Getting this wrong is the obvious way to make the method look
good by accident.

TWO SIMILARITY MECHANISMS (both are "clustering" in the loose sense; they
differ in whether the neighbourhood is per-query or precomputed):

    analog_knn      k nearest neighbours by weighted Euclidean distance
    analog_kmeans   KMeans on the same space; the target inherits its
                    cluster's ratio distribution

Plus the pool-restriction question the long history exists to answer:

    analog_knn_recent   identical to analog_knn but analogs restricted to
                        2023+ -- does reaching back to 2015 actually help?

BENCHMARKS
    dow_base   mean of same-weekday demand over the 5 same-weekday days
               available at O (the simple baseline already in use)
    lgbm       the shipped long-history model, config.py defaults, with
               production's moving CNY anchor applied exactly as production
               applies it

LEAKAGE GUARDS (each asserted at runtime by --self-check)
  - analog pool, baselines and LightGBM training are all restricted to
    dates <= O. Nothing reads T, nor anything between O+1 and T.
  - similarity features contain NO demand -- only calendar/holiday/mainland
    facts knowable years ahead.
  - COVID (2020-01-23..2022-12-31) and full-closure days are excluded from
    the analog pool, from baselines and from scoring.
  - k, cluster count and feature weights are FIXED A PRIORI (see DEFAULTS
    below) and are not tuned on the evaluation set. The k-sensitivity table
    at the end is a diagnostic, not a selection step.

READ-ONLY / NON-DESTRUCTIVE
    Imports long_history/{config,engine} read-only. Writes only under
    research/analog_days_output/. Touches nothing in v2/ or long_history/.

RESULT (run 2026-08-19: 88 origins, 2023-01-01..2026-05-03, h=1/2/4/7/10/14,
517 scored days) -- REJECTED as a replacement, one thread worth keeping.

  Pooled over the whole window the analog method looks competitive
  (8.41% vs lgbm 8.68%, and 16.25% vs 18.77% on holidays). That is an
  ARTIFACT and should not be quoted. The LightGBM arm is not a fixed
  competitor: its training set grows with the origin, so early origins
  measure a data-starved model, not the model that exists now.

    era              ALL                normal             holiday
    2023-24    analog 8.10 / lgbm 9.87  7.20 / 7.20   15.26 / 30.95
    2025-26    analog 8.85 / lgbm 6.99  7.36 / 7.19   17.31 /  5.83

  CNY, lgbm by year: 2023 37.5% -> 2024 86.9% -> 2025 5.6% -> 2026 6.7%.
  The 2024 blow-up is genuine (at that origin the model had one usable
  recent CNY, the 2023 recovery anomaly) but it cannot recur -- history
  only accumulates. In the current regime the model beats the analog
  method on every segment, and by 3x on holidays.

  Per-origin gate: 43/87 origins worse than lgbm -- FAIL.

  Two findings that DO survive:
    - The long history genuinely helps the analog method: restricting the
      pool to 2023+ degrades it 8.41% -> 9.98%, and 16.25% -> 23.02% on
      holidays. Ratio-to-local-baseline really does carry shape across the
      level shift, which is the premise the long-history pipeline rests on.
    - A 50/50 analog+lgbm blend beats lgbm on NORMAL days in 4 of 4 years
      (7.16/6.14/7.95/3.59 vs 7.97/6.50/8.17/4.13). Consistent direction,
      but the 0.5 weight was chosen AFTER seeing these results and the
      blend still fails the per-origin gate (32/86 normal-day origins
      worse), so it is a lead to test properly, not a result. On holidays
      the blend is much worse than lgbm today (10.42% vs 5.83%).

Usage:
    uv run python research/analog_days_longhistory.py --self-check
    uv run python research/analog_days_longhistory.py --skip-lgbm    # fast arms only
    uv run python research/analog_days_longhistory.py                # full run
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
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "long_history"))

import config as C                  # noqa: E402
from engine import core as S        # long_history's VENDORED engine -- read-only
from engine import features as F    # noqa: E402

OUT = ROOT / "research" / "analog_days_output"

# ── fixed a priori; NOT tuned on the evaluation set ───────────────────────
K_NEIGHBOURS = 15
N_CLUSTERS = 40
BASELINE_WINDOW = 28          # days ending at the cutoff
BASELINE_MIN_DAYS = 5         # widen the window until this many normal days
BASELINE_WIDEN = (28, 56, 84)
RECENCY_HALF_LIFE = 1095      # matches config.DEFAULT_HALF_LIFE
DIST_SOFTENER = 0.5           # w = 1/(dist + DIST_SOFTENER); avoids /0 on exact ties
RECENT_POOL_START = pd.Timestamp("2023-01-01")

COVID_LO, COVID_HI = pd.Timestamp("2020-01-23"), pd.Timestamp("2022-12-31")

# Similarity weights. Holiday identity and position dominate, then weekday,
# then season -- the ordering the domain implies (a CNY d+1 Saturday is far
# more like another CNY d+1 than like an ordinary Saturday).
WEIGHTS = {
    "dow_sin": 3.0, "dow_cos": 3.0,
    "is_friday": 1.5, "is_saturday": 1.5, "is_sunday": 1.5,
    "doy_sin": 1.0, "doy_cos": 1.0,
    "is_holiday": 3.0, "hol_offset": 3.0,
    "mainland_block_len": 1.0, "mainland_block_pos": 1.0, "mainland_workday": 1.0,
    "log_days_to_hol": 0.8, "log_days_from_hol": 0.8,
}
HOLIDAY_ONEHOT_WEIGHT = 4.0


def usable(d: pd.Timestamp) -> bool:
    return not (COVID_LO <= d <= COVID_HI)


# ══════════════════════════════════════════════════════════════════════════
# Similarity space -- calendar + holiday facts only. NO DEMAND.
# ══════════════════════════════════════════════════════════════════════════
def holiday_name_offset_map() -> dict:
    """date -> (holiday_name, offset_from_anchor) over ALL 9 holidays.

    core._holiday_alignment_map() covers only the MOVING set (it exists to
    align year-over-year lags); the similarity space wants every holiday,
    so this builds the full map. First anchor wins on the rare overlap.
    """
    out = {}
    for name, anchors in S.HOLIDAY_ANCHORS.items():
        ws, we = S.HOLIDAY_WINDOWS[name]
        for a in anchors:
            for off in range(ws, we + 1):
                out.setdefault(a + pd.Timedelta(days=off), (name, off))
    return out


def build_similarity_space(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per date, columns = raw (unstandardised) similarity features.

    Every column here is a pure calendar/holiday/mainland-calendar fact,
    knowable years in advance. If a demand-derived column ever appears in
    this function the whole experiment is invalid -- check_no_demand_in_space
    asserts it stays that way.
    """
    df = pd.DataFrame({"target_date": dates})
    with F.extended_holidays():
        df = S.add_calendar(df)
        df = S.add_holiday_flags(df)
        df = S.add_mainland_block(df)
        hol_map = holiday_name_offset_map()
    names = sorted(S.HOLIDAY_ANCHORS.keys())

    t = pd.to_datetime(df["target_date"])
    sp = pd.DataFrame(index=df.index)
    for c in ("dow_sin", "dow_cos", "doy_sin", "doy_cos",
              "is_friday", "is_saturday", "is_sunday"):
        sp[c] = df[c].astype(float)

    hit = t.map(lambda d: hol_map.get(d))
    sp["is_holiday"] = hit.notna().astype(float)
    # Offset normalised by the holiday's own window half-width, so "1 day
    # into a 4-day holiday" and "1 day into an 18-day holiday" are not
    # forced onto the same scale.
    def _off_norm(d):
        v = hol_map.get(d)
        if v is None:
            return 0.0
        name, off = v
        ws, we = S.HOLIDAY_WINDOWS[name]
        span = max(abs(ws), abs(we), 1)
        return off / float(span)
    sp["hol_offset"] = t.map(_off_norm).astype(float)

    for n in names:
        sp[f"hol_{n}"] = t.map(lambda d, n=n: 1.0 if (hol_map.get(d) or ("", 0))[0] == n
                               else 0.0).astype(float)

    sp["mainland_block_len"] = df["mainland_block_length"].astype(float)
    sp["mainland_block_pos"] = df["mainland_block_pos_norm"].astype(float)
    sp["mainland_workday"] = df["mainland_is_workday"].astype(float)
    sp["log_days_to_hol"] = np.log1p(df["days_to_next_holiday"].clip(0, 120)).astype(float)
    sp["log_days_from_hol"] = np.log1p(df["days_from_last_holiday"].clip(0, 120)).astype(float)

    sp.index = pd.DatetimeIndex(t)
    return sp


def weight_vector(cols: list) -> np.ndarray:
    return np.array([WEIGHTS.get(c, HOLIDAY_ONEHOT_WEIGHT if c.startswith("hol_") else 1.0)
                     for c in cols], dtype=float)


def standardise(sp: pd.DataFrame) -> tuple[np.ndarray, list]:
    """z-score each column over the whole space, then apply the weights.

    Standardising over the FULL date index (not the per-origin pool) is
    deliberate: the scaling is a property of the calendar, which is known in
    advance for every date including future ones, so it carries no
    information about demand and cannot leak.
    """
    cols = list(sp.columns)
    X = sp[cols].to_numpy(dtype=float)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    return Z * weight_vector(cols), cols


# ══════════════════════════════════════════════════════════════════════════
# Baselines
# ══════════════════════════════════════════════════════════════════════════
class Baselines:
    """normal_baseline(D, cutoff) and the same-DOW benchmark, memoised.

    `normal_baseline` deliberately does NOT depend on D except through the
    cutoff -- it is "the recent normal level as known at `cutoff`". D is kept
    in the signature only to make call sites read the way the method is
    described.
    """

    def __init__(self, demand: pd.DataFrame):
        self.dm = dict(zip(demand["date"], demand["demand"].astype(float)))
        with F.extended_holidays():
            self.hol = S.holiday_date_set()
        self.closed = set(demand.loc[(demand.floortables <= 0) |
                                     (demand.demand <= 0), "date"])
        self.dow = {d: d.weekday() for d in self.dm}
        self._nb_cache: dict = {}
        self._dow_cache: dict = {}

    def is_normal(self, d: pd.Timestamp) -> bool:
        v = self.dm.get(d)
        return (v is not None and not np.isnan(v) and v > 0
                and d not in self.hol and d not in self.closed and usable(d))

    def normal_baseline(self, cutoff: pd.Timestamp) -> float:
        """Mean demand over normal days in the window ending at `cutoff`."""
        hit = self._nb_cache.get(cutoff)
        if hit is not None:
            return hit
        out = np.nan
        for win in BASELINE_WIDEN:
            vals = [self.dm[d] for d in pd.date_range(cutoff - pd.Timedelta(days=win - 1), cutoff)
                    if self.is_normal(d)]
            if len(vals) >= BASELINE_MIN_DAYS:
                out = float(np.mean(vals))
                break
        self._nb_cache[cutoff] = out
        return out

    def same_dow_baseline(self, target: pd.Timestamp, cutoff: pd.Timestamp,
                          n_weeks: int = 5) -> float:
        """The benchmark already in use: mean of the last `n_weeks` same-weekday
        demands available at `cutoff`. Holiday days are kept out -- including
        them is what makes this benchmark collapse over CNY."""
        key = (target.weekday(), cutoff)
        hit = self._dow_cache.get(key)
        if hit is not None:
            return hit
        vals, d = [], cutoff
        while d > cutoff - pd.Timedelta(days=90) and len(vals) < n_weeks:
            if d.weekday() == target.weekday() and self.is_normal(d):
                vals.append(self.dm[d])
            d -= pd.Timedelta(days=1)
        out = float(np.mean(vals)) if len(vals) >= 2 else np.nan
        self._dow_cache[key] = out
        return out


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


# ══════════════════════════════════════════════════════════════════════════
# The analog forecasters
# ══════════════════════════════════════════════════════════════════════════
def ratio_for(bl: Baselines, S_day: pd.Timestamp, h: int) -> float:
    """demand(S) / normal_baseline(S - h). NaN when either side is unusable.

    The `- h` is the horizon-parallel origin: the analog is measured from the
    same information distance the live forecast has.
    """
    v = bl.dm.get(S_day)
    if v is None or np.isnan(v) or v <= 0 or S_day in bl.closed or not usable(S_day):
        return np.nan
    b = bl.normal_baseline(S_day - pd.Timedelta(days=h))
    if not np.isfinite(b) or b <= 0:
        return np.nan
    return v / b


def analog_predict(target: pd.Timestamp, origin: pd.Timestamp, h: int,
                   bl: Baselines, Z: np.ndarray, pos: dict, all_dates: np.ndarray,
                   k: int = K_NEIGHBOURS, pool_start: pd.Timestamp | None = None,
                   return_detail: bool = False):
    """KNN analog forecast. Pool is strictly <= origin (and >= pool_start)."""
    base_t = bl.normal_baseline(origin)
    if not np.isfinite(base_t) or base_t <= 0:
        return (np.nan, {}) if return_detail else np.nan

    lo = pool_start if pool_start is not None else pd.Timestamp.min
    cand = [d for d in all_dates
            if d <= origin and d >= lo and usable(d) and d in pos]
    if not cand:
        return (np.nan, {}) if return_detail else np.nan

    idx = np.array([pos[d] for d in cand])
    q = Z[pos[target]] if target in pos else None
    if q is None:
        return (np.nan, {}) if return_detail else np.nan
    dist = np.sqrt(((Z[idx] - q) ** 2).sum(axis=1))

    order = np.argsort(dist)
    picked_r, picked_w, picked_d = [], [], []
    for j in order:
        d = cand[j]
        r = ratio_for(bl, d, h)
        if not np.isfinite(r):
            continue
        age = (origin - d).days
        w = (1.0 / (dist[j] + DIST_SOFTENER)) * 0.5 ** (age / RECENCY_HALF_LIFE)
        picked_r.append(r); picked_w.append(w); picked_d.append(d)
        if len(picked_r) >= k:
            break
    if not picked_r:
        return (np.nan, {}) if return_detail else np.nan

    r = weighted_median(np.array(picked_r), np.array(picked_w))
    pred = base_t * r
    if return_detail:
        return pred, {"ratio": r, "n": len(picked_r), "analogs": picked_d}
    return pred


def kmeans_predict(target: pd.Timestamp, origin: pd.Timestamp, h: int,
                   bl: Baselines, Z: np.ndarray, pos: dict, labels: np.ndarray,
                   cand: list, k_fallback_args: tuple):
    """Cluster-based variant: the target inherits its cluster's ratio
    distribution. Falls back to KNN when the cluster is too thin to trust."""
    base_t = bl.normal_baseline(origin)
    if not np.isfinite(base_t) or base_t <= 0:
        return np.nan
    if target not in pos:
        return np.nan
    lab = labels[pos[target]] if pos[target] < len(labels) else None
    if lab is None:
        return np.nan

    rs, ws = [], []
    for d in cand:
        if labels[pos[d]] != lab:
            continue
        r = ratio_for(bl, d, h)
        if not np.isfinite(r):
            continue
        rs.append(r); ws.append(0.5 ** ((origin - d).days / RECENCY_HALF_LIFE))
    if len(rs) < BASELINE_MIN_DAYS:
        return analog_predict(target, origin, h, bl, Z, pos, *k_fallback_args)
    return base_t * weighted_median(np.array(rs), np.array(ws))


# ══════════════════════════════════════════════════════════════════════════
# LightGBM arm -- the shipped long-history model
# ══════════════════════════════════════════════════════════════════════════
def lgbm_origin(mat, feats, dm, origin, demand, horizons):
    """Train on data <= origin, predict this origin's horizons, then apply
    production's moving CNY anchor exactly as production does.

    dropna is on "y" ONLY. Long-lag columns are legitimately NaN by design
    and LightGBM handles NaN natively -- requiring every feature non-null
    silently empties the training set.
    """
    train = mat[mat["target_date"] <= origin].dropna(subset=["y"])
    rows = []
    for h in horizons:
        sub = train[train["horizon"] == h]
        if len(sub) < 60:
            continue
        T = origin + pd.Timedelta(days=h)
        v = dm.get(T)
        if v is None or np.isnan(v) or v <= 0:
            continue
        pr = mat[(mat["target_date"] == T) & (mat["horizon"] == h)]
        if pr.empty:
            continue
        w = S.make_sample_weights(sub["target_date"],
                                  is_holiday=S.holiday_mask_from_matrix(sub),
                                  half_life_days=C.DEFAULT_HALF_LIFE)
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
        a = S.apply_moving_anchor(p, demand, origin)
    df["pred"] = a["p50"].values
    return df


# ══════════════════════════════════════════════════════════════════════════
# Self-checks -- these must pass before any number below is worth reading
# ══════════════════════════════════════════════════════════════════════════
def self_check(bl: Baselines, sp: pd.DataFrame, Z, pos, all_dates) -> int:
    fails = []

    # 1. no demand-derived column ever enters the similarity space.
    #    A whitelist, not a substring heuristic -- an earlier substring version
    #    flagged `doy_sin` (it contains "y_") while a genuinely demand-derived
    #    column named e.g. `level` would have slipped straight through.
    allowed = {"dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_friday",
               "is_saturday", "is_sunday", "is_holiday", "hol_offset",
               "mainland_block_len", "mainland_block_pos", "mainland_workday",
               "log_days_to_hol", "log_days_from_hol"}
    allowed |= {f"hol_{n}" for n in S.HOLIDAY_ANCHORS}
    unexpected = sorted(set(sp.columns) - allowed)
    if unexpected:
        fails.append(f"unrecognised columns in similarity space (must be "
                     f"calendar/holiday only): {unexpected}")

    # 2. an analog's ratio must not read anything after its own parallel origin
    d = pd.Timestamp("2025-03-15")
    seen = []
    orig_nb = bl.normal_baseline
    def spy(cutoff):
        seen.append(cutoff)
        return orig_nb(cutoff)
    bl.normal_baseline = spy
    ratio_for(bl, d, 7)
    bl.normal_baseline = orig_nb
    if any(c > d - pd.Timedelta(days=7) for c in seen):
        fails.append(f"ratio_for read a baseline after its parallel origin: {seen}")

    # 3. the prediction pool never contains a date after the origin
    O = pd.Timestamp("2025-06-01")
    T = O + pd.Timedelta(days=10)
    _, det = analog_predict(T, O, 10, bl, Z, pos, all_dates, return_detail=True)
    late = [x for x in det.get("analogs", []) if x > O]
    if late:
        fails.append(f"analogs after the origin: {late[:5]}")

    # 4. COVID days must never be analogs
    covid = [x for x in det.get("analogs", []) if not usable(x)]
    if covid:
        fails.append(f"COVID-period analogs selected: {covid[:5]}")

    # 5. baseline excludes holidays
    b = bl.normal_baseline(pd.Timestamp("2026-02-20"))   # mid-CNY 2026
    raw = np.mean([bl.dm[x] for x in pd.date_range("2026-01-24", "2026-02-20")
                   if x in bl.dm and bl.dm[x] > 0])
    if np.isfinite(b) and abs(b - raw) < 1e-9:
        fails.append("normal_baseline matched the raw window mean over CNY "
                     "-- holiday exclusion is not taking effect")

    print("SELF-CHECKS")
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
    else:
        print("  5/5 pass  (no demand in similarity space; no post-origin or "
              "COVID analogs; horizon-parallel baselines; holidays excluded)")
    return 1 if fails else 0


# ══════════════════════════════════════════════════════════════════════════
def mape(d: pd.DataFrame) -> float:
    d = d[(d.actual > 0) & d.pred.notna()]
    return float(np.mean(np.abs(d.actual - d.pred) / d.actual)) * 100 if len(d) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-start", default="2023-01-01")
    ap.add_argument("--stride", type=int, default=14, help="days between origins")
    ap.add_argument("--horizons", default="1,2,4,7,10,14")
    ap.add_argument("--skip-lgbm", action="store_true",
                    help="run only the cheap arms (seconds instead of ~40 min)")
    ap.add_argument("--self-check", action="store_true",
                    help="run leakage self-checks and exit")
    a = ap.parse_args()
    horizons = [int(x) for x in a.horizons.split(",")]

    print("=" * 78)
    print("ANALOG-DAY (SIMILAR-DAY RATIO) FORECAST -- LONG-HISTORY VALIDATION")
    print("=" * 78)

    demand = F.load_long_demand()
    print(f"data: {demand.date.min().date()} .. {demand.date.max().date()} "
          f"({len(demand):,} days)")

    all_dates = np.array(sorted(demand["date"]))
    span = pd.DatetimeIndex(pd.date_range(demand.date.min(), demand.date.max()))
    sp = build_similarity_space(span)
    Z, cols = standardise(sp)
    pos = {d: i for i, d in enumerate(sp.index)}
    bl = Baselines(demand)
    print(f"similarity space: {Z.shape[0]:,} days x {Z.shape[1]} features "
          f"(calendar/holiday only)")

    if a.self_check:
        return self_check(bl, sp, Z, pos, all_dates)
    rc = self_check(bl, sp, Z, pos, all_dates)
    if rc:
        print("\nABORTING -- self-checks failed, results would not be trustworthy.")
        return 1

    origins = [d for d in pd.date_range(a.eval_start,
                                        demand.date.max() - pd.Timedelta(days=max(horizons)),
                                        freq=f"{a.stride}D") if usable(d)]
    print(f"\norigins: {len(origins)}  ({origins[0].date()} .. {origins[-1].date()})"
          f"   horizons: {horizons}")

    with F.extended_holidays():
        hol_set = S.holiday_date_set()
        hol_map = holiday_name_offset_map()

    # ── cheap arms ────────────────────────────────────────────────────────
    t0 = time.time()
    rows = []
    for O in origins:
        cand = [d for d in all_dates if d <= O and usable(d) and d in pos]
        if len(cand) < 200:
            continue
        cidx = np.array([pos[d] for d in cand])
        km = KMeans(n_clusters=min(N_CLUSTERS, len(cand) // 10), n_init=4,
                    random_state=123).fit(Z[cidx])
        labels_full = np.full(Z.shape[0], -1)
        labels_full[cidx] = km.labels_
        for T in [O + pd.Timedelta(days=h) for h in horizons]:
            labels_full[pos[T]] = int(km.predict(Z[pos[T]].reshape(1, -1))[0])

        for h in horizons:
            T = O + pd.Timedelta(days=h)
            act = bl.dm.get(T)
            if act is None or np.isnan(act) or act <= 0 or T in bl.closed or not usable(T):
                continue
            rows.append({
                "origin": O, "target": T, "horizon": h, "actual": act,
                "is_holiday": T in hol_set,
                "holiday": (hol_map.get(T) or ("none", 0))[0],
                "analog_knn": analog_predict(T, O, h, bl, Z, pos, all_dates),
                "analog_knn_recent": analog_predict(T, O, h, bl, Z, pos, all_dates,
                                                    pool_start=RECENT_POOL_START),
                "analog_kmeans": kmeans_predict(T, O, h, bl, Z, pos, labels_full, cand,
                                                (all_dates, K_NEIGHBOURS)),
                "dow_base": bl.same_dow_baseline(T, O),
            })
    df = pd.DataFrame(rows)
    print(f"  cheap arms: {len(df)} predictions  [{time.time()-t0:.0f}s]")

    # ── LightGBM arm ──────────────────────────────────────────────────────
    arms = ["analog_knn", "analog_knn_recent", "analog_kmeans", "dow_base"]
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.skip_lgbm:
        t0 = time.time()
        print("\nbuilding long-history matrix (cached after the first run)...", flush=True)
        base = F.build_base(demand, history_start=None, quiet=True)
        mat = F.apply_exclusions(base, demand, [])
        feats = F.feature_columns(mat)
        print(f"  matrix {mat.shape[0]:,} x {mat.shape[1]}  [{time.time()-t0:.0f}s]")
        print(f"  fitting {len(origins)*len(horizons):,} models "
              f"(~{len(origins)*len(horizons)*4/60:.0f} min)...", flush=True)
        t0, parts = time.time(), []
        for i, O in enumerate(origins, 1):
            r = lgbm_origin(mat, feats, bl.dm, O, demand, horizons)
            if not r.empty:
                parts.append(r)
            if i % 10 == 0:
                print(f"    {i}/{len(origins)} origins  [{time.time()-t0:.0f}s]", flush=True)
        lg = pd.concat(parts, ignore_index=True)[["origin", "target", "horizon", "pred"]]
        lg = lg.rename(columns={"pred": "lgbm"})
        lg.to_csv(OUT / "lgbm_arm.csv", index=False)
        df = df.merge(lg, on=["origin", "target", "horizon"], how="left")
        arms.append("lgbm")
        print(f"  lgbm arm done  [{time.time()-t0:.0f}s]")
    elif (OUT / "lgbm_arm.csv").exists():
        lg = pd.read_csv(OUT / "lgbm_arm.csv", parse_dates=["origin", "target"])
        df = df.merge(lg, on=["origin", "target", "horizon"], how="left")
        arms.append("lgbm")
        print("  reused cached lgbm arm from a previous run")

    # score only rows every arm produced, so the comparison is like-for-like
    complete = df[arms].notna().all(axis=1)
    dropped = int((~complete).sum())
    df = df[complete].reset_index(drop=True)
    df.to_csv(OUT / "predictions.csv", index=False)
    if dropped:
        print(f"  ({dropped} rows dropped -- at least one arm had no prediction)")

    def tbl(sub: pd.DataFrame, label: str, n_col=True):
        line = f"  {label:<22s}"
        if n_col:
            line += f"{len(sub):>6d}"
        for arm in arms:
            line += f"{mape(sub.rename(columns={arm:'pred'})):10.2f}%"
        print(line)

    hdr = f"  {'segment':<22s}{'n':>6s}" + "".join(f"{x:>11s}" for x in arms)
    print("\n" + "=" * 78)
    print("RESULTS -- MAPE (lower is better)")
    print("=" * 78)
    print(hdr)
    tbl(df, "ALL")
    tbl(df[~df.is_holiday], "normal days")
    tbl(df[df.is_holiday], "holiday days")

    print("\nBy holiday:")
    print(hdr)
    for name, sub in sorted(df[df.is_holiday].groupby("holiday"),
                            key=lambda kv: -len(kv[1])):
        tbl(sub, name)

    print("\nBy year (all days):")
    print(hdr)
    for y, sub in df.groupby(df.target.dt.year):
        tbl(sub, str(y))

    # THE DECISIVE TABLE. The pooled numbers above are misleading, because the
    # LightGBM arm is not a fixed competitor -- its training set grows with the
    # origin. At a Feb-2024 origin it had 2015-2019, nothing from 2020-2022
    # (COVID-excluded) and only the 2023 recovery year, i.e. effectively ONE
    # usable recent CNY, itself an anomaly. It scored 86.9% MAPE on CNY 2024.
    # By 2025-2026 it has three clean CNYs and scores ~6%. Pooling the two eras
    # hands the analog method a win it does not have today, over a
    # data-starvation condition that cannot recur (history only accumulates).
    print("\nBy era -- the model's training set GROWS, so pooling eras misleads:")
    print(hdr)
    for lo, hi, lbl in ((2023, 2024, "2023-24 data-starved"),
                        (2025, 2026, "2025-26 current")):
        s = df[df.target.dt.year.between(lo, hi)]
        if s.empty:
            continue
        tbl(s, lbl + " ALL")
        tbl(s[~s.is_holiday], lbl + " normal")
        tbl(s[s.is_holiday], lbl + " holiday")

    print("\nCNY by year (where the eras diverge most):")
    print(hdr)
    for y, sub in df[df.holiday == "CNY"].groupby(df.target.dt.year):
        tbl(sub, f"CNY {y}")

    print("\nBy horizon (all days):")
    print(hdr)
    for h, sub in df.groupby("horizon"):
        tbl(sub, f"h={h}")

    if "lgbm" in arms:
        print("\nPer-origin gate vs lgbm (does the analog method ever lose):")
        for arm in ("analog_knn", "analog_kmeans"):
            worse = tot = 0
            for O, sub in df.groupby("origin"):
                x = mape(sub.rename(columns={"lgbm": "pred"}))
                y = mape(sub.rename(columns={arm: "pred"}))
                if np.isnan(x) or np.isnan(y):
                    continue
                tot += 1
                worse += int(y > x + 1e-9)
            print(f"  {arm:<20s} {worse}/{tot} origins worse than lgbm -- "
                  f"{'FAIL' if worse else 'PASS'}")

    print("\nDIAGNOSTIC ONLY (not a selection step -- k was fixed at "
          f"{K_NEIGHBOURS} before this ran):")
    print(f"  {'k':<22s}{'n':>6s}{'all':>11s}{'normal':>11s}{'holiday':>11s}")
    for k in (5, 10, 15, 25, 40):
        p = [analog_predict(r.target, r.origin, r.horizon, bl, Z, pos, all_dates, k=k)
             for r in df.itertuples()]
        d2 = df.assign(pred=p)
        print(f"  k={k:<20d}{len(d2):>6d}{mape(d2):10.2f}%"
              f"{mape(d2[~d2.is_holiday]):10.2f}%{mape(d2[d2.is_holiday]):10.2f}%")

    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
