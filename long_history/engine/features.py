"""features.py — builds the feature matrix for the long-history experiment.

Thin layer over `core.py` -- this folder's VENDORED copy of the feature engine.
Nothing here imports from production, so experiments cannot affect it. Three
things are layered on top of core:

  1. Holiday anchors extended to 2016-2030 (production ships 2024+ only).
  2. ISO week-of-year -- absent from production. `month`, `week_of_month`,
     `quarter` and `day_of_year` (+ sin/cos) already exist; week-of-year is a
     useful middle granularity (52 buckets vs 12 months vs 365 days) that is
     noise on 2 years of data but learnable on 10.
  3. A `regime` feature, so the model can condition on pre-covid / covid /
     recovery / current rather than being forced to average across them.

core.py already ships the 2016-2030 anchors. `extended_holidays()` remains as
a context manager so the anchor set is explicit at the call site and always
restored afterwards.
"""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/ -- this package owns its data
sys.path.insert(0, str(ROOT))

from engine import core as S
import config as C
from engine.holidays import extended_anchors

_ORIGINAL_ANCHORS = {k: list(v) for k, v in S.HOLIDAY_ANCHORS.items()}


@contextlib.contextmanager
def extended_holidays():
    """Temporarily swap production's 2024+ anchors for the 2016-2030 set."""
    try:
        S.HOLIDAY_ANCHORS.clear()
        S.HOLIDAY_ANCHORS.update(extended_anchors())
        yield
    finally:
        S.HOLIDAY_ANCHORS.clear()
        S.HOLIDAY_ANCHORS.update(_ORIGINAL_ANCHORS)


def load_long_demand(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else ROOT / C.DATA_FILE
    d = pd.read_csv(p, parse_dates=["date"])
    d["demand"] = pd.to_numeric(d["demand"], errors="coerce")
    d["floortables"] = pd.to_numeric(d["floortables"], errors="coerce")
    return d.sort_values("date").reset_index(drop=True)


def add_week_of_year(mat: pd.DataFrame) -> pd.DataFrame:
    """ISO week number + cyclical encoding. Captures annual seasonality at a
    coarser grain than day_of_year, which matters once several years of the
    same week can be compared."""
    t = pd.to_datetime(mat["target_date"])
    wk = t.dt.isocalendar().week.astype(int)
    mat["week_of_year"] = wk
    mat["woy_sin"] = np.sin(2 * np.pi * wk / 52.0)
    mat["woy_cos"] = np.cos(2 * np.pi * wk / 52.0)
    return mat


def add_regime(mat: pd.DataFrame) -> pd.DataFrame:
    """Integer-coded regime, so the model can separate structural periods."""
    names = [r[0] for r in C.REGIMES]
    code = {n: i for i, n in enumerate(names)}
    reg = pd.to_datetime(mat["target_date"]).map(C.regime_of)
    mat["regime"] = reg.map(code).fillna(-1).astype(int)
    return mat


def engine_fingerprint() -> str:
    """Hash of the feature engine itself.

    Covers BOTH files that decide what ends up in the matrix: core.py (the
    bulk of the features) and this file (week_of_year, regime, exclusions).
    Hashing only core.py would leave a gap -- a change to add_regime here
    would not invalidate anything.

    Any cache whose contents depend on how features are computed MUST include
    this, or editing the engine silently serves stale results. Used by the
    matrix cache below and by the evaluation cache in training/evaluate.py.
    """
    h = hashlib.sha1()
    for name in ("core.py", "features.py"):
        h.update((HERE / name).read_bytes())
    return h.hexdigest()[:16]


def _matrix_cache_key(demand: pd.DataFrame, history_start, add_woy, add_reg) -> str:
    """Identity of a built matrix.

    Includes a hash of core.py itself, so editing the feature engine
    invalidates every cached matrix automatically. Without that, a stale
    matrix could be served after a feature change -- silently invalidating
    every downstream number.
    """
    parts = {
        "core": engine_fingerprint(),
        "rows": len(demand),
        "first": str(demand.date.min().date()),
        "last": str(demand.date.max().date()),
        "sum": float(pd.to_numeric(demand.demand, errors="coerce").fillna(0).sum()),
        "history_start": str(history_start),
        "woy": bool(add_woy), "reg": bool(add_reg),
        "holdout": int(S.HOLDOUT_DAYS),
    }
    blob = json.dumps(parts, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def build_base(demand: pd.DataFrame, history_start: str | None = None,
               add_woy: bool | None = None, add_reg: bool | None = None,
               quiet: bool = True, use_cache: bool | None = None) -> pd.DataFrame:
    """EXPENSIVE part: the feature matrix, with no exclusions applied yet.

    Split out so several exclusion variants can share one build -- the matrix
    takes ~6 minutes on 11 years of data, while applying exclusions is a
    fraction of a second.

    Cached to disk when config.CACHE_MATRIX is on (see _matrix_cache_key for
    how staleness is prevented).
    """
    add_woy = C.ADD_WEEK_OF_YEAR if add_woy is None else add_woy
    add_reg = C.ADD_REGIME_FEATURE if add_reg is None else add_reg
    use_cache = getattr(C, "CACHE_MATRIX", False) if use_cache is None else use_cache

    cache_path = None
    if use_cache:
        key = _matrix_cache_key(demand, history_start, add_woy, add_reg)
        cache_dir = ROOT / C.OUT_DIR / "matrix_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"matrix_{key}.pkl"
        if cache_path.exists():
            if not quiet:
                print(f"  [matrix] cache hit: {cache_path.name}")
            return pd.read_pickle(cache_path)

    d = demand
    if history_start:
        d = d[d.date >= pd.Timestamp(history_start)].reset_index(drop=True)

    with extended_holidays():
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                mat = S.build_matrix(d, holdout_days=S.HOLDOUT_DAYS,
                                     as_of=d.date.max())
        else:
            mat = S.build_matrix(d, holdout_days=S.HOLDOUT_DAYS, as_of=d.date.max())

    if add_woy:
        mat = add_week_of_year(mat)
    if add_reg:
        mat = add_regime(mat)

    if cache_path is not None:
        mat.to_pickle(cache_path)
        if not quiet:
            print(f"  [matrix] cached -> {cache_path.name}")
    return mat


def apply_exclusions(mat: pd.DataFrame, demand: pd.DataFrame,
                     extra_ranges: list | None = None) -> pd.DataFrame:
    """CHEAP part: blank y over excluded periods + closures, then guard the
    yearly lookbacks. Returns a COPY so the base matrix stays reusable."""
    mat = mat.copy()
    t = pd.to_datetime(mat["target_date"])
    ranges = list(C.EXCLUDE_FROM_TRAINING) + list(extra_ranges or [])
    excluded_dates = set()

    if ranges:
        mask = pd.Series(False, index=mat.index)
        for lo, hi in ranges:
            mask |= (t >= pd.Timestamp(lo)) & (t <= pd.Timestamp(hi))
            excluded_dates |= set(pd.date_range(lo, hi))
        mat.loc[mask, "y"] = np.nan

    if getattr(C, "EXCLUDE_CLOSURES", True):
        closed = set(demand.loc[(demand.floortables <= 0) |
                                (demand.demand <= 0), "date"])
        if closed:
            mat.loc[t.isin(closed), "y"] = np.nan
            excluded_dates |= closed

    if getattr(C, "GUARD_LAGS_CROSSING_EXCLUDED", True) and excluded_dates:
        _guard_yearly(mat, t, excluded_dates)
    return mat


def build(demand: pd.DataFrame, history_start: str | None = None,
          exclude_from_training: bool = True,
          add_woy: bool | None = None, add_reg: bool | None = None,
          quiet: bool = True) -> pd.DataFrame:
    """Feature matrix for one experiment variant.

    history_start        drop rows before this date ENTIRELY (changes what the
                         model can see at all, including via lags)
    exclude_from_training set y=NaN over config.EXCLUDE_FROM_TRAINING -- those
                         rows stop being learned from but their demand values
                         REMAIN available to lag features, keeping the calendar
                         contiguous. Deleting them instead would NaN out every
                         lag reaching across the gap.
    """
    add_woy = C.ADD_WEEK_OF_YEAR if add_woy is None else add_woy
    add_reg = C.ADD_REGIME_FEATURE if add_reg is None else add_reg

    d = demand
    if history_start:
        d = d[d.date >= pd.Timestamp(history_start)].reset_index(drop=True)

    with extended_holidays():
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                mat = S.build_matrix(d, holdout_days=S.HOLDOUT_DAYS,
                                     as_of=d.date.max())
        else:
            mat = S.build_matrix(d, holdout_days=S.HOLDOUT_DAYS, as_of=d.date.max())

    if add_woy:
        mat = add_week_of_year(mat)
    if add_reg:
        mat = add_regime(mat)

    # y is already set by build_matrix; blank it over excluded ranges so those
    # rows drop out of training (training does .dropna(subset=["y"])).
    t = pd.to_datetime(mat["target_date"])
    excluded_dates = set()

    if exclude_from_training and C.EXCLUDE_FROM_TRAINING:
        mask = pd.Series(False, index=mat.index)
        for lo, hi in C.EXCLUDE_FROM_TRAINING:
            mask |= (t >= pd.Timestamp(lo)) & (t <= pd.Timestamp(hi))
        mat.loc[mask, "y"] = np.nan
        excluded_dates |= set(pd.date_range(*C.EXCLUDE_FROM_TRAINING[0])) if \
            len(C.EXCLUDE_FROM_TRAINING) == 1 else set()
        for lo, hi in C.EXCLUDE_FROM_TRAINING:
            excluded_dates |= set(pd.date_range(lo, hi))

    # Full closures: floortables == 0 means zero tables open. Supply-zero, not
    # a demand observation -- and demand == 0 breaks MAPE outright.
    if getattr(C, "EXCLUDE_CLOSURES", True):
        closed = set(d.loc[(d.floortables <= 0) | (d.demand <= 0), "date"])
        if closed:
            mat.loc[t.isin(closed), "y"] = np.nan
            excluded_dates |= closed

    # Guard yearly lookbacks whose SOURCE date lands in an excluded period.
    # lag_365/lag_728 and yoy_ratio reach back far enough that 2023-2024 rows
    # would otherwise read a collapsed COVID value as "the same day last year".
    if getattr(C, "GUARD_LAGS_CROSSING_EXCLUDED", True) and excluded_dates:
        _guard_yearly(mat, t, excluded_dates)

    return mat


def _guard_yearly(mat: pd.DataFrame, t: pd.Series, bad: set) -> None:
    """NaN out yearly features whose source date falls in an excluded period."""
    h = mat["horizon"].astype(int)
    for col, offset in (("lag_365", 365), ("lag_anchor_365", 365), ("lag_728", 728)):
        if col in mat.columns:
            mat.loc[(t - pd.to_timedelta(offset, unit="D")).isin(bad), col] = np.nan
    if "yoy_ratio" in mat.columns:
        recent = t - pd.to_timedelta(h + 1, unit="D")
        mat.loc[recent.isin(bad) | (recent - pd.Timedelta(days=365)).isin(bad),
                "yoy_ratio"] = np.nan
    # holiday YoY block: source is the previous occurrence of the same holiday
    for col in ("holiday_yoy_lag", "holiday_yoy_lift", "same_holiday_lastyear_lag"):
        if col not in mat.columns:
            continue
        src = t - pd.Timedelta(days=364)          # legacy lag uses 364
        if col.startswith("holiday_yoy"):
            align = S._holiday_alignment_map()
            prev_src = t.map(lambda x: (align.get(pd.Timestamp(x)) or (None, 0, None))[2])
            off = t.map(lambda x: (align.get(pd.Timestamp(x)) or (None, 0, None))[1])
            src = pd.Series([p + pd.Timedelta(days=o) if p is not None else pd.NaT
                             for p, o in zip(prev_src, off)], index=mat.index)
        mat.loc[src.isin(bad), col] = np.nan


def feature_columns(mat: pd.DataFrame) -> list:
    return [c for c in mat.columns if c not in {"target_date", "horizon", "y"}]
