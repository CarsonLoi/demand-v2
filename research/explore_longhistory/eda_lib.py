from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "long_history"))

import config as C                           # noqa: E402
from engine import core as S                 # noqa: E402
from engine import features as F             # noqa: E402

CLEAN_YEARS = [2015, 2016, 2017, 2018, 2019, 2023, 2024, 2025, 2026]
COVID_YEARS = {2020, 2021, 2022}


def load_frames() -> pd.DataFrame:
    """Raw daily series + the derived per-table column and calendar parts."""
    d = F.load_long_demand()
    d = d[["date", "demand", "floortables"]].sort_values("date").reset_index(drop=True)
    d["demand_per_table"] = d["demand"] / d["floortables"].clip(lower=1)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["dow"] = d["date"].dt.weekday
    return d


def holiday_date_set() -> set[pd.Timestamp]:
    with F.extended_holidays():
        return set(S.holiday_date_set())


def normal_day_mask(df: pd.DataFrame) -> pd.Series:
    """demand>0, floortables>0, not a holiday-window day."""
    hol = holiday_date_set()
    return (
        (df["demand"] > 0)
        & (df["floortables"] > 0)
        & (~df["date"].isin(hol))
    )


def alignment_cv(values) -> float:
    a = np.asarray(values, dtype="float64")
    a = a[~np.isnan(a)]
    if a.size < 2:
        return float("nan")
    m = a.mean()
    if m == 0:
        return float("nan")
    return float(a.std() / m)


_GROUP_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("yearly_lookback", re.compile(r"^(lag_365|lag_728|lag_anchor_365|yoy_ratio|yoy_anchor_diff)$")),
    ("holiday_yoy",     re.compile(r"^(holiday_yoy_lag|holiday_yoy_lift|same_holiday_lastyear_lag)$")),
    ("recency_aggregate", re.compile(r"^(rolling_|ewma_|same_dow_mean_|trend_|same_dow_vs_overall$)")),
    ("capacity",        re.compile(r"^(floortables$|lag_anchor_\d+_per_table$)")),
    ("short_lag",       re.compile(r"^(lag_\d+$|lag_anchor_\d+$|lag_anchor_\d+_zscore$)")),
    ("mainland_block",  re.compile(r"^mainland_")),
    ("regime",          re.compile(r"^regime$")),
    ("holiday_flag",    re.compile(
        r"^(is_(CNY|MidAutumn|DragonBoat|ChingMing|Easter|GoldenWeek|Labour|NewYear|Christmas)$"
        r"|(CNY|GoldenWeek|Labour|DragonBoat)_d[mp]\d+$"
        r"|(CNY|GoldenWeek|Labour|MidAutumn)_dow\d$"
        r"|holiday_dow\d$"
        r"|days_(to_next|from_last)_holiday$"
        r"|(CNY|GoldenWeek|Labour|MidAutumn)_x_recent$)")),
    ("calendar",        re.compile(
        r"^(dow|day$|day_of_year|week_of_month|week_of_year|month|quarter|year$"
        r"|is_weekend|is_friday|is_saturday|is_sunday|is_month_start|is_month_end"
        r"|dow_sin|dow_cos|month_sin|month_cos|doy_sin|doy_cos|woy_sin|woy_cos)")),
]


def feature_group(name: str) -> str:
    for group, pat in _GROUP_PATTERNS:
        if pat.match(name):
            return group
    return "other"


# ── Part A1 — descriptive statistics ─────────────────────────────────────
def describe_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].astype("float64")
    full = pd.date_range(df["date"].min(), df["date"].max())
    return pd.Series({
        "count": int(s.notna().sum()),
        "span_days": len(full),
        "gaps": int(len(full) - df["date"].nunique()),
        "min": s.min(), "p10": s.quantile(0.10), "median": s.median(),
        "mean": s.mean(), "p90": s.quantile(0.90), "max": s.max(),
        "std": s.std(), "cv": s.std() / s.mean() if s.mean() else np.nan,
        "skew": s.skew(), "kurtosis": s.kurt(),
    })


def describe_by(df: pd.DataFrame, by: str, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by)
    out = pd.DataFrame(index=sorted(df[by].unique()))
    for c in cols:
        out[f"{c}_mean"] = g[c].mean()
        out[f"{c}_std"] = g[c].std()
        out[f"{c}_cv"] = g[c].std() / g[c].mean()
        out[f"{c}_min"] = g[c].min()
        out[f"{c}_max"] = g[c].max()
    out["n"] = g.size()
    out.index.name = by
    return out


def describe_by_regime(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["demand", "demand_per_table"]
    d = df.copy()
    d["regime"] = d["date"].map(C.regime_of)
    order = [r[0] for r in C.REGIMES]
    t = describe_by(d, "regime", cols)
    return t.reindex([r for r in order if r in t.index])


def closure_days(df: pd.DataFrame) -> pd.DataFrame:
    m = (df["floortables"] == 0) | (df["demand"] == 0)
    return df.loc[m, ["date", "demand", "floortables"]].reset_index(drop=True)


# ── Part A2 — monthly seasonality by year + alignment ────────────────────
def monthly_index_by_year(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_year"] = d["date"].dt.year
    d["_month"] = d["date"].dt.month
    d = d[d["_year"].isin(CLEAN_YEARS)]
    rows = {}
    for yr, sub in d.groupby("_year"):
        ann = sub["demand_per_table"].mean()
        if ann == 0 or np.isnan(ann):
            continue
        m = sub.groupby("_month")["demand_per_table"].mean() / ann
        rows[int(yr)] = m.reindex(range(1, 13))
    return pd.DataFrame(rows).T.sort_index()


def monthly_alignment(index_table: pd.DataFrame) -> dict:
    per_month_std = index_table.std(axis=0)
    shape_corr = index_table.T.corr()               # year x year, over the 12-vectors
    n = len(shape_corr)
    offdiag = shape_corr.values[~np.eye(n, dtype=bool)] if n > 1 else np.array([np.nan])
    return {
        "per_month_std": per_month_std,
        "shape_corr": shape_corr,
        "mean_offdiag_corr": float(np.nanmean(offdiag)),
        "unstable_months": [int(m) for m, v in per_month_std.items() if v > 0.10],
    }


# ── Part A3 — holiday impact by year + alignment ─────────────────────────
WIDE_WINDOW_HOLIDAYS = ["CNY", "GoldenWeek", "Labour", "Christmas", "Easter", "DragonBoat"]


def holiday_windows_by_year() -> dict[tuple[str, int], list[pd.Timestamp]]:
    out: dict[tuple[str, int], list[pd.Timestamp]] = {}
    with F.extended_holidays():
        for name, anchors in S.HOLIDAY_ANCHORS.items():
            ws, we = S.HOLIDAY_WINDOWS[name]
            for a in anchors:
                dates = [a + pd.Timedelta(days=k) for k in range(ws, we + 1)]
                out[(name, int(a.year))] = dates
    return out


def pre_holiday_baseline(df: pd.DataFrame, window_dates, *, min_days: int = 5,
                         widen: tuple[int, ...] = (28, 56, 84)) -> dict:
    start = min(pd.Timestamp(d) for d in window_dates)
    normal = df.loc[normal_day_mask(df)]
    for lb in widen:
        lo = start - pd.Timedelta(days=lb)
        sub = normal[(normal["date"] < start) & (normal["date"] >= lo)]
        if len(sub) >= min_days:
            return {"demand": float(sub["demand"].mean()),
                    "demand_per_table": float(sub["demand_per_table"].mean()),
                    "n_days": int(len(sub)), "lookback_used": lb}
    sub = normal[(normal["date"] < start) & (normal["date"] >= start - pd.Timedelta(days=widen[-1]))]
    return {"demand": float("nan"), "demand_per_table": float("nan"),
            "n_days": int(len(sub)), "lookback_used": None}


def holiday_multiplier_by_year(df: pd.DataFrame) -> pd.DataFrame:
    wins = holiday_windows_by_year()
    dm = dict(zip(df["date"], df["demand"].astype(float)))
    dpt = dict(zip(df["date"], df["demand_per_table"].astype(float)))
    rows = []
    for (name, year), dates in sorted(wins.items()):
        if year not in CLEAN_YEARS:          # COVID years -> the separate A4 section
            continue
        in_range = [d for d in dates if d in dm]
        if not in_range:
            continue
        b = pre_holiday_baseline(df, dates)
        w_raw = np.nanmean([dm[d] for d in in_range])
        w_pt = np.nanmean([dpt[d] for d in in_range])
        rows.append({
            "holiday": name, "year": year, "n_window_days": len(in_range),
            "baseline_n_days": b["n_days"], "baseline_lookback": b["lookback_used"],
            "mult_raw": w_raw / b["demand"] if b["demand"] else np.nan,
            "mult_per_table": w_pt / b["demand_per_table"] if b["demand_per_table"] else np.nan,
        })
    return pd.DataFrame(rows)


def holiday_offset_shape_by_year(df: pd.DataFrame, holiday: str) -> pd.DataFrame:
    with F.extended_holidays():
        ws, we = S.HOLIDAY_WINDOWS[holiday]
        anchors = list(S.HOLIDAY_ANCHORS[holiday])
    dpt = dict(zip(df["date"], df["demand_per_table"].astype(float)))
    offsets = list(range(ws, we + 1))
    rows = {}
    for a in anchors:
        yr = int(a.year)
        if yr not in CLEAN_YEARS:
            continue
        dates = [a + pd.Timedelta(days=k) for k in offsets]
        b = pre_holiday_baseline(df, dates)
        base = b["demand_per_table"]
        if not base or np.isnan(base):
            continue
        rows[yr] = pd.Series(
            {k: (dpt.get(a + pd.Timedelta(days=k), np.nan) / base) for k in offsets})
    return pd.DataFrame(rows).T.sort_index()


def holiday_alignment(mult_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in mult_table.groupby("holiday"):
        clean = sub[sub["year"].isin(CLEAN_YEARS)]
        cv_pt = alignment_cv(clean["mult_per_table"])
        n = int(clean["mult_per_table"].notna().sum())
        if n < 3:
            verdict = "insufficient"
        elif not np.isnan(cv_pt) and cv_pt < 0.10:
            verdict = "consistent"
        else:
            verdict = "variable"
        rows.append({
            "holiday": name, "n_clean_years": n,
            "cv_per_table": cv_pt, "cv_raw": alignment_cv(clean["mult_raw"]),
            "min_year_mult": clean["mult_per_table"].min(),
            "max_year_mult": clean["mult_per_table"].max(),
            "verdict": verdict,
        })
    return pd.DataFrame(rows).set_index("holiday").sort_values("cv_per_table")


def cny_trough_by_year(df: pd.DataFrame) -> pd.DataFrame:
    shape = holiday_offset_shape_by_year(df, "CNY")
    pre = [c for c in shape.columns if -7 <= c <= -1]
    rows = {}
    for yr in shape.index:
        s = shape.loc[yr, pre].astype(float)
        if s.notna().any():
            rows[int(yr)] = {"min_mult_offset": int(s.idxmin()), "min_mult_value": float(s.min())}
    return pd.DataFrame(rows).T


# ── Part A4 — COVID section (kept out of the year-over-year views) ────────
def covid_timeline(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2024-12-31")].copy()
    d["period"] = d["date"].dt.to_period("M")
    g = d.groupby("period")
    out = pd.DataFrame({
        "mean_demand": g["demand"].mean(),
        "mean_per_table": g["demand_per_table"].mean(),
        "n_days": g.size(),
        "n_closure_days": g.apply(
            lambda x: int(((x["floortables"] == 0) | (x["demand"] == 0)).sum())),
    })
    out.index = out.index.astype(str)
    return out


def recovery_2023_check(df: pd.DataFrame) -> dict:
    d = df[df["date"].dt.year == 2023]
    jan = d.loc[d["date"].dt.month == 1, "demand_per_table"].mean()
    dec = d.loc[d["date"].dt.month == 12, "demand_per_table"].mean()
    h1_2024 = df.loc[(df["date"].dt.year == 2024) & (df["date"].dt.month <= 6),
                     "demand_per_table"].mean()
    return {
        "jan_per_table": float(jan), "dec_per_table": float(dec),
        "climbed": bool(dec > jan),
        "norm_2024_h1": float(h1_2024),
        "reached_norm": bool(dec >= 0.95 * h1_2024),
    }
