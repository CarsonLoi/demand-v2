from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

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
    d["_is_closure"] = ((d["floortables"] == 0) | (d["demand"] == 0)).astype(int)
    g = d.groupby("period")
    out = pd.DataFrame({
        "mean_demand": g["demand"].mean(),
        "mean_per_table": g["demand_per_table"].mean(),
        "n_days": g.size(),
        "n_closure_days": g["_is_closure"].sum(),
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


# ── Part B1 — feature-matrix sample + data dictionary ────────────────────
_MATRIX_CACHE: pd.DataFrame | None = None


def build_matrix() -> pd.DataFrame:
    global _MATRIX_CACHE
    if _MATRIX_CACHE is None:
        demand = F.load_long_demand()
        base = F.build_base(demand, history_start=None, quiet=True)
        _MATRIX_CACHE = F.apply_exclusions(base, demand, [])
    return _MATRIX_CACHE


def feature_names(mat: pd.DataFrame) -> list[str]:
    return F.feature_columns(mat)


def sample_tall(mat: pd.DataFrame, n: int = 500, seed: int = 0) -> pd.DataFrame:
    d = mat.sort_values(["target_date", "horizon"]).reset_index(drop=True)
    idx = np.linspace(0, len(d) - 1, num=min(n, len(d))).round().astype(int)
    return d.iloc[np.unique(idx)].reset_index(drop=True)


def pick_sample_dates() -> dict[str, pd.Timestamp]:
    return {
        "normal_midweek":  pd.Timestamp("2025-11-19"),
        "normal_saturday": pd.Timestamp("2025-11-22"),
        "cny_dm3":         pd.Timestamp("2026-02-14"),
        "cny_dp1":         pd.Timestamp("2026-02-18"),
        "goldenweek":      pd.Timestamp("2025-10-02"),
        "jan2026_a":       pd.Timestamp("2026-01-26"),
        "jan2026_b":       pd.Timestamp("2026-01-29"),
    }


def sample_wide(mat: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    feats = feature_names(mat)
    cols = {}
    for label, d in pick_sample_dates().items():
        row = mat[(mat["target_date"] == d) & (mat["horizon"] == horizon)]
        cols[label] = (row[feats].iloc[0] if len(row)
                       else pd.Series(np.nan, index=feats))
    out = pd.DataFrame(cols)
    out["__group__"] = [feature_group(f) for f in out.index]
    return out


def feature_dictionary(mat: pd.DataFrame) -> pd.DataFrame:
    feats = feature_names(mat)
    trainable = mat[mat["y"].notna()]
    rows = []
    for f in feats:
        col = mat[f]
        rows.append({
            "feature": f, "group": feature_group(f), "dtype": str(col.dtype),
            "pct_nonnull_all": 100 * col.notna().mean(),
            "pct_nonnull_trainable": 100 * trainable[f].notna().mean(),
            "min": col.min(skipna=True), "mean": col.mean(skipna=True),
            "max": col.max(skipna=True),
        })
    return pd.DataFrame(rows).set_index("feature")


# ── Part B2 — feature correlation ───────────────────────────────────────
def corr_frames(mat: pd.DataFrame) -> dict:
    feats = feature_names(mat)
    tr = mat[mat["y"].notna()]
    num = tr[feats].select_dtypes(include=[np.number])
    num = num.loc[:, num.std(numeric_only=True) > 0]
    return {"pearson": num.corr(method="pearson"),
            "spearman": num.corr(method="spearman")}


def high_corr_pairs(pearson: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    rows = []
    cols = list(pearson.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = pearson.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"feat_a": a, "feat_b": b,
                             "group_a": feature_group(a), "group_b": feature_group(b),
                             "r": float(r)})
    if not rows:
        return pd.DataFrame(columns=["feat_a", "feat_b", "group_a", "group_b", "r"])
    return (pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True))


def target_corr(mat: pd.DataFrame) -> pd.DataFrame:
    feats = feature_names(mat)
    tr = mat[mat["y"].notna()]
    y = tr["y"].astype(float)
    rows = []
    for f in feats:
        col = tr[f]
        if col.notna().sum() < 30 or col.std(skipna=True) == 0:
            rows.append({"feature": f, "pearson": np.nan, "spearman": np.nan})
            continue
        rows.append({"feature": f,
                     "pearson": float(col.corr(y, method="pearson")),
                     "spearman": float(col.corr(y, method="spearman"))})
    out = pd.DataFrame(rows).set_index("feature")
    return out.reindex(out["spearman"].abs().sort_values(ascending=False).index)


def group_corr(pearson: pd.DataFrame) -> pd.DataFrame:
    groups = sorted({feature_group(c) for c in pearson.columns})
    members = {g: [c for c in pearson.columns if feature_group(c) == g] for g in groups}
    out = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for g1 in groups:
        for g2 in groups:
            block = pearson.loc[members[g1], members[g2]].abs()
            if g1 == g2:
                vals = (block.values[~np.eye(len(members[g1]), dtype=bool)]
                        if len(members[g1]) > 1 else np.array([]))
            else:
                vals = block.values.ravel()
            vals = vals[~np.isnan(vals)] if vals.size else vals
            out.loc[g1, g2] = float(vals.mean()) if vals.size else np.nan
    return out


# ── Part B3 — SHAP via LightGBM native pred_contrib ─────────────────────
def train_horizon_model(mat: pd.DataFrame, feats: list[str], horizon: int,
                        as_of: pd.Timestamp) -> "lgb.LGBMRegressor":
    train = mat[(mat["horizon"] == horizon) & (mat["target_date"] <= as_of)].dropna(subset=["y"])
    w = S.make_sample_weights(train["target_date"],
                              is_holiday=S.holiday_mask_from_matrix(train),
                              half_life_days=C.DEFAULT_HALF_LIFE)
    m = lgb.LGBMRegressor(**C.LGBM_PARAMS)
    m.fit(train[feats], train["y"], sample_weight=w)
    m._n_train_rows = int(len(train))
    return m


def _contrib_frame(model, X: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, float]:
    c = model.predict(X[feats], pred_contrib=True)
    return c[:, :-1], float(c[0, -1])            # (contribs, bias) -- bias identical per row


def shap_global(mat: pd.DataFrame, feats: list[str], horizons: list[int],
                sample: int = 4000, seed: int = 0) -> pd.DataFrame:
    as_of = mat["target_date"].max()
    rng = np.random.default_rng(seed)
    per_h = {}
    pooled = np.zeros(len(feats))
    total_rows = 0
    for h in horizons:
        m = train_horizon_model(mat, feats, h, as_of)
        pool = mat[(mat["horizon"] == h) & (mat["y"].notna())]
        take = pool.iloc[rng.choice(len(pool), size=min(sample, len(pool)), replace=False)]
        contribs, _ = _contrib_frame(m, take, feats)
        absmean = np.abs(contribs).mean(axis=0)
        per_h[f"mean_abs_shap_{h}"] = pd.Series(absmean, index=feats)
        pooled += absmean * len(take)
        total_rows += len(take)
    out = pd.DataFrame(per_h)
    out["mean_abs_shap"] = pooled / total_rows
    return out.sort_values("mean_abs_shap", ascending=False)


def shap_local(mat: pd.DataFrame, feats: list[str], horizon: int = 7) -> dict:
    as_of = mat["target_date"].max()
    m = train_horizon_model(mat, feats, horizon, as_of)
    out = {}
    for label, d in pick_sample_dates().items():
        row = mat[(mat["target_date"] == d) & (mat["horizon"] == horizon)]
        if row.empty:
            continue
        contribs, bias = _contrib_frame(m, row, feats)
        s = pd.DataFrame({"feature_value": row[feats].iloc[0].values,
                          "shap": contribs[0]}, index=feats)
        s = s.reindex(s["shap"].abs().sort_values(ascending=False).index)
        out[label] = s
        out[f"{label}__meta__"] = pd.DataFrame(
            {"base_value": [bias], "prediction": [float(m.predict(row[feats])[0])]})
    return out
