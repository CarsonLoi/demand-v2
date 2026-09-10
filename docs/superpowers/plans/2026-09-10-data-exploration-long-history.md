# Data Exploration for the Long-History Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone exploration deliverable under `research/explore_longhistory/` that profiles the long-history pipeline's raw series and engineered feature matrix, with explicit year-over-year alignment verdicts for seasonality and holiday impact, a readable feature-matrix sample, feature correlations, and SHAP feature importance.

**Architecture:** A pure-function library (`eda_lib.py`) holds every computation; a chart module (`plots.py`) holds every matplotlib figure; an orchestrator (`run_explore.py`) calls both and writes all CSV/PNG artifacts plus an `EXPLORATION.md` skeleton; a notebook (`explore.ipynb`) imports the same library. Assert-based checks live in `checks.py`. Nothing in `long_history/` or `v2/` is modified — the engine is imported read-only.

**Tech Stack:** Python 3.11, pandas 2.2, numpy 1.26, matplotlib 3.8, lightgbm 4.3 (native `pred_contrib` for SHAP — no `shap` library). `uv run python ...` to execute.

## Global Constraints

- Standalone: `research/explore_longhistory/` is entirely new. Imports `long_history/config.py` and `long_history/engine/*` **read-only**. Modifies nothing in `long_history/` or `v2/`. (spec: Non-goals)
- No new pip dependency. SHAP via `lightgbm` `model.predict(X, pred_contrib=True)`. (spec: Non-goals)
- No pytest — it is not installed. Checks are `(bool, str)`-returning functions collected by a `run_all()` main, matching `long_history/hourly/checks.py`. (repo convention)
- Not wired into `run_pipeline.py`. (spec: Non-goals)
- Analysis only — no model change. Redundancy or contamination found is *reported* in `EXPLORATION.md`, never acted on here. (spec: Non-goals)
- Clean years = calendar years 2015-2019 and 2023-2026. Calendar years 2020, 2021, 2022 are excluded from every year-over-year output. (spec: Definitions)
- Normal day = `demand > 0` and `floortables > 0` and date not in `S.holiday_date_set()` (extended anchors). (spec: Definitions)
- Pre-holiday baseline = mean over normal days in the 28 calendar days immediately **before** the holiday window start; widen to 56 then 84 if under 5 normal days; return NaN (not a partial mean) if still under 5. Lead-in only — never a day inside or after the window. (spec: Definitions)
- `demand_per_table = demand / floortables.clip(lower=1)`. (spec: Data)
- SHAP models: trained with `config.LGBM_PARAMS`, `S.make_sample_weights(..., half_life_days=config.DEFAULT_HALF_LIFE)`, `S.holiday_mask_from_matrix`, `dropna(subset=["y"])` **only**, `as_of` = data max date (explains the full-history model, disclosed in the report). (spec: Part B3)
- `ruff` line-length 100, `from __future__ import annotations` at the top of every module (repo style).
- Never `git commit` unless a step says to; run `git status --short` before every `git add`. (session standing rule)

---

## File Structure

| File | Responsibility |
|---|---|
| `research/explore_longhistory/__init__.py` | empty package marker |
| `research/explore_longhistory/eda_lib.py` | every computation as a pure function; no file writes, no plotting |
| `research/explore_longhistory/plots.py` | every matplotlib figure; each takes a DataFrame + an output `Path`, writes a PNG, returns nothing |
| `research/explore_longhistory/checks.py` | assert-based checks, `(bool, str)` per check, `run_all()` main |
| `research/explore_longhistory/run_explore.py` | orchestrator: build matrix, call `eda_lib` + `plots`, write CSVs/PNGs + `EXPLORATION.md` skeleton, `--self-check` |
| `research/explore_longhistory/EXPLORATION.md` | the written report — tables from CSVs + hand prose + chart refs |
| `research/explore_longhistory/explore.ipynb` | notebook: every analysis as a cell, imports `eda_lib` + `plots` |
| `research/explore_longhistory/output/` | generated `.csv` and `.png` (created at run time) |

Shared import header used by every `.py` module in the folder:

```python
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root
sys.path.insert(0, str(ROOT / "long_history"))

import config as C                           # noqa: E402
from engine import core as S                 # noqa: E402
from engine import features as F             # noqa: E402
```

(The notebook uses the same three `sys.path.insert` / imports in its first cell, with `HERE = Path.cwd()` fallback.)

---

## Task 1: Scaffold — package, shared helpers, chart style, checks runner

**Files:**
- Create: `research/explore_longhistory/__init__.py` (empty)
- Create: `research/explore_longhistory/eda_lib.py`
- Create: `research/explore_longhistory/plots.py`
- Create: `research/explore_longhistory/checks.py`

**Interfaces:**
- Produces:
  - `eda_lib.CLEAN_YEARS: list[int]` → `[2015,2016,2017,2018,2019,2023,2024,2025,2026]`
  - `eda_lib.COVID_YEARS: set[int]` → `{2020, 2021, 2022}`
  - `eda_lib.load_frames() -> pd.DataFrame` — columns `date, demand, floortables, demand_per_table, year, month, dow`, sorted, one row per day
  - `eda_lib.normal_day_mask(df: pd.DataFrame) -> pd.Series` — bool Series aligned to `df.index`
  - `eda_lib.holiday_date_set() -> set[pd.Timestamp]` — `S.holiday_date_set()` under `F.extended_holidays()`
  - `eda_lib.alignment_cv(values) -> float` — `nanstd / nanmean` of a 1-D array-like; `0.0` when all equal; `nan` when fewer than 2 non-nan
  - `eda_lib.feature_group(name: str) -> str` — one of `recency_aggregate, short_lag, yearly_lookback, holiday_yoy, holiday_flag, calendar, mainland_block, regime, capacity, other`
  - `plots.style_ax(ax) -> None` and palette constants `NAVY, TEAL, GREY, GOLD, RED, GREEN`
  - `checks.run_all() -> int` — prints `[PASS]/[FAIL]` per check, returns 1 if any failed

- [ ] **Step 1: Create the package marker and `eda_lib.py` with helpers**

Create `research/explore_longhistory/__init__.py` empty.

Create `research/explore_longhistory/eda_lib.py`:

```python
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
```

- [ ] **Step 2: Create `plots.py` with shared style**

Create `research/explore_longhistory/plots.py`:

```python
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

NAVY, TEAL, GREY, GOLD, RED, GREEN = (
    "#0F2942", "#2E8BA8", "#5F7183", "#E8B547", "#C74B4B", "#2FA87C")

# a stable colour per clean year for the year-over-year line charts
YEAR_COLOURS = {
    2015: "#6B8CAE", 2016: "#4E7A9E", 2017: "#2E8BA8", 2018: "#1F6F86",
    2019: "#0F2942", 2023: "#E8B547", 2024: "#D98C3A", 2025: "#C74B4B", 2026: "#2FA87C",
}


def style_ax(ax) -> None:
    ax.grid(alpha=0.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")
```

- [ ] **Step 3: Create `checks.py` with the runner and the first helper checks**

Create `research/explore_longhistory/checks.py`:

```python
"""checks.py -- assert-based checks for the exploration library.

No pytest in this repo; matches long_history/hourly/checks.py style. Each
check returns (ok: bool, message: str). Run:

    uv run python research/explore_longhistory/checks.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eda_lib as L          # noqa: E402


def check_alignment_cv_zero_when_equal() -> tuple[bool, str]:
    v = L.alignment_cv([3.0, 3.0, 3.0])
    return v == 0.0, f"alignment_cv([3,3,3]) = {v} (want 0.0)"


def check_alignment_cv_known_spread() -> tuple[bool, str]:
    # values 1,2,3 -> mean 2, population std sqrt(2/3) ~ 0.8165 -> cv ~ 0.4082
    v = L.alignment_cv([1.0, 2.0, 3.0])
    return abs(v - 0.40825) < 1e-4, f"alignment_cv([1,2,3]) = {v:.5f} (want 0.40825)"


def check_alignment_cv_nan_when_thin() -> tuple[bool, str]:
    v = L.alignment_cv([5.0])
    return np.isnan(v), f"alignment_cv([5]) = {v} (want nan)"


def check_feature_group_known_names() -> tuple[bool, str]:
    cases = {
        "lag_365": "yearly_lookback", "yoy_ratio": "yearly_lookback",
        "lag_2": "short_lag", "lag_anchor_7": "short_lag",
        "rolling_max_28": "recency_aggregate", "ewma_14": "recency_aggregate",
        "same_dow_mean_8w": "recency_aggregate",
        "holiday_yoy_lag": "holiday_yoy",
        "is_CNY": "holiday_flag", "CNY_dm7": "holiday_flag", "holiday_dow3": "holiday_flag",
        "days_to_next_holiday": "holiday_flag", "CNY_x_recent": "holiday_flag",
        "dow_sin": "calendar", "month": "calendar", "week_of_year": "calendar",
        "mainland_block_pos_norm": "mainland_block",
        "regime": "regime",
        "floortables": "capacity", "lag_anchor_7_per_table": "capacity",
    }
    bad = {k: L.feature_group(k) for k, want in cases.items() if L.feature_group(k) != want}
    return not bad, ("misgrouped: " + str(bad) if bad else f"{len(cases)} feature names grouped correctly")


def check_load_frames_shape() -> tuple[bool, str]:
    d = L.load_frames()
    ok = (
        list(d.columns[:5]) == ["date", "demand", "floortables", "demand_per_table", "year"]
        and d["date"].is_monotonic_increasing
        and d["date"].duplicated().sum() == 0
    )
    return ok, f"load_frames: {len(d):,} rows, {d.date.min().date()}..{d.date.max().date()}"


def check_normal_day_mask_excludes_holidays() -> tuple[bool, str]:
    d = L.load_frames()
    hol = L.holiday_date_set()
    m = L.normal_day_mask(d)
    leaked = d.loc[m & d["date"].isin(hol)]
    return leaked.empty, (f"{len(leaked)} holiday days marked normal" if len(leaked)
                          else "no holiday-window day is marked normal")


CHECKS = [
    check_alignment_cv_zero_when_equal,
    check_alignment_cv_known_spread,
    check_alignment_cv_nan_when_thin,
    check_feature_group_known_names,
    check_load_frames_shape,
    check_normal_day_mask_excludes_holidays,
]


def run_all() -> int:
    failed = 0
    for fn in CHECKS:
        try:
            ok, msg = fn()
        except Exception as e:                       # noqa: BLE001
            ok, msg = False, f"raised {type(e).__name__}: {e}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {fn.__name__}: {msg}")
        failed += 0 if ok else 1
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all())
```

- [ ] **Step 4: Run the checks — expect all PASS**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `6/6 checks passed`. If `check_feature_group_known_names` fails, fix the regexes in `eda_lib._GROUP_PATTERNS` until the listed names map correctly — those 21 names are the contract.

- [ ] **Step 5: Commit**

```bash
git status --short
git add research/explore_longhistory/__init__.py research/explore_longhistory/eda_lib.py research/explore_longhistory/plots.py research/explore_longhistory/checks.py
git commit -m "Scaffold long-history exploration: helpers, chart style, checks runner"
```

---

## Task 2: Part A1 — descriptive statistics

**Files:**
- Modify: `research/explore_longhistory/eda_lib.py` (append functions)
- Modify: `research/explore_longhistory/plots.py` (append `plot_per_table_by_year`)
- Modify: `research/explore_longhistory/checks.py` (append checks)

**Interfaces:**
- Consumes: `load_frames`, `normal_day_mask` (Task 1)
- Produces:
  - `eda_lib.describe_series(df, col: str) -> pd.Series` — index `count,span_days,gaps,min,p10,median,mean,p90,max,std,cv,skew,kurtosis`
  - `eda_lib.describe_by(df, by: str, cols: list[str]) -> pd.DataFrame` — MultiIndex-free; one row per group value, columns `<col>_mean, <col>_std, <col>_cv, <col>_min, <col>_max, n`
  - `eda_lib.describe_by_regime(df) -> pd.DataFrame` — like `describe_by` but grouped by `C.regime_of(date)`
  - `eda_lib.closure_days(df) -> pd.DataFrame` — rows where `floortables == 0` or `demand == 0`; columns `date, demand, floortables`
  - `plots.plot_per_table_by_year(df, out: Path) -> None`

- [ ] **Step 1: Add the check functions to `checks.py`**

Append to `checks.py` (and add all four names to `CHECKS`):

```python
def check_describe_series_gaps_zero() -> tuple[bool, str]:
    d = L.load_frames()
    s = L.describe_series(d, "demand")
    return s["gaps"] == 0, f"gaps in daily series = {s['gaps']} (want 0)"


def check_closure_days_match_known() -> tuple[bool, str]:
    d = L.load_frames()
    cl = set(pd.to_datetime(L.closure_days(d)["date"]).dt.date)
    known = {pd.Timestamp("2018-09-16").date()}
    known |= {x.date() for x in pd.date_range("2020-02-05", "2020-02-19")}
    known |= {x.date() for x in pd.date_range("2022-07-11", "2022-07-22")}
    missing = known - cl
    return not missing, (f"known closures not detected: {sorted(missing)[:3]}..." if missing
                         else f"{len(cl)} closure/zero days, all known closures present")


def check_describe_by_year_all_years_present() -> tuple[bool, str]:
    d = L.load_frames()
    t = L.describe_by(d, "year", ["demand_per_table"])
    yrs = set(t.index.astype(int))
    return {2015, 2020, 2026}.issubset(yrs), f"years in table: {sorted(yrs)}"


def check_describe_by_dow_seven_rows() -> tuple[bool, str]:
    d = L.load_frames()
    t = L.describe_by(d, "dow", ["demand"])
    return len(t) == 7, f"dow table has {len(t)} rows (want 7)"
```

- [ ] **Step 2: Run checks — expect the four new ones to FAIL**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: the four new checks FAIL with `raised AttributeError: module 'eda_lib' has no attribute 'describe_series'` (etc.). Task 1's six still PASS.

- [ ] **Step 3: Implement the functions in `eda_lib.py`**

Append to `eda_lib.py`:

```python
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
```

- [ ] **Step 4: Add the chart to `plots.py`**

Append:

```python
def plot_per_table_by_year(df: pd.DataFrame, out: Path) -> None:
    """Monthly mean demand-per-table, one line per year (all years shown)."""
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    g = d.groupby([d["date"].dt.year, d["date"].dt.month])["demand_per_table"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for yr in sorted({i[0] for i in g.index}):
        sub = g.loc[yr]
        ax.plot(sub.index, sub.values, "o-", ms=3, lw=1.5,
                color=YEAR_COLOURS.get(yr, GREY), label=str(yr),
                alpha=0.55 if yr in (2020, 2021, 2022) else 1.0)
    ax.set_xlabel("month", color=GREY, fontsize=9)
    ax.set_ylabel("mean demand / table", color=GREY, fontsize=9)
    ax.set_title("Demand per table by month and year (faded = COVID years)",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=4)
    style_ax(ax)
    _save(fig, out)
```

- [ ] **Step 5: Run checks — expect all PASS**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `10/10 checks passed`.

- [ ] **Step 6: Commit**

```bash
git status --short
git add research/explore_longhistory/eda_lib.py research/explore_longhistory/plots.py research/explore_longhistory/checks.py
git commit -m "A1: descriptive statistics (overall / year / dow / regime) + closure list"
```

---

## Task 3: Part A2 — monthly seasonality by year + alignment

**Files:**
- Modify: `research/explore_longhistory/eda_lib.py`
- Modify: `research/explore_longhistory/plots.py`
- Modify: `research/explore_longhistory/checks.py`

**Interfaces:**
- Consumes: `load_frames`, `CLEAN_YEARS`, `COVID_YEARS`, `alignment_cv` (Task 1)
- Produces:
  - `eda_lib.monthly_index_by_year(df) -> pd.DataFrame` — index = clean years, columns = months 1..12, value = `(month mean demand_per_table) / (that year's annual mean demand_per_table)`. COVID years absent.
  - `eda_lib.monthly_alignment(index_table) -> dict` — `{"per_month_std": pd.Series(index=1..12), "shape_corr": pd.DataFrame(year x year), "mean_offdiag_corr": float, "unstable_months": list[int]}` (unstable = per-month std > 0.10)
  - `plots.plot_monthly_seasonality(index_table, out) -> None`
  - `plots.plot_monthly_shape_corr(shape_corr, out) -> None`

- [ ] **Step 1: Add checks**

Append to `checks.py`, add names to `CHECKS`:

```python
def check_monthly_index_flat_series_is_one() -> tuple[bool, str]:
    dates = pd.date_range("2015-01-01", "2016-12-31")
    df = pd.DataFrame({"date": dates, "demand": 100.0, "floortables": 10.0})
    df["demand_per_table"] = df["demand"] / df["floortables"].clip(lower=1)
    df["year"] = df["date"].dt.year
    t = L.monthly_index_by_year(df)
    return np.allclose(t.values, 1.0), f"flat series -> index range [{t.values.min():.3f}, {t.values.max():.3f}] (want all 1.0)"


def check_monthly_index_doubled_month() -> tuple[bool, str]:
    dates = pd.date_range("2015-01-01", "2015-12-31")
    df = pd.DataFrame({"date": dates, "demand": 100.0, "floortables": 10.0})
    df.loc[df["date"].dt.month == 6, "demand"] = 200.0
    df["demand_per_table"] = df["demand"] / df["floortables"].clip(lower=1)
    df["year"] = df["date"].dt.year
    t = L.monthly_index_by_year(df)
    ann = df["demand_per_table"].mean() / 10.0     # annual mean / per-table base
    want_june = 20.0 / (df["demand_per_table"].mean())
    return abs(t.loc[2015, 6] - want_june) < 1e-6, f"June index = {t.loc[2015,6]:.4f} (want {want_june:.4f})"


def check_monthly_no_covid_years() -> tuple[bool, str]:
    t = L.monthly_index_by_year(L.load_frames())
    bad = L.COVID_YEARS & set(t.index)
    return not bad, f"covid years in monthly table: {bad or 'none'}"
```

- [ ] **Step 2: Run checks — expect the three new ones to FAIL**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: 3 FAIL (`AttributeError` on `monthly_index_by_year`), 10 PASS.

- [ ] **Step 3: Implement in `eda_lib.py`**

```python
def monthly_index_by_year(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["year"].isin(CLEAN_YEARS)].copy()
    rows = {}
    for yr, sub in d.groupby("year"):
        ann = sub["demand_per_table"].mean()
        if ann == 0 or np.isnan(ann):
            continue
        m = sub.groupby("month")["demand_per_table"].mean() / ann
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
```

- [ ] **Step 4: Add charts to `plots.py`**

```python
def plot_monthly_seasonality(index_table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for yr in index_table.index:
        ax.plot(range(1, 13), index_table.loc[yr].values, "o-", ms=3.5, lw=1.6,
                color=YEAR_COLOURS.get(int(yr), GREY), label=str(int(yr)))
    ax.axhline(1.0, color=GREY, lw=1, ls="--")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month", color=GREY, fontsize=9)
    ax.set_ylabel("monthly index (1.0 = year's average)", color=GREY, fontsize=9)
    ax.set_title("Monthly seasonality by year — clean years only",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=5)
    style_ax(ax)
    _save(fig, out)


def plot_monthly_shape_corr(shape_corr: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(shape_corr.values, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(shape_corr)))
    ax.set_yticks(range(len(shape_corr)))
    ax.set_xticklabels([str(int(y)) for y in shape_corr.columns], rotation=45, fontsize=8)
    ax.set_yticklabels([str(int(y)) for y in shape_corr.index], fontsize=8)
    for i in range(len(shape_corr)):
        for j in range(len(shape_corr)):
            ax.text(j, i, f"{shape_corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if shape_corr.values[i, j] > 0.6 else "black")
    ax.set_title("Year-to-year correlation of the 12-month shape",
                 color=NAVY, fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out)
```

- [ ] **Step 5: Run checks — expect all PASS**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `13/13 checks passed`.

- [ ] **Step 6: Commit**

```bash
git status --short
git add research/explore_longhistory/eda_lib.py research/explore_longhistory/plots.py research/explore_longhistory/checks.py
git commit -m "A2: monthly seasonality by year + alignment (spread + shape correlation)"
```

---

## Task 4: Part A3 — holiday impact by year + alignment

**Files:**
- Modify: `research/explore_longhistory/eda_lib.py`
- Modify: `research/explore_longhistory/plots.py`
- Modify: `research/explore_longhistory/checks.py`

**Interfaces:**
- Consumes: `load_frames`, `normal_day_mask`, `holiday_date_set`, `CLEAN_YEARS`, `alignment_cv`
- Produces:
  - `eda_lib.holiday_windows_by_year() -> dict[tuple[str,int], list[pd.Timestamp]]` — `(holiday_name, year) -> [window dates]`, from `S.HOLIDAY_ANCHORS`/`S.HOLIDAY_WINDOWS` under `F.extended_holidays()`
  - `eda_lib.pre_holiday_baseline(df, window_dates, *, min_days=5, widen=(28,56,84)) -> dict` — `{"demand": float|nan, "demand_per_table": float|nan, "n_days": int, "lookback_used": int|None}`. Window start = `min(window_dates)`. Considers only rows with `date < window_start` AND `date >= window_start - lookback`, restricted to normal days.
  - `eda_lib.holiday_multiplier_by_year(df) -> pd.DataFrame` — one row per `(holiday, year)`; columns `holiday, year, n_window_days, baseline_n_days, baseline_lookback, mult_raw, mult_per_table`
  - `eda_lib.holiday_offset_shape_by_year(df, holiday: str) -> pd.DataFrame` — index = year, columns = day offsets from `S.HOLIDAY_WINDOWS[holiday]`, value = `(demand_per_table on that offset day) / (pre-holiday baseline demand_per_table)`
  - `eda_lib.holiday_alignment(mult_table) -> pd.DataFrame` — one row per holiday; columns `n_clean_years, cv_per_table, cv_raw, min_year_mult, max_year_mult, verdict` (verdict: `"consistent"` if `cv_per_table < 0.10` and `n_clean_years >= 3`, else `"variable"`, else `"insufficient"`)
  - `eda_lib.cny_trough_by_year(df) -> pd.DataFrame` — index = year, columns `min_mult_offset, min_mult_value` over CNY offsets −7..−1
  - `plots.plot_holiday_impact(mult_table, out)`, `plots.plot_holiday_offset_shape(shape_by_holiday: dict[str, pd.DataFrame], out)`, `plots.plot_cny_trough(trough_table, out)`

WIDE_WINDOW_HOLIDAYS constant: `["CNY", "GoldenWeek", "Labour", "Christmas", "Easter", "DragonBoat"]`

- [ ] **Step 1: Add checks**

Append, add to `CHECKS`:

```python
def _synth_holiday_frame():
    """A flat 100/day series with a known holiday window and a known dip
    just before it, so the baseline and multiplier are hand-checkable."""
    dates = pd.date_range("2015-01-01", "2015-12-31")
    df = pd.DataFrame({"date": dates, "demand": 100.0, "floortables": 10.0})
    # holiday window 2015-06-10 .. 2015-06-16, demand doubles inside it
    win = pd.date_range("2015-06-10", "2015-06-16")
    df.loc[df["date"].isin(win), "demand"] = 200.0
    # a POISON value on 2015-06-12 (inside window) and on 2015-07-01 (after) --
    # neither may enter the pre-holiday baseline
    df.loc[df["date"] == pd.Timestamp("2015-07-01"), "demand"] = 9999.0
    df["demand_per_table"] = df["demand"] / df["floortables"].clip(lower=1)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.weekday
    return df, list(win)


def check_pre_holiday_baseline_is_leadin_only() -> tuple[bool, str]:
    df, win = _synth_holiday_frame()
    # treat only `win` as holiday days for this synthetic check
    orig = L.holiday_date_set
    L.holiday_date_set = lambda: set(win)                     # type: ignore
    try:
        b = L.pre_holiday_baseline(df, win)
    finally:
        L.holiday_date_set = orig                             # type: ignore
    ok = abs(b["demand"] - 100.0) < 1e-9 and b["n_days"] >= 5
    return ok, f"baseline demand = {b['demand']} (want 100.0 -- no window/after day leaked), n={b['n_days']}"


def check_pre_holiday_baseline_widens_and_nans() -> tuple[bool, str]:
    dates = pd.date_range("2015-05-01", "2015-06-16")
    df = pd.DataFrame({"date": dates, "demand": 0.0, "floortables": 10.0})  # all closures
    win = list(pd.date_range("2015-06-10", "2015-06-16"))
    df["demand_per_table"] = 0.0
    orig = L.holiday_date_set
    L.holiday_date_set = lambda: set(win)                     # type: ignore
    try:
        b = L.pre_holiday_baseline(df, win)
    finally:
        L.holiday_date_set = orig                             # type: ignore
    return np.isnan(b["demand"]) and b["n_days"] < 5, f"all-closure lead-in -> demand={b['demand']}, n={b['n_days']} (want nan, <5)"


def check_holiday_multiplier_no_covid() -> tuple[bool, str]:
    t = L.holiday_multiplier_by_year(L.load_frames())
    bad = L.COVID_YEARS & set(t["year"].astype(int))
    return not bad, f"covid years in holiday multiplier table: {bad or 'none'}"


def check_holiday_alignment_verdict_columns() -> tuple[bool, str]:
    t = L.holiday_alignment(L.holiday_multiplier_by_year(L.load_frames()))
    need = {"n_clean_years", "cv_per_table", "cv_raw", "verdict"}
    return need.issubset(t.columns), f"holiday_alignment columns: {list(t.columns)}"


def check_cny_trough_below_one() -> tuple[bool, str]:
    t = L.cny_trough_by_year(L.load_frames())
    # every clean CNY year should show a pre-CNY trough < 1.0 (demand below baseline)
    vals = t["min_mult_value"].dropna()
    return (vals < 1.0).all() and len(vals) >= 3, f"pre-CNY min multipliers: {vals.round(2).to_dict()}"
```

- [ ] **Step 2: Run checks — expect the six new ones to FAIL**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: 6 FAIL (`AttributeError`), 13 PASS.

- [ ] **Step 3: Implement in `eda_lib.py`**

```python
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
```

- [ ] **Step 4: Add charts to `plots.py`**

```python
def plot_holiday_impact(mult_table: pd.DataFrame, out: Path) -> None:
    hols = sorted(mult_table["holiday"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, name in zip(axes.flat, hols):
        sub = mult_table[mult_table["holiday"] == name].sort_values("year")
        clean = sub[sub["year"].isin(YEAR_COLOURS)]
        ax.bar(clean["year"].astype(str), clean["mult_per_table"],
               color=[YEAR_COLOURS.get(int(y), GREY) for y in clean["year"]])
        ax.axhline(1.0, color=GREY, lw=1, ls="--")
        ax.set_title(name, color=NAVY, fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        style_ax(ax)
    for ax in axes.flat[len(hols):]:
        ax.set_visible(False)
    fig.suptitle("Holiday demand multiplier vs pre-holiday baseline (per table), clean years",
                 color=NAVY, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out)


def plot_holiday_offset_shape(shape_by_holiday: dict, out: Path) -> None:
    names = list(shape_by_holiday)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.4 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        tbl = shape_by_holiday[name]
        for yr in tbl.index:
            ax.plot(tbl.columns, tbl.loc[yr].values, "o-", ms=3, lw=1.5,
                    color=YEAR_COLOURS.get(int(yr), GREY), label=str(int(yr)))
        ax.axhline(1.0, color=GREY, lw=1, ls="--")
        ax.set_title(f"{name} — demand/table by day offset, ÷ pre-holiday baseline",
                     color=NAVY, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("days from anchor", color=GREY, fontsize=8)
        ax.legend(fontsize=7, ncol=5)
        style_ax(ax)
    fig.tight_layout()
    _save(fig, out)


def plot_cny_trough(trough_table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    yrs = [int(y) for y in trough_table.index]
    ax.bar([str(y) for y in yrs], trough_table["min_mult_value"],
           color=[YEAR_COLOURS.get(y, GREY) for y in yrs])
    ax.axhline(1.0, color=GREY, lw=1, ls="--")
    for x, (v, o) in enumerate(zip(trough_table["min_mult_value"], trough_table["min_mult_offset"])):
        ax.text(x, v + 0.01, f"d{int(o)}", ha="center", fontsize=8, color=GREY)
    ax.set_ylabel("min demand/table in CNY d-7..d-1  ÷ baseline", color=GREY, fontsize=9)
    ax.set_title("Pre-CNY trough depth by year — the mid-January failure mode",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    style_ax(ax)
    _save(fig, out)
```

- [ ] **Step 5: Run checks — expect all PASS**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `19/19 checks passed`. If `check_cny_trough_below_one` fails because a clean year is missing, inspect `holiday_offset_shape_by_year(load_frames(), "CNY")` — a missing year means its pre-holiday baseline came back NaN (too few normal days), which is legitimate; relax the check to `len(vals) >= 3` only if that is confirmed the cause.

- [ ] **Step 6: Commit**

```bash
git status --short
git add research/explore_longhistory/eda_lib.py research/explore_longhistory/plots.py research/explore_longhistory/checks.py
git commit -m "A3: holiday impact by year, per-offset shape, CNY trough, alignment verdict"
```

---

## Task 5: Part A4 — COVID section

**Files:**
- Modify: `research/explore_longhistory/eda_lib.py`
- Modify: `research/explore_longhistory/plots.py`
- Modify: `research/explore_longhistory/checks.py`

**Interfaces:**
- Consumes: `load_frames`
- Produces:
  - `eda_lib.covid_timeline(df) -> pd.DataFrame` — index = `YYYY-MM` period 2019-01..2024-12, columns `mean_demand, mean_per_table, n_days, n_closure_days`
  - `eda_lib.recovery_2023_check(df) -> dict` — `{"jan_per_table": float, "dec_per_table": float, "climbed": bool, "norm_2024_h1": float, "reached_norm": bool}`
  - `plots.plot_covid_timeline(timeline, out) -> None`

- [ ] **Step 1: Add checks**

```python
def check_covid_timeline_spans_2019_2024() -> tuple[bool, str]:
    t = L.covid_timeline(L.load_frames())
    lo, hi = str(t.index.min()), str(t.index.max())
    return lo == "2019-01" and hi == "2024-12", f"covid timeline span {lo}..{hi}"


def check_recovery_2023_climbed() -> tuple[bool, str]:
    r = L.recovery_2023_check(L.load_frames())
    return r["climbed"] and r["jan_per_table"] < r["dec_per_table"], \
        f"2023 per-table jan={r['jan_per_table']:.1f} dec={r['dec_per_table']:.1f} climbed={r['climbed']}"
```

- [ ] **Step 2: Run checks — expect 2 FAIL**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: 2 FAIL (`AttributeError`), 19 PASS.

- [ ] **Step 3: Implement in `eda_lib.py`**

```python
def covid_timeline(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2024-12-31")].copy()
    d["period"] = d["date"].dt.to_period("M")
    g = d.groupby("period")
    out = pd.DataFrame({
        "mean_demand": g["demand"].mean(),
        "mean_per_table": g["demand_per_table"].mean(),
        "n_days": g.size(),
        "n_closure_days": g.apply(lambda x: int(((x["floortables"] == 0) | (x["demand"] == 0)).sum())),
    })
    out.index = out.index.astype(str)
    return out


def recovery_2023_check(df: pd.DataFrame) -> dict:
    d = df[df["year"] == 2023]
    jan = d.loc[d["month"] == 1, "demand_per_table"].mean()
    dec = d.loc[d["month"] == 12, "demand_per_table"].mean()
    h1_2024 = df.loc[(df["year"] == 2024) & (df["month"] <= 6), "demand_per_table"].mean()
    return {
        "jan_per_table": float(jan), "dec_per_table": float(dec),
        "climbed": bool(dec > jan),
        "norm_2024_h1": float(h1_2024),
        "reached_norm": bool(dec >= 0.95 * h1_2024),
    }
```

- [ ] **Step 4: Add chart to `plots.py`**

```python
def plot_covid_timeline(timeline: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = range(len(timeline))
    ax.plot(x, timeline["mean_per_table"], "o-", color=NAVY, lw=1.7, ms=3)
    ax.axvspan(list(timeline.index).index("2020-01"), list(timeline.index).index("2022-12"),
               color=RED, alpha=0.10)
    ax.set_xticks(list(x)[::3])
    ax.set_xticklabels(list(timeline.index)[::3], rotation=45, fontsize=8)
    ax.set_ylabel("mean demand / table", color=GREY, fontsize=9)
    ax.set_title("COVID collapse and recovery — monthly demand per table, 2019–2024",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    style_ax(ax)
    _save(fig, out)
```

- [ ] **Step 5: Run checks — expect all PASS**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `21/21 checks passed`.

- [ ] **Step 6: Commit**

```bash
git status --short
git add research/explore_longhistory/eda_lib.py research/explore_longhistory/plots.py research/explore_longhistory/checks.py
git commit -m "A4: COVID timeline + 2023 recovery-not-normal check"
```

---

## Task 6: Part B1 — feature-matrix sample + data dictionary

**Files:**
- Modify: `research/explore_longhistory/eda_lib.py`
- Modify: `research/explore_longhistory/checks.py`

**Interfaces:**
- Consumes: `load_frames`, `feature_group`
- Produces:
  - `eda_lib.build_matrix() -> pd.DataFrame` — `F.apply_exclusions(F.build_base(load_long_demand, history_start=None, quiet=True), demand, [])`; cached by the engine
  - `eda_lib.feature_names(mat) -> list[str]` — `F.feature_columns(mat)`
  - `eda_lib.sample_tall(mat, n=500, seed=0) -> pd.DataFrame` — `n` rows spread evenly across sorted `target_date`, all columns
  - `eda_lib.pick_sample_dates() -> dict[str, pd.Timestamp]` — labelled target dates: `normal_midweek, normal_saturday, cny_dm3, cny_dp1, goldenweek, jan2026_a, jan2026_b` (jan2026 = 2026-01-26, 2026-01-29; cny anchor 2026-02-17 → dm3 2026-02-14, dp1 2026-02-18; normal_midweek 2025-11-19 Wed; normal_saturday 2025-11-22; goldenweek 2025-10-02)
  - `eda_lib.sample_wide(mat, horizon=7) -> pd.DataFrame` — index = feature name, columns = the labels from `pick_sample_dates()`, values = that feature for `(target_date, horizon)`; a trailing `__group__` column
  - `eda_lib.feature_dictionary(mat) -> pd.DataFrame` — index = feature, columns `group, dtype, pct_nonnull_all, pct_nonnull_trainable, min, mean, max`

- [ ] **Step 1: Add checks**

```python
def check_matrix_feature_count() -> tuple[bool, str]:
    mat = L.build_matrix()
    n = len(L.feature_names(mat))
    return n == 174, f"matrix has {n} features (want 174)"


def check_sample_wide_has_jan2026_and_all_features() -> tuple[bool, str]:
    mat = L.build_matrix()
    w = L.sample_wide(mat)
    cols_ok = {"jan2026_a", "jan2026_b", "cny_dp1"}.issubset(w.columns)
    rows_ok = len(w) == len(L.feature_names(mat))
    return cols_ok and rows_ok, f"wide sample: {w.shape}, cols {list(w.columns)}"


def check_feature_dictionary_groups_and_nonnull() -> tuple[bool, str]:
    mat = L.build_matrix()
    fd = L.feature_dictionary(mat)
    groups = set(fd["group"])
    known = {"recency_aggregate", "short_lag", "yearly_lookback", "holiday_yoy",
             "holiday_flag", "calendar", "mainland_block", "regime", "capacity", "other"}
    ok = groups.issubset(known) and fd["pct_nonnull_all"].between(0, 100).all()
    return ok, f"{len(fd)} features, groups {sorted(groups)}"


def check_sample_tall_shape() -> tuple[bool, str]:
    mat = L.build_matrix()
    t = L.sample_tall(mat, n=500)
    return len(t) == 500 and "target_date" in t.columns, f"tall sample {t.shape}"
```

- [ ] **Step 2: Run checks — expect 4 FAIL**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: 4 FAIL, 21 PASS. First run builds the matrix (~2s on a warm cache; up to 8 min cold) — let it finish.

- [ ] **Step 3: Implement in `eda_lib.py`**

```python
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
```

- [ ] **Step 4: Run checks — expect all PASS**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `25/25 checks passed`. If `check_matrix_feature_count` reports a number other than 174, do NOT change the assertion — re-read `long_history/config.py` (`ADD_WEEK_OF_YEAR`, `ADD_REGIME_FEATURE` must both be `True`) and confirm nothing in `long_history/engine` changed; the 174 is the spec's ground truth.

- [ ] **Step 5: Commit**

```bash
git status --short
git add research/explore_longhistory/eda_lib.py research/explore_longhistory/checks.py
git commit -m "B1: feature-matrix tall/wide sample + data dictionary"
```

---

## Task 7: Part B2 — feature correlation

**Files:**
- Modify: `research/explore_longhistory/eda_lib.py`
- Modify: `research/explore_longhistory/plots.py`
- Modify: `research/explore_longhistory/checks.py`

**Interfaces:**
- Consumes: `build_matrix`, `feature_names`, `feature_group`
- Produces:
  - `eda_lib.corr_frames(mat) -> dict` — `{"pearson": pd.DataFrame, "spearman": pd.DataFrame}` over numeric features on trainable rows (`y` notna), computed on the subset of features with > 0 variance
  - `eda_lib.high_corr_pairs(pearson, threshold=0.9) -> pd.DataFrame` — columns `feat_a, feat_b, group_a, group_b, r`; each unordered pair once; `|r| >= threshold`; sorted by `|r|` desc
  - `eda_lib.target_corr(mat) -> pd.DataFrame` — index = feature, columns `pearson, spearman`; correlation with `y` on trainable rows; sorted by `|spearman|` desc
  - `eda_lib.group_corr(pearson) -> pd.DataFrame` — mean `|r|` between every pair of feature groups (square, symmetric)
  - `plots.plot_group_corr(group_corr, out)`, `plots.plot_target_corr(target_corr, out, top=30)`

- [ ] **Step 1: Add checks**

```python
def check_high_corr_pairs_dedup_and_synthetic() -> tuple[bool, str]:
    rng = np.random.default_rng(0)
    base = rng.normal(size=2000)
    p = pd.DataFrame({
        "a": base, "b": base * 2 + 1e-6 * rng.normal(size=2000),   # r ~ 1
        "c": rng.normal(size=2000),                                 # independent
    }).corr()
    hp = L.high_corr_pairs(p, threshold=0.9)
    pair = set(hp.iloc[0][["feat_a", "feat_b"]]) if len(hp) else set()
    return len(hp) == 1 and pair == {"a", "b"}, f"high_corr_pairs -> {hp.to_dict('records')}"


def check_target_corr_ranked_and_covers_all() -> tuple[bool, str]:
    mat = L.build_matrix()
    tc = L.target_corr(mat)
    ranked = tc["spearman"].abs().is_monotonic_decreasing
    covers = len(tc) == len(L.feature_names(mat))
    return ranked and covers, f"target_corr: {len(tc)} rows, ranked={ranked}"


def check_group_corr_symmetric() -> tuple[bool, str]:
    mat = L.build_matrix()
    gc = L.group_corr(L.corr_frames(mat)["pearson"])
    return np.allclose(gc.values, gc.values.T, equal_nan=True), f"group_corr shape {gc.shape}"
```

- [ ] **Step 2: Run checks — expect 3 FAIL**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: 3 FAIL, 25 PASS.

- [ ] **Step 3: Implement in `eda_lib.py`**

```python
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
                vals = block.values[~np.eye(len(members[g1]), dtype=bool)] if len(members[g1]) > 1 else [np.nan]
                out.loc[g1, g2] = np.nanmean(vals)
            else:
                out.loc[g1, g2] = np.nanmean(block.values)
    return out
```

- [ ] **Step 4: Add charts to `plots.py`**

```python
def plot_group_corr(group_corr: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(group_corr.values.astype(float), vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(group_corr))); ax.set_yticks(range(len(group_corr)))
    ax.set_xticklabels(group_corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(group_corr.index, fontsize=8)
    for i in range(len(group_corr)):
        for j in range(len(group_corr)):
            v = group_corr.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "black")
    ax.set_title("Mean |correlation| between feature groups", color=NAVY,
                 fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out)


def plot_target_corr(target_corr: pd.DataFrame, out: Path, top: int = 30) -> None:
    s = target_corr["spearman"].dropna()
    top_s = pd.concat([s.head(top), s.tail(0)]).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top_s))))
    ax.barh(range(len(top_s)), top_s.values,
            color=[TEAL if v >= 0 else RED for v in top_s.values])
    ax.set_yticks(range(len(top_s))); ax.set_yticklabels(top_s.index, fontsize=8)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("Spearman correlation with y", color=GREY, fontsize=9)
    ax.set_title(f"Top {top} features by |correlation| with the target",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    style_ax(ax)
    _save(fig, out)
```

- [ ] **Step 5: Run checks — expect all PASS**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `28/28 checks passed`.

- [ ] **Step 6: Commit**

```bash
git status --short
git add research/explore_longhistory/eda_lib.py research/explore_longhistory/plots.py research/explore_longhistory/checks.py
git commit -m "B2: feature correlation — high-corr pairs, target corr, group heatmap"
```

---

## Task 8: Part B3 — SHAP via LightGBM native `pred_contrib`

**Files:**
- Modify: `research/explore_longhistory/eda_lib.py`
- Modify: `research/explore_longhistory/plots.py`
- Modify: `research/explore_longhistory/checks.py`

**Interfaces:**
- Consumes: `build_matrix`, `feature_names`, `pick_sample_dates`
- Produces:
  - `eda_lib.train_horizon_model(mat, feats, horizon: int, as_of: pd.Timestamp) -> lgb.LGBMRegressor` — trained on `mat` rows with `target_date <= as_of` and `horizon == h`, `dropna(subset=["y"])` only, sample weights per Global Constraints
  - `eda_lib.shap_global(mat, feats, horizons: list[int], sample: int = 4000, seed: int = 0) -> pd.DataFrame` — index = feature, columns `mean_abs_shap` (mean over the pooled per-horizon samples of `|shap|`), `mean_abs_shap_<h>` per horizon; sorted desc by `mean_abs_shap`
  - `eda_lib.shap_local(mat, feats, horizon: int = 7) -> dict[str, pd.DataFrame]` — label → DataFrame(index=feature, columns `feature_value, shap`), plus a one-row `__meta__` frame carrying `base_value` and `prediction`
  - `plots.plot_shap_summary(mat, feats, global_table, out, top=25)` — strip plot of SHAP value vs rank-normalised feature value for the top features (recomputes contribs on a fresh 2000-row sample at horizon 7)
  - `plots.plot_shap_waterfall(local_df, meta, label, out)` — base → ±contributions of the top 15 features → prediction

- [ ] **Step 1: Add checks**

```python
import lightgbm as lgb  # add to checks.py imports


def check_shap_additivity() -> tuple[bool, str]:
    mat = L.build_matrix()
    feats = L.feature_names(mat)
    as_of = mat["target_date"].max()
    m = L.train_horizon_model(mat, feats, horizon=7, as_of=as_of)
    rows = mat[(mat["horizon"] == 7) & (mat["y"].notna())].head(200)
    contrib = m.predict(rows[feats], pred_contrib=True)      # (n, n_feat + 1)
    recon = contrib.sum(axis=1)
    pred = m.predict(rows[feats])
    return np.allclose(recon, pred, atol=1e-6), \
        f"max |sum(contrib) - predict| = {np.abs(recon - pred).max():.2e} (want < 1e-6)"


def check_train_horizon_model_dropna_y_only() -> tuple[bool, str]:
    mat = L.build_matrix()
    feats = L.feature_names(mat)
    as_of = mat["target_date"].max()
    m = L.train_horizon_model(mat, feats, horizon=7, as_of=as_of)
    want = int(mat[(mat["horizon"] == 7) & (mat["target_date"] <= as_of)]["y"].notna().sum())
    got = int(m.booster_.num_trees() > 0) and m.fitted_
    n_train = getattr(m, "_n_train_rows", None)
    return n_train == want, f"trained on {n_train} rows (want {want} = dropna on y only)"


def check_shap_global_covers_features_and_ranked() -> tuple[bool, str]:
    mat = L.build_matrix()
    feats = L.feature_names(mat)
    g = L.shap_global(mat, feats, horizons=[7], sample=800)
    ranked = g["mean_abs_shap"].is_monotonic_decreasing
    return ranked and set(g.index) == set(feats), \
        f"shap_global: {len(g)} features, ranked={ranked}"
```

Note for `check_train_horizon_model_dropna_y_only`: `train_horizon_model` must set `m._n_train_rows = len(train_df)` before returning, so the check can read it.

- [ ] **Step 2: Run checks — expect 3 FAIL**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: 3 FAIL, 28 PASS.

- [ ] **Step 3: Implement in `eda_lib.py`**

Add `import lightgbm as lgb` to `eda_lib.py` imports, then append:

```python
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
    return c[:, :-1], float(c[0, -1])            # (contribs, bias) — bias identical per row


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
```

- [ ] **Step 4: Add charts to `plots.py`**

```python
def plot_shap_summary(mat, feats, global_table, out: Path, top: int = 25) -> None:
    import eda_lib as L
    as_of = mat["target_date"].max()
    m = L.train_horizon_model(mat, feats, 7, as_of)
    pool = mat[(mat["horizon"] == 7) & (mat["y"].notna())]
    take = pool.sample(min(2000, len(pool)), random_state=0)
    contribs = m.predict(take[feats], pred_contrib=True)[:, :-1]
    sh = pd.DataFrame(contribs, columns=feats)
    order = list(global_table.head(top).index)
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(order) + 1))
    for i, f in enumerate(order[::-1]):
        vals = take[f].to_numpy(dtype="float64")
        rank = pd.Series(vals).rank(pct=True).to_numpy()
        ax.scatter(sh[f], np.full(len(sh), i) + (rank - 0.5) * 0.7,
                   c=rank, cmap="coolwarm", s=6, alpha=0.5, linewidths=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1], fontsize=8)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("SHAP value (impact on predicted patron-hours)", color=GREY, fontsize=9)
    ax.set_title("SHAP summary — horizon 7, top features (colour = feature value)",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    style_ax(ax)
    _save(fig, out)


def plot_shap_waterfall(local_df: pd.DataFrame, meta: pd.DataFrame, label: str, out: Path) -> None:
    top = local_df.head(15).iloc[::-1]
    base = float(meta["base_value"].iloc[0]); pred = float(meta["prediction"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 6))
    running = base
    for i, (feat, r) in enumerate(top.iterrows()):
        ax.barh(i, r["shap"], left=running, color=TEAL if r["shap"] >= 0 else RED)
        running += r["shap"]
    ax.axvline(base, color=GREY, ls="--", lw=1, label=f"base {base:,.0f}")
    ax.axvline(pred, color=NAVY, ls="-", lw=1.4, label=f"prediction {pred:,.0f}")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{f} = {v:,.2f}" for f, v in zip(top.index, top['feature_value'])], fontsize=8)
    ax.set_title(f"Why the {label} forecast landed where it did (horizon 7)",
                 color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    style_ax(ax)
    _save(fig, out)
```

- [ ] **Step 5: Run checks — expect all PASS**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `31/31 checks passed`. This task's checks train several LightGBM models — allow ~1-2 min.

- [ ] **Step 6: Commit**

```bash
git status --short
git add research/explore_longhistory/eda_lib.py research/explore_longhistory/plots.py research/explore_longhistory/checks.py
git commit -m "B3: SHAP via LightGBM pred_contrib — global ranking + local waterfalls"
```

---

## Task 9: Orchestrator `run_explore.py` + `--self-check`

**Files:**
- Create: `research/explore_longhistory/run_explore.py`

**Interfaces:**
- Consumes: everything in `eda_lib` and `plots`
- Produces: CLI `run_explore.py` writing `output/*.csv`, `output/*.png`, and `EXPLORATION.md` (skeleton if absent, never overwrites an existing one)

- [ ] **Step 1: Write `run_explore.py`**

Create `research/explore_longhistory/run_explore.py`:

```python
"""run_explore.py -- build every exploration artifact for the long-history
pipeline. Standalone; imports long_history/{config,engine} read-only.

    uv run python research/explore_longhistory/run_explore.py
    uv run python research/explore_longhistory/run_explore.py --self-check
    uv run python research/explore_longhistory/run_explore.py --skip-shap
    uv run python research/explore_longhistory/run_explore.py --horizons all
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eda_lib as L          # noqa: E402
import plots as P            # noqa: E402

OUT = HERE / "output"


def _self_check() -> int:
    d = L.load_frames()
    problems = []
    full = pd.date_range(d["date"].min(), d["date"].max())
    if len(full) != d["date"].nunique():
        problems.append(f"calendar gaps: {len(full) - d['date'].nunique()} missing days")
    cl = set(pd.to_datetime(L.closure_days(d)["date"]).dt.date)
    known = {pd.Timestamp('2018-09-16').date()}
    known |= {x.date() for x in pd.date_range('2020-02-05', '2020-02-19')}
    known |= {x.date() for x in pd.date_range('2022-07-11', '2022-07-22')}
    if known - cl:
        problems.append(f"known closures missing: {sorted(known - cl)[:3]}")
    mult = L.holiday_multiplier_by_year(d)
    missing_base = mult[mult["baseline_lookback"].isna()]
    for _, r in missing_base.iterrows():
        print(f"  note: {r['holiday']} {int(r['year'])} has no pre-holiday baseline "
              f"(only {int(r['baseline_n_days'])} normal lead-in days)")
    print("SELF-CHECK:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--horizons", default="1,4,7,14,21,28")
    ap.add_argument("--shap-sample", type=int, default=4000)
    a = ap.parse_args()

    if a.self_check:
        return _self_check()

    OUT.mkdir(parents=True, exist_ok=True)
    d = L.load_frames()
    print(f"data: {d.date.min().date()} .. {d.date.max().date()}  ({len(d):,} days)")

    # ---- Part A ----------------------------------------------------------
    ov = pd.DataFrame({c: L.describe_series(d, c)
                       for c in ("demand", "floortables", "demand_per_table")})
    ov.to_csv(OUT / "stats_overall.csv")
    L.describe_by(d, "year", ["demand", "demand_per_table", "floortables"]).to_csv(OUT / "stats_by_year.csv")
    L.describe_by(d, "dow", ["demand", "demand_per_table"]).to_csv(OUT / "stats_by_dow.csv")
    L.describe_by_regime(d).to_csv(OUT / "stats_by_regime.csv")
    L.closure_days(d).to_csv(OUT / "closure_days.csv", index=False)

    mi = L.monthly_index_by_year(d)
    mi.to_csv(OUT / "monthly_index_by_year.csv")
    ma = L.monthly_alignment(mi)
    pd.DataFrame({"per_month_std": ma["per_month_std"]}).to_csv(OUT / "monthly_alignment.csv")
    ma["shape_corr"].to_csv(OUT / "monthly_shape_corr.csv")
    P.plot_monthly_seasonality(mi, OUT / "monthly_seasonality.png")
    P.plot_monthly_shape_corr(ma["shape_corr"], OUT / "monthly_shape_corr.png")
    print(f"monthly alignment: mean year-to-year shape corr = {ma['mean_offdiag_corr']:.3f}; "
          f"unstable months = {ma['unstable_months'] or 'none'}")

    mult = L.holiday_multiplier_by_year(d)
    mult.to_csv(OUT / "holiday_multiplier_by_year.csv", index=False)
    align = L.holiday_alignment(mult)
    align.to_csv(OUT / "holiday_alignment.csv")
    shapes = {h: L.holiday_offset_shape_by_year(d, h) for h in L.WIDE_WINDOW_HOLIDAYS}
    for h, tbl in shapes.items():
        tbl.to_csv(OUT / f"holiday_offset_shape_{h}.csv")
    trough = L.cny_trough_by_year(d)
    trough.to_csv(OUT / "cny_trough_by_year.csv")
    P.plot_holiday_impact(mult, OUT / "holiday_impact.png")
    P.plot_holiday_offset_shape(shapes, OUT / "holiday_offset_shape.png")
    P.plot_cny_trough(trough, OUT / "cny_trough_by_year.png")
    print("holiday alignment:\n" + align[["n_clean_years", "cv_per_table", "verdict"]].to_string())

    ct = L.covid_timeline(d)
    ct.to_csv(OUT / "covid_timeline.csv")
    P.plot_covid_timeline(ct, OUT / "covid_timeline.png")
    rec = L.recovery_2023_check(d)
    print(f"2023 recovery: per-table {rec['jan_per_table']:.1f} -> {rec['dec_per_table']:.1f}, "
          f"reached 2024-H1 norm ({rec['norm_2024_h1']:.1f}): {rec['reached_norm']}")

    # ---- Part B ----------------------------------------------------------
    mat = L.build_matrix()
    feats = L.feature_names(mat)
    L.sample_tall(mat).to_csv(OUT / "sample_matrix_tall.csv", index=False)
    L.sample_wide(mat).to_csv(OUT / "sample_matrix_wide.csv")
    L.feature_dictionary(mat).to_csv(OUT / "feature_dictionary.csv")

    cf = L.corr_frames(mat)
    L.high_corr_pairs(cf["pearson"]).to_csv(OUT / "corr_high_pairs.csv", index=False)
    tc = L.target_corr(mat)
    tc.to_csv(OUT / "corr_target.csv")
    gc = L.group_corr(cf["pearson"])
    gc.to_csv(OUT / "corr_group.csv")
    P.plot_group_corr(gc, OUT / "corr_heatmap_grouped.png")
    P.plot_target_corr(tc, OUT / "corr_target.png")
    print(f"high-correlation pairs (|r|>=0.9): {len(L.high_corr_pairs(cf['pearson']))}")

    if not a.skip_shap:
        horizons = list(range(1, 29)) if a.horizons == "all" else [int(x) for x in a.horizons.split(",")]
        g = L.shap_global(mat, feats, horizons, sample=a.shap_sample)
        g.to_csv(OUT / "shap_importance.csv")
        P.plot_shap_summary(mat, feats, g, OUT / "shap_summary.png")
        loc = L.shap_local(mat, feats, horizon=7)
        for label in L.pick_sample_dates():
            if label in loc:
                loc[label].to_csv(OUT / f"shap_local_{label}.csv")
                P.plot_shap_waterfall(loc[label], loc[f"{label}__meta__"], label,
                                      OUT / f"shap_waterfall_{label}.png")
        print("SHAP top 10 (mean |shap|, pooled horizons):")
        print(g["mean_abs_shap"].head(10).to_string())

    _write_report_skeleton(OUT, ma, align, rec)
    print(f"\n-> {OUT}")
    return 0


def _write_report_skeleton(out: Path, ma: dict, align: pd.DataFrame, rec: dict) -> None:
    report = HERE / "EXPLORATION.md"
    if report.exists():
        print(f"  ({report.name} exists — not overwriting)")
        return
    report.write_text(
        "# Long-history pipeline — data exploration\n\n"
        "_Generated skeleton. Fill every `TODO(prose)` with the reading of the "
        "referenced table/chart._\n\n"
        "## A1 Descriptive statistics\n\nTODO(prose) — see `output/stats_*.csv`.\n\n"
        "## A2 Monthly seasonality\n\n"
        f"Mean year-to-year shape correlation: **{ma['mean_offdiag_corr']:.3f}**. "
        f"Unstable months: {ma['unstable_months'] or 'none'}. "
        "![](output/monthly_seasonality.png)\n\nTODO(prose)\n\n"
        "## A3 Holiday impact\n\n"
        f"```\n{align[['n_clean_years','cv_per_table','verdict']].to_string()}\n```\n"
        "![](output/holiday_impact.png) ![](output/cny_trough_by_year.png)\n\nTODO(prose)\n\n"
        "## A4 COVID\n\n"
        f"2023 per-table {rec['jan_per_table']:.1f} → {rec['dec_per_table']:.1f}; "
        f"reached 2024-H1 norm: {rec['reached_norm']}. "
        "![](output/covid_timeline.png)\n\nTODO(prose)\n\n"
        "## B1 Feature-matrix sample\n\n`output/sample_matrix_wide.csv`, "
        "`output/feature_dictionary.csv`.\n\nTODO(prose)\n\n"
        "## B2 Feature correlation\n\n`output/corr_high_pairs.csv`. "
        "![](output/corr_heatmap_grouped.png) ![](output/corr_target.png)\n\nTODO(prose)\n\n"
        "## B3 SHAP feature importance\n\n`output/shap_importance.csv`. "
        "![](output/shap_summary.png)\n\n"
        "Local explanations for the mid-Jan-2026 under-forecast: "
        "![](output/shap_waterfall_jan2026_a.png) "
        "![](output/shap_waterfall_jan2026_b.png)\n\nTODO(prose)\n\n"
        "## Notes on method / honesty\n\n"
        "- SHAP models trained on the full history (`as_of` = data max) — this "
        "explains what the deployed model learned, not held-out accuracy.\n"
        "- Pre-holiday baseline is lead-in only (normal days before the window).\n"
        "- Nothing here changes the model; redundancy/contamination is recorded, not acted on.\n",
        encoding="utf-8")
    print(f"  wrote skeleton {report.name}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the self-check**

Run: `uv run python research/explore_longhistory/run_explore.py --self-check`
Expected: `SELF-CHECK: PASS`, possibly with `note:` lines for holiday-years that legitimately lack a baseline.

- [ ] **Step 3: Run the fast path (no SHAP)**

Run: `uv run python research/explore_longhistory/run_explore.py --skip-shap`
Expected: prints the monthly + holiday alignment tables and the high-corr count; `output/` now holds the Part A + B1 + B2 CSVs and PNGs; `EXPLORATION.md` skeleton written. Confirm no traceback and that each PNG is non-empty (`ls -la research/explore_longhistory/output/`).

- [ ] **Step 4: Run the full path**

Run: `uv run python research/explore_longhistory/run_explore.py`
Expected: additionally writes `shap_importance.csv`, `shap_summary.png`, `shap_local_*.csv`, `shap_waterfall_*.png`; prints the SHAP top 10. ~5 min.

- [ ] **Step 5: Commit**

```bash
git status --short
git add research/explore_longhistory/run_explore.py
git commit -m "Orchestrator: run_explore.py writes every artifact + --self-check"
```

---

## Task 10: `EXPLORATION.md` prose + `explore.ipynb`

**Files:**
- Modify: `research/explore_longhistory/EXPLORATION.md` (replace every `TODO(prose)`)
- Create: `research/explore_longhistory/explore.ipynb`

**Interfaces:**
- Consumes: the `output/` artifacts from Task 9

- [ ] **Step 1: Fill the report prose**

Open `research/explore_longhistory/EXPLORATION.md`. For each section, replace `TODO(prose)` with 2-5 sentences stating what the numbers say, read from the actual CSVs in `output/`:
- A1: the demand/per-table range, the CV by year, which regime is most volatile, the closure count.
- A2: is the mean shape correlation high (>0.85 = aligned)? name the unstable months and a likely reason.
- A3: for each holiday, state its verdict and CV; call out the CNY trough depths by year from `cny_trough_by_year.csv`; note which holidays are `variable` and what that implies for pooling years.
- A4: state whether 2023 climbed and whether it reached the 2024 norm; confirm/deny the config's "recovery not normal" claim with the numbers.
- B1: describe one wide-sample column end to end (e.g. `cny_dp1`) — which features fire, which are NaN and why (horizon gating).
- B2: how many `|r|>=0.9` pairs, which groups they cluster in, the top 3 target-correlated features.
- B3: the SHAP top 10, how it compares to the gain ranking already in `research/lag_holiday_features_output/feature_importance.csv`, and — reading `shap_waterfall_jan2026_*.png` — which features pushed the mid-Jan-2026 forecast down (expect `yoy_ratio` / `lag_365` / recency aggregates).

Keep the honesty notes section as written.

- [ ] **Step 2: Create the notebook**

Create `research/explore_longhistory/explore.ipynb` as JSON. Use this exact structure (7 code cells + 1 setup), each preceded by a markdown cell restating the question. Build it with a small script rather than by hand:

```bash
uv run python - <<'PY'
import nbformat as nbf
nb = nbf.v4.new_notebook()
def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
nb.cells = [
    md("# Long-history pipeline — data exploration\nRuns every analysis from `eda_lib`. Same cached matrix as `run_explore.py`."),
    code("import sys\nfrom pathlib import Path\nHERE = Path.cwd()\nsys.path.insert(0, str(HERE))\nimport eda_lib as L, plots as P\nimport pandas as pd, numpy as np\nimport matplotlib.pyplot as plt\n%matplotlib inline\nd = L.load_frames()\nd.head()"),
    md("## A1 — descriptive statistics: what does the raw series look like?"),
    code("pd.DataFrame({c: L.describe_series(d, c) for c in ['demand','floortables','demand_per_table']})"),
    code("L.describe_by(d, 'year', ['demand_per_table']).round(2)"),
    md("## A2 — is monthly seasonality the same shape every year?"),
    code("mi = L.monthly_index_by_year(d)\nma = L.monthly_alignment(mi)\nprint('mean year-to-year shape corr:', round(ma['mean_offdiag_corr'], 3))\nprint('unstable months:', ma['unstable_months'])\nfor yr in mi.index:\n    plt.plot(range(1,13), mi.loc[yr], 'o-', label=int(yr))\nplt.axhline(1, ls='--', c='grey'); plt.legend(ncol=5); plt.title('monthly index by year'); plt.show()"),
    md("## A3 — is each holiday's impact the same every year?"),
    code("mult = L.holiday_multiplier_by_year(d)\nL.holiday_alignment(mult)[['n_clean_years','cv_per_table','verdict']]"),
    code("L.cny_trough_by_year(d)   # pre-CNY trough depth by year"),
    code("shape = L.holiday_offset_shape_by_year(d, 'CNY')\nfor yr in shape.index:\n    plt.plot(shape.columns, shape.loc[yr], 'o-', label=int(yr))\nplt.axhline(1, ls='--', c='grey'); plt.legend(ncol=5); plt.title('CNY demand/table by day offset ÷ baseline'); plt.show()"),
    md("## A4 — the COVID years, kept separate"),
    code("ct = L.covid_timeline(d)\nplt.plot(range(len(ct)), ct['mean_per_table'], 'o-')\nplt.title('per-table 2019-2024'); plt.show()\nL.recovery_2023_check(d)"),
    md("## B1 — one row of the feature matrix after feature engineering"),
    code("mat = L.build_matrix()\nfeats = L.feature_names(mat)\nprint(len(feats), 'features')\nL.sample_wide(mat)"),
    code("L.feature_dictionary(mat).head(20)"),
    md("## B2 — which features are redundant, which track the target"),
    code("cf = L.corr_frames(mat)\nL.high_corr_pairs(cf['pearson']).head(20)"),
    code("L.target_corr(mat).head(15)"),
    md("## B3 — SHAP feature importance (LightGBM native pred_contrib)"),
    code("g = L.shap_global(mat, feats, horizons=[1,7,14,28], sample=2000)\ng['mean_abs_shap'].head(20)"),
    code("loc = L.shap_local(mat, feats, horizon=7)\nP.plot_shap_waterfall(loc['jan2026_a'], loc['jan2026_a__meta__'], 'jan2026_a', HERE/'output'/'nb_waterfall.png')\nfrom IPython.display import Image\nImage(str(HERE/'output'/'nb_waterfall.png'))"),
]
nbf.write(nb, "research/explore_longhistory/explore.ipynb")
print("wrote notebook")
PY
```

If `nbformat` is missing: `uv run --with nbformat python - <<'PY' ... PY`.

- [ ] **Step 3: Execute the notebook end to end to confirm it runs**

Run: `uv run --with nbconvert --with jupyter jupyter nbconvert --to notebook --execute --inplace research/explore_longhistory/explore.ipynb`
Expected: completes with no cell errors. (If `jupyter`/`nbconvert` are not available and cannot be added, instead open and run it manually, or skip execution and note in the commit that the notebook is unexecuted — the `.py` orchestrator is the tested path.)

- [ ] **Step 4: Final self-check + full run**

Run: `uv run python research/explore_longhistory/checks.py`
Expected: `31/31 checks passed`.

Run: `uv run python research/explore_longhistory/run_explore.py`
Expected: clean, all artifacts refreshed.

- [ ] **Step 5: Commit**

```bash
git status --short
git add research/explore_longhistory/EXPLORATION.md research/explore_longhistory/explore.ipynb
git commit -m "Exploration report prose + runnable notebook"
```

- [ ] **Step 6: Commit the generated artifacts (ask the user first)**

The user asked specifically for "a sample data after feature engineering, and the feature importance" as deliverables. Confirm whether to commit `research/explore_longhistory/output/` (CSVs + PNGs) or leave it regenerable/untracked. If yes:

```bash
git status --short
git add research/explore_longhistory/output/
git commit -m "Exploration outputs: stats, alignment tables, correlations, SHAP"
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| A1 descriptive stats (overall/year/dow/regime, closures) | Task 2 |
| A2 monthly seasonality by year + alignment (spread + shape corr) | Task 3 |
| A3 holiday multiplier by year, per-offset shape, CNY trough, CV verdict | Task 4 |
| A4 COVID timeline + 2023 recovery check | Task 5 |
| B1 tall/wide sample + data dictionary | Task 6 |
| B2 high-corr pairs, target corr, group heatmap | Task 7 |
| B3 SHAP global + local via `pred_contrib` | Task 8 |
| pre-holiday baseline = lead-in only, widen 28→56→84, NaN under 5 | Task 4 (impl + 2 checks) |
| clean years exclude 2020/21/22 everywhere | Tasks 3, 4 (checks assert absence) |
| SHAP additivity test | Task 8 (`check_shap_additivity`) |
| `train_horizon_model` dropna on y only | Task 8 (`check_train_horizon_model_dropna_y_only`) |
| `--self-check` on real data (gaps, known closures, baseline coverage) | Task 9 |
| CLI flags (`--horizons`, `--shap-sample`, `--skip-shap`) | Task 9 (`--baseline-window` dropped — see note) |
| EXPLORATION.md report | Task 10 |
| explore.ipynb notebook | Task 10 |
| honesty notes (as_of disclosed, in-sample SHAP, no model change) | Task 9 skeleton + Task 10 prose |
| charts (~12 PNGs) | folded into Tasks 2-8, rendered by Task 9 |

Deviations from the spec, all minor:
- **pytest → `checks.py`**: pytest is not installed; the repo convention is assert-based check scripts. Same coverage, no new dependency.
- **`--baseline-window` CLI flag dropped**: the widen tuple `(28,56,84)` is a function default in `pre_holiday_baseline`; exposing it as a flag adds a plumbing path through six functions for a knob the spec's own definition fixes. If the user wants it, it is a one-line `argparse` add later.
- **`covid_notes.md` fragment**: folded directly into the `EXPLORATION.md` A4 section rather than a separate file (spec allowed either).
- Spec's "print actual vs a quick reference forecast for the Jan-2026 rows" (Risks section): covered by the SHAP local waterfalls, which show the prediction and base value against the feature values for those exact rows — a stronger anchor than a bare number. Not adding a separate mini-forecast.

**2. Placeholder scan:** No `TODO`/`TBD` in code. `TODO(prose)` appears only as intentional markers in the generated `EXPLORATION.md` skeleton, which Task 10 Step 1 replaces. All code steps contain complete code.

**3. Type consistency:**
- `pre_holiday_baseline` returns `{"demand", "demand_per_table", "n_days", "lookback_used"}` — consumed with those keys in `holiday_multiplier_by_year`, `holiday_offset_shape_by_year`, and Task 4/9 checks. Consistent.
- `holiday_multiplier_by_year` columns (`holiday, year, n_window_days, baseline_n_days, baseline_lookback, mult_raw, mult_per_table`) — consumed by `holiday_alignment` (`mult_per_table`, `mult_raw`, `year`, `holiday`) and Task 9 (`baseline_lookback`). Consistent.
- `shap_global` returns index=feature with `mean_abs_shap` column — consumed by `plot_shap_summary` (`.head(top).index`) and Task 9 (`.head(10)`). Consistent.
- `shap_local` returns `{label: df, f"{label}__meta__": df}` — consumed by Task 9 and `plot_shap_waterfall(local_df, meta, label, out)`. Consistent.
- `monthly_alignment` returns `{"per_month_std", "shape_corr", "mean_offdiag_corr", "unstable_months"}` — consumed by Task 9 and `plot_monthly_shape_corr(shape_corr)`. Consistent.
- `build_matrix()` / `feature_names()` names identical across Tasks 6, 7, 8, 9.

All checks from earlier tasks stay green as later tasks only append.
