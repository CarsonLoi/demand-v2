# Hourly Demand Splitting for `long_history/` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the existing repo-root `hourly/` prototype (daily→hourly demand splitting via an empirical distribution — local weekday baselines plus a holiday-specific profile chosen by Total Variation Distance) into a fully independent `long_history/hourly/` package, and build a leakage-free validation script that measures the empirical method against a lightweight comparison model, so "the actual distribution is good enough, no complicated model needed" is a measured claim, not an assertion.

**Architecture:** Five new modules under `long_history/hourly/` — `patterns.py` (shared utilities, ported), `analyze.py` (quarterly decision builder, ported with one new `as_of` parameter), `split.py` (daily splitter, ported), `comparison_model.py` (new — the strawman regressor), `validate.py` (new — the honest backtest). A `checks.py` script grows across tasks holding assert-based sanity checks, matching this repo's existing convention (`long_history/data/validate.py`'s PASS/WARN/FAIL style) — there is no pytest anywhere in this repo, so this plan does not introduce one.

**Tech Stack:** Python, pandas, numpy, matplotlib (existing repo stack — no new dependencies).

## Global Constraints

- Nothing under `long_history/hourly/` may import from `v2/`, the repo root, or any file outside `long_history/`. Holiday dates come from `long_history/engine/core.py`, never `v2/_shared.py`.
- All outputs land under `long_history/output/hourly/`, or beside a forecast run (`long_history/output/run_<date>/predictions_hourly.csv`).
- No real hourly data exists yet (`long_history/data/raw/hourly_demand.csv` is not present). Every script must run cleanly and report honestly against the 10-day format-reference sample (`long_history/hourly/sample_hourly_data.csv`, copied in during Task 1) without crashing — but any accuracy number produced against that sample must be labeled a **mechanism check, not evidence** in both code output and docs. Real validation is deferred until real data exists.
- The comparison model in `comparison_model.py` is never used to produce an actual forecast — it exists solely to be measured against inside `validate.py`.
- Match existing `long_history/` house style exactly: each one-level-deep module opens with
  ```python
  HERE = Path(__file__).resolve().parent
  ROOT = HERE.parents[0]
  sys.path.insert(0, str(ROOT))
  ```
  then `import config as C`, `from engine import core as S` / `from engine import features as F` as needed.
- Reuse `engine.core.holiday_date_set()` for "every date inside any holiday window" — do not reimplement it.

---

## File Structure

```
long_history/hourly/
├── __init__.py
├── patterns.py            shared utilities: load hourly data, TVD, local DOW
│                           baselines, holiday membership (Task 1)
├── checks.py               assert-based sanity checks, grows across Tasks 1/3/4
├── analyze.py               quarterly: builds decisions.csv + holiday_profiles.csv
│                           + comparison plots (Task 2)
├── split.py                 daily: applies decisions to a saved forecast (Task 3)
├── comparison_model.py      the strawman LightGBM regressor, used only by
│                           validate.py (Task 4)
├── validate.py               the leave-one-out honest backtest (Task 5)
├── sample_hourly_data.csv   10-day format reference, copied from hourly/ (Task 1)
├── README.md                 module overview (Task 7)
└── TUTORIAL.md                standalone step-by-step tutorial (Task 7)

long_history/run_pipeline.py   modified: add opt-in "hourly-analyze" / "hourly-split"
                              steps (Task 6)
long_history/docs/COMPLETE_GUIDE.md   modified: new §18 on the hourly module (Task 7)
long_history/README.md                modified: one-line pointer (Task 7)
```

---

### Task 1: Scaffold `hourly/` package + `patterns.py` + `checks.py`

**Files:**
- Create: `long_history/hourly/__init__.py` (empty)
- Create: `long_history/hourly/patterns.py`
- Create: `long_history/hourly/checks.py`
- Create: `long_history/hourly/sample_hourly_data.csv` (copy of `hourly/sample_hourly_data.csv`)

**Interfaces:**
- Produces (used by Tasks 2, 3, 4, 5):
  - `DOW_BUCKETS: dict[str, list[int]]`
  - `TVD_CLOSE: float`, `TVD_BORDERLINE: float`, `LOCAL_WINDOW_DAYS: int`, `MIN_DOW_SAMPLES: int`, `MIN_HOLIDAY_OCCURRENCES: int`
  - `DATA_HOURLY: Path`, `DERIVED: Path` (= `long_history/output/hourly/`, created if missing)
  - `load_hourly(path: Path | None = None) -> pd.DataFrame` — columns `date, hour, demand`
  - `tvd(p, q) -> float`
  - `share_vector(hourly_demand_24) -> np.ndarray`
  - `date_to_hourly_array(df: pd.DataFrame) -> dict[pd.Timestamp, np.ndarray]`
  - `compute_local_dow_baselines(reference_date, date_to_hours, holiday_dates, days_back=LOCAL_WINDOW_DAYS) -> dict[str, np.ndarray | None]`
  - `holiday_occurrence_dates(holiday_name: str, day_offset: int) -> list[pd.Timestamp]`
  - `find_holiday_membership(date: pd.Timestamp) -> tuple[str | None, int | None]`

- [ ] **Step 1: Create the package and copy the sample data**

```bash
mkdir -p "long_history/hourly"
touch "long_history/hourly/__init__.py"
cp "hourly/sample_hourly_data.csv" "long_history/hourly/sample_hourly_data.csv"
```

- [ ] **Step 2: Write `long_history/hourly/checks.py` with the first three checks (will fail — `patterns` doesn't exist yet)**

```python
"""checks.py -- assert-based sanity checks for the hourly module.

Matches long_history/data/validate.py's PASS/WARN/FAIL style. There is no
pytest anywhere in this repo; this file grows across tasks as each new
module lands, and is the closest thing to a test suite this package has.

Usage:
    uv run python long_history/hourly/checks.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import patterns as P  # noqa: E402


def check_share_vector() -> tuple[bool, str]:
    v = P.share_vector([10, 20, 30, 40])
    ok = abs(v.sum() - 1.0) < 1e-9 and np.allclose(v, [0.1, 0.2, 0.3, 0.4])
    return ok, f"share_vector([10,20,30,40]) = {v.tolist()}"


def check_share_vector_zero_total() -> tuple[bool, str]:
    v = P.share_vector(np.zeros(24))
    ok = np.isnan(v).all()
    return ok, "share_vector(all-zero) returns all-NaN (a zero day cannot be split)"


def check_tvd_bounds() -> tuple[bool, str]:
    same = P.tvd([0.5, 0.5], [0.5, 0.5])
    disjoint = P.tvd([1.0, 0.0], [0.0, 1.0])
    ok = abs(same - 0.0) < 1e-9 and abs(disjoint - 1.0) < 1e-9
    return ok, f"tvd(identical)={same:.4f} (want 0)  tvd(disjoint)={disjoint:.4f} (want 1)"


def check_local_dow_baseline_no_leakage() -> tuple[bool, str]:
    """The baseline for `reference_date` must never read reference_date or
    anything on/after it -- that would be using the future to predict itself."""
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="D")
    date_to_hours = {}
    for d in dates:
        # sentinel: hour 0 share == 999 only on/after a poison date, so if the
        # baseline ever includes a poisoned day, its hour-0 share is huge.
        arr = np.full(24, 1.0)
        if d >= pd.Timestamp("2024-05-01"):
            arr[0] = 999.0
        date_to_hours[d] = arr

    ref = pd.Timestamp("2024-05-01")  # the day the poison starts
    baselines = P.compute_local_dow_baselines(ref, date_to_hours, holiday_dates=set())
    bad = [b for b, v in baselines.items() if v is not None and v[0] > 10]
    ok = not bad
    return ok, (f"poisoned buckets leaked into baseline: {bad}" if bad
               else "no poisoned (on/after reference_date) day reached the baseline")


def run_all() -> int:
    checks = [check_share_vector, check_share_vector_zero_total,
              check_tvd_bounds, check_local_dow_baseline_no_leakage]
    failed = 0
    for fn in checks:
        ok, msg = fn()
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {fn.__name__}: {msg}")
        failed += 0 if ok else 1
    print(f"\n{len(checks)-failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all())
```

- [ ] **Step 3: Run it to confirm it fails (patterns.py does not exist)**

Run: `uv run python long_history/hourly/checks.py`
Expected: `ModuleNotFoundError: No module named 'patterns'`

- [ ] **Step 4: Write `long_history/hourly/patterns.py`**

```python
"""patterns.py -- shared utilities for the hourly-splitting module.

Reuses long_history's own holiday config (engine/core.py), never v2/_shared.py
-- this module is part of the fully self-contained long_history/ package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/
sys.path.insert(0, str(ROOT))

from engine.core import HOLIDAY_ANCHORS, HOLIDAY_WINDOWS  # noqa: E402

DATA_HOURLY = ROOT / "data" / "raw" / "hourly_demand.csv"
DERIVED = ROOT / "output" / "hourly"
DERIVED.mkdir(parents=True, exist_ok=True)

# Four DOW buckets used everywhere
DOW_BUCKETS = {
    "weekday":  [0, 1, 2, 3],   # Mon-Thu
    "friday":   [4],
    "saturday": [5],
    "sunday":   [6],
}

# TVD thresholds for the decision cascade
TVD_CLOSE = 0.05         # below = DOW pattern is essentially equivalent
TVD_BORDERLINE = 0.10    # below = DOW with manual-review flag
LOCAL_WINDOW_DAYS = 90   # prior-N-days local DOW baseline
MIN_DOW_SAMPLES = 5      # minimum non-holiday days per DOW bucket
MIN_HOLIDAY_OCCURRENCES = 2  # minimum historical samples per holiday-day cell


def load_hourly(path: Path | None = None) -> pd.DataFrame:
    """Load an hourly_demand.csv with columns date, hour, demand.

    `path` is overridable (e.g. for tests / the sample file); production
    callers omit it and get DATA_HOURLY.
    """
    p = Path(path) if path is not None else DATA_HOURLY
    df = pd.read_csv(p, parse_dates=["date"])
    df["hour"] = df["hour"].astype(int)
    df["demand"] = df["demand"].astype(float)
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def tvd(p, q) -> float:
    """Total Variation Distance between two probability vectors. Bounded [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def share_vector(hourly_demand_24) -> np.ndarray:
    """Normalize a 24-hour demand vector to shares summing to 1.0."""
    arr = np.asarray(hourly_demand_24, dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.full(24, np.nan)
    return arr / total


def get_holiday_window_dates() -> set:
    """All dates falling within ANY holiday window -- used to exclude them
    when computing 'normal' DOW baselines."""
    dates = set()
    for name, anchors in HOLIDAY_ANCHORS.items():
        ws, we = HOLIDAY_WINDOWS[name]
        for a in anchors:
            for d in pd.date_range(a + pd.Timedelta(days=ws),
                                   a + pd.Timedelta(days=we)):
                dates.add(pd.Timestamp(d))
    return dates


def date_to_hourly_array(df: pd.DataFrame) -> dict:
    """date -> 24-vector of demand. Days with fewer than 24 rows get NaNs."""
    out = {}
    for d, g in df.groupby("date"):
        arr = np.full(24, np.nan)
        for _, r in g.iterrows():
            arr[int(r["hour"])] = float(r["demand"])
        out[pd.Timestamp(d)] = arr
    return out


def compute_local_dow_baselines(
    reference_date: pd.Timestamp,
    date_to_hours: dict,
    holiday_dates: set,
    days_back: int = LOCAL_WINDOW_DAYS,
) -> dict:
    """For a reference date, compute the four DOW share vectors using only
    non-holiday days from the prior `days_back` days -- strictly BEFORE
    reference_date. Returns {bucket: 24-vector or None}."""
    window_start = reference_date - pd.Timedelta(days=days_back)
    window_end = reference_date - pd.Timedelta(days=1)

    bucket_shares = {b: [] for b in DOW_BUCKETS}
    for d, hours in date_to_hours.items():
        if d < window_start or d > window_end:
            continue
        if d in holiday_dates:
            continue
        if np.isnan(hours).any():
            continue
        dow = d.weekday()
        s = share_vector(hours)
        if np.isnan(s).any():
            continue
        for bucket, dows in DOW_BUCKETS.items():
            if dow in dows:
                bucket_shares[bucket].append(s)
                break

    out = {}
    for bucket, shares in bucket_shares.items():
        if len(shares) < MIN_DOW_SAMPLES:
            out[bucket] = None
        else:
            out[bucket] = np.mean(shares, axis=0)
    return out


def holiday_occurrence_dates(holiday_name: str, day_offset: int) -> list:
    """All historical dates matching a (holiday, day_offset) cell."""
    anchors = HOLIDAY_ANCHORS.get(holiday_name, [])
    return [a + pd.Timedelta(days=day_offset) for a in anchors]


def find_holiday_membership(date: pd.Timestamp) -> tuple:
    """For a given date, return (holiday_name, day_offset) if it falls in
    any holiday window, else (None, None). If multiple holidays overlap,
    returns the first match in HOLIDAY_ANCHORS' insertion order."""
    for name, anchors in HOLIDAY_ANCHORS.items():
        ws, we = HOLIDAY_WINDOWS[name]
        for a in anchors:
            offset = (date - a).days
            if ws <= offset <= we:
                return name, int(offset)
    return None, None
```

- [ ] **Step 5: Run checks.py again to confirm it passes**

Run: `uv run python long_history/hourly/checks.py`
Expected: `4/4 checks passed`, exit code 0

- [ ] **Step 6: Commit**

```bash
git add long_history/hourly/__init__.py long_history/hourly/patterns.py \
        long_history/hourly/checks.py long_history/hourly/sample_hourly_data.csv
git commit -m "Add hourly/patterns.py: shared utilities for hourly demand splitting"
```

---

### Task 2: Port `analyze.py` (quarterly decision builder)

**Files:**
- Create: `long_history/hourly/analyze.py`

**Interfaces:**
- Consumes: everything from Task 1's `patterns.py` (imported as `P`)
- Produces (used by Tasks 3, 5):
  - `analyze_one_cell(holiday_name: str, day_offset: int, date_to_hours: dict, holiday_dates: set, as_of: pd.Timestamp | None = None) -> dict | None` — the `as_of` parameter is new versus the prototype: when given, only occurrences with `date <= as_of` are considered. This lets Task 5's honest backtest reuse this exact function instead of duplicating the decision cascade.
  - The result dict's keys: `holiday, day_offset, n_occurrences, per_occ, tvd_mean, tvd_max, modal_bucket, modal_count, decision, reason, holiday_profile_shares`
  - `HOLIDAYS_TO_ANALYZE: list[str]`
  - Writes `long_history/output/hourly/decisions.csv`, `long_history/output/hourly/holiday_profiles.csv`, `long_history/output/hourly/pattern_plots/*.png`

- [ ] **Step 1: Write `long_history/hourly/analyze.py`**

```python
"""analyze.py -- decide, per (holiday, day_offset), whether the hourly
pattern should inherit a local DOW pattern or use its own holiday-specific
pattern.

For each historical occurrence of a holiday-day, compares its hourly share
distribution against the four DOW patterns computed from the PRIOR 90 DAYS
LOCAL TO THAT OCCURRENCE (excluding other holidays). Aggregates the
per-occurrence Total Variation Distance (TVD) across years and emits a
decision.

Outputs (under long_history/output/hourly/):
    decisions.csv         per-cell decisions
    holiday_profiles.csv  recency-weighted holiday shares
    pattern_plots/*.png   side-by-side charts for borderline / holiday cases

Usage:
    uv run python long_history/hourly/analyze.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import patterns as P  # noqa: E402
from engine.core import HOLIDAY_WINDOWS  # noqa: E402

# Which holidays to analyze (single-day low-impact ones get DOW patterns by default)
HOLIDAYS_TO_ANALYZE = [
    "CNY", "GoldenWeek", "Labour", "MidAutumn", "Christmas", "Easter",
]


def analyze_one_cell(holiday_name: str, day_offset: int,
                     date_to_hours: dict, holiday_dates: set,
                     as_of: pd.Timestamp | None = None) -> dict | None:
    """Returns a result dict with per-occurrence TVDs and an aggregated
    decision, or None if there isn't enough data to decide.

    `as_of`: when given, only occurrences with date <= as_of are considered.
    This is what makes the function reusable for an honest walk-forward
    backtest (validate.py) -- without it, a decision made "today" could be
    informed by holiday occurrences that haven't happened yet.
    """
    occ_dates = P.holiday_occurrence_dates(holiday_name, day_offset)
    if as_of is not None:
        occ_dates = [d for d in occ_dates if d <= as_of]
    occ_dates = [d for d in occ_dates if d in date_to_hours
                 and not np.isnan(date_to_hours[d]).any()]

    if len(occ_dates) < P.MIN_HOLIDAY_OCCURRENCES:
        return None

    per_occ = []
    p_hol_shares = []
    for occ_date in occ_dates:
        p_occ = P.share_vector(date_to_hours[occ_date])
        if np.isnan(p_occ).any():
            continue
        p_hol_shares.append(p_occ)

        local = P.compute_local_dow_baselines(occ_date, date_to_hours, holiday_dates)
        bucket_tvds = {b: P.tvd(p_occ, v) for b, v in local.items() if v is not None}
        if not bucket_tvds:
            continue

        best_bucket = min(bucket_tvds, key=bucket_tvds.get)
        per_occ.append({
            "date": occ_date.date().isoformat(),
            "p_hol": p_occ.tolist(),
            "tvds": bucket_tvds,
            "best_bucket": best_bucket,
            "best_tvd": float(bucket_tvds[best_bucket]),
        })

    if len(per_occ) < P.MIN_HOLIDAY_OCCURRENCES:
        return None

    tvd_values = [o["best_tvd"] for o in per_occ]
    tvd_mean = float(np.mean(tvd_values))
    tvd_max = float(np.max(tvd_values))

    bucket_votes = {}
    for o in per_occ:
        bucket_votes[o["best_bucket"]] = bucket_votes.get(o["best_bucket"], 0) + 1
    modal_bucket = max(bucket_votes, key=bucket_votes.get)
    modal_count = bucket_votes[modal_bucket]
    majority = max(2, len(per_occ) // 2 + 1)

    # Decision cascade
    if (tvd_mean < P.TVD_CLOSE and tvd_max < P.TVD_CLOSE + 0.03
            and modal_count >= majority):
        decision = f"USE_{modal_bucket.upper()}"
        reason = "consistent_close_match"
    elif tvd_mean < P.TVD_BORDERLINE and modal_count >= majority:
        decision = f"USE_{modal_bucket.upper()}"
        reason = "borderline_flag_review"
    elif tvd_mean >= P.TVD_BORDERLINE:
        decision = "USE_HOLIDAY_PROFILE"
        reason = "strong_difference"
    else:
        decision = "USE_HOLIDAY_PROFILE"
        reason = "inconsistent_match"

    # Recency-weighted holiday profile: more recent occurrences carry more weight
    n = len(p_hol_shares)
    weights = np.linspace(1.0, 1.5, n)
    weights /= weights.sum()
    holiday_profile = np.average(p_hol_shares, axis=0, weights=weights)

    return {
        "holiday": holiday_name,
        "day_offset": day_offset,
        "n_occurrences": len(per_occ),
        "per_occ": per_occ,
        "tvd_mean": tvd_mean,
        "tvd_max": tvd_max,
        "modal_bucket": modal_bucket,
        "modal_count": modal_count,
        "decision": decision,
        "reason": reason,
        "holiday_profile_shares": holiday_profile.tolist(),
    }


def plot_comparison(result: dict, date_to_hours: dict, holiday_dates: set,
                    out_path: Path) -> None:
    """Hourly chart showing each occurrence + recency-weighted holiday avg +
    the best DOW baseline (relative to the most recent occurrence)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for o in result["per_occ"]:
        ax.plot(range(24), o["p_hol"], "-", alpha=0.45, linewidth=1,
                label=f"{o['date']}")

    ax.plot(range(24), result["holiday_profile_shares"], "o-", linewidth=2.5,
            color="C3", label="Holiday avg (recency-weighted)")

    last_occ = pd.Timestamp(result["per_occ"][-1]["date"])
    local = P.compute_local_dow_baselines(last_occ, date_to_hours, holiday_dates)
    best = result["modal_bucket"]
    if local.get(best) is not None:
        ax.plot(range(24), local[best], "s-", linewidth=2, color="C0",
                label=f"{best} baseline (prior 90d before {last_occ.date()})")

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour")
    ax.set_ylabel("share of daily demand")
    ax.set_title(f"{result['holiday']} d{result['day_offset']:+d}  |  "
                 f"TVD_mean={result['tvd_mean']:.3f}  |  "
                 f"decision={result['decision']}  ({result['reason']})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def main() -> int:
    print("=== Hourly pattern analyzer (long_history) ===\n")
    df = P.load_hourly()
    print(f"  Loaded {len(df):,} hourly rows  "
          f"({df.date.min().date()} .. {df.date.max().date()})")

    date_to_hours = P.date_to_hourly_array(df)
    holiday_dates = P.get_holiday_window_dates()
    print(f"  Identified {len(holiday_dates)} holiday-window dates to exclude "
          "from DOW baselines\n")

    results = []
    for holiday in HOLIDAYS_TO_ANALYZE:
        ws, we = HOLIDAY_WINDOWS[holiday]
        for offset in range(ws, we + 1):
            r = analyze_one_cell(holiday, offset, date_to_hours, holiday_dates)
            if r is None:
                continue
            results.append(r)
            print(f"  {r['holiday']:12s} d{r['day_offset']:+d}  "
                  f"n={r['n_occurrences']}  "
                  f"tvd_mean={r['tvd_mean']:.3f}  "
                  f"tvd_max={r['tvd_max']:.3f}  "
                  f"modal={r['modal_bucket']:8s} -> {r['decision']:24s} "
                  f"({r['reason']})")

    if not results:
        print("\n  [WARN] No (holiday, day_offset) cells had enough data to decide.")
        print(f"  Need at least {P.MIN_HOLIDAY_OCCURRENCES} occurrences with full "
              "24-hour data per cell. This is EXPECTED against the 10-day sample "
              "file -- it has zero holiday coverage. Not an error.")
        # Still write empty output files so downstream steps (split.py) have
        # something to read rather than crashing on a missing file.
        pd.DataFrame(columns=["holiday", "day_offset", "n_occurrences", "tvd_mean",
                              "tvd_max", "modal_bucket", "modal_count", "decision",
                              "reason"]).to_csv(P.DERIVED / "decisions.csv", index=False)
        pd.DataFrame(columns=["holiday", "day_offset", "hour", "share"]).to_csv(
            P.DERIVED / "holiday_profiles.csv", index=False)
        return 0

    # 1) Decisions CSV
    rows = []
    max_occ = max(r["n_occurrences"] for r in results)
    for r in results:
        row = {
            "holiday": r["holiday"], "day_offset": r["day_offset"],
            "n_occurrences": r["n_occurrences"],
            "tvd_mean": round(r["tvd_mean"], 4), "tvd_max": round(r["tvd_max"], 4),
            "modal_bucket": r["modal_bucket"], "modal_count": r["modal_count"],
            "decision": r["decision"], "reason": r["reason"],
        }
        for i in range(max_occ):
            if i < len(r["per_occ"]):
                o = r["per_occ"][i]
                row[f"occ{i+1}_date"] = o["date"]
                row[f"occ{i+1}_best_bucket"] = o["best_bucket"]
                row[f"occ{i+1}_best_tvd"] = round(o["best_tvd"], 4)
            else:
                row[f"occ{i+1}_date"] = ""
                row[f"occ{i+1}_best_bucket"] = ""
                row[f"occ{i+1}_best_tvd"] = ""
        rows.append(row)

    decisions_csv = P.DERIVED / "decisions.csv"
    pd.DataFrame(rows).to_csv(decisions_csv, index=False)
    print(f"\n  -> {decisions_csv}")

    # 2) Holiday profiles CSV (long format: one row per hour per cell)
    prof_rows = []
    for r in results:
        for h, share in enumerate(r["holiday_profile_shares"]):
            prof_rows.append({"holiday": r["holiday"], "day_offset": r["day_offset"],
                              "hour": h, "share": round(float(share), 6)})
    profiles_csv = P.DERIVED / "holiday_profiles.csv"
    pd.DataFrame(prof_rows).to_csv(profiles_csv, index=False)
    print(f"  -> {profiles_csv}")

    # 3) Side-by-side plots for non-trivial decisions
    plot_dir = P.DERIVED / "pattern_plots"
    plot_dir.mkdir(exist_ok=True)
    flagged = [r for r in results
               if r["reason"] in ("borderline_flag_review", "strong_difference",
                                  "inconsistent_match")]
    for r in flagged:
        fname = (f"{r['holiday']}_d{r['day_offset']:+d}.png"
                 .replace("+", "p").replace("-", "m"))
        plot_comparison(r, date_to_hours, holiday_dates, plot_dir / fname)
    print(f"  -> {len(flagged)} comparison plot(s) under {plot_dir}")

    print(f"\n=== Done -- review {decisions_csv} and PNGs in {plot_dir} ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it against the sample data to confirm graceful handling of insufficient holiday data**

```bash
cp long_history/hourly/sample_hourly_data.csv long_history/data/raw/hourly_demand.csv
uv run python long_history/hourly/analyze.py
```
Expected: prints `[WARN] No (holiday, day_offset) cells had enough data to decide` (the 10-day sample has zero holiday occurrences, and `MIN_HOLIDAY_OCCURRENCES=2` can't be met) — exit code 0, not a crash. Confirms `long_history/output/hourly/decisions.csv` and `holiday_profiles.csv` were written (empty, header-only).

```bash
rm long_history/data/raw/hourly_demand.csv   # remove the temp copy; real data lands here later
```

- [ ] **Step 3: Commit**

```bash
git add long_history/hourly/analyze.py
git commit -m "Add hourly/analyze.py: quarterly holiday-vs-DOW decision builder"
```

---

### Task 3: Port `split.py` (daily splitter) + conservation/leakage checks

**Files:**
- Create: `long_history/hourly/split.py`
- Modify: `long_history/hourly/checks.py` — append two check functions

**Interfaces:**
- Consumes: `patterns.py` (Task 1), reads `long_history/output/hourly/decisions.csv` and `holiday_profiles.csv` (Task 2's output)
- Produces: `long_history/output/run_<date>/predictions_hourly.csv`
  - `split_day(date, p10, p50, p90, decisions_lookup, profile_lookup, local_baselines) -> tuple[list[dict], str]` — returns (24 output rows, strategy label). Extracted from the original prototype's inline loop so `checks.py` can call it directly instead of running the whole CLI.

- [ ] **Step 1: Write `long_history/hourly/split.py`**

```python
"""split.py -- split a daily forecast into hourly demand using the
per-(holiday, day_offset) decisions from analyze.py and the most-recent
prior-90d local DOW baseline for non-holiday days.

Reads:
    long_history/output/run_<date>/predictions.csv
    long_history/output/hourly/decisions.csv
    long_history/output/hourly/holiday_profiles.csv
    long_history/data/raw/hourly_demand.csv

Writes:
    long_history/output/run_<date>/predictions_hourly.csv

Usage:
    uv run python long_history/hourly/split.py --run-date 2026-06-09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
import patterns as P  # noqa: E402


def _get_dow_baseline(date: pd.Timestamp, local_baselines: dict) -> tuple:
    """Return (24-vector, bucket_name) for this date's DOW bucket."""
    dow = date.weekday()
    for bucket, dows in P.DOW_BUCKETS.items():
        if dow in dows:
            return local_baselines.get(bucket), bucket
    return None, None


def split_day(date: pd.Timestamp, p10: float, p50: float, p90: float,
             decisions_lookup: dict, profile_lookup: dict,
             local_baselines: dict) -> tuple[list, str]:
    """Split one day's (p10, p50, p90) into 24 hourly rows.

    Returns (rows, strategy). rows is [] if no usable share vector was found
    (caller should warn and skip -- never silently emit garbage).
    """
    share_vec = None
    strategy = None

    holiday, offset = P.find_holiday_membership(date)
    if holiday is not None and (holiday, offset) in decisions_lookup:
        decision = decisions_lookup[(holiday, offset)]["decision"]
        if decision == "USE_HOLIDAY_PROFILE":
            share_vec = profile_lookup.get((holiday, offset))
            strategy = f"{holiday}_d{offset:+d}_holiday_profile"
        elif decision.startswith("USE_"):
            bucket = decision.replace("USE_", "").lower()
            share_vec = local_baselines.get(bucket)
            strategy = f"{holiday}_d{offset:+d}_use_{bucket}"

    if share_vec is None:
        share_vec, bucket = _get_dow_baseline(date, local_baselines)
        strategy = f"dow_{bucket}" if bucket else "missing"

    if share_vec is None or np.isnan(share_vec).any():
        return [], strategy

    rows = []
    for h in range(24):
        s = float(share_vec[h])
        rows.append({"date": date.date().isoformat(), "hour": h,
                     "p10": round(p10 * s, 4), "p50": round(p50 * s, 4),
                     "p90": round(p90 * s, 4), "strategy": strategy})
    return rows, strategy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-date", type=str, required=True,
                        help="ISO date matching output/run_<YYYYMMDD>/")
    args = parser.parse_args()

    run_date = pd.Timestamp(args.run_date).normalize()
    stamp = run_date.strftime("%Y%m%d")
    run_dir = ROOT / C.OUT_DIR / f"run_{stamp}"
    preds_path = run_dir / "predictions.csv"
    if not preds_path.exists():
        print(f"ERROR: {preds_path} not found. Run training/forecast.py first.")
        return 1

    preds = pd.read_csv(preds_path, parse_dates=["date"])
    print(f"  loaded daily forecast: {len(preds)} day(s) from {preds_path}")

    decisions_csv = P.DERIVED / "decisions.csv"
    profiles_csv = P.DERIVED / "holiday_profiles.csv"
    if not decisions_csv.exists() or not profiles_csv.exists():
        print(f"ERROR: missing {decisions_csv} or {profiles_csv}.")
        print("       Run hourly/analyze.py first.")
        return 1

    decisions = pd.read_csv(decisions_csv)
    decisions_lookup = decisions.set_index(["holiday", "day_offset"]).to_dict("index")
    print(f"  loaded {len(decisions)} decision row(s) from {decisions_csv.name}")

    profiles = pd.read_csv(profiles_csv)
    profile_lookup = {}
    for (hol, off), g in profiles.groupby(["holiday", "day_offset"]):
        profile_lookup[(hol, int(off))] = g.sort_values("hour")["share"].to_numpy()
    print(f"  loaded {len(profile_lookup)} holiday-profile cell(s)")

    df_hourly = P.load_hourly()
    date_to_hours = P.date_to_hourly_array(df_hourly)
    holiday_dates = P.get_holiday_window_dates()
    local_baselines = P.compute_local_dow_baselines(
        run_date + pd.Timedelta(days=1), date_to_hours, holiday_dates)

    missing = [b for b, v in local_baselines.items() if v is None]
    if missing:
        print(f"  [WARN] insufficient prior-90d samples for DOW bucket(s): {missing}")
    print(f"  computed local DOW baselines for {run_date.date()} reference")

    out_rows = []
    strategy_count = {}
    for _, row in preds.iterrows():
        rows, strategy = split_day(row["date"], float(row["p10"]), float(row["p50"]),
                                   float(row["p90"]), decisions_lookup, profile_lookup,
                                   local_baselines)
        if not rows:
            print(f"  [WARN] {row['date'].date()}: no share vector ({strategy}); skipping")
            continue
        strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
        out_rows.extend(rows)

    out_df = pd.DataFrame(out_rows)
    out_path = run_dir / "predictions_hourly.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  -> {out_path}: {len(out_df):,} row(s)")

    print("\n  Strategy summary (days assigned to each pattern):")
    for strat, n in sorted(strategy_count.items(), key=lambda kv: -kv[1]):
        print(f"    {strat:40s} {n:3d}")

    if not out_df.empty:
        check = (out_df.groupby("date")["p50"].sum().reset_index()
                 .merge(preds.assign(date=preds.date.dt.date.astype(str)),
                        on="date", suffixes=("_hourly_sum", "_daily")))
        diffs = (check["p50_hourly_sum"] - check["p50_daily"]).abs()
        bad = check[diffs > 1.0]
        if len(bad):
            print(f"\n  [WARN] {len(bad)} day(s) where hourly p50 sum != daily p50 "
                  f"(beyond rounding). Investigate.")

    print(f"\n=== Done -- open {out_path} ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Append conservation + leakage checks to `checks.py`**

Add these imports near the top (below the existing `import patterns as P`):

```python
import split as SPLIT  # noqa: E402
```

Add these two functions, and add both to the `checks` list inside `run_all()`:

```python
def check_split_conservation() -> tuple[bool, str]:
    """The 24 hourly p50s produced for a day must sum back to that day's
    daily p50 -- a split that doesn't conserve the total is silently wrong."""
    share_vec = np.array([1 / 24] * 24)  # a flat day, trivial to check by hand
    decisions_lookup, profile_lookup = {}, {}
    local_baselines = {"weekday": share_vec, "friday": share_vec,
                       "saturday": share_vec, "sunday": share_vec}
    rows, strategy = SPLIT.split_day(pd.Timestamp("2024-01-03"), 100.0, 200.0, 300.0,
                                     decisions_lookup, profile_lookup, local_baselines)
    total_p50 = sum(r["p50"] for r in rows)
    ok = abs(total_p50 - 200.0) < 0.01
    return ok, f"24-hour p50 sum = {total_p50:.2f} (want 200.00), strategy={strategy}"


def check_split_no_future_leakage() -> tuple[bool, str]:
    """split.py must build its local DOW baseline using run_date+1 as the
    reference -- i.e. it may see run_date, but nothing after it."""
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="D")
    date_to_hours = {}
    for d in dates:
        arr = np.full(24, 1.0)
        if d > pd.Timestamp("2024-05-01"):   # strictly AFTER the run date
            arr[0] = 999.0
        date_to_hours[d] = arr
    run_date = pd.Timestamp("2024-05-01")
    baselines = P.compute_local_dow_baselines(run_date + pd.Timedelta(days=1),
                                              date_to_hours, holiday_dates=set())
    bad = [b for b, v in baselines.items() if v is not None and v[0] > 10]
    ok = not bad
    return ok, (f"future-dated buckets leaked into the split baseline: {bad}" if bad
               else "no day after run_date reached the split's baseline")
```

Update `run_all()`'s `checks` list to include both new functions.

- [ ] **Step 3: Run it — expect a failure first if the import is wrong, then pass**

Run: `uv run python long_history/hourly/checks.py`
Expected: `6/6 checks passed`

- [ ] **Step 4: Commit**

```bash
git add long_history/hourly/split.py long_history/hourly/checks.py
git commit -m "Add hourly/split.py: daily-to-hourly splitter, with conservation and leakage checks"
```

---

### Task 4: `comparison_model.py` — the strawman regressor

**Files:**
- Create: `long_history/hourly/comparison_model.py`
- Modify: `long_history/hourly/checks.py` — append one check function

**Interfaces:**
- Consumes: `patterns.py` (Task 1)
- Produces (used by Task 5):
  - `FEATURE_COLS: list[str]`
  - `build_hourly_feature_matrix(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame` — one row per (date, hour), columns `FEATURE_COLS + ["date", "hour", "share"]`
  - `train_comparison_model(feat: pd.DataFrame, train_mask: pd.Series) -> lightgbm.LGBMRegressor`
  - `predict_day_shares(model, feat: pd.DataFrame, date: pd.Timestamp) -> np.ndarray` — always returns a 24-vector that sums to 1.0 (normalized)

- [ ] **Step 1: Write `long_history/hourly/comparison_model.py`**

```python
"""comparison_model.py -- a lightweight regressor, used ONLY inside
validate.py to give the empirical-distribution method's "good enough" claim
a real reference point.

NEVER used to produce an actual forecast. One LightGBM regressor predicting
a single hour's SHARE of the daily total, from: hour-of-day, day-of-week,
month, a holiday flag + day-offset-within-holiday-window, and the
prior-N-days average share for that same hour (an honest lag feature -- it
only reads days strictly before the row's own date, same discipline as
every other lag in this project).

Deliberately small and untuned: its only job is being a fair "the
alternative you'd otherwise have to build," not a competitive product.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import patterns as P  # noqa: E402

FEATURE_COLS = ["hour", "dow", "month", "is_holiday", "holiday_offset", "lag_share_90d"]

LGBM_PARAMS = dict(
    objective="regression", n_estimators=200, learning_rate=0.05,
    num_leaves=15, max_depth=4, min_child_samples=10,
    n_jobs=-1, verbose=-1, random_state=123,
)


def build_hourly_feature_matrix(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    """df: long-format date,hour,demand (from patterns.load_hourly()).

    Returns one row per (date, hour) with FEATURE_COLS + date, hour, share.
    Days with fewer than 24 hours, or a zero/NaN daily total, are dropped --
    their share is undefined.
    """
    date_to_hours = P.date_to_hourly_array(df)
    rows = []
    for d, hours in date_to_hours.items():
        if np.isnan(hours).any():
            continue
        shares = P.share_vector(hours)
        if np.isnan(shares).any():
            continue

        # Honest lag feature: mean share for THIS hour over the prior 90
        # non-holiday days, same window discipline as compute_local_dow_baselines.
        prior_shares = {h: [] for h in range(24)}
        window_start = d - pd.Timedelta(days=P.LOCAL_WINDOW_DAYS)
        for d2, hours2 in date_to_hours.items():
            if d2 < window_start or d2 >= d:      # strictly before d
                continue
            if d2 in holiday_dates or np.isnan(hours2).any():
                continue
            s2 = P.share_vector(hours2)
            if np.isnan(s2).any():
                continue
            for h in range(24):
                prior_shares[h].append(s2[h])

        is_hol = d in holiday_dates
        hol_name, hol_offset = P.find_holiday_membership(d) if is_hol else (None, None)

        for h in range(24):
            lag = float(np.mean(prior_shares[h])) if prior_shares[h] else np.nan
            rows.append({
                "date": d, "hour": h, "dow": d.weekday(), "month": d.month,
                "is_holiday": int(is_hol),
                "holiday_offset": hol_offset if hol_offset is not None else 999,
                "lag_share_90d": lag,
                "share": float(shares[h]),
            })
    return pd.DataFrame(rows)


def train_comparison_model(feat: pd.DataFrame, train_mask: pd.Series) -> lgb.LGBMRegressor:
    """Fits one LightGBM regressor to predict `share` from FEATURE_COLS,
    using only rows where train_mask is True."""
    sub = feat[train_mask].dropna(subset=FEATURE_COLS + ["share"])
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(sub[FEATURE_COLS], sub["share"])
    return model


def predict_day_shares(model: lgb.LGBMRegressor, feat: pd.DataFrame,
                       date: pd.Timestamp) -> np.ndarray:
    """Predicts and NORMALIZES a 24-vector of shares (sums to 1.0) for `date`.

    The model predicts each hour independently, so raw outputs will not sum
    to exactly 1 -- normalizing here keeps the comparison fair: both methods'
    final step is "a 24-vector that sums to 1, applied to the daily total."
    """
    day_rows = feat[feat["date"] == date].sort_values("hour")
    if len(day_rows) != 24 or day_rows[FEATURE_COLS].isna().any().any():
        return np.full(24, np.nan)
    raw = model.predict(day_rows[FEATURE_COLS])
    raw = np.clip(raw, 0, None)   # a negative predicted share is meaningless
    total = raw.sum()
    if total <= 0:
        return np.full(24, np.nan)
    return raw / total
```

- [ ] **Step 2: Append the comparison-model check to `checks.py`**

Add near the top:
```python
import comparison_model as CM  # noqa: E402
```

Add:
```python
def check_comparison_model_shares_sum_to_one() -> tuple[bool, str]:
    """predict_day_shares must always return a normalized 24-vector."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame([
        {"date": d, "hour": h, "demand": max(1.0, 100 + 20 * np.sin(h / 3) + rng.normal(0, 5))}
        for d in dates for h in range(24)
    ])
    feat = CM.build_hourly_feature_matrix(df, holiday_dates=set())
    train_mask = feat["date"] < dates[30]
    model = CM.train_comparison_model(feat, train_mask)
    test_date = dates[35]
    shares = CM.predict_day_shares(model, feat, test_date)
    ok = not np.isnan(shares).any() and abs(shares.sum() - 1.0) < 1e-6
    return ok, f"predicted shares for {test_date.date()} sum to {shares.sum():.6f}"
```

Add `check_comparison_model_shares_sum_to_one` to the `checks` list in `run_all()`.

- [ ] **Step 3: Run it**

Run: `uv run python long_history/hourly/checks.py`
Expected: `7/7 checks passed`

- [ ] **Step 4: Commit**

```bash
git add long_history/hourly/comparison_model.py long_history/hourly/checks.py
git commit -m "Add hourly/comparison_model.py: lightweight strawman used only in validation"
```

---

### Task 5: `validate.py` — the leave-one-out honest backtest

**Files:**
- Create: `long_history/hourly/validate.py`

**Interfaces:**
- Consumes: `patterns.py`, `analyze.analyze_one_cell(..., as_of=...)`, `comparison_model.py`
- Produces: `long_history/output/hourly/validation_report.csv` (per date/hour/method), `long_history/output/hourly/validation_summary.csv` (per day-type, per method)

- [ ] **Step 1: Write `long_history/hourly/validate.py`**

```python
"""validate.py -- honest, leakage-free validation of the hourly-splitting
approach: does the empirical distribution method actually beat, or at least
match, a lightweight comparison model?

METHOD
------
Pick a holdout window (the last `--holdout-days` days with full 24-hour
data). Everything before the holdout trains the comparison model ONCE (not
refit per day -- see comparison_model.py's docstring for why). The empirical
method needs no training, so for EACH day inside the holdout it is
recomputed walk-forward, using only data strictly before that day:
    - the local DOW baseline (compute_local_dow_baselines, already leakage-
      safe by construction)
    - the holiday decision cascade (analyze.analyze_one_cell(..., as_of=
      day-1), which now supports an as_of cutoff for exactly this purpose)

Both methods' predicted share-vectors are applied to that day's ACTUAL daily
total (not a forecast total) -- this isolates hourly-split error from
daily-forecast error, which is a separately measured concern.

STATUS
------
Run today against long_history/hourly/sample_hourly_data.csv, this is a
MECHANISM CHECK ONLY -- 10 days, zero holiday coverage, no statistical
power. The numbers it prints are NOT evidence that the empirical method is
"good enough." Real validation requires long_history/data/raw/hourly_demand.csv
with real history (18-24+ months, multiple holiday occurrences per cell) --
this script does not change when that data lands; you just point it there
and the numbers become real.

Usage:
    uv run python long_history/hourly/validate.py
    uv run python long_history/hourly/validate.py --holdout-days 90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))

import patterns as P  # noqa: E402
import comparison_model as CM  # noqa: E402
from analyze import analyze_one_cell  # noqa: E402


def classify_day_type(is_holiday: bool, decision: str | None, dow: int) -> str:
    if not is_holiday:
        return "ordinary_weekday" if dow < 4 else "friday_saturday_sunday"
    if decision == "USE_HOLIDAY_PROFILE":
        return "holiday_specific_profile"
    return "holiday_inherited_dow"


def empirical_prediction(date: pd.Timestamp, date_to_hours: dict,
                         holiday_dates: set) -> tuple[np.ndarray | None, str]:
    """Walk-forward empirical-method prediction for `date`, using only data
    strictly before it. Returns (24-share-vector or None, day_type)."""
    local = P.compute_local_dow_baselines(date, date_to_hours, holiday_dates)
    hol_name, hol_offset = P.find_holiday_membership(date)

    if hol_name is None:
        bucket = "weekday" if date.weekday() < 4 else \
                 ["friday", "saturday", "sunday"][date.weekday() - 4]
        return local.get(bucket), classify_day_type(False, None, date.weekday())

    result = analyze_one_cell(hol_name, hol_offset, date_to_hours, holiday_dates,
                              as_of=date - pd.Timedelta(days=1))
    if result is None:
        # not enough PRIOR occurrences yet to decide -- fall through to DOW,
        # same fallback split.py uses
        bucket = "weekday" if date.weekday() < 4 else \
                 ["friday", "saturday", "sunday"][date.weekday() - 4]
        return local.get(bucket), classify_day_type(False, None, date.weekday())

    day_type = classify_day_type(True, result["decision"], date.weekday())
    if result["decision"] == "USE_HOLIDAY_PROFILE":
        return np.array(result["holiday_profile_shares"]), day_type
    bucket = result["modal_bucket"]
    return local.get(bucket), day_type


def mape_tvd(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    mask = actual > 0
    if not mask.any():
        return np.nan, np.nan
    ape = np.abs(pred[mask] - actual[mask]) / actual[mask]
    return float(np.mean(ape)) * 100, P.tvd(pred, P.share_vector(actual))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-days", type=int, default=90,
                    help="how many of the most recent full days to validate on")
    a = ap.parse_args()

    print("=" * 78)
    print("HOURLY SPLIT VALIDATION")
    print("=" * 78)

    df = P.load_hourly()
    date_to_hours = P.date_to_hourly_array(df)
    holiday_dates = P.get_holiday_window_dates()
    full_dates = sorted(d for d, h in date_to_hours.items() if not np.isnan(h).any())

    holdout_days = min(a.holdout_days, max(1, len(full_dates) // 3))
    if holdout_days < a.holdout_days:
        print(f"  [note] only {len(full_dates)} full days available; shrinking "
              f"--holdout-days {a.holdout_days} -> {holdout_days}")
    holdout = full_dates[-holdout_days:]
    train_end = holdout[0]
    print(f"  data: {len(full_dates)} full days ({full_dates[0].date()} .. "
          f"{full_dates[-1].date()})")
    print(f"  holdout: {len(holdout)} days ({holdout[0].date()} .. {holdout[-1].date()})")

    if len(holiday_dates & set(holdout)) == 0:
        print("  [note] holdout window contains ZERO holiday days -- the "
              "holiday_* segments below will be empty. This is expected "
              "against the sample file; not an error.")

    # Train the comparison model ONCE, on everything before the holdout.
    feat = CM.build_hourly_feature_matrix(df, holiday_dates)
    train_mask = feat["date"] < train_end
    n_train = int(train_mask.sum())
    if n_train < 24 * 5:
        print(f"  [WARN] only {n_train} training rows for the comparison model "
              f"({n_train // 24} full days) -- far below what a real model "
              "needs. Proceeding anyway; treat its numbers as illustrative only.")
    model = CM.train_comparison_model(feat, train_mask) if n_train >= 24 else None

    rows = []
    for date in holdout:
        actual = date_to_hours[date]
        emp_shares, day_type = empirical_prediction(date, date_to_hours, holiday_dates)
        cmp_shares = (CM.predict_day_shares(model, feat, date)
                     if model is not None else np.full(24, np.nan))

        for method, shares in (("empirical", emp_shares), ("comparison_model", cmp_shares)):
            if shares is None or np.isnan(shares).any():
                rows.append({"date": date, "method": method, "day_type": day_type,
                            "mape": np.nan, "tvd": np.nan, "n_hours": 0})
                continue
            mape, tvd_ = mape_tvd(shares, actual)
            rows.append({"date": date, "method": method, "day_type": day_type,
                        "mape": mape, "tvd": tvd_, "n_hours": 24})

    report = pd.DataFrame(rows)
    P.DERIVED.mkdir(parents=True, exist_ok=True)
    report_path = P.DERIVED / "validation_report.csv"
    report.to_csv(report_path, index=False)

    summary = (report.dropna(subset=["mape"])
              .groupby(["day_type", "method"])["mape"]
              .agg(["mean", "count"]).reset_index()
              .rename(columns={"mean": "mape_pct", "count": "n_days"}))
    summary_path = P.DERIVED / "validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("RESULTS BY DAY TYPE  (MECHANISM CHECK ONLY IF RUN AGAINST THE SAMPLE "
          "FILE -- see module docstring)")
    print("=" * 78)
    if summary.empty:
        print("  No scoreable days -- every prediction was missing data. "
              "Expected on a very short/sparse hourly file.")
    else:
        for day_type in sorted(summary.day_type.unique()):
            sub = summary[summary.day_type == day_type]
            print(f"\n  {day_type}:")
            for _, r in sub.iterrows():
                print(f"    {r['method']:18s} MAPE {r['mape_pct']:6.2f}%  "
                      f"(n={int(r['n_days'])} day-method rows)")

    print(f"\n-> {report_path}")
    print(f"-> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the mechanism smoke test against the sample data**

```bash
cp long_history/hourly/sample_hourly_data.csv long_history/data/raw/hourly_demand.csv
uv run python long_history/hourly/validate.py --holdout-days 3
```
Expected: runs to completion without a traceback, exit code 0. Prints the `[note]` about zero holiday coverage, prints a results table for `ordinary_weekday` / `friday_saturday_sunday` segments (values will look poor/noisy — 10 days total is far too little data — this is fine, the goal is confirming the pipeline runs, not a good number), writes `long_history/output/hourly/validation_report.csv` and `validation_summary.csv`.

```bash
rm long_history/data/raw/hourly_demand.csv
```

- [ ] **Step 3: Commit**

```bash
git add long_history/hourly/validate.py
git commit -m "Add hourly/validate.py: leakage-free empirical-vs-comparison-model backtest"
```

---

### Task 6: Wire into `run_pipeline.py` as opt-in steps

**Files:**
- Modify: `long_history/run_pipeline.py`

**Interfaces:**
- Consumes: `long_history/hourly/analyze.py`, `split.py` as subprocess targets (same pattern as the existing `SCRIPTS` dict)

- [ ] **Step 1: Add two entries to `SCRIPTS` and `ALL_STEPS`**

In `long_history/run_pipeline.py`, find the existing block:
```python
SCRIPTS = {
    "validate":   "data/validate.py",
    "weather":    "data/scrape_typhoon.py",
    "experiment": "training/experiment.py",
    "evaluate":   "training/evaluate.py",
    "forecast":   "training/forecast.py",
}
ALL_STEPS = ["validate", "weather", "experiment", "evaluate", "forecast"]
DEFAULT_STEPS = ["validate", "evaluate", "forecast"]
```

Replace with:
```python
SCRIPTS = {
    "validate":       "data/validate.py",
    "weather":        "data/scrape_typhoon.py",
    "experiment":     "training/experiment.py",
    "evaluate":       "training/evaluate.py",
    "forecast":       "training/forecast.py",
    "hourly-analyze": "hourly/analyze.py",
    "hourly-split":   "hourly/split.py",
}
ALL_STEPS = ["validate", "weather", "experiment", "evaluate", "forecast",
            "hourly-analyze", "hourly-split"]
# NOT in DEFAULT_STEPS: long_history/data/raw/hourly_demand.csv does not
# exist yet, so these would fail on a routine run. Opt in explicitly with
# --steps ... hourly-analyze hourly-split once real hourly data lands.
DEFAULT_STEPS = ["validate", "evaluate", "forecast"]
```

- [ ] **Step 2: Add dispatch branches**

Find the `for step in a.steps:` block's `elif step == "forecast":` branch and add two new branches immediately after it, before the `else: continue`:

```python
        elif step == "hourly-analyze":
            rc = run([SCRIPTS["hourly-analyze"]], "hourly pattern analysis")
        elif step == "hourly-split":
            cmd = [SCRIPTS["hourly-split"], "--run-date",
                  a.run_date or eval_end]
            rc = run(cmd, "hourly split")
```

- [ ] **Step 3: Verify the new steps are recognized (data doesn't exist yet, so expect a clean, informative failure — not a crash)**

Run: `uv run python long_history/run_pipeline.py --steps hourly-analyze`
Expected: the `validate` gate is not in `--steps`, so this runs `hourly-analyze` directly; `analyze.py` will fail with a clear `FileNotFoundError`-style message pointing at the missing `data/raw/hourly_demand.csv` (from `patterns.load_hourly()`'s `pd.read_csv`), and `run_pipeline.py` reports it as `!! hourly pattern analysis FAILED` and stops — not a Python traceback dump. Confirms the wiring is correct even though the step can't succeed until real data exists.

- [ ] **Step 4: Commit**

```bash
git add long_history/run_pipeline.py
git commit -m "Wire hourly-analyze / hourly-split as opt-in run_pipeline.py steps"
```

---

### Task 7: Documentation

**Files:**
- Create: `long_history/hourly/README.md`
- Create: `long_history/hourly/TUTORIAL.md` (standalone, per explicit request — not merged into `long_history/docs/TUTORIAL.md`)
- Modify: `long_history/docs/COMPLETE_GUIDE.md` — new §18
- Modify: `long_history/README.md` — one-line pointer

- [ ] **Step 1: Write `long_history/hourly/README.md`**

```markdown
# Hourly demand splitting

Splits the 28-day daily forecast produced by `training/forecast.py` into
hourly demand, using the actual historical hourly distribution — not a
trained model. For each forecast day, picks between:

1. **A locally-current DOW pattern** — built from the prior 90 non-holiday
   days relative to the run date (weekday / Friday / Saturday / Sunday).
2. **A holiday-specific pattern** — a recency-weighted average of past
   occurrences of that exact holiday-day (e.g. CNY day +1 across all years).

The choice is data-driven: `analyze.py` compares each historical
holiday-day against its own contemporary DOW baselines using Total
Variation Distance (TVD). When the holiday looks indistinguishable from a
normal DOW, the system inherits the DOW pattern; otherwise it uses the
holiday-specific pattern. Full detail: [TUTORIAL.md](TUTORIAL.md).

## Status: pipeline complete, validation pending real data

⚠️ **`long_history/data/raw/hourly_demand.csv` does not exist yet.** Every
script here runs and has been smoke-tested against
`sample_hourly_data.csv` (10 days, format reference only — the README
for that file is explicit that it is NOT enough data to run the
analyzer meaningfully). Nothing here has been validated against real
hourly history. See `validate.py`'s module docstring and
[COMPLETE_GUIDE.md §18](../docs/COMPLETE_GUIDE.md) for what "validated"
will mean once real data lands.

## Files

| File | Purpose |
|---|---|
| `patterns.py` | Shared functions: data loading, TVD, local DOW baselines, holiday membership |
| `analyze.py` | Quarterly: runs the TVD analysis, emits decisions and plots |
| `split.py` | Daily: applies decisions + current local DOW baseline to a saved forecast |
| `comparison_model.py` | A lightweight regressor, used ONLY inside `validate.py` — never produces an actual forecast |
| `validate.py` | Leakage-free backtest: empirical method vs the comparison model, by day-type |
| `checks.py` | Assert-based sanity checks (conservation, leakage, share-vector correctness) |
| `sample_hourly_data.csv` | 10-day format reference for `data/raw/hourly_demand.csv` — NOT sufficient for real validation |

## Independence

Holiday dates come from `long_history/engine/core.py` — the extended,
corrected 2016-2030 set — not `v2/_shared.py`. Nothing here imports from
outside `long_history/`.

## Quick start

See [TUTORIAL.md](TUTORIAL.md).
```

- [ ] **Step 2: Write `long_history/hourly/TUTORIAL.md`**

```markdown
# Tutorial — hourly demand splitting

Step-by-step commands for the hourly module. For *why* it works this way,
read [README.md](README.md) and
[../docs/COMPLETE_GUIDE.md §18](../docs/COMPLETE_GUIDE.md).

All commands run from the **repository root**.

---

## 0. Before you start

⚠️ **You need `long_history/data/raw/hourly_demand.csv`** — real hourly
history, at least 18 months, ideally 24+, covering multiple occurrences of
each major holiday (CNY, Golden Week, Labour, Mid-Autumn). Format:

```csv
date,hour,demand
2024-01-01,0,234
2024-01-01,1,189
...
2024-01-01,23,412
```

Every date needs all 24 hours present. `long_history/hourly/sample_hourly_data.csv`
shows the format but is only 10 days — not enough to run anything
meaningfully. Until the real file exists, the commands below will either
fail with a clear "file not found," or (for `analyze.py` /
`validate.py` against the sample) run as a **mechanism check only** — see
the warnings each script prints.

---

## 1. One-time / quarterly: build the decisions

```bash
uv run python long_history/hourly/analyze.py
```

Produces, under `long_history/output/hourly/`:

| Output | Contents |
|---|---|
| `decisions.csv` | Per `(holiday, day_offset)`: TVD per occurrence, modal best-DOW match, final decision |
| `holiday_profiles.csv` | Recency-weighted hourly shares for cells decided to use holiday-specific patterns |
| `pattern_plots/*.png` | Side-by-side hourly charts for any borderline or holiday-specific cell — **review these manually** |

Re-run quarterly as new holiday samples accumulate.

## 2. Every forecast: split daily into hourly

```bash
# After training/forecast.py has produced output/run_<date>/predictions.csv:
uv run python long_history/hourly/split.py --run-date 2026-06-09
```

Produces `long_history/output/run_20260609/predictions_hourly.csv`:

```csv
date,hour,p10,p50,p90,strategy
2026-06-10,0,12.4,15.8,19.2,dow_weekday
...
2026-07-01,0,18.7,23.5,28.4,CNY_d+1_holiday_profile
```

The `strategy` column shows which pattern was applied — useful when
something looks wrong.

## 3. Validating the approach

```bash
uv run python long_history/hourly/validate.py --holdout-days 90
```

Runs an honest, walk-forward backtest: for each day in the holdout window,
using only data strictly before it, computes what the empirical method
would have predicted and what a lightweight comparison model would have
predicted, applies both to that day's *actual* total, and reports MAPE by
day-type (ordinary weekday, Friday/Saturday/Sunday, holiday-inherited-DOW,
holiday-specific-profile).

Produces, under `long_history/output/hourly/`:
- `validation_report.csv` — one row per (date, method)
- `validation_summary.csv` — MAPE by (day-type, method)

**Read the day-type breakdown, not just the average.** The empirical
method is only "good enough, no complicated model needed" if it holds up
across *every* segment — a good pooled average hiding one bad segment is
not a pass.

## 4. Running the sanity checks

```bash
uv run python long_history/hourly/checks.py
```

Confirms the mechanism itself is sound: shares sum to 1, TVD behaves
correctly, no split or baseline ever reads same-day or future data. Fast,
no real hourly data required — safe to run any time.

## 5. Wiring into the main pipeline

```bash
uv run python long_history/run_pipeline.py --steps hourly-analyze hourly-split
```

Not part of the default `run_pipeline.py` steps (`validate`, `evaluate`,
`forecast`) — opt in explicitly once real hourly data exists.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `FileNotFoundError` on `hourly_demand.csv` | you haven't added real hourly history yet — see §0 |
| `analyze.py` prints "No cells had enough data" | expected until you have ≥2 occurrences per `(holiday, day_offset)` — needs real, multi-year data |
| `split.py` warns "no share vector" for a date | that date's DOW bucket had fewer than 5 non-holiday days in the prior 90 — expected on a short history |
| `validate.py` shows empty holiday segments | your holdout window doesn't contain any holiday days — widen `--holdout-days` or pick a different window |
```

- [ ] **Step 3: Add §18 to `long_history/docs/COMPLETE_GUIDE.md`**

Insert a new section immediately before the existing final section
(renumber that section from whatever number it currently has to one higher
— check the file's current section numbering with `grep -n "^## " long_history/docs/COMPLETE_GUIDE.md`
before inserting, and confirm the new section's number continues that
sequence without a collision):

```markdown
## 18. Hourly demand splitting

Splits each day's forecast into 24 hourly values, using the actual
historical hourly distribution — no trained model does the splitting
itself.

### Why an empirical distribution, not a model

For each forecast day, the split picks between two *measured* shapes: the
average hourly pattern from the last 90 similar (non-holiday) days, or —
for holidays whose shape genuinely differs — a recency-weighted average of
past occurrences of that exact holiday-day. The choice is data-driven,
using Total Variation Distance to check whether a holiday actually looks
different from an ordinary day of that type.

A small comparison model exists (`long_history/hourly/comparison_model.py`)
but is used **only inside the validation script**, never to produce an
actual forecast — its whole purpose is giving the "the empirical approach
is good enough" claim a real number to be measured against, rather than
leaving it asserted.

### Current status: pipeline complete, validation pending real data

`long_history/data/raw/hourly_demand.csv` does not exist yet. Every script
in `long_history/hourly/` runs cleanly and has been smoke-tested against a
10-day format-reference sample — but that is a **mechanism check**, not
evidence of accuracy. Real validation requires real hourly history (18-24+
months, multiple occurrences of each major holiday) and runs with one
command once that data lands: `uv run python long_history/hourly/validate.py`.

See [long_history/hourly/README.md](../hourly/README.md) and
[long_history/hourly/TUTORIAL.md](../hourly/TUTORIAL.md) for full detail.
```

- [ ] **Step 4: Add a pointer in `long_history/README.md`**

In the existing "Layout" section's file tree, add an `hourly/` line after
`viz/`. In the "Documentation" table, add a row pointing at
`hourly/TUTORIAL.md`.

- [ ] **Step 5: Verify all new markdown links resolve**

```bash
uv run python -c "
import re, pathlib
base = pathlib.Path('long_history')
for md in [base/'README.md', base/'hourly/README.md', base/'hourly/TUTORIAL.md',
          base/'docs/COMPLETE_GUIDE.md']:
    t = md.read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)#]+)(#[^)]*)?\)', t):
        target = (md.parent / link[0]).resolve()
        print(f\"  {'OK  ' if target.exists() else 'BAD '} {md} -> {link[0]}\")
"
```
Expected: every line prints `OK`.

- [ ] **Step 6: Commit**

```bash
git add long_history/hourly/README.md long_history/hourly/TUTORIAL.md \
        long_history/docs/COMPLETE_GUIDE.md long_history/README.md
git commit -m "Document the hourly-splitting module: README, standalone tutorial, guide section"
```

---

## Post-plan: what's still deferred

Not part of this plan, intentionally:

- **Real validation.** `validate.py` cannot prove anything until
  `long_history/data/raw/hourly_demand.csv` exists with real history. When
  it does: `uv run python long_history/hourly/validate.py --holdout-days 90`
  (or a larger window) is the whole re-run.
- **Deleting the repo-root `hourly/` prototype.** Left as-is per the
  design doc — revisit once this package is verified against real data.
