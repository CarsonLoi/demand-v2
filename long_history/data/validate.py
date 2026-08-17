"""validate_data.py — hard gate on the demand CSV. Run this FIRST.

The single most damaging failure mode in this model is a gap in the date
series. Lag features are CALENDAR lookups (`T - N days`), not positional ones,
so one missing day silently NaNs out every lag, rolling window and EWMA that
reaches across it -- for up to two years afterwards (lag_728). Production's
`load_demand()` is a bare read_csv and checks none of this.

Exit code 0 = safe to proceed. Non-zero = do not run the experiment.

Usage:
    uv run python long_history/validate_data.py
    uv run python long_history/validate_data.py --file path/to/other.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]          # long_history/ -- this package owns its data
sys.path.insert(0, str(ROOT))
import config as C


def validate(path: Path, verbose: bool = True) -> tuple[int, int]:
    """Returns (n_errors, n_warnings). Errors block; warnings inform."""
    errs, warns = [], []

    if not path.exists():
        print(f"FATAL: {path} does not exist.")
        print(f"       Put your long-history CSV there, or edit DATA_FILE in "
              f"long_history/config.py")
        return 1, 0

    df = pd.read_csv(path)

    # ---- schema
    need = {"date", "demand", "floortables"}
    missing = need - set(df.columns)
    if missing:
        errs.append(f"missing column(s): {sorted(missing)}")
        for e in errs:
            print(f"  ERROR  {e}")
        return len(errs), len(warns)

    # ---- types
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        errs.append(f"date column will not parse as dates ({e})")
        print(f"  ERROR  {errs[-1]}")
        return len(errs), len(warns)

    for col in ("demand", "floortables"):
        before = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lost = before - df[col].notna().sum()
        if lost:
            errs.append(f"{col}: {lost} value(s) are not numeric")

    df = df.sort_values("date").reset_index(drop=True)

    # ---- the big one: calendar contiguity
    full = pd.date_range(df.date.min(), df.date.max())
    gaps = full.difference(df.date)
    if len(gaps):
        errs.append(
            f"{len(gaps)} MISSING DAY(S) in the date range -- lags will break. "
            f"First few: {[str(g.date()) for g in gaps[:5]]}")

    dupes = df.date[df.date.duplicated()].unique()
    if len(dupes):
        errs.append(f"{len(dupes)} duplicate date(s), e.g. "
                    f"{[str(pd.Timestamp(d).date()) for d in dupes[:5]]}")

    # ---- values
    # A day with demand == 0 AND floortables == 0 is a FULL CLOSURE: zero tables
    # open, so it is a supply-zero observation, not a demand observation. Those
    # are expected and are excluded from training targets downstream
    # (config.EXCLUDE_CLOSURES). Only demand <= 0 while tables WERE open is a
    # genuine data error.
    closure = (df.demand <= 0) & (df.floortables <= 0)
    bad_zero = (df.demand <= 0) & (df.floortables > 0)

    if closure.any():
        n = int(closure.sum())
        spans = (df.loc[closure, "date"].dt.to_period("M")
                 .value_counts().sort_index())
        warns.append(
            f"{n} full-closure day(s) (demand=0 AND floortables=0) — expected, "
            f"excluded from training: "
            + ", ".join(f"{p} x{c}" for p, c in spans.items()))
    if bad_zero.any():
        n = int(bad_zero.sum())
        ex = [str(x.date()) for x in df.loc[bad_zero, "date"].head(5)]
        errs.append(f"{n} row(s) with demand <= 0 while tables were OPEN "
                    f"(floortables > 0) — real data error, e.g. {ex}")
    if df.demand.isna().any():
        errs.append(f"{int(df.demand.isna().sum())} row(s) with blank demand")
    if df.floortables.isna().any():
        warns.append(f"{int(df.floortables.isna().sum())} row(s) with blank floortables")

    # ---- structural-break sniff test (informational, never blocking)
    df["regime"] = df.date.map(C.regime_of)
    by_reg = df.groupby("regime").demand.agg(["count", "mean", "min", "max"])

    if verbose:
        print(f"\n=== {path} ===")
        print(f"  rows        : {len(df):,}")
        print(f"  range       : {df.date.min().date()} .. {df.date.max().date()} "
              f"({(df.date.max()-df.date.min()).days + 1:,} calendar days)")
        print(f"  demand      : mean {df.demand.mean():,.0f}  "
              f"min {df.demand.min():,.0f}  max {df.demand.max():,.0f}")
        print(f"\n  by regime (check these look like different regimes to you):")
        print(by_reg.to_string())

        # year-over-year level, to make a regime break visible
        yr = df.groupby(df.date.dt.year).demand.mean()
        print(f"\n  mean demand by year:")
        for y, v in yr.items():
            base = yr.iloc[0]
            bar = "#" * max(1, int(40 * v / yr.max()))
            print(f"    {y}  {v:8,.0f}  {bar}")

        print()
        for w in warns:
            print(f"  WARN   {w}")
        for e in errs:
            print(f"  ERROR  {e}")
        print()
        if errs:
            print(f"  FAILED -- {len(errs)} error(s). Fix before running the experiment.")
            if len(gaps):
                print(f"           Do NOT 'fix' gaps by deleting surrounding rows -- "
                      f"fill them.\n           If a day genuinely had no trading, that is a "
                      f"real zero and\n           belongs in EXCLUDE_FROM_TRAINING, not deleted.")
        else:
            print(f"  PASSED{' with ' + str(len(warns)) + ' warning(s)' if warns else ''} "
                  f"-- safe to run the experiment.")
    return len(errs), len(warns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None, help="CSV to check (default: config.DATA_FILE)")
    a = ap.parse_args()
    path = Path(a.file) if a.file else ROOT / C.DATA_FILE
    n_err, _ = validate(path)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
