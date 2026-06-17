"""Scrape HKO's Tropical Cyclone Warning Signals Database for historical
typhoon records (T8 and above) since a given start year, and save them as
data/raw/typhoons.csv with one row per affected DATE (not per signal-period).

Source: https://www.hko.gov.hk/dps/wxinfo/climat/warndb/tc.dat
(The raw data file backing https://www.hko.gov.hk/en/wxinfo/climat/warndb/warndb1.shtml)

For each calendar day a T8/T9/T10 signal was active for at least part of the
day, we record the HIGHEST signal level reached that day. If a single date
had multiple typhoons named (very rare in practice), they're concatenated.

Output schema (data/raw/typhoons.csv):
    date         ISO YYYY-MM-DD
    typhoon_name uppercase typhoon name(s), e.g. SAOLA, KOINU
    highest_signal one of 8, 9, 10

Usage:
    uv run python scrape_typhoon.py                 # default: 2023-01-01 onward
    uv run python scrape_typhoon.py --since 2020
    uv run python scrape_typhoon.py --min-signal 3  # also include T3+ events
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from datetime import date as _date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA_TYPHOONS = ROOT / "data" / "raw" / "typhoons.csv"
HKO_TC_DATA = "https://www.hko.gov.hk/dps/wxinfo/climat/warndb/tc.dat"


def fetch_raw_data(url: str = HKO_TC_DATA) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; weather-scrape/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    # HKO file uses UTF-8 with BOM
    return raw.decode("utf-8-sig")


def parse_records(text: str) -> pd.DataFrame:
    """Parse the tab-separated HKO file into a DataFrame.

    Each row schema (16 fields):
        record_id type name signal direction
        start_HHMM start_day start_month start_year start_flag
        end_HHMM   end_day   end_month   end_year   end_flag
        duration_HHMM
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 16:
            continue
        try:
            signal = int(parts[3])
        except ValueError:
            continue
        rows.append({
            "record_id":   parts[0],
            "type":        parts[1],
            "name":        parts[2],
            "signal":      signal,
            "direction":   parts[4],
            "start_date":  pd.Timestamp(year=int(parts[8]), month=int(parts[7]),
                                         day=int(parts[6])),
            "end_date":    pd.Timestamp(year=int(parts[13]), month=int(parts[12]),
                                         day=int(parts[11])),
        })
    return pd.DataFrame(rows)


def expand_to_daily(df: pd.DataFrame, since_year: int, min_signal: int) -> pd.DataFrame:
    """For each signal period, enumerate every calendar day it covered.
    Then group by date keeping the highest signal."""
    df = df[(df["start_date"].dt.year >= since_year) & (df["signal"] >= min_signal)].copy()

    per_day = []
    for _, r in df.iterrows():
        for d in pd.date_range(r["start_date"], r["end_date"], freq="D"):
            per_day.append({
                "date": d.normalize(),
                "name": str(r["name"]).strip().upper() if r["name"] else "UNNAMED",
                "signal": int(r["signal"]),
            })

    daily = pd.DataFrame(per_day)
    if daily.empty:
        return daily

    # Group by date: keep highest signal; concatenate distinct names if any
    out = (daily.groupby("date", as_index=False)
           .agg(typhoon_name=("name", lambda s: ",".join(sorted(set(s)))),
                highest_signal=("signal", "max")))
    out = out.sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--since", type=int, default=2023,
                        help="Start year (inclusive). Default 2023.")
    parser.add_argument("--min-signal", type=int, default=8,
                        help="Minimum signal level to include (1, 3, 8, 9, 10). "
                             "Default 8.")
    parser.add_argument("--out", type=str, default=str(DATA_TYPHOONS),
                        help=f"Output CSV path. Default {DATA_TYPHOONS}")
    args = parser.parse_args()

    print(f"=== HKO typhoon scrape ===\n")
    print(f"  source: {HKO_TC_DATA}")
    print(f"  filter: year >= {args.since}, signal >= {args.min_signal}")

    text = fetch_raw_data()
    df = parse_records(text)
    print(f"  parsed: {len(df):,} total signal records (since 1946)")

    out = expand_to_daily(df, args.since, args.min_signal)
    print(f"  output: {len(out)} affected day(s)\n")

    if not out.empty:
        # Print a short summary table
        for _, r in out.iterrows():
            print(f"    {r['date']}  T{r['highest_signal']:<2d}  {r['typhoon_name']}")
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as e:
        print(f"\nNETWORK ERROR: {e}", file=sys.stderr)
        sys.exit(2)
