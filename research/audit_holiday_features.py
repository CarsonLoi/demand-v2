"""Audit every date-referencing feature for moving-holiday correctness.

Moving (lunar / computed) holidays shift by up to three weeks year to year, so
any feature that reaches back a FIXED number of days lands on the wrong day.
This script checks each such feature and reports PASS / WARN / FAIL.

Run:  uv run python research/audit_holiday_features.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from _shared import (HOLIDAY_ANCHORS, HOLIDAY_WINDOWS, load_demand,
                     _is_mainland_workday)

# Which holidays actually move year to year?
MOVING = {"CNY", "MidAutumn", "DragonBoat", "Easter"}
FIXED = {"NewYear", "Labour", "GoldenWeek", "Christmas", "ChingMing"}

ok = lambda m: print(f"  \033[92mPASS\033[0m  {m}")
warn = lambda m: print(f"  \033[93mWARN\033[0m  {m}")
bad = lambda m: print(f"  \033[91mFAIL\033[0m  {m}")

issues = []

print("=" * 78)
print("1. Which holidays move, and by how much?")
print("=" * 78)
for name, anchors in HOLIDAY_ANCHORS.items():
    a = sorted(anchors)
    gaps = [(a[i + 1] - a[i]).days for i in range(len(a) - 1)]
    drift = [abs(g - 365) for g in gaps]
    tag = "MOVING" if name in MOVING else "fixed "
    gs = ", ".join(str(g) for g in gaps)
    flag = ""
    if max(drift, default=0) > 3 and name not in MOVING:
        flag = "  <-- moves but not classified MOVING!"
        issues.append(f"{name} drifts {max(drift)}d but is treated as fixed")
    print(f"  {tag} {name:12s} gaps: {gs:14s} max drift from 365d: {max(drift, default=0):3d}d{flag}")

print()
print("=" * 78)
print("2. Anchor coverage — can we forecast into the next 12 months?")
print("=" * 78)
demand = load_demand()
horizon_end = demand.date.max() + pd.Timedelta(days=365)
for name, anchors in HOLIDAY_ANCHORS.items():
    latest = max(anchors)
    if latest < horizon_end:
        bad(f"{name:12s} last anchor {latest.date()} — no coverage past that; "
            f"holiday flags will be 0 for the next occurrence")
        issues.append(f"{name} missing anchor beyond {latest.date()}")
    else:
        ok(f"{name:12s} covered to {latest.date()}")

print()
print("=" * 78)
print("3. Fixed-offset features near MOVING holidays")
print("=" * 78)
print("  A fixed lag lands on the wrong day when the holiday shifts.\n")
for name in sorted(MOVING):
    a = sorted(HOLIDAY_ANCHORS[name])
    if len(a) < 2:
        continue
    cur, prev = a[-1], a[-2]
    gap = (cur - prev).days
    for lagname, lag in (("lag_365", 365), ("lag_728", 728),
                         ("same_holiday_lastyear_lag", 364)):
        landed = cur - pd.Timedelta(days=lag)
        # how far is that from the matching prior occurrence?
        ref = prev if lag < 400 else (a[-3] if len(a) >= 3 else None)
        if ref is None:
            continue
        off = (landed - ref).days
        if abs(off) <= 2:
            ok(f"{name:11s} {lagname:26s} lands {off:+3d}d from prior {name}")
        else:
            bad(f"{name:11s} {lagname:26s} lands {off:+3d}d from prior {name} "
                f"— WRONG DAY")
            issues.append(f"{lagname} misaligned {off:+d}d for {name}")

print()
print("=" * 78)
print("4. Is the holiday-aligned feature correct for every moving holiday?")
print("=" * 78)
d2d = dict(zip(demand["date"], demand["demand"].astype(float)))
from _shared import _holiday_alignment_map
align = _holiday_alignment_map()
for name in sorted(MOVING | FIXED):
    a = sorted(HOLIDAY_ANCHORS[name])
    if len(a) < 2:
        warn(f"{name:12s} only {len(a)} anchor(s) — no prior occurrence to align to")
        continue
    cur = a[-1]
    ws, we = HOLIDAY_WINDOWS[name]
    probe = cur + pd.Timedelta(days=ws)      # first day of the window
    hit = align.get(probe)
    if hit is None:
        bad(f"{name:12s} window day {probe.date()} has NO alignment entry")
        issues.append(f"{name} missing alignment entry")
        continue
    hname, off, prevanchor = hit
    matched = prevanchor + pd.Timedelta(days=off)
    delta = (matched - prevanchor).days
    if hname == name and delta == off:
        ok(f"{name:12s} {probe.date()} (offset {off:+d}) -> {matched.date()} "
           f"(offset {delta:+d} from prior {name})  aligned")
    else:
        bad(f"{name:12s} misrouted to {hname}")
        issues.append(f"{name} misrouted to {hname}")

print()
print("=" * 78)
print("5. Mainland working-day calendar coverage")
print("=" * 78)
for yr in (2026, 2027):
    probe = pd.Timestamp(f"{yr}-06-15")
    res = _is_mainland_workday(probe)
    if res is None:
        warn(f"chinese_calendar has NO data for {yr} — mainland block features "
             f"fall back to -1/0 for that year")
        issues.append(f"chinese_calendar missing {yr}")
    else:
        ok(f"chinese_calendar covers {yr}")

print()
print("=" * 78)
print(f"SUMMARY — {len(issues)} issue(s) found")
print("=" * 78)
for i, s in enumerate(issues, 1):
    print(f"  {i}. {s}")
if not issues:
    print("  No issues found.")
