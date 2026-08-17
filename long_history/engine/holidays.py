"""holidays_extended.py — holiday anchors covering 2016-2030.

Production ships anchors for 2024+ only (ANCHOR_YEARS = range(2024, 2030) for
fixed/computed holidays, and 3 hand-entered anchors each for the moving ones).
Extending history to 2016 requires 8 more years of MOVING holiday dates.

Three classes, handled differently:

  FIXED     New Year, Labour, Golden Week, Christmas -- same calendar date
            every year, so they are generated and can never go stale.
  COMPUTED  Easter -- production already computes it for any year via the
            anonymous Gregorian algorithm. Nothing to maintain.
  MOVING    CNY, Mid-Autumn, Dragon Boat, Ching Ming -- lunar or solar-term,
            must be listed explicitly.

The MOVING dates below were cross-checked against `chinese_calendar`'s
holiday blocks rather than typed from memory, and verify_anchors() re-checks
them at runtime. A WRONG anchor is worse than a missing one: it teaches the
model a holiday shape on the wrong days.

NOTE ON CHING MING: it is a solar term and lands on 4, 5 or occasionally 6
April. Production hardcodes 04-04 for every year, which is WRONG for 2018,
2019, 2022 and 2023. Corrected here.
"""
from __future__ import annotations
import datetime as dt
import pandas as pd

# ── MOVING holidays: day 1 of the festival ────────────────────────────────
CNY_DATES = [
    "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25",
    "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29",
    "2026-02-17",
]
MIDAUTUMN_DATES = [
    "2016-09-15", "2017-10-04", "2018-09-24", "2019-09-13", "2020-10-01",
    "2021-09-21", "2022-09-10", "2023-09-29", "2024-09-17", "2025-10-06",
    "2026-09-25",
]
DRAGONBOAT_DATES = [
    "2016-06-09", "2017-05-30", "2018-06-18", "2019-06-07", "2020-06-25",
    "2021-06-14", "2022-06-03", "2023-06-22", "2024-06-10", "2025-05-31",
    "2026-06-19",
]
# Solar term -- NOT always 4 April.
CHINGMING_DATES = [
    "2016-04-04", "2017-04-04", "2018-04-05", "2019-04-05", "2020-04-04",
    "2021-04-04", "2022-04-05", "2023-04-05", "2024-04-04", "2025-04-04",
    "2026-04-04",
]

EXTENDED_YEARS = range(2016, 2031)


def _easter(year: int) -> pd.Timestamp:
    """Western (Gregorian) Easter -- same algorithm production uses."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    month, day = divmod(h + L - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def extended_anchors() -> dict:
    """Anchors for 2016-2030, in the same shape production expects."""
    return {
        "CNY":        [pd.Timestamp(d) for d in CNY_DATES],
        "MidAutumn":  [pd.Timestamp(d) for d in MIDAUTUMN_DATES],
        "DragonBoat": [pd.Timestamp(d) for d in DRAGONBOAT_DATES],
        "ChingMing":  [pd.Timestamp(d) for d in CHINGMING_DATES],
        "Easter":     [_easter(y) for y in EXTENDED_YEARS],
        "GoldenWeek": [pd.Timestamp(f"{y}-10-01") for y in EXTENDED_YEARS],
        "Labour":     [pd.Timestamp(f"{y}-05-01") for y in EXTENDED_YEARS],
        "NewYear":    [pd.Timestamp(f"{y}-01-01") for y in EXTENDED_YEARS],
        "Christmas":  [pd.Timestamp(f"{y}-12-25") for y in EXTENDED_YEARS],
    }


def verify_anchors(verbose: bool = True) -> list:
    """Re-check every MOVING anchor against chinese_calendar's holiday blocks.

    Returns a list of problems (empty = all good). chinese_calendar covers
    2004-2026; anchors outside that range are reported as unverifiable rather
    than silently trusted.
    """
    import chinese_calendar as cc
    from collections import defaultdict

    name_map = {"CNY": "Spring Festival", "MidAutumn": "Mid-autumn Festival",
                "DragonBoat": "Dragon Boat Festival", "ChingMing": "Tomb-sweeping Day"}
    blocks = defaultdict(list)
    d = dt.date(2016, 1, 1)
    while d <= dt.date(2026, 12, 31):
        try:
            on, nm = cc.get_holiday_detail(d)
            if on and nm:
                blocks[(nm, d.year)].append(d)
        except Exception:
            pass
        d += dt.timedelta(days=1)

    problems, unverifiable = [], []
    anchors = extended_anchors()
    for key, cc_name in name_map.items():
        for a in anchors[key]:
            if not (2016 <= a.year <= 2026):
                unverifiable.append(f"{key} {a.date()}")
                continue
            blk = blocks.get((cc_name, a.year))
            if not blk:
                # Mid-Autumn 2020 fell inside National Day and has no block of
                # its own -- a known, legitimate case rather than a data error.
                if key == "MidAutumn" and a.year == 2020:
                    continue
                problems.append(f"{key} {a.date()}: no {cc_name} block in {a.year}")
                continue
            lo, hi = blk[0], blk[-1]
            if not (lo <= a.date() <= hi):
                problems.append(
                    f"{key} {a.date()}: outside {cc_name} block {lo}..{hi}")

    if verbose:
        if problems:
            print("  [anchors] PROBLEMS FOUND:")
            for p in problems:
                print(f"    {p}")
        else:
            print("  [anchors] all moving-holiday anchors verified "
                  "against chinese_calendar (2016-2026)")
        if unverifiable:
            print(f"  [anchors] {len(unverifiable)} anchor(s) outside "
                  f"chinese_calendar's range -- NOT verified, check by hand:")
            print(f"            {', '.join(unverifiable[:8])}"
                  f"{' ...' if len(unverifiable) > 8 else ''}")
    return problems


if __name__ == "__main__":
    a = extended_anchors()
    print("Extended holiday anchors\n" + "=" * 60)
    for k, v in sorted(a.items()):
        print(f"  {k:11s} {len(v):3d} anchors  "
              f"{min(v).date()} .. {max(v).date()}")
    print()
    verify_anchors()
