"""blend.py — honest ensemble blending: per-(DOW,horizon) top-2 selection plus
fixed-date holiday anchoring.

All calibration here is done on a VALIDATION window (recent data with known
labels), never on the forecast target. The two techniques:

1. Selection: for each (day-of-week, horizon) cell, pick the 2 base models with
   lowest validation MAPE and average them. Beats equal-weighting because model
   strengths vary by weekday and lead time.

2. Holiday anchor: for fixed-date holiday windows (Labour, Golden Week, New Year,
   Christmas, Ching Ming), blend the model prediction toward
       lag_365 * recent_YoY_growth
   because base models systematically under-predict holiday spikes. Lunar
   holidays (CNY, Mid-Autumn) are left alone — lag_365 doesn't align for them.

Used by forecast.py when --blend selection is passed. See research/ for the
backtests that validated these choices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "v2"))
from _shared import HOLDOUT_DAYS, HOLIDAY_ANCHORS, HOLIDAY_WINDOWS  # noqa: E402

MODEL_NAMES = ["lgbm_l2", "lgbm_q", "xgb", "cat", "lgbm_bag", "v3_2stage"]

# Fixed-date holidays fall on the same month/day each year, so lag_365 aligns
# to the same holiday phase. Lunar holidays (CNY, MidAutumn) move and are excluded.
FIXED_HOLIDAYS = {"NewYear", "Christmas", "ChingMing", "Labour", "GoldenWeek"}


def mape(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = (y > 0) & ~np.isnan(p)
    return float(np.mean(np.abs(y[m] - p[m]) / y[m])) if m.any() else np.nan


def yoy_growth(demand: pd.DataFrame, before_date: pd.Timestamp, window: int = 28) -> float:
    """Trailing-window YoY growth ratio from data strictly before `before_date`."""
    d = demand[demand["date"] < before_date]
    if d.empty:
        return 1.0
    end = d["date"].max()
    recent = d[d["date"] > end - pd.Timedelta(days=window)]["demand"].mean()
    ya = end - pd.Timedelta(days=365)
    prior = d[(d["date"] > ya - pd.Timedelta(days=window)) & (d["date"] <= ya)]["demand"].mean()
    if prior and prior > 0 and not np.isnan(recent):
        return float(recent / prior)
    return 1.0


def fixed_influence_mask(dates, post: int) -> np.ndarray:
    """True for dates within [window_start, window_end + post] of any fixed-date
    holiday anchor."""
    out = []
    for date in pd.to_datetime(dates):
        hit = False
        for name in FIXED_HOLIDAYS:
            ws, we = HOLIDAY_WINDOWS[name]
            for a in HOLIDAY_ANCHORS[name]:
                if ws <= (date - a).days <= we + post:
                    hit = True
                    break
            if hit:
                break
        out.append(hit)
    return np.array(out)


def build_selection_table(val_wide: pd.DataFrame, pool=MODEL_NAMES) -> dict:
    """Return {(dow, horizon): [top-2 model names]} from validation predictions."""
    table = {}
    for (dow, h), s in val_wide.groupby(["dow", "horizon"]):
        if len(s) < 3:
            continue
        scores = {n: mape(s["y"], s[n]) for n in pool}
        table[(int(dow), int(h))] = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1])[:2]]
    return table


def apply_selection(wide: pd.DataFrame, table: dict, pool=MODEL_NAMES) -> np.ndarray:
    """Blend each row using its (dow, horizon) top-2; fall back to full-pool mean."""
    out = []
    for _, r in wide.iterrows():
        names = table.get((int(r["dow"]), int(r["horizon"])), pool)
        out.append(float(np.mean([r[n] for n in names])))
    return np.array(out)


def calibrate_anchor(val_wide: pd.DataFrame, val_blend: np.ndarray,
                     demand: pd.DataFrame, val_start: pd.Timestamp) -> tuple:
    """Choose (growth_window, post, alpha) minimizing MAPE on validation
    fixed-date-holiday rows. Returns the tuple; alpha=0 means 'no anchoring'."""
    best = (np.inf, (28, 0, 0.0))
    for window in (28, 42, 56):
        Gv = yoy_growth(demand, val_start, window)
        for post in (0, 1, 2, 3):
            vmask = fixed_influence_mask(val_wide["target_date"], post)
            if vmask.sum() < 10:
                continue
            anchor = val_wide["lag_365"].values * Gv
            for alpha in np.linspace(0, 1, 21):
                p = val_blend.copy()
                m = vmask & ~np.isnan(anchor)
                p[m] = (1 - alpha) * val_blend[m] + alpha * anchor[m]
                sc = mape(val_wide["y"].values[vmask], p[vmask])
                if sc < best[0]:
                    best = (sc, (window, post, float(alpha)))
    return best[1]


def apply_anchor(wide: pd.DataFrame, blend: np.ndarray, demand: pd.DataFrame,
                 forecast_start: pd.Timestamp, params: tuple) -> np.ndarray:
    """Apply the calibrated anchor to fixed-date holiday rows of a forecast."""
    window, post, alpha = params
    if alpha <= 0:
        return blend
    G = yoy_growth(demand, forecast_start, window)
    mask = fixed_influence_mask(wide["target_date"], post)
    anchor = wide["lag_365"].values * G
    out = blend.copy()
    m = mask & ~np.isnan(anchor)
    out[m] = (1 - alpha) * blend[m] + alpha * anchor[m]
    return out
