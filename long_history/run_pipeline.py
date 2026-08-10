"""run_pipeline.py — the single entry point for the long-history model.

Runs the steps in the right order and stops at the first failure, so you
cannot accidentally forecast from data that did not pass validation.

THE STEPS
---------
  validate   Check the data file: no gaps, no duplicates, no impossible
             values. A missing day silently corrupts every lag feature, so
             this is a HARD GATE -- nothing else runs if it fails.

  experiment Compare settings (history length x half-life) over held-out
             windows and report which wins. SLOW and only needed when the
             data or the business changes -- not part of a routine run.

  evaluate   Measure how accurate the chosen setup actually is, over a date
             range you pick, at a lead time you pick. Produces charts and a
             per-holiday breakdown.

  forecast   Produce the real 28-day forecast from the latest data.

DEFAULT RUN = validate -> evaluate -> forecast. `experiment` is opt-in.

The feature matrix is cached (config.CACHE_MATRIX), so the first step pays
the ~5-8 minute build and the rest reuse it.

USAGE
-----
    # Everything you normally want
    uv run python long_history/run_pipeline.py

    # Just check the data
    uv run python long_history/run_pipeline.py --steps validate

    # Full run including the settings search
    uv run python long_history/run_pipeline.py --steps validate experiment evaluate forecast

    # Evaluate a specific range at a 7-day lead
    uv run python long_history/run_pipeline.py --steps evaluate \
        --eval-start 2026-01-01 --eval-end 2026-05-28 --eval-n 6

    # Forecast from a past date (backtest-style rerun)
    uv run python long_history/run_pipeline.py --steps forecast --run-date 2026-04-30
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent   # long_history/ -- this package owns everything
sys.path.insert(0, str(ROOT))

import config as C

# step name -> script, relative to this package
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


def run(cmd: list[str], label: str) -> int:
    print("\n" + "=" * 78)
    print(f"STEP: {label}")
    print("=" * 78, flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=str(ROOT))
    dt = time.time() - t0
    if r.returncode:
        print(f"\n!! {label} FAILED (exit {r.returncode}) after {dt:.0f}s")
    else:
        print(f"\n-- {label} ok  [{dt:.0f}s]")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=DEFAULT_STEPS,
                    help=f"steps to run, in order (default: {' '.join(DEFAULT_STEPS)})")
    ap.add_argument("--eval-start", default="2026-01-01", help="evaluate: first target date")
    ap.add_argument("--eval-end", default=None,
                    help="evaluate: last target date (default = last date in the data)")
    ap.add_argument("--eval-n", type=int, default=6,
                    help="evaluate: window depth; lead = n+1 (default 6 -> 7-day lead)")
    ap.add_argument("--eval-mode", choices=["fixed_lead", "rolling_window"],
                    default="rolling_window")
    ap.add_argument("--run-date", default=None, help="forecast: run date (default last row)")
    ap.add_argument("--half-life", type=int, default=None,
                    help=f"override recency half-life (default {C.DEFAULT_HALF_LIFE})")
    ap.add_argument("--no-anchor", action="store_true", help="skip the CNY correction")
    ap.add_argument("--quick", action="store_true", help="experiment: sampled horizons")
    a = ap.parse_args()

    eval_end = a.eval_end
    if eval_end is None:
        import pandas as pd
        d = pd.read_csv(ROOT / C.DATA_FILE, parse_dates=["date"])
        eval_end = str(d["date"].max().date())

    print("=" * 78)
    print("LONG-HISTORY PIPELINE")
    print("=" * 78)
    print(f"  data       : {C.DATA_FILE}")
    print(f"  steps      : {' -> '.join(a.steps)}")
    print(f"  half-life  : {a.half_life or C.DEFAULT_HALF_LIFE}d")
    print(f"  CNY anchor : {'OFF' if a.no_anchor else 'on'}")
    print(f"  matrix cache: {'on' if getattr(C,'CACHE_MATRIX',False) else 'off'}")

    for step in a.steps:
        if step == "validate":
            rc = run([SCRIPTS["validate"]], "validate data")
        elif step == "weather":
            rc = run([SCRIPTS["weather"]], "refresh typhoon records")
        elif step == "experiment":
            cmd = [SCRIPTS["experiment"]] + (["--quick"] if a.quick else [])
            rc = run(cmd, "experiment (settings comparison)")
        elif step == "evaluate":
            cmd = [SCRIPTS["evaluate"], "--start", a.eval_start,
                   "--end", eval_end, "--n", str(a.eval_n), "--mode", a.eval_mode,
                   "--by-lead"]
            if a.half_life:
                cmd += ["--half-life", str(a.half_life)]
            if a.no_anchor:
                cmd += ["--no-anchor"]
            rc = run(cmd, f"evaluate ({a.eval_start}..{eval_end}, lead {a.eval_n+1}d)")
        elif step == "forecast":
            cmd = [SCRIPTS["forecast"]]
            if a.run_date:
                cmd += ["--run-date", a.run_date]
            if a.half_life:
                cmd += ["--half-life", str(a.half_life)]
            if a.no_anchor:
                cmd += ["--no-anchor"]
            rc = run(cmd, "forecast (28 days)")
        elif step == "hourly-analyze":
            rc = run([SCRIPTS["hourly-analyze"]], "hourly pattern analysis")
        elif step == "hourly-split":
            cmd = [SCRIPTS["hourly-split"], "--run-date",
                  a.run_date or eval_end]
            rc = run(cmd, "hourly split")
        else:
            continue

        if rc:
            print(f"\nPipeline stopped at '{step}'. Fix the problem above and rerun.")
            return rc

    print("\n" + "=" * 78)
    print(f"PIPELINE COMPLETE — outputs in {C.OUT_DIR}/")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
