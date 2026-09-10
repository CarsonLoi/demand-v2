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

    # ---- Part A --------------------------------------------------------------
    ov = pd.DataFrame({c: L.describe_series(d, c)
                       for c in ("demand", "floortables", "demand_per_table")})
    ov.to_csv(OUT / "stats_overall.csv")
    L.describe_by(d, "year", ["demand", "demand_per_table", "floortables"]).to_csv(OUT / "stats_by_year.csv")
    L.describe_by(d, "dow", ["demand", "demand_per_table"]).to_csv(OUT / "stats_by_dow.csv")
    L.describe_by_regime(d).to_csv(OUT / "stats_by_regime.csv")
    L.closure_days(d).to_csv(OUT / "closure_days.csv", index=False)
    P.plot_per_table_by_year(d, OUT / "per_table_by_year.png")

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

    # ---- Part B --------------------------------------------------------------
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
        actual_by_date = dict(zip(d["date"], d["demand"].astype(float)))
        g = L.shap_global(mat, feats, horizons, sample=a.shap_sample)
        g.to_csv(OUT / "shap_importance.csv")
        P.plot_shap_summary(mat, feats, g, OUT / "shap_summary.png")

        loc = L.shap_local(mat, feats, horizon=7, actual_by_date=actual_by_date)
        for label in L.pick_sample_dates():
            if label in loc:
                loc[label].to_csv(OUT / f"shap_local_{label}.csv")
                P.plot_shap_waterfall(loc[label], loc[f"{label}__meta__"], label,
                                      OUT / f"shap_waterfall_{label}.png")

        wf = L.shap_local_walkforward(mat, feats, ["jan2026_a", "jan2026_b"],
                                      horizon=7, actual_by_date=actual_by_date)
        for label in ("jan2026_a", "jan2026_b"):
            if label in wf:
                wf[label].to_csv(OUT / f"shap_local_walkforward_{label}.csv")
                P.plot_shap_waterfall(wf[label], wf[f"{label}__meta__"],
                                      f"{label} (walk-forward)",
                                      OUT / f"shap_waterfall_walkforward_{label}.png")

        print("SHAP top 10 (mean |shap|, pooled horizons):")
        print(g["mean_abs_shap"].head(10).to_string())
        print("\nsample-day accuracy (horizon 7):")
        for label in L.pick_sample_dates():
            if label in loc:
                mt = loc[f"{label}__meta__"].iloc[0]
                print(f"  {label:16s} actual={mt['actual']:7.0f}  pred={mt['prediction']:7.0f}  "
                      f"err={mt['pct_error']:+.1f}%")
        for label in ("jan2026_a", "jan2026_b"):
            if label in wf:
                mt = wf[f"{label}__meta__"].iloc[0]
                print(f"  {label:16s} (walk-fwd) actual={mt['actual']:7.0f}  "
                      f"pred={mt['prediction']:7.0f}  err={mt['pct_error']:+.1f}%")

    _write_report_skeleton(OUT, ma, align, rec)
    print(f"\n-> {OUT}")
    return 0


def _write_report_skeleton(out: Path, ma: dict, align: pd.DataFrame, rec: dict) -> None:
    report = HERE / "EXPLORATION.md"
    if report.exists():
        print(f"  ({report.name} exists -- not overwriting)")
        return
    report.write_text(
        "# Long-history pipeline -- data exploration\n\n"
        "_Generated skeleton. Fill every `TODO(prose)` with the reading of the "
        "referenced table/chart._\n\n"
        "## A1 Descriptive statistics\n\nTODO(prose) -- see `output/stats_*.csv`.\n\n"
        "## A2 Monthly seasonality\n\n"
        f"Mean year-to-year shape correlation: **{ma['mean_offdiag_corr']:.3f}**. "
        f"Unstable months: {ma['unstable_months'] or 'none'}. "
        "![](output/monthly_seasonality.png)\n\nTODO(prose)\n\n"
        "## A3 Holiday impact\n\n"
        f"```\n{align[['n_clean_years','cv_per_table','verdict']].to_string()}\n```\n"
        "![](output/holiday_impact.png) ![](output/cny_trough_by_year.png)\n\nTODO(prose)\n\n"
        "## A4 COVID\n\n"
        f"2023 per-table {rec['jan_per_table']:.1f} -> {rec['dec_per_table']:.1f}; "
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
        "- SHAP models trained on the full history (`as_of` = data max) -- this "
        "explains what the deployed model learned, not held-out accuracy.\n"
        "- Pre-holiday baseline is lead-in only (normal days before the window).\n"
        "- Nothing here changes the model; redundancy/contamination is recorded, not acted on.\n",
        encoding="utf-8")
    print(f"  wrote skeleton {report.name}")


if __name__ == "__main__":
    sys.exit(main())
